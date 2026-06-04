# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Raft -> etcd DCS migration orchestrator.

Drives a leader-coordinated, brief-downtime cutover via peer-data phase flags. Every
unit reacts to phase changes in ``peer-relation-changed`` so a reboot mid-migration
resumes from the recorded phase. Reverts to Raft on any failure before commit; Raft
journals are deleted only at the final commit step (point of no return).
"""

import logging
from typing import TYPE_CHECKING

from tenacity import Retrying, stop_after_delay, wait_fixed

from constants import (
    DCS_BACKEND_KEY,
    DCS_BACKEND_TARGET_KEY,
    DCS_ETCD,
    DCS_MIGRATION_KEY,
    MIGRATION_ACK_KEY,
    MIGRATION_DONE,
    MIGRATION_PHASES,
    MIGRATION_PREPARING,
    MIGRATION_REPOINTED,
    MIGRATION_RESUME_PENDING_KEY,
    MIGRATION_STARTED,
    MIGRATION_STOPPED,
)

if TYPE_CHECKING:
    from charm import PostgresqlOperatorCharm

logger = logging.getLogger(__name__)


class DCSMigration:
    """Owns the Raft->etcd migration state machine."""

    def __init__(self, charm: "PostgresqlOperatorCharm"):
        self.charm = charm

    @property
    def phase(self) -> str | None:
        """The current migration phase, or ``None`` when no migration is in progress."""
        return self.charm.app_peer_data.get(DCS_MIGRATION_KEY)

    @property
    def in_progress(self) -> bool:
        """True while a Raft->etcd migration is underway."""
        return self.phase is not None

    def reconcile(self) -> bool:
        """Drive the migration forward on each hook; return True if one is in progress.

        Returning True signals the caller to skip its normal reconfigure flow (which would
        fight the migration by starting Patroni or handling process failures). When a
        replica is not yet satisfied it simply does not ack, and a later hook retries.
        """
        if self.in_progress:
            try:
                self.apply_phase()
                self.advance_if_ready()
            except Exception as e:
                logger.exception("DCS migration step failed")
                if self.phase != MIGRATION_DONE:
                    self.rollback(str(e))
            return True
        if self.charm.dcs.is_etcd:
            # Post-migration housekeeping (per-unit, idempotent): commit() clears only the
            # leader's own ack, so each unit prunes its now-orphaned ack and stale Raft
            # journals on its next hook.
            self.charm._patroni.remove_raft_journals()
            self.charm.unit_peer_data.pop(MIGRATION_ACK_KEY, None)
        elif self.charm.dcs.is_raft:
            # Post-rollback housekeeping: a rollback whose in-hook unpause could not finish
            # leaves the cluster in the maintenance mode set by the `preparing` phase. Lift
            # it here so a later hook elects a primary instead of hanging on "awaiting for
            # member to start".
            self._resume_after_rollback()
        return False

    def check_preconditions(self) -> tuple[bool, str]:
        """Validate the cluster can safely begin a Raft->etcd migration."""
        if self.in_progress:
            return False, "a migration is already in progress"
        if not self.charm.dcs.is_raft:
            return False, "cluster is already on etcd (or not on Raft)"
        if not self.charm.etcd.is_ready:
            return False, "etcd relation is not ready (need endpoints, CA and username)"
        if self.charm.etcd.username != self.charm.etcd._common_name:
            return False, (
                "etcd username does not match the client certificate CN "
                f"({self.charm.etcd.username!r} != {self.charm.etcd._common_name!r})"
            )
        if not self.charm._patroni.has_raft_quorum():
            return False, "Raft cluster does not currently have quorum"
        if not self.charm._patroni.are_all_members_ready():
            return False, "not all cluster members are ready"
        return True, ""

    def begin(self) -> None:
        """Phase 0 -> preparing: record the target backend and start the machine."""
        self.charm.app_peer_data[DCS_BACKEND_TARGET_KEY] = DCS_ETCD
        self.charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_PREPARING
        logger.info("DCS migration started (target=etcd, phase=preparing)")

    def _next_phase(self) -> str | None:
        if self.phase is None:
            return None
        idx = MIGRATION_PHASES.index(self.phase)
        if idx + 1 >= len(MIGRATION_PHASES):
            return None
        return MIGRATION_PHASES[idx + 1]

    def _all_units_acked(self) -> bool:
        """True when every peer unit (incl. leader) acked the current phase."""
        relation = self.charm._peers
        if relation is None:
            return False
        expected = self.phase
        # Read every unit's ack uniformly through the peer databag; the local unit's
        # bucket here is the same one ack_phase() writes via charm.unit_peer_data.
        units = [self.charm.unit, *relation.units]
        return all(relation.data[unit].get(MIGRATION_ACK_KEY) == expected for unit in units)

    def ack_phase(self) -> None:
        """Publish this unit's ack for the current phase."""
        if self.phase is not None:
            self.charm.unit_peer_data[MIGRATION_ACK_KEY] = self.phase

    def apply_phase(self) -> None:
        """Perform this unit's local action for the current phase, then ack.

        Idempotent: re-running for the same phase is safe (supports reboot resume).
        """
        phase = self.phase
        if phase == MIGRATION_PREPARING:
            # Maintenance mode is cluster-wide; only the leader needs to set it.
            if self.charm.unit.is_leader():
                self.charm._patroni.pause()
        elif phase == MIGRATION_STOPPED:
            # Stop the topology observer before Patroni: with the REST API gone it would
            # crash and be respawned every hook, storming peer-relation-changed.
            self.charm._observer.stop_observer()
            self.charm._patroni.stop_patroni()
        elif phase == MIGRATION_REPOINTED:
            # The DCS seam already resolves to the etcd target in this phase, so stage the
            # etcd TLS files and re-render Patroni's config onto etcd while it is stopped;
            # the `started` phase then brings Patroni up on the new backend.
            # If the etcd client cert/key/CA cannot be materialized, abort before touching
            # Patroni's config: starting it against a backend it cannot authenticate to
            # would fail messily later. Raising here routes through the pre-commit rollback.
            if not self.charm.etcd.write_tls_files():
                raise RuntimeError("etcd TLS files unavailable; aborting repoint to etcd")
            self.charm.update_config()
            # Drop the stale paused dynamic-config cache (Patroni is stopped here) so the
            # `started` phase bootstraps the empty etcd keyspace unpaused. A carried-over
            # pause:true would deadlock: paused => Patroni never writes /config => the
            # unpause can never take effect (see _resume_with_retry).
            self.charm._patroni.remove_dynamic_config_cache()
        elif phase == MIGRATION_STARTED:
            # Former Raft leader starts first to seed the fresh etcd keyspace and claim
            # the leader key; replicas wait until that primary is visible before starting
            # AND acking, so the leader cannot commit while a replica is still down.
            if self.charm.unit.is_leader():
                self.charm._patroni.start_patroni()
                self._resume_with_retry()
            else:
                if self.charm._patroni.get_primary() is None:
                    return  # no primary yet: skip ack, retry on next reconcile
                self.charm._patroni.start_patroni()
        self.ack_phase()

    def _resume_with_retry(self) -> None:
        """Unpause Patroni, retrying while its REST API settles after a (re)start.

        ``start_patroni()`` returns before Patroni's REST API is reachable. On the forward
        ``started`` phase the ``repointed`` phase removed the stale ``patroni.dynamic.json``
        cache, so Patroni bootstraps the empty etcd keyspace unpaused and writes ``/config``;
        until that key exists ``PATCH /config`` returns 503, so a 200 also confirms the leader
        has seeded the keyspace and claimed leadership. The rollback path calls this right
        after restarting Patroni on Raft, where the REST API is likewise briefly unreachable.
        Either way a transient ``ReadTimeout``/503 is expected, so retry until the unpause is
        accepted; a hard failure after the deadline reraises to the caller.
        """
        for attempt in Retrying(stop=stop_after_delay(60), wait=wait_fixed(3), reraise=True):
            with attempt:
                self.charm._patroni.resume()

    def _resume_after_rollback(self) -> None:
        """Lift a lingering maintenance-mode pause left by a deferred rollback.

        ``rollback()`` records ``MIGRATION_RESUME_PENDING_KEY`` when it cannot unpause the
        cluster in-hook (Patroni's REST API still settling after the restart, or PostgreSQL
        not yet back). The pause came from the ``preparing`` phase and persists in the Raft
        DCS, so without lifting it Patroni stays in maintenance mode and never elects a
        primary. Only the leader can clear the cluster-wide pause; it retries here on each
        ``peer-relation-changed``/``update-status`` until the unpause is accepted, then
        clears the flag. A still-failing unpause is swallowed so the hook completes cleanly
        and the flag is left in place for the next retry.
        """
        if not self.charm.unit.is_leader():
            return
        if not self.charm.app_peer_data.get(MIGRATION_RESUME_PENDING_KEY):
            return
        try:
            self._resume_with_retry()
        except Exception:
            logger.exception("Post-rollback unpause still failing; will retry next hook")
            return
        self.charm.app_peer_data.pop(MIGRATION_RESUME_PENDING_KEY, None)
        logger.info("Post-rollback unpause completed; Raft cluster resumed")

    def commit(self) -> None:
        """Point of no return: persist etcd backend, delete Raft data, resume failover."""
        self.charm.app_peer_data[DCS_BACKEND_KEY] = DCS_ETCD
        if self.charm.unit.is_leader():
            self.charm._patroni.resume()
        # Patroni is already running on etcd here; drop the stale journals without
        # stopping it (remove_raft_data() would halt the just-migrated cluster).
        self.charm._patroni.remove_raft_journals()
        # A present watcher relation is a Raft-only tie-breaker; disable it.
        if getattr(self.charm, "watcher_offer", None) is not None:
            self.charm.watcher_offer.disable_watcher()
        self.charm.app_peer_data.pop(DCS_MIGRATION_KEY, None)
        self.charm.app_peer_data.pop(DCS_BACKEND_TARGET_KEY, None)
        self.charm.unit_peer_data.pop(MIGRATION_ACK_KEY, None)
        logger.info("DCS migration committed: now on etcd; Raft data removed")

    def rollback(self, reason: str) -> None:
        """Revert an in-flight migration to Raft. Valid only before commit (no data loss)."""
        logger.warning("Rolling back DCS migration to Raft: %s", reason)
        if self.charm.unit.is_leader():
            self.charm.app_peer_data.pop(DCS_BACKEND_TARGET_KEY, None)
            self.charm.app_peer_data.pop(DCS_MIGRATION_KEY, None)
        self.charm.unit_peer_data.pop(MIGRATION_ACK_KEY, None)
        # Only revert local Patroni once the active backend is Raft again. The leader
        # cleared the target keys above, so dcs.active resolves to Raft for it. A
        # non-leader cannot clear those app-level keys, so during the repointed/started
        # window dcs.active still resolves to etcd -- rendering now would start Patroni
        # on the very backend we are abandoning. Such a unit defers: once the leader
        # clears the keys, the normal peer-relation-changed/update_status flow re-renders
        # Raft and restarts Patroni on its next hook.
        if self.charm.dcs.is_raft:
            # Best-effort local recovery: re-render Raft, restart Patroni, and leave
            # maintenance mode. This block MUST NOT raise. The phase-clear pops above are
            # still uncommitted in ops' in-memory databag (it flushes only on a clean hook),
            # so an exception escaping here would abort the hook, discard those pops, and
            # strand the migration in its failed phase -- which juju would then retry
            # forever. Patroni's REST API is typically still restarting here, so the unpause
            # is retried (a transient ReadTimeout/503 is expected); if recovery still cannot
            # finish this hook, log and defer to the next normal peer-relation-changed/
            # update_status, which re-renders and restarts Patroni once the keys are gone.
            try:
                self.charm.update_config()
                self.charm._patroni.start_patroni()
                if self.charm.unit.is_leader():
                    self._resume_with_retry()
            except Exception:
                logger.exception("Rollback local Patroni recovery deferred to next hook")
                # The cluster is still in the maintenance mode set by `preparing`. Record
                # that so reconcile() retries the unpause on later hooks; otherwise the
                # restarted-on-Raft cluster stays paused and never elects a primary. Only
                # the leader can clear the cluster-wide pause and write app data.
                if self.charm.unit.is_leader():
                    self.charm.app_peer_data[MIGRATION_RESUME_PENDING_KEY] = "true"

    def advance_if_ready(self) -> None:
        """Leader-only: advance to the next phase once all units acked the current one."""
        if not self.charm.unit.is_leader() or self.phase is None:
            return
        if self.phase == MIGRATION_DONE:
            # A previous hook reached DONE but commit() may not have finished; commit()
            # is idempotent, so re-run it to converge (then the migration keys are gone).
            self.commit()
            return
        if not self._all_units_acked():
            relation = self.charm._peers
            if relation is not None:
                acks = {
                    unit.name: relation.data[unit].get(MIGRATION_ACK_KEY)
                    for unit in [self.charm.unit, *relation.units]
                }
                logger.info("DCS migration phase=%s waiting on acks: %s", self.phase, acks)
            return
        nxt = self._next_phase()
        if nxt is not None:
            self.charm.app_peer_data[DCS_MIGRATION_KEY] = nxt
            logger.info("DCS migration advancing to phase=%s", nxt)
            if nxt == MIGRATION_DONE:
                self.commit()
