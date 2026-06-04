# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Unit tests for the Raft->etcd DCS migration orchestrator."""

from unittest.mock import MagicMock

import pytest

from src.constants import (
    DCS_BACKEND_KEY,
    DCS_BACKEND_TARGET_KEY,
    DCS_ETCD,
    DCS_MIGRATION_KEY,
    DCS_RAFT,
    MIGRATION_ACK_KEY,
    MIGRATION_DONE,
    MIGRATION_PAUSED,
    MIGRATION_PREPARING,
    MIGRATION_REPOINTED,
    MIGRATION_RESUME_PENDING_KEY,
    MIGRATION_STARTED,
    MIGRATION_STOPPED,
)
from src.migration import DCSMigration


class _LeaderOnlyDict(dict):
    """A dict that raises on mutation, emulating ops' leader-only app databag."""

    def __setitem__(self, key, value):
        raise RuntimeError("non-leader write to app databag")

    def pop(self, *args, **kwargs):
        raise RuntimeError("non-leader pop from app databag")


def make_charm(
    active=DCS_RAFT,
    etcd_ready=True,
    username="postgresql",
    cn="postgresql",
    all_ready=True,
    quorum=True,
):
    charm = MagicMock()
    charm.app_peer_data = {"dcs-backend": active}
    charm.unit_peer_data = {}
    charm.unit.is_leader.return_value = True
    charm.dcs.active = active
    charm.dcs.is_raft = active == DCS_RAFT
    charm.etcd.is_ready = etcd_ready
    charm.etcd.username = username
    charm.etcd._common_name = cn
    charm._patroni.are_all_members_ready.return_value = all_ready
    charm._patroni.has_raft_quorum.return_value = quorum
    charm._patroni.get_primary.return_value = "postgresql-0"
    charm.etcd.write_tls_files.return_value = True
    return charm


class TestPreconditions:
    def test_ok_when_healthy_raft_and_etcd_ready(self):
        m = DCSMigration(make_charm())
        ok, reason = m.check_preconditions()
        assert ok is True
        assert reason == ""

    def test_refuses_when_already_etcd(self):
        m = DCSMigration(make_charm(active=DCS_ETCD))
        ok, reason = m.check_preconditions()
        assert ok is False
        assert "already" in reason.lower()

    def test_refuses_when_etcd_not_ready(self):
        m = DCSMigration(make_charm(etcd_ready=False))
        ok, reason = m.check_preconditions()
        assert ok is False
        assert "etcd" in reason.lower()

    def test_refuses_when_cn_mismatch(self):
        m = DCSMigration(make_charm(username="postgresql", cn="other"))
        ok, reason = m.check_preconditions()
        assert ok is False
        assert "cn" in reason.lower() or "username" in reason.lower()

    def test_refuses_when_cluster_unhealthy(self):
        m = DCSMigration(make_charm(all_ready=False))
        ok, reason = m.check_preconditions()
        assert ok is False
        assert "ready" in reason.lower()

    def test_refuses_when_no_raft_quorum(self):
        m = DCSMigration(make_charm(quorum=False))
        ok, reason = m.check_preconditions()
        assert ok is False
        assert "quorum" in reason.lower()

    def test_refuses_quorum_before_members_ready(self):
        # Both guards fail; quorum must be reported first (it is the more
        # conservative state, so it gates ahead of the members-ready check).
        m = DCSMigration(make_charm(quorum=False, all_ready=False))
        ok, reason = m.check_preconditions()
        assert ok is False
        assert "quorum" in reason.lower()

    def test_refuses_when_migration_already_in_progress(self):
        # A second action fired during the pre-target phases (where dcs.is_raft is still
        # True) must not be allowed to begin() and reset the phase mid-migration.
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_PREPARING
        m = DCSMigration(charm)
        ok, reason = m.check_preconditions()
        assert ok is False
        assert "in progress" in reason.lower()


class TestPhaseAndInProgress:
    def test_phase_none_and_not_in_progress_by_default(self):
        m = DCSMigration(make_charm())
        assert m.phase is None
        assert m.in_progress is False

    def test_phase_and_in_progress_track_peer_data(self):
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_PREPARING
        m = DCSMigration(charm)
        assert m.phase == MIGRATION_PREPARING
        assert m.in_progress is True


class TestBegin:
    def test_begin_sets_target_and_preparing_phase(self):
        charm = make_charm()
        m = DCSMigration(charm)
        m.begin()
        assert charm.app_peer_data[DCS_BACKEND_TARGET_KEY] == DCS_ETCD
        assert charm.app_peer_data[DCS_MIGRATION_KEY] == MIGRATION_PREPARING


class TestUnitPhaseActions:
    def test_apply_preparing_pauses_on_leader(self):
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_PREPARING
        m = DCSMigration(charm)
        m.apply_phase()
        charm._patroni.pause.assert_called_once()

    def test_apply_stopped_stops_patroni(self):
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_STOPPED
        m = DCSMigration(charm)
        m.apply_phase()
        charm._patroni.stop_patroni.assert_called_once()

    def test_apply_repointed_writes_etcd_files_and_renders(self):
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_REPOINTED
        m = DCSMigration(charm)
        m.apply_phase()
        charm.etcd.write_tls_files.assert_called_once()
        charm.update_config.assert_called_once()
        # Stale paused dynamic-config cache must be dropped so the started phase
        # bootstraps the empty etcd keyspace unpaused (avoids the PATCH /config 503 deadlock).
        charm._patroni.remove_dynamic_config_cache.assert_called_once()

    def test_apply_repointed_raises_when_tls_files_missing(self):
        # If the etcd client cert/key/CA cannot be materialized, repoint must abort before
        # reconfiguring Patroni onto etcd so reconcile() routes through the rollback path.
        charm = make_charm()
        charm.etcd.write_tls_files.return_value = False
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_REPOINTED
        m = DCSMigration(charm)
        with pytest.raises(RuntimeError):
            m.apply_phase()
        charm.update_config.assert_not_called()
        charm._patroni.remove_dynamic_config_cache.assert_not_called()

    def test_apply_preparing_does_not_pause_on_non_leader(self):
        charm = make_charm()
        charm.unit.is_leader.return_value = False
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_PREPARING
        m = DCSMigration(charm)
        m.apply_phase()
        charm._patroni.pause.assert_not_called()

    def test_unit_acks_current_phase(self):
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_STOPPED
        m = DCSMigration(charm)
        m.ack_phase()
        assert charm.unit_peer_data[MIGRATION_ACK_KEY] == MIGRATION_STOPPED


class TestNextPhase:
    def test_none_when_no_phase(self):
        m = DCSMigration(make_charm())
        assert m._next_phase() is None

    def test_follows_phase_order(self):
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_PREPARING
        m = DCSMigration(charm)
        assert m._next_phase() == MIGRATION_PAUSED

    def test_none_at_last_phase(self):
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_DONE
        m = DCSMigration(charm)
        assert m._next_phase() is None


class TestLeaderAdvance:
    def test_advances_when_all_acked(self):
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_PREPARING
        m = DCSMigration(charm)
        m._all_units_acked = MagicMock(return_value=True)
        m.advance_if_ready()
        assert charm.app_peer_data[DCS_MIGRATION_KEY] == MIGRATION_PAUSED

    def test_does_not_advance_when_not_all_acked(self):
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_PREPARING
        m = DCSMigration(charm)
        m._all_units_acked = MagicMock(return_value=False)
        m.advance_if_ready()
        assert charm.app_peer_data[DCS_MIGRATION_KEY] == MIGRATION_PREPARING

    def test_does_not_advance_when_not_leader(self):
        charm = make_charm()
        charm.unit.is_leader.return_value = False
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_PREPARING
        m = DCSMigration(charm)
        m._all_units_acked = MagicMock(return_value=True)
        m.advance_if_ready()
        assert charm.app_peer_data[DCS_MIGRATION_KEY] == MIGRATION_PREPARING


class TestStartResumeCommit:
    def test_apply_started_starts_leader_first(self):
        charm = make_charm()
        charm.unit.name = "postgresql/0"
        charm._patroni.get_primary.return_value = "postgresql-0"
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_STARTED
        m = DCSMigration(charm)
        m.apply_phase()
        charm._patroni.start_patroni.assert_called_once()

    def test_apply_started_leader_resumes_to_clear_carried_over_pause(self):
        # Patroni restores the paused dynamic config onto the fresh etcd keyspace, so the
        # leader must resume to run leader election and become primary.
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_STARTED
        m = DCSMigration(charm)
        m.apply_phase()
        charm._patroni.resume.assert_called_once()

    def test_apply_started_leader_starts_without_primary(self):
        # The leader must start unconditionally to seed etcd, even with no primary visible.
        charm = make_charm()
        charm._patroni.get_primary.return_value = None
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_STARTED
        m = DCSMigration(charm)
        m.apply_phase()
        charm._patroni.start_patroni.assert_called_once()
        charm._patroni.resume.assert_called_once()
        assert charm.unit_peer_data[MIGRATION_ACK_KEY] == MIGRATION_STARTED

    def test_apply_started_replica_waits_for_primary(self):
        # A replica must not start until the former leader is primary in etcd.
        charm = make_charm()
        charm.unit.is_leader.return_value = False
        charm._patroni.get_primary.return_value = None
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_STARTED
        m = DCSMigration(charm)
        m.apply_phase()
        charm._patroni.start_patroni.assert_not_called()
        charm._patroni.resume.assert_not_called()
        assert MIGRATION_ACK_KEY not in charm.unit_peer_data

    def test_apply_started_replica_starts_once_primary_visible(self):
        # Only the leader resumes; a replica just starts once a primary is visible.
        charm = make_charm()
        charm.unit.is_leader.return_value = False
        charm._patroni.get_primary.return_value = "postgresql-0"
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_STARTED
        m = DCSMigration(charm)
        m.apply_phase()
        charm._patroni.start_patroni.assert_called_once()
        charm._patroni.resume.assert_not_called()

    def test_commit_sets_backend_and_removes_raft(self):
        charm = make_charm()
        m = DCSMigration(charm)
        m.commit()
        assert charm.app_peer_data[DCS_BACKEND_KEY] == DCS_ETCD
        # Journals are dropped without stopping the just-migrated cluster.
        charm._patroni.remove_raft_journals.assert_called_once()
        charm._patroni.remove_raft_data.assert_not_called()
        assert "dcs-migration" not in charm.app_peer_data
        assert "dcs-backend-target" not in charm.app_peer_data

    def test_commit_resumes_failover(self):
        charm = make_charm()
        m = DCSMigration(charm)
        m.commit()
        charm._patroni.resume.assert_called_once()

    def test_commit_non_leader_does_not_resume(self):
        charm = make_charm()
        charm.unit.is_leader.return_value = False
        m = DCSMigration(charm)
        m.commit()
        charm._patroni.resume.assert_not_called()

    def test_commit_disables_present_watcher(self):
        charm = make_charm()
        m = DCSMigration(charm)
        m.commit()
        charm.watcher_offer.disable_watcher.assert_called_once()

    def test_commit_tolerates_absent_watcher(self):
        charm = make_charm()
        charm.watcher_offer = None
        m = DCSMigration(charm)
        m.commit()  # must not raise when no watcher relation is present


class TestRollback:
    def test_leader_rollback_clears_app_and_unit_state(self):
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_STOPPED
        charm.app_peer_data[DCS_BACKEND_TARGET_KEY] = DCS_ETCD
        charm.unit_peer_data[MIGRATION_ACK_KEY] = MIGRATION_STOPPED
        m = DCSMigration(charm)
        m.rollback("boom")
        assert DCS_MIGRATION_KEY not in charm.app_peer_data
        assert DCS_BACKEND_TARGET_KEY not in charm.app_peer_data
        assert MIGRATION_ACK_KEY not in charm.unit_peer_data
        charm.update_config.assert_called_once()
        charm._patroni.start_patroni.assert_called_once()
        charm._patroni.resume.assert_called_once()

    def test_non_leader_rollback_in_pre_target_phase_reverts_locally(self):
        # is_raft is True here (make_charm default), i.e. a pre-target phase
        # (preparing/paused/stopped) where the backend is still Raft, so a non-leader
        # may safely re-render Raft and restart Patroni itself.
        charm = make_charm()
        charm.unit.is_leader.return_value = False
        # A non-leader writing/popping app_peer_data would raise in production; emulate
        # that by making the app databag reject mutation so the test catches a regression.
        charm.app_peer_data = _LeaderOnlyDict()
        charm.unit_peer_data[MIGRATION_ACK_KEY] = MIGRATION_STOPPED
        m = DCSMigration(charm)
        m.rollback("boom")  # must not raise
        assert MIGRATION_ACK_KEY not in charm.unit_peer_data
        charm.update_config.assert_called_once()
        charm._patroni.start_patroni.assert_called_once()
        charm._patroni.resume.assert_not_called()

    def test_non_leader_rollback_in_target_phase_defers_local_revert(self):
        # During repointed/started the app-level target keys are still set and a
        # non-leader cannot clear them, so dcs.active still resolves to etcd (is_raft
        # False). Re-rendering now would start Patroni on the backend we are abandoning,
        # so the unit must clear only its ack and defer to the leader-coordinated revert.
        charm = make_charm(active=DCS_ETCD)  # dcs.is_raft is False
        charm.unit.is_leader.return_value = False
        charm.app_peer_data = _LeaderOnlyDict()
        charm.unit_peer_data[MIGRATION_ACK_KEY] = MIGRATION_REPOINTED
        m = DCSMigration(charm)
        m.rollback("boom")  # must not raise
        assert MIGRATION_ACK_KEY not in charm.unit_peer_data
        charm.update_config.assert_not_called()
        charm._patroni.start_patroni.assert_not_called()
        charm._patroni.resume.assert_not_called()

    def test_leader_rollback_commits_phase_clear_when_patroni_recovery_fails(self):
        # Regression: while Patroni is still restarting after the stopped phase its REST
        # API times out, so the leader's unpause during local recovery raises. That
        # exception must NOT escape rollback(): the phase-clear pops happen first but are
        # still uncommitted in ops' in-memory databag, so a raise here would abort the
        # hook, discard the pops, and strand the migration in 'repointed' (which juju then
        # retries forever). rollback() must clear the state durably and swallow the
        # recovery failure, deferring Patroni recovery to the next normal hook.
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_REPOINTED
        charm.app_peer_data[DCS_BACKEND_TARGET_KEY] = DCS_ETCD
        charm.unit_peer_data[MIGRATION_ACK_KEY] = MIGRATION_REPOINTED
        # Patroni REST unreachable during recovery, injected at both the retried unpause
        # helper and a direct resume() so the test is robust to how recovery unpauses.
        charm._patroni.resume.side_effect = RuntimeError("Patroni REST read timed out")
        m = DCSMigration(charm)
        m._resume_with_retry = MagicMock(side_effect=RuntimeError("Patroni REST read timed out"))
        m.rollback("etcd TLS files unavailable")  # must not raise
        assert DCS_MIGRATION_KEY not in charm.app_peer_data
        assert DCS_BACKEND_TARGET_KEY not in charm.app_peer_data
        assert MIGRATION_ACK_KEY not in charm.unit_peer_data

    def test_leader_rollback_flags_pending_resume_when_unpause_defers(self):
        # The `preparing` phase put the cluster in maintenance mode. When the in-hook
        # unpause cannot complete (Patroni REST still settling), rollback must record a
        # durable resume-pending flag so a later hook lifts the lingering pause. Without
        # it the Raft cluster stays paused -> no primary is elected and every unit hangs
        # on "awaiting for member to start".
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_REPOINTED
        charm.app_peer_data[DCS_BACKEND_TARGET_KEY] = DCS_ETCD
        charm.unit_peer_data[MIGRATION_ACK_KEY] = MIGRATION_REPOINTED
        m = DCSMigration(charm)
        m._resume_with_retry = MagicMock(side_effect=RuntimeError("Patroni REST read timed out"))
        m.rollback("etcd TLS files unavailable")  # must not raise
        assert charm.app_peer_data.get(MIGRATION_RESUME_PENDING_KEY) == "true"

    def test_leader_rollback_does_not_flag_pending_resume_on_success(self):
        # A clean in-hook unpause leaves no pending flag for reconcile() to act on.
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_STOPPED
        charm.app_peer_data[DCS_BACKEND_TARGET_KEY] = DCS_ETCD
        m = DCSMigration(charm)
        m.rollback("boom")
        assert MIGRATION_RESUME_PENDING_KEY not in charm.app_peer_data


class TestReconcile:
    def test_drives_apply_and_advance_when_in_progress(self):
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_PREPARING
        m = DCSMigration(charm)
        m.apply_phase = MagicMock()
        m.advance_if_ready = MagicMock()
        assert m.reconcile() is True
        m.apply_phase.assert_called_once()
        m.advance_if_ready.assert_called_once()

    def test_rolls_back_on_error_before_done(self):
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_STOPPED
        m = DCSMigration(charm)
        m.apply_phase = MagicMock(side_effect=RuntimeError("kaboom"))
        m.rollback = MagicMock()
        assert m.reconcile() is True
        m.rollback.assert_called_once()

    def test_does_not_roll_back_on_error_at_done(self):
        charm = make_charm()
        charm.app_peer_data[DCS_MIGRATION_KEY] = MIGRATION_DONE
        m = DCSMigration(charm)
        m.apply_phase = MagicMock(side_effect=RuntimeError("kaboom"))
        m.rollback = MagicMock()
        assert m.reconcile() is True
        m.rollback.assert_not_called()

    def test_gcs_journals_when_etcd_and_not_in_progress(self):
        charm = make_charm(active=DCS_ETCD)
        charm.dcs.is_etcd = True
        m = DCSMigration(charm)
        assert m.reconcile() is False
        charm._patroni.remove_raft_journals.assert_called_once()

    def test_prunes_orphaned_ack_when_etcd_and_not_in_progress(self):
        # commit() clears only the leader's own ack; each non-leader prunes its stale
        # 'started' ack here on its next post-migration hook.
        charm = make_charm(active=DCS_ETCD)
        charm.dcs.is_etcd = True
        charm.unit_peer_data[MIGRATION_ACK_KEY] = MIGRATION_STARTED
        m = DCSMigration(charm)
        assert m.reconcile() is False
        assert MIGRATION_ACK_KEY not in charm.unit_peer_data

    def test_noop_when_raft_and_not_in_progress(self):
        charm = make_charm()
        charm.dcs.is_etcd = False
        m = DCSMigration(charm)
        m._resume_with_retry = MagicMock()
        assert m.reconcile() is False
        charm._patroni.remove_raft_journals.assert_not_called()
        # No pending-resume flag => reconcile must not poke Patroni's REST API.
        m._resume_with_retry.assert_not_called()

    def test_resumes_pending_rollback_pause_when_raft_and_not_in_progress(self):
        # After a deferred rollback the Raft cluster can be stranded in maintenance mode.
        # The next hook must lift it: reconcile() retries the unpause and clears the flag
        # once the cluster resumes.
        charm = make_charm()
        charm.dcs.is_etcd = False
        charm.app_peer_data[MIGRATION_RESUME_PENDING_KEY] = "true"
        m = DCSMigration(charm)
        m._resume_with_retry = MagicMock()
        assert m.reconcile() is False
        m._resume_with_retry.assert_called_once()
        assert MIGRATION_RESUME_PENDING_KEY not in charm.app_peer_data

    def test_keeps_pending_resume_flag_when_unpause_still_fails(self):
        # If Patroni's REST API is still unreachable the flag must persist so a later
        # hook retries; reconcile() must swallow the failure rather than abort the hook.
        charm = make_charm()
        charm.dcs.is_etcd = False
        charm.app_peer_data[MIGRATION_RESUME_PENDING_KEY] = "true"
        m = DCSMigration(charm)
        m._resume_with_retry = MagicMock(side_effect=RuntimeError("still settling"))
        assert m.reconcile() is False  # must not raise
        assert charm.app_peer_data.get(MIGRATION_RESUME_PENDING_KEY) == "true"

    def test_non_leader_does_not_resume_pending_rollback_pause(self):
        # Only the leader can lift the cluster-wide pause; a non-leader must not try, and
        # must not touch the leader-only app databag.
        charm = make_charm()
        charm.dcs.is_etcd = False
        charm.unit.is_leader.return_value = False
        charm.app_peer_data = _LeaderOnlyDict()
        m = DCSMigration(charm)
        m._resume_with_retry = MagicMock()
        assert m.reconcile() is False  # must not raise
        m._resume_with_retry.assert_not_called()
