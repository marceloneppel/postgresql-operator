# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""DCS backend-selection seam.

Single source of truth for which Distributed Configuration Store (DCS) backend
Patroni uses — Raft (default) or etcd — and for the template variables each backend
needs. Call sites ask this seam instead of branching on `if etcd` inline.

Shaped to be promoted later into `RaftDCS`/`EtcdDCS` strategy classes without
changing any caller (see the fresh-cluster plan's "A -> B later" decision).
"""

from typing import TYPE_CHECKING

from constants import (
    DCS_BACKEND_KEY,
    DCS_BACKEND_TARGET_KEY,
    DCS_ETCD,
    DCS_MIGRATION_KEY,
    DCS_RAFT,
    ETCD_CA_FILE,
    ETCD_CLIENT_CERT_FILE,
    ETCD_CLIENT_KEY_FILE,
    MIGRATION_REPOINTED,
    MIGRATION_STARTED,
)

if TYPE_CHECKING:
    from charm import PostgresqlOperatorCharm

# Phases where the migration target should already drive config rendering.
_TARGET_ACTIVE_PHASES = frozenset({MIGRATION_REPOINTED, MIGRATION_STARTED})


class DCSBackend:
    """Resolves the active DCS backend and its Patroni template variables."""

    def __init__(self, charm: "PostgresqlOperatorCharm"):
        self.charm = charm

    @property
    def active(self) -> str:
        """The active backend.

        During the repoint->start window of a migration, the *target* backend wins so
        that config renders for the new DCS before it is committed to ``dcs-backend``.
        """
        migration_phase = self.charm.app_peer_data.get(DCS_MIGRATION_KEY)
        target = self.charm.app_peer_data.get(DCS_BACKEND_TARGET_KEY)
        if target and migration_phase in _TARGET_ACTIVE_PHASES:
            return target
        return self.charm.app_peer_data.get(DCS_BACKEND_KEY, DCS_RAFT)

    @property
    def is_etcd(self) -> bool:
        """True when Patroni should use the etcd3 backend."""
        return self.active == DCS_ETCD

    @property
    def is_raft(self) -> bool:
        """True when Patroni should use the Raft backend."""
        return self.active == DCS_RAFT

    def template_vars(self) -> dict:
        """Return the DCS variables for ``templates/patroni.yml.j2``.

        The etcd filename vars are always present (even on Raft) so the template
        never references an undefined variable. The etcd ``hosts``/``prefix`` come
        from the etcd relation only when etcd is the active backend.
        """
        etcd_hosts: list[str] = []
        etcd_prefix = ""
        etcd_username = ""
        etcd_password = ""
        if self.is_etcd:
            etcd_hosts = self.charm.etcd.hosts
            etcd_prefix = self.charm.etcd.prefix
            etcd_username = self.charm.etcd.username or ""
            etcd_password = self.charm.etcd.password or ""
        return {
            "dcs_backend": self.active,
            # Patroni's etcd3 backend ignores the ``etcd3.namespace`` key; it derives the
            # keyspace from the *top-level* ``namespace`` + ``scope``. So the per-app etcd
            # prefix (which is also the granted RBAC range) must drive the top-level
            # namespace, otherwise Patroni reads/writes outside the granted range and etcd
            # denies every request.
            "patroni_namespace": etcd_prefix if self.is_etcd else "/db/",
            "etcd_hosts": etcd_hosts,
            "etcd_prefix": etcd_prefix,
            "etcd_username": etcd_username,
            "etcd_password": etcd_password,
            "etcd_ca_file": ETCD_CA_FILE,
            "etcd_client_cert_file": ETCD_CLIENT_CERT_FILE,
            "etcd_client_key_file": ETCD_CLIENT_KEY_FILE,
        }
