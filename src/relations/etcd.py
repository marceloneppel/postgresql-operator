# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""PostgreSQL etcd DCS relation implementation.

This module owns the relation between Charmed PostgreSQL and Charmed etcd, used as
Patroni's Distributed Configuration Store (DCS) in place of the deprecated Raft DCS.

It is the ONLY module that touches the etcd data-platform-libs internals, so the
v0/v1 requirer choice (see the fresh-cluster plan, Task 1 spike) is swappable here
without affecting any other module.
"""

import logging
from functools import cached_property
from typing import TYPE_CHECKING

from charmlibs.interfaces.tls_certificates import (
    CertificateRequestAttributes,
    TLSCertificatesRequiresV4,
)
from charms.data_platform_libs.v0.data_interfaces import EtcdRequires
from ops import EventBase, EventSource, Object, Relation
from single_kernel_postgresql.config.literals import Substrates
from single_kernel_postgresql.utils import render_file

from constants import (
    ETCD_CA_FILE,
    ETCD_CLIENT_CERT_FILE,
    ETCD_CLIENT_CERTIFICATES_RELATION,
    ETCD_CLIENT_KEY_FILE,
    ETCD_RELATION,
    PATRONI_CONF_PATH,
)

if TYPE_CHECKING:
    from charm import PostgresqlOperatorCharm

logger = logging.getLogger(__name__)


def parse_etcd_hosts(endpoints: str | None) -> list[str]:
    """Turn an etcd `endpoints`/`uris` string into a sorted Patroni `hosts` list.

    Accepts comma-separated `ip:port` (from `endpoints`) or `scheme://ip:port`
    (from `uris`); the scheme is stripped because Patroni's `etcd3.hosts` wants
    bare `ip:port` and the protocol is set separately.
    """
    if not endpoints:
        return []
    hosts = []
    for raw in endpoints.split(","):
        host = raw.strip()
        if not host:
            continue
        if "://" in host:
            host = host.split("://", 1)[1]
        hosts.append(host)
    return sorted(hosts)


class EtcdReadyEvent(EventBase):
    """Emitted when etcd is ready to be used as the DCS (endpoints + ca + username)."""


class EtcdEndpointsChangedEvent(EventBase):
    """Emitted when the etcd endpoints change (re-render hosts and reload Patroni)."""


# Task 1 spike decision: use data-platform-libs *v0* ``EtcdRequires``. The etcd charm
# runs the v1 provider but detects a v0 requirer and writes a v0-compatible databag.
# ``fetch_relation_field`` transparently resolves the secret-wrapped fields
# (``username``/``tls-ca``/``uris``/``password``), so readers below treat every field
# uniformly. mTLS is mandatory; ``set_mtls_cert`` writes the app databag (leader-only).
class PostgreSQLEtcd(Object):
    """Owns the etcd DCS relation and its dedicated client-certificate relation."""

    on_etcd_ready = EventSource(EtcdReadyEvent)
    on_etcd_endpoints_changed = EventSource(EtcdEndpointsChangedEvent)

    def __init__(self, charm: "PostgresqlOperatorCharm"):
        super().__init__(charm, ETCD_RELATION)
        self.charm = charm

        # The etcd charm derives our RBAC username from the related application name
        # (this charm's app name), which equals this value, so it doubles as the
        # expected username. The client certificate itself is CN-less: etcd's gRPC
        # gateway -- the only etcd API Patroni's etcd3 backend speaks -- rejects client
        # certs that carry a CommonName, so identity is carried by username/password
        # instead. A SAN is required for the certificate request to be valid.
        self._common_name = self.charm.app.name

        self.certificate = TLSCertificatesRequiresV4(
            self.charm,
            ETCD_CLIENT_CERTIFICATES_RELATION,
            certificate_requests=[
                CertificateRequestAttributes(sans_dns=frozenset({self._common_name})),
            ],
        )

        self._etcd = EtcdRequires(
            self.charm,
            ETCD_RELATION,
            prefix=self.prefix,
            mtls_cert=None,
        )

        self.framework.observe(self._etcd.on.etcd_ready, self._on_etcd_ready)
        self.framework.observe(self._etcd.on.endpoints_changed, self._on_endpoints_changed)
        self.framework.observe(
            self.certificate.on.certificate_available, self._on_certificate_available
        )
        self.framework.observe(
            self.charm.on[ETCD_RELATION].relation_broken, self._on_relation_broken
        )
        self.framework.observe(
            self.charm.on[ETCD_CLIENT_CERTIFICATES_RELATION].relation_broken,
            self._on_relation_broken,
        )

    @cached_property
    def _relation(self) -> Relation | None:
        return self.model.get_relation(ETCD_RELATION)

    @property
    def prefix(self) -> str:
        """App-unique etcd keyspace; doubles as Patroni's ``namespace``."""
        return f"/{self.charm.app.name}/{self.charm.model.uuid}/"

    def _read(self, field: str) -> str | None:
        if not self._relation:
            return None
        return self._etcd.fetch_relation_field(self._relation.id, field)

    @property
    def endpoints(self) -> str | None:
        """The etcd client endpoints string (prefers ``endpoints``, falls back to ``uris``)."""
        return self._read("endpoints") or self._read("uris")

    @property
    def hosts(self) -> list[str]:
        """The etcd endpoints as a sorted list of bare ``ip:port`` for Patroni."""
        return parse_etcd_hosts(self.endpoints)

    @property
    def username(self) -> str | None:
        """The etcd RBAC username published by the etcd charm."""
        return self._read("username")

    @property
    def password(self) -> str | None:
        """The etcd RBAC password published by the etcd charm.

        Patroni authenticates to etcd with username/password (token auth) because the
        gRPC gateway cannot use the client certificate's CommonName for identity.
        """
        return self._read("password")

    @property
    def tls_ca(self) -> str | None:
        """The etcd server CA certificate (PEM) published by the etcd charm."""
        return self._read("tls-ca")

    @property
    def is_ready(self) -> bool:
        """True when etcd has published endpoints + CA + username + password."""
        if not self._relation:
            return False
        return bool(self.endpoints and self.tls_ca and self.username and self.password)

    def _assigned_cert_and_key(self) -> tuple[str | None, str | None]:
        """Return the (leaf cert, private key) PEMs from a single assignment snapshot.

        Read both in one call so a renewal landing between two reads cannot yield a
        mismatched cert/key pair.
        """
        certs, private_key = self.certificate.get_assigned_certificates()
        cert = str(certs[0].certificate) if certs else None
        key = str(private_key) if private_key else None
        return cert, key

    @property
    def client_cert(self) -> str | None:
        """Our assigned leaf client certificate (PEM), or None if not yet issued."""
        return self._assigned_cert_and_key()[0]

    @property
    def client_key(self) -> str | None:
        """Our client private key (PEM), or None if not yet issued."""
        return self._assigned_cert_and_key()[1]

    def publish_mtls_cert(self) -> None:
        """Publish our leaf client cert so etcd can create the RBAC user (CN = username).

        Writes the relation's app databag, so only the leader does this.
        """
        if not self._relation or not self.charm.unit.is_leader():
            return
        leaf = self.client_cert
        if leaf:
            self._etcd.set_mtls_cert(self._relation.id, leaf)

    def write_tls_files(self) -> bool:
        """Materialize etcd CA + client cert + key under PATRONI_CONF_PATH.

        Uses the repo's existing file-render helper (the same one Patroni config and
        peer certs use). Returns True only when all three files were written.
        """
        ca = self.tls_ca
        cert, key = self._assigned_cert_and_key()
        if not (ca and cert and key):
            return False
        render_file(Substrates.VM, f"{PATRONI_CONF_PATH}/{ETCD_CA_FILE}", ca, 0o600)
        render_file(Substrates.VM, f"{PATRONI_CONF_PATH}/{ETCD_CLIENT_CERT_FILE}", cert, 0o600)
        render_file(Substrates.VM, f"{PATRONI_CONF_PATH}/{ETCD_CLIENT_KEY_FILE}", key, 0o600)
        return True

    def template_vars(self) -> dict:
        """The etcd-specific Patroni template variables."""
        return {"etcd_hosts": self.hosts, "etcd_prefix": self.prefix}

    def _on_etcd_ready(self, event: EventBase) -> None:
        self.publish_mtls_cert()
        if self.is_ready:
            self.on_etcd_ready.emit()

    def _on_endpoints_changed(self, event: EventBase) -> None:
        self.on_etcd_endpoints_changed.emit()

    def _on_certificate_available(self, event: EventBase) -> None:
        self.publish_mtls_cert()
        self.write_tls_files()
        if self.is_ready:
            self.charm.update_config()

    def _on_relation_broken(self, event: EventBase) -> None:
        # No auto fall-back to Raft (mTLS DCS is mandatory once selected).
        self.charm._set_dcs_blocked_status()
