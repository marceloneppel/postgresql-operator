# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Unit tests for the DCS backend-selection seam."""

from unittest.mock import MagicMock

from src.constants import (
    DCS_ETCD,
    DCS_RAFT,
    ETCD_CA_FILE,
    ETCD_CLIENT_CERT_FILE,
    ETCD_CLIENT_KEY_FILE,
)
from src.dcs import DCSBackend


def make_charm(active=None, etcd_hosts=None, etcd_prefix=""):
    charm = MagicMock()
    charm.app_peer_data = {} if active is None else {"dcs-backend": active}
    charm.etcd.hosts = etcd_hosts or []
    charm.etcd.prefix = etcd_prefix
    return charm


class TestDCSBackend:
    def test_defaults_to_raft(self):
        dcs = DCSBackend(make_charm())
        assert dcs.active == DCS_RAFT
        assert dcs.is_raft is True
        assert dcs.is_etcd is False

    def test_reads_etcd_from_peer_data(self):
        dcs = DCSBackend(make_charm(active=DCS_ETCD))
        assert dcs.active == DCS_ETCD
        assert dcs.is_etcd is True
        assert dcs.is_raft is False

    def test_template_vars_raft_has_empty_etcd_fields(self):
        dcs = DCSBackend(make_charm())
        v = dcs.template_vars()
        assert v["dcs_backend"] == DCS_RAFT
        assert v["etcd_hosts"] == []
        assert v["etcd_prefix"] == ""
        # Raft keeps Patroni's historical top-level namespace.
        assert v["patroni_namespace"] == "/db/"
        # File-name vars are always present so the template never NameErrors.
        assert v["etcd_ca_file"] == ETCD_CA_FILE
        assert v["etcd_client_cert_file"] == ETCD_CLIENT_CERT_FILE
        assert v["etcd_client_key_file"] == ETCD_CLIENT_KEY_FILE

    def test_template_vars_etcd_pulls_from_relation(self):
        charm = make_charm(
            active=DCS_ETCD, etcd_hosts=["10.0.0.1:2379"], etcd_prefix="/postgresql/abc/"
        )
        dcs = DCSBackend(charm)
        v = dcs.template_vars()
        assert v["dcs_backend"] == DCS_ETCD
        assert v["etcd_hosts"] == ["10.0.0.1:2379"]
        assert v["etcd_prefix"] == "/postgresql/abc/"
        # Etcd drives Patroni's top-level namespace from the per-app prefix so the keys
        # Patroni reads/writes fall inside the granted RBAC range.
        assert v["patroni_namespace"] == "/postgresql/abc/"


def test_active_follows_migration_target_during_repoint():
    charm = make_charm(active="raft")
    charm.app_peer_data["dcs-backend-target"] = "etcd"
    charm.app_peer_data["dcs-migration"] = "repointed"
    dcs = DCSBackend(charm)
    assert dcs.active == DCS_ETCD


def test_active_follows_migration_target_during_started():
    charm = make_charm(active="raft")
    charm.app_peer_data["dcs-backend-target"] = "etcd"
    charm.app_peer_data["dcs-migration"] = "started"
    dcs = DCSBackend(charm)
    assert dcs.active == DCS_ETCD


def test_active_ignores_target_before_repoint():
    charm = make_charm(active="raft")
    charm.app_peer_data["dcs-backend-target"] = "etcd"
    charm.app_peer_data["dcs-migration"] = "preparing"
    dcs = DCSBackend(charm)
    assert dcs.active == "raft"
