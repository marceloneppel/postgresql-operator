# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Integration tests for using Charmed etcd as Patroni's DCS (fresh-cluster bootstrap)."""

import logging

import pytest
from pytest_operator.plugin import OpsTest
from tenacity import Retrying, stop_after_attempt, wait_fixed

from .helpers import get_leader_unit, get_primary, run_command_on_unit
from .new_relations.helpers import get_application_relation_data

logger = logging.getLogger(__name__)

ETCD = "charmed-etcd"
CERTS = "self-signed-certificates"
PG = "postgresql"
# charmed-etcd's client-facing relation endpoint; confirm against its metadata if the
# databag guard ever fails after a charmed-etcd refactor.
ETCD_CLIENT_ENDPOINT = "etcd-client"
PATRONICTL = (
    "charmed-postgresql.patronictl "
    "-c /var/snap/charmed-postgresql/current/etc/patroni/patroni.yaml"
)
PATRONI_SERVICE = "snap.charmed-postgresql.patroni.service"


@pytest.mark.abort_on_fail
async def test_fresh_cluster_bootstraps_on_etcd(ops_test: OpsTest, charm, etcd_charm):
    """A fresh PostgreSQL cluster bootstraps with etcd (not Raft) as its DCS."""
    # Single-node etcd: multi-node password (token) auth needs `--auth-token jwt`; one node
    # uses the default per-member `simple` token, which is enough to validate the
    # CN-less-cert + username/password gRPC-gateway auth path the patched etcd publishes.
    await ops_test.model.deploy(
        etcd_charm, application_name=ETCD, base="ubuntu@24.04", num_units=1
    )
    await ops_test.model.deploy(CERTS, channel="1/stable")
    await ops_test.model.deploy(charm, application_name=PG, num_units=3, base="ubuntu@24.04")

    # charmed-etcd exposes three certificate endpoints, so each integration must name
    # both endpoints explicitly; a bare integrate(CERTS, ETCD) is ambiguous.
    await ops_test.model.integrate(f"{CERTS}:certificates", f"{ETCD}:peer-certificates")
    await ops_test.model.integrate(f"{CERTS}:certificates", f"{ETCD}:client-certificates")
    await ops_test.model.integrate(f"{CERTS}:send-ca-cert", f"{ETCD}:client-cas")
    await ops_test.model.integrate(f"{CERTS}:certificates", f"{PG}:etcd-client-certificates")
    await ops_test.model.integrate(f"{ETCD}:{ETCD_CLIENT_ENDPOINT}", f"{PG}:etcd")

    await ops_test.model.wait_for_idle(apps=[PG, ETCD], status="active", timeout=1800)

    # Patroni must report a healthy cluster (backed by etcd).
    leader = await get_leader_unit(ops_test, PG)
    output = await run_command_on_unit(ops_test, leader.name, f"{PATRONICTL} list")
    assert "Leader" in output


@pytest.mark.abort_on_fail
async def test_etcd_databag_shape_guard(ops_test: OpsTest):
    """Codify the Task 1 wire-shape spike so a lib bump can't silently break it."""
    # Our (requirer) side publishes the app-scoped etcd keyspace prefix.
    prefix = await get_application_relation_data(ops_test, PG, "etcd", "prefix")
    assert prefix is not None
    assert prefix.startswith(f"/{PG}/")

    # etcd publishes its client endpoints (carrying the 2379 client port) into the
    # requirer side of the relation -- this is the exact field src/relations/etcd.py
    # reads to build Patroni's etcd host list, so guard it on the PG side.
    endpoints = await get_application_relation_data(ops_test, PG, "etcd", "endpoints")
    assert endpoints is not None
    assert ":2379" in endpoints


async def test_failover_on_etcd(ops_test: OpsTest):
    """Stopping Patroni on the primary elects a new primary through etcd."""
    leader = await get_leader_unit(ops_test, PG)
    old_primary = await get_primary(ops_test, leader.name)

    # Query a unit that stays up so the get-primary action keeps working.
    surviving = next(
        unit.name for unit in ops_test.model.applications[PG].units if unit.name != old_primary
    )

    await run_command_on_unit(ops_test, old_primary, f"sudo systemctl stop {PATRONI_SERVICE}")

    # etcd-backed Patroni must elect a different primary.
    for attempt in Retrying(stop=stop_after_attempt(30), wait=wait_fixed(10), reraise=True):
        with attempt:
            new_primary = await get_primary(ops_test, surviving)
            assert new_primary != old_primary

    # Restore the stopped member so the model stays healthy for later checks.
    await run_command_on_unit(ops_test, old_primary, f"sudo systemctl start {PATRONI_SERVICE}")
    await ops_test.model.wait_for_idle(apps=[PG], status="active", timeout=900)
