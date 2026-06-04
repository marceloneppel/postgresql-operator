# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Integration tests for the live Raft->etcd DCS migration (migrate-dcs-to-etcd action)."""

import logging
import time

import pytest
from pytest_operator.plugin import OpsTest
from tenacity import Retrying, stop_after_attempt, stop_after_delay, wait_fixed

from .helpers import (
    execute_query_on_unit,
    get_leader_unit,
    get_password,
    get_primary,
    get_unit_address,
    run_command_on_unit,
    write_query_on_unit,
)
from .new_relations.helpers import get_application_relation_data

logger = logging.getLogger(__name__)

ETCD = "charmed-etcd"
CERTS = "self-signed-certificates"
PG = "postgresql"
# Second PostgreSQL app, used only by the rollback test so it never disturbs the
# successfully-migrated cluster from the first test.
PG_RB = "postgresql-rb"
# charmed-etcd's client-facing relation endpoint (see test_etcd_dcs.py).
ETCD_CLIENT_ENDPOINT = "etcd-client"
PATRONI_SERVICE = "snap.charmed-postgresql.patroni.service"
# Peer relation carrying the charm's app-scoped migration state (dcs-backend, dcs-migration).
PEER_RELATION = "database-peers"
# On-disk Raft journal directory; the migration deletes it only at the commit step.
RAFT_PATH = "/var/snap/charmed-postgresql/current/etc/patroni/raft"
ROW_COUNT = 1000


async def _integrate_etcd(ops_test: OpsTest, pg_app: str) -> None:
    """Wire the shared etcd + certificates apps into the given PostgreSQL app.

    charmed-etcd exposes three certificate endpoints, so each integration must name
    both endpoints explicitly; a bare integrate(CERTS, ETCD) is ambiguous.
    """
    await ops_test.model.integrate(f"{CERTS}:certificates", f"{pg_app}:etcd-client-certificates")
    await ops_test.model.integrate(f"{ETCD}:{ETCD_CLIENT_ENDPOINT}", f"{pg_app}:etcd")


async def _get_dcs_backend(ops_test: OpsTest, pg_app: str) -> str | None:
    """Committed DCS backend ('raft'/'etcd') from peer app-data, or None if unset."""
    return await get_application_relation_data(ops_test, pg_app, PEER_RELATION, "dcs-backend")


async def _get_migration_phase(ops_test: OpsTest, pg_app: str) -> str | None:
    """In-flight migration phase from peer app-data (None once no migration is underway)."""
    return await get_application_relation_data(ops_test, pg_app, PEER_RELATION, "dcs-migration")


async def _raft_data_absent(ops_test: OpsTest, unit_name: str) -> bool:
    """True when the on-disk Raft journal directory is gone on the given unit."""
    output = await run_command_on_unit(
        ops_test, unit_name, f"test -e {RAFT_PATH} && echo present || echo absent"
    )
    return "absent" in output


async def _primary_unit(ops_test: OpsTest, pg_app: str) -> str:
    """Name of the unit currently holding the Patroni leader (primary) role."""
    leader = await get_leader_unit(ops_test, pg_app)
    return await get_primary(ops_test, leader.name)


async def _write_rows(ops_test: OpsTest, pg_app: str, count: int) -> None:
    """Create a known table on the primary and insert `count` rows into it."""
    primary = await _primary_unit(ops_test, pg_app)
    address = get_unit_address(ops_test, primary)
    password = await get_password(ops_test, database_app_name=pg_app)
    # Adjacent literals concatenate into one two-statement string; psycopg2 runs both
    # in a single execute().
    await write_query_on_unit(
        address,
        password,
        "CREATE TABLE IF NOT EXISTS migration_test (id integer);"
        f"INSERT INTO migration_test SELECT generate_series(1, {count});",
    )


async def _count_rows(ops_test: OpsTest, pg_app: str) -> int:
    """Count the rows previously written by `_write_rows` on the current primary."""
    primary = await _primary_unit(ops_test, pg_app)
    address = get_unit_address(ops_test, primary)
    password = await get_password(ops_test, database_app_name=pg_app)
    result = await execute_query_on_unit(address, password, "SELECT count(*) FROM migration_test;")
    return result[0]


@pytest.mark.abort_on_fail
async def test_migrate_raft_cluster_to_etcd(ops_test: OpsTest, charm, etcd_charm):
    """An existing Raft cluster with data migrates to etcd with no data loss."""
    # 1. Deploy a Raft cluster (no etcd relation yet) and write a known dataset.
    await ops_test.model.deploy(charm, application_name=PG, num_units=3, base="ubuntu@24.04")
    await ops_test.model.wait_for_idle(apps=[PG], status="active", timeout=1800)
    assert await _get_dcs_backend(ops_test, PG) in (None, "raft")
    await _write_rows(ops_test, PG, ROW_COUNT)
    primary_before = await _primary_unit(ops_test, PG)

    # 2. Relate the shared etcd + certificates apps. This alone does NOT migrate.
    # Single-node patched etcd: same constraint as test_etcd_dcs.py -- multi-node token
    # (password) auth needs `--auth-token jwt`; one node's default per-member `simple`
    # token is enough to validate the migration path against the CN-less-cert auth flow.
    await ops_test.model.deploy(
        etcd_charm, application_name=ETCD, base="ubuntu@24.04", num_units=1
    )
    await ops_test.model.deploy(CERTS, channel="1/stable")
    await ops_test.model.integrate(f"{CERTS}:certificates", f"{ETCD}:peer-certificates")
    await ops_test.model.integrate(f"{CERTS}:certificates", f"{ETCD}:client-certificates")
    await ops_test.model.integrate(f"{CERTS}:send-ca-cert", f"{ETCD}:client-cas")
    await _integrate_etcd(ops_test, PG)
    await ops_test.model.wait_for_idle(apps=[PG, ETCD], status="active", timeout=1800)
    # Still on Raft: relating etcd is a prerequisite, not the trigger.
    assert await _get_dcs_backend(ops_test, PG) in (None, "raft")

    # 3. Run the migration action on the leader and time the cutover.
    leader = await get_leader_unit(ops_test, PG)
    t0 = time.monotonic()
    action = await leader.run_action("migrate-dcs-to-etcd")
    await action.wait()
    assert action.results.get("status") == "migration-started"

    # Wait for the leader-coordinated state machine to commit onto etcd.
    for attempt in Retrying(stop=stop_after_delay(20 * 60), wait=wait_fixed(10), reraise=True):
        with attempt:
            assert await _get_dcs_backend(ops_test, PG) == "etcd"
    # Upper bound on the brief write-unavailability window (spec section 7): wall-clock
    # from action start until the backend is committed to etcd. The true downtime is
    # shorter, since PostgreSQL only pauses across the Patroni restart, not the whole run.
    logger.info("Raft->etcd migration committed in <= %.1fs (upper bound)", time.monotonic() - t0)
    await ops_test.model.wait_for_idle(apps=[PG], status="active", timeout=1800)

    # 4. No data loss, leader preserved, etcd-backed, and Raft journals removed.
    assert await _count_rows(ops_test, PG) == ROW_COUNT
    assert await _primary_unit(ops_test, PG) == primary_before
    assert await _get_dcs_backend(ops_test, PG) == "etcd"
    assert await _get_migration_phase(ops_test, PG) is None
    for unit in ops_test.model.applications[PG].units:
        assert await _raft_data_absent(ops_test, unit.name), f"Raft data left on {unit.name}"

    # 5. Failover now flows through etcd: stop Patroni on the primary, expect a new one.
    surviving = next(
        unit.name for unit in ops_test.model.applications[PG].units if unit.name != primary_before
    )
    await run_command_on_unit(ops_test, primary_before, f"sudo systemctl stop {PATRONI_SERVICE}")
    for attempt in Retrying(stop=stop_after_attempt(30), wait=wait_fixed(10), reraise=True):
        with attempt:
            assert await get_primary(ops_test, surviving) != primary_before
    # Restore the stopped member so the model stays healthy for any later checks.
    await run_command_on_unit(ops_test, primary_before, f"sudo systemctl start {PATRONI_SERVICE}")
    await ops_test.model.wait_for_idle(apps=[PG], status="active", timeout=900)


@pytest.mark.abort_on_fail
async def test_rollback_returns_to_raft(ops_test: OpsTest, charm):
    """A failure before commit must return the cluster to healthy Raft with data intact."""
    # Fresh Raft cluster with its own data, sharing the etcd + certificates apps from
    # the first test so we exercise rollback without touching the migrated cluster.
    await ops_test.model.deploy(charm, application_name=PG_RB, num_units=3, base="ubuntu@24.04")
    await ops_test.model.wait_for_idle(apps=[PG_RB], status="active", timeout=1800)
    await _write_rows(ops_test, PG_RB, ROW_COUNT)
    primary_before = await _primary_unit(ops_test, PG_RB)

    await _integrate_etcd(ops_test, PG_RB)
    await ops_test.model.wait_for_idle(apps=[PG_RB, ETCD], status="active", timeout=1800)

    # Arm a deterministic pre-commit failure BEFORE starting the migration, rather than
    # racing a relation removal against the (now fast) cutover. Dropping ONLY the client-
    # certificate relation leaves the etcd relation -- and thus the migration preconditions
    # (endpoints/CA/username/password all come from the etcd relation) -- satisfied, so the
    # action still starts. But the repoint phase can no longer materialize the etcd client
    # cert, so write_tls_files() fails and the orchestrator rolls back to Raft before the
    # commit (point of no return). Removing this relation while the cluster is still on Raft
    # does not block the unit (_set_dcs_blocked_status only blocks an already-etcd cluster).
    await ops_test.juju(
        "remove-relation", f"{CERTS}:certificates", f"{PG_RB}:etcd-client-certificates"
    )
    await ops_test.model.wait_for_idle(apps=[PG_RB], status="active", timeout=1800)

    leader = await get_leader_unit(ops_test, PG_RB)
    action = await leader.run_action("migrate-dcs-to-etcd")
    await action.wait()
    assert action.results.get("status") == "migration-started"

    # The state machine must converge back to Raft: no migration in flight and the
    # committed backend is Raft (never etcd).
    for attempt in Retrying(stop=stop_after_delay(20 * 60), wait=wait_fixed(10), reraise=True):
        with attempt:
            assert await _get_migration_phase(ops_test, PG_RB) is None
            assert await _get_dcs_backend(ops_test, PG_RB) in (None, "raft")
    await ops_test.model.wait_for_idle(apps=[PG_RB], status="active", timeout=1800)

    # Rollback is non-destructive: all rows survive and the primary still serves them.
    assert await _count_rows(ops_test, PG_RB) == ROW_COUNT
    assert await _primary_unit(ops_test, PG_RB) == primary_before
