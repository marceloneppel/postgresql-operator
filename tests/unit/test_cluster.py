# Copyright 2021 Canonical Ltd.
# See LICENSE file for licensing details.

from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, mock_open, patch, sentinel

import pytest
import requests
from charmlibs import snap
from jinja2 import Template
from ops.testing import Harness
from pysyncobj.utility import UtilityException
from single_kernel_postgresql.config.literals import REWIND_USER, Substrates
from tenacity import (
    RetryError,
    stop_after_delay,
    wait_fixed,
)

from charm import PostgresqlOperatorCharm
from cluster import (
    PATRONI_TIMEOUT,
    Patroni,
    RemoveRaftMemberFailedError,
    SwitchoverFailedError,
    SwitchoverNotSyncError,
)
from constants import (
    DCS_RAFT,
    ETCD_CA_FILE,
    ETCD_CLIENT_CERT_FILE,
    ETCD_CLIENT_KEY_FILE,
    PATRONI_CONF_PATH,
    PATRONI_LOGS_PATH,
    POSTGRESQL_DATA_PATH,
    POSTGRESQL_LOGS_PATH,
    RAFT_PARTNER_PREFIX,
)

PATRONI_SERVICE = "patroni"
CREATE_CLUSTER_CONF_PATH = "/var/snap/charmed-postgresql/current/etc/postgresql/postgresql.conf"


# This method will be used by the mock to replace requests.get
def mocked_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, json_data):
            self.json_data = json_data

        def json(self):
            return self.json_data

        elapsed = Mock()

    data = {
        "http://server1/cluster": {
            "members": [{"name": "postgresql-0", "host": "1.1.1.1", "role": "leader", "lag": "1"}]
        },
        "http://server1/health": {"state": "running"},
        "http://server4/cluster": {"members": []},
    }
    if args[0] in data:
        return MockResponse(data[args[0]])

    raise requests.exceptions.Timeout()


@pytest.fixture(autouse=True)
def peers_ips():
    peers_ips = {"2.2.2.2", "3.3.3.3"}
    yield peers_ips


@pytest.fixture()
def harness():
    harness = Harness(PostgresqlOperatorCharm)
    harness.begin()
    yield harness
    harness.cleanup()


@pytest.fixture(autouse=True)
def patroni(harness, peers_ips):
    patroni = Patroni(
        harness.charm,
        "1.1.1.1",
        "postgresql",
        "postgresql-0",
        1,
        peers_ips,
        "fake-superuser-password",
        "fake-replication-password",
        "fake-rewind-password",
        "fake-raft-password",
        "fake-patroni-password",
    )
    yield patroni


def test_get_member_ip(peers_ips, patroni):
    with patch(
        "cluster.parallel_patroni_get_request", return_value=None
    ) as _parallel_patroni_get_request:
        # No IP if no members
        assert patroni.get_member_ip(patroni.member_name) is None

        _parallel_patroni_get_request.return_value = {
            "members": [
                {
                    "name": "postgresql-1",
                    "host": "2.2.2.2",
                },
                {
                    "name": "postgresql-0",
                    "host": "1.1.1.1",
                },
            ]
        }
        assert patroni.get_member_ip(patroni.member_name) == "1.1.1.1"


def test_get_patroni_health(peers_ips, patroni):
    with (
        patch("cluster.stop_after_delay", new_callable=PropertyMock) as _stop_after_delay,
        patch("cluster.wait_fixed", new_callable=PropertyMock) as _wait_fixed,
        patch("charm.Patroni._patroni_url", new_callable=PropertyMock) as _patroni_url,
        patch("requests.get", side_effect=mocked_requests_get) as _get,
    ):
        # Test when the Patroni API is reachable.
        _patroni_url.return_value = "http://server1"
        health = patroni.get_patroni_health()

        # Check needed to ensure a fast charm deployment.
        _stop_after_delay.assert_called_once_with(60)
        _wait_fixed.assert_called_once_with(7)

        assert health == {"state": "running"}

        # Test when the Patroni API is not reachable.
        _patroni_url.return_value = "http://server2"
        with pytest.raises(RetryError):
            patroni.get_patroni_health()
            assert False


def test_get_postgresql_version(peers_ips, patroni):
    assert patroni.get_postgresql_version() == "16.14"


def test_dict_to_hba_string(harness, patroni):
    mock_data = {
        "ldapbasedn": "dc=example,dc=net",
        "ldapbinddn": "cn=serviceuser,dc=example,dc=net",
        "ldapbindpasswd": "password",
        "ldaptls": False,
        "ldapurl": "ldap://0.0.0.0:3893",
    }

    assert patroni._dict_to_hba_string(mock_data) == (
        'ldapbasedn="dc=example,dc=net" '
        'ldapbinddn="cn=serviceuser,dc=example,dc=net" '
        'ldapbindpasswd="password" '
        "ldaptls=0 "
        'ldapurl="ldap://0.0.0.0:3893"'
    )


def test_get_primary(peers_ips, patroni):
    with (
        patch(
            "cluster.parallel_patroni_get_request", return_value=None
        ) as _parallel_patroni_get_request,
    ):
        # No primary if no members
        assert patroni.get_primary() is None

        _parallel_patroni_get_request.return_value = {
            "members": [
                {
                    "name": "postgresql-1",
                    "role": "replica",
                },
                {
                    "name": "postgresql-0",
                    "role": "leader",
                },
            ]
        }
        # Test using the current Patroni URL.
        assert patroni.get_primary() == "postgresql-0"

        # Test requesting the primary in the unit name pattern.
        assert patroni.get_primary(unit_name_pattern=True) == "postgresql/0"


def test_is_creating_backup(peers_ips, patroni):
    with patch("charm.Patroni.cluster_status") as _cluster_status:
        # Test when one member is creating a backup.
        _cluster_status.return_value = [
            {"name": "postgresql-0"},
            {"name": "postgresql-1", "tags": {"is_creating_backup": True}},
        ]
        assert patroni.is_creating_backup

        # Test when no member is creating a backup.
        del patroni.cached_cluster_status
        _cluster_status.return_value = [{"name": "postgresql-0"}, {"name": "postgresql-1"}]
        assert not patroni.is_creating_backup


def test_is_replication_healthy(peers_ips, patroni):
    with (
        patch("requests.get") as _get,
        patch("charm.Patroni.get_primary") as _get_primary,
        patch("charm.Patroni.get_standby_leader") as _get_standby_leader,
        patch("charm.Patroni.get_member_ip"),
        patch("cluster.stop_after_delay", return_value=stop_after_delay(0)),
    ):
        # Test when replication is healthy.
        _get.return_value.status_code = 200
        assert patroni.is_replication_healthy()

        # Test when replication is not healthy.
        _get.side_effect = [
            MagicMock(status_code=200),
            MagicMock(status_code=200),
            MagicMock(status_code=503),
        ]
        assert not patroni.is_replication_healthy()

        # Test no primary
        _get.side_effect = None
        _get.return_value.status_code = 200
        _get_primary.return_value = None
        _get_standby_leader.return_value = None
        assert not patroni.is_replication_healthy()

        # Standby leader
        _get_standby_leader.return_value = "standby"
        assert patroni.is_replication_healthy()


def test_is_member_isolated(peers_ips, patroni):
    with (
        patch("cluster.stop_after_delay", return_value=stop_after_delay(0)),
        patch("cluster.wait_fixed", return_value=wait_fixed(0)),
        patch("requests.get", side_effect=mocked_requests_get) as _get,
        patch("charm.Patroni._patroni_url", new_callable=PropertyMock) as _patroni_url,
    ):
        # Test when it wasn't possible to connect to the Patroni API.
        _patroni_url.return_value = "http://server3"
        assert not patroni.is_member_isolated

        # Test when the member isn't isolated from the cluster.
        _patroni_url.return_value = "http://server1"
        assert not patroni.is_member_isolated

        # Test when the member is isolated from the cluster.
        _patroni_url.return_value = "http://server4"
        assert patroni.is_member_isolated


def test_render_patroni_yml_file(peers_ips, patroni):
    with (
        patch(
            "relations.async_replication.PostgreSQLAsyncReplication.get_partner_addresses",
            return_value=["2.2.2.2", "3.3.3.3"],
        ),
        patch("charm.Patroni.get_postgresql_version") as _get_postgresql_version,
        patch("cluster.render_file") as _render_file,
        patch(
            "charm.PostgresqlOperatorCharm.listen_ips",
            new_callable=PropertyMock,
            return_value=["1.1.1.1", "192.168.0.1"],
        ),
    ):
        _get_postgresql_version.return_value = "16.6"

        # Define variables to render in the template.
        member_name = "postgresql-0"
        scope = "postgresql"
        superuser_password = "fake-superuser-password"
        replication_password = "fake-replication-password"
        rewind_password = "fake-rewind-password"
        raft_password = "fake-raft-password"
        patroni_password = "fake-patroni-password"
        postgresql_version = "16"

        # Get the expected content from a file.
        with open("templates/patroni.yml.j2") as file:
            contents = file.read()
            template = Template(contents)
            # Setup a mock for the `open` method, set returned data to patroni.yml template.
            mock = mock_open(read_data=contents)

        expected_content = template.render(
            conf_path=PATRONI_CONF_PATH,
            data_path=POSTGRESQL_DATA_PATH,
            log_path=PATRONI_LOGS_PATH,
            postgresql_log_path=POSTGRESQL_LOGS_PATH,
            member_name=member_name,
            partner_addrs=["2.2.2.2", "3.3.3.3"],
            peers_ips=sorted(peers_ips),
            scope=scope,
            self_ip=patroni.unit_ip,
            listen_ips=["1.1.1.1", "192.168.0.1"],
            superuser="operator",
            superuser_password=superuser_password,
            replication_password=replication_password,
            rewind_user=REWIND_USER,
            rewind_password=rewind_password,
            version=postgresql_version,
            synchronous_node_count=0,
            maximum_lag_on_failover=1048576,
            raft_password=raft_password,
            patroni_password=patroni_password,
            instance_password_encryption="scram-sha-256",
            slots={},
            # DCS seam vars (raft-mode defaults); etcd filename vars are always present.
            dcs_backend=DCS_RAFT,
            patroni_namespace="/db/",
            etcd_hosts=[],
            etcd_prefix="",
            etcd_username="",
            etcd_password="",
            etcd_ca_file=ETCD_CA_FILE,
            etcd_client_cert_file=ETCD_CLIENT_CERT_FILE,
            etcd_client_key_file=ETCD_CLIENT_KEY_FILE,
        )

        # Patch the `open` method with our mock.
        with patch("builtins.open", mock, create=True):
            # Call the method.
            patroni.render_patroni_yml_file()

        # Check the template is opened read-only in the call to open.
        assert mock.call_args_list[0][0] == ("templates/patroni.yml.j2",)
        # Ensure the correct rendered template is sent to _render_file method.
        _render_file.assert_called_once_with(
            Substrates.VM,
            "/var/snap/charmed-postgresql/current/etc/patroni/patroni.yaml",
            expected_content,
            0o600,
        )


def test_render_patroni_yml_passes_dcs_vars(harness, patroni):
    """render_patroni_yml_file feeds the real DCS seam vars into the template."""
    from unittest.mock import PropertyMock, patch

    captured = {}

    class _T:
        def __init__(self, text):
            self._text = text

        def render(self, **kwargs):
            captured.update(kwargs)
            return "rendered"

    with (
        patch("cluster.Template", _T),
        patch("cluster.render_file"),
        patch(
            "relations.async_replication.PostgreSQLAsyncReplication.get_partner_addresses",
            return_value=[],
        ),
        patch("charm.Patroni.get_postgresql_version", return_value="16.0"),
        patch(
            "charm.PostgresqlOperatorCharm.listen_ips",
            new_callable=PropertyMock,
            return_value=["1.1.1.1"],
        ),
    ):
        patroni.render_patroni_yml_file()

    # The real DCSBackend.template_vars() flows through: with no etcd relation the
    # default backend is raft and the etcd host/prefix vars are empty.
    assert captured["dcs_backend"] == "raft"
    assert captured["etcd_hosts"] == []
    assert captured["etcd_prefix"] == ""


def test_render_patroni_yml_etcd_backend():
    """Etcd backend renders etcd3 block and omits the raft block."""
    from jinja2 import Template

    with open("templates/patroni.yml.j2") as f:
        template = Template(f.read())

    rendered = template.render(
        conf_path="/etc/patroni",
        scope="postgresql",
        patroni_namespace="/postgresql/abc/",
        member_name="postgresql-0",
        self_ip="1.1.1.1",
        log_path="/logs",
        postgresql_log_path="/pglogs",
        data_path="/data",
        listen_ips=["1.1.1.1"],
        superuser="operator",
        superuser_password="x",
        replication_password="x",
        rewind_user="rewind",
        rewind_password="x",
        patroni_password="x",
        version="16",
        synchronous_node_count=1,
        maximum_lag_on_failover=1048576,
        connectivity=True,
        is_creating_backup=False,
        enable_ldap=False,
        enable_tls=False,
        partner_addrs=[],
        peers_ips=set(),
        extra_replication_endpoints=set(),
        pgbackrest_configuration_file="/pgbackrest.conf",
        enable_pgbackrest_archiving=False,
        restoring_backup=False,
        primary_cluster_endpoint=None,
        stanza=None,
        restore_stanza=None,
        backup_id=None,
        pitr_target=None,
        restore_timeline=None,
        restore_to_latest=False,
        pg_parameters=None,
        user_databases_map={},
        slots={},
        instance_password_encryption="scram-sha-256",
        ldap_parameters="",
        raft_password="x",
        watcher=None,
        # DCS seam vars:
        dcs_backend="etcd",
        etcd_hosts=["10.0.0.1:2379", "10.0.0.2:2379"],
        etcd_prefix="/postgresql/abc/",
        etcd_username="postgresql",
        etcd_password="s3cr3tpw",
        etcd_ca_file="etcd_ca.pem",
        etcd_client_cert_file="etcd_client_cert.pem",
        etcd_client_key_file="etcd_client_key.pem",
    )

    assert "etcd3:" in rendered
    assert "raft:" not in rendered
    assert "10.0.0.1:2379" in rendered
    assert "10.0.0.2:2379" in rendered
    assert "namespace: /postgresql/abc/" in rendered
    assert "/etc/patroni/etcd_ca.pem" in rendered
    assert "/etc/patroni/etcd_client_cert.pem" in rendered
    assert "/etc/patroni/etcd_client_key.pem" in rendered

    # The whole rendered file must be valid YAML and the etcd3 block must carry the
    # username/password used for token auth through the gRPC gateway. Patroni's etcd3
    # backend ignores an ``etcd3.namespace`` key, so the keyspace is carried by the
    # top-level ``namespace`` instead (which must match the granted RBAC prefix).
    import yaml

    parsed = yaml.safe_load(rendered)
    assert parsed["namespace"] == "/postgresql/abc/"
    assert "namespace" not in parsed["etcd3"]
    assert parsed["etcd3"]["username"] == "postgresql"
    assert parsed["etcd3"]["password"] == "s3cr3tpw"
    assert parsed["etcd3"]["hosts"] == ["10.0.0.1:2379", "10.0.0.2:2379"]


def test_render_patroni_yml_raft_backend_default():
    """Without dcs_backend == etcd, the raft block renders and etcd3 does not."""
    from jinja2 import Template

    with open("templates/patroni.yml.j2") as f:
        template = Template(f.read())

    rendered = template.render(
        conf_path="/etc/patroni",
        scope="postgresql",
        patroni_namespace="/db/",
        member_name="postgresql-0",
        self_ip="1.1.1.1",
        log_path="/logs",
        postgresql_log_path="/pglogs",
        data_path="/data",
        listen_ips=["1.1.1.1"],
        superuser="operator",
        superuser_password="x",
        replication_password="x",
        rewind_user="rewind",
        rewind_password="x",
        patroni_password="x",
        version="16",
        synchronous_node_count=1,
        maximum_lag_on_failover=1048576,
        connectivity=True,
        is_creating_backup=False,
        enable_ldap=False,
        enable_tls=False,
        partner_addrs=[],
        peers_ips=set(),
        extra_replication_endpoints=set(),
        pgbackrest_configuration_file="/pgbackrest.conf",
        enable_pgbackrest_archiving=False,
        restoring_backup=False,
        primary_cluster_endpoint=None,
        stanza=None,
        restore_stanza=None,
        backup_id=None,
        pitr_target=None,
        restore_timeline=None,
        restore_to_latest=False,
        pg_parameters=None,
        user_databases_map={},
        slots={},
        instance_password_encryption="scram-sha-256",
        ldap_parameters="",
        raft_password="raftpw",
        watcher=None,
        dcs_backend="raft",
        etcd_hosts=[],
        etcd_prefix="",
        etcd_ca_file="etcd_ca.pem",
        etcd_client_cert_file="etcd_client_cert.pem",
        etcd_client_key_file="etcd_client_key.pem",
    )

    assert "raft:" in rendered
    assert "etcd3:" not in rendered
    assert "password: raftpw" in rendered
    assert "namespace: /db/" in rendered


def test_start_patroni(peers_ips, patroni):
    with patch("charm.snap.SnapCache") as _snap_cache:
        _cache = _snap_cache.return_value
        _selected_snap = _cache.__getitem__.return_value
        _selected_snap.start.side_effect = [None, snap.SnapError]

        # Test a success scenario.
        assert patroni.start_patroni()
        _cache.__getitem__.assert_called_once_with("charmed-postgresql")
        _selected_snap.start.assert_called_once_with(services=[PATRONI_SERVICE])

        # Test a fail scenario.
        assert not patroni.start_patroni()


def test_stop_patroni(peers_ips, patroni):
    with patch("charm.snap.SnapCache") as _snap_cache:
        _cache = _snap_cache.return_value
        _selected_snap = _cache.__getitem__.return_value
        _selected_snap.stop.side_effect = [None, snap.SnapError]
        _selected_snap.services.__getitem__.return_value.__getitem__.return_value = False

        # Test a success scenario.
        assert patroni.stop_patroni()
        _cache.__getitem__.assert_called_once_with("charmed-postgresql")
        _selected_snap.stop.assert_called_once_with(services=[PATRONI_SERVICE])
        _selected_snap.services.__getitem__.return_value.__getitem__.assert_called_once_with(
            "active"
        )

        # Test a fail scenario.
        assert not patroni.stop_patroni()


def test_reinitialize_postgresql(peers_ips, patroni):
    with patch("requests.post") as _post:
        patroni.reinitialize_postgresql()
        _post.assert_called_once_with(
            f"https://{patroni.unit_ip}:8008/reinitialize",
            verify=patroni.verify,
            auth=patroni._patroni_auth,
            timeout=PATRONI_TIMEOUT,
        )


def test_switchover(peers_ips, patroni):
    with (
        patch("requests.post") as _post,
        patch("cluster.Patroni.get_primary", return_value="primary"),
    ):
        response = _post.return_value
        response.status_code = 200

        patroni.switchover()

        _post.assert_called_once_with(
            "https://1.1.1.1:8008/switchover",
            json={"leader": "primary"},
            verify=patroni.verify,
            auth=patroni._patroni_auth,
            timeout=PATRONI_TIMEOUT,
        )
        _post.reset_mock()

        # Test candidate
        patroni.switchover("candidate")

        _post.assert_called_once_with(
            "https://1.1.1.1:8008/switchover",
            json={"leader": "primary", "candidate": "candidate"},
            verify=patroni.verify,
            auth=patroni._patroni_auth,
            timeout=PATRONI_TIMEOUT,
        )

        # Test candidate, not sync
        response = _post.return_value
        response.status_code = 412
        response.text = "candidate name does not match with sync_standby"
        with pytest.raises(SwitchoverNotSyncError):
            patroni.switchover("candidate")
            assert False

        # Test general error
        response = _post.return_value
        response.status_code = 412
        response.text = "something else "
        with pytest.raises(SwitchoverFailedError):
            patroni.switchover()
            assert False


def test_pause_patches_patroni_config(patroni):
    from unittest.mock import patch

    with patch("cluster.requests.patch") as req:
        req.return_value.status_code = 200
        patroni.pause()
        assert req.call_args.kwargs["json"] == {"pause": True}


def test_resume_patches_patroni_config(patroni):
    from unittest.mock import patch

    with patch("cluster.requests.patch") as req:
        req.return_value.status_code = 200
        patroni.resume()
        assert req.call_args.kwargs["json"] == {"pause": False}


def test_update_synchronous_node_count(peers_ips, patroni):
    with (
        patch("cluster.stop_after_delay", return_value=stop_after_delay(0)) as _wait_fixed,
        patch("cluster.wait_fixed", return_value=wait_fixed(0)) as _wait_fixed,
        patch("requests.patch") as _patch,
    ):
        response = _patch.return_value
        response.status_code = 200

        patroni.update_synchronous_node_count()

        _patch.assert_called_once_with(
            "https://1.1.1.1:8008/config",
            json={"synchronous_node_count": 0, "synchronous_mode_strict": False},
            verify=patroni.verify,
            auth=patroni._patroni_auth,
            timeout=PATRONI_TIMEOUT,
        )

        # Test when the request fails.
        response.status_code = 500
        with pytest.raises(RetryError):
            patroni.update_synchronous_node_count()
            assert False


@pytest.mark.parametrize(
    ("members_ips", "node_count", "config_strict", "expected_strict"),
    [
        # Bootstrap: only the leader is a member, count is 2 -> strict must stay off.
        ('["1.1.1.1"]', 2, True, False),
        # Mid scale-up: leader + 1 member, count 2 -> still off (the deadlock guard).
        ('["1.1.1.1", "2.2.2.2"]', 2, True, False),
        # Full membership: leader + 2 members, count 2 -> strict on.
        ('["1.1.1.1", "2.2.2.2", "3.3.3.3"]', 2, True, True),
        # Two-unit cluster, count 1 -> strict on.
        ('["1.1.1.1", "2.2.2.2"]', 1, True, True),
        # Operator disabled strict via config -> stays off regardless of membership.
        ('["1.1.1.1", "2.2.2.2", "3.3.3.3"]', 2, False, False),
        # Single-unit cluster, count 0 -> strict off.
        ('["1.1.1.1"]', 0, True, False),
    ],
)
def test_synchronous_configuration_strict_guard(
    harness, patroni, members_ips, node_count, config_strict, expected_strict
):
    with (
        patch(
            "cluster.Patroni._synchronous_node_count",
            new_callable=PropertyMock,
            return_value=node_count,
        ),
        patch.object(
            type(harness.charm),
            "app_peer_data",
            new_callable=PropertyMock,
            return_value={"members_ips": members_ips},
        ),
        patch.object(
            type(harness.charm),
            "config",
            new_callable=PropertyMock,
            return_value=MagicMock(synchronous_mode_strict=config_strict),
        ),
    ):
        config = patroni.synchronous_configuration

    assert config["synchronous_node_count"] == node_count
    assert config["synchronous_mode_strict"] is expected_strict


def test_set_max_timelines_history(peers_ips, patroni):
    with (
        patch("requests.patch") as _patch,
    ):
        patroni.set_max_timelines_history()

        _patch.assert_called_once_with(
            f"{patroni._patroni_url}/config",
            json={"max_timelines_history": 50},
            verify=patroni.verify,
            auth=patroni._patroni_auth,
            timeout=PATRONI_TIMEOUT,
        )


def test_configure_patroni_on_unit(peers_ips, patroni):
    with (
        patch("os.chmod") as _chmod,
        patch("builtins.open") as _open,
        patch("os.chown") as _chown,
        patch("pwd.getpwnam") as _getpwnam,
    ):
        _getpwnam.return_value.pw_uid = sentinel.uid
        _getpwnam.return_value.pw_gid = sentinel.gid

        patroni.configure_patroni_on_unit()

        _getpwnam.assert_called_once_with("_daemon_")

        _chown.assert_any_call(
            "/var/snap/charmed-postgresql/common/var/lib/postgresql",
            uid=sentinel.uid,
            gid=sentinel.gid,
        )

        _open.assert_called_once_with(CREATE_CLUSTER_CONF_PATH, "a")
        _chmod.assert_called_once_with(
            "/var/snap/charmed-postgresql/common/var/lib/postgresql", 448
        )


def test_member_started_true(peers_ips, patroni):
    with (
        patch("cluster.requests.get") as _get,
        patch("cluster.stop_after_delay", return_value=stop_after_delay(0)),
        patch("cluster.wait_fixed", return_value=wait_fixed(0)),
        patch("charm.Patroni.is_patroni_running", return_value=True),
    ):
        _get.return_value.json.return_value = {"state": "running"}

        assert patroni.member_started

        _get.assert_called_once_with(
            "https://1.1.1.1:8008/health",
            verify=patroni.verify,
            timeout=5,
            auth=patroni._patroni_auth,
        )


def test_member_started_false(peers_ips, patroni):
    with (
        patch("cluster.requests.get") as _get,
        patch("cluster.stop_after_delay", return_value=stop_after_delay(0)),
        patch("cluster.wait_fixed", return_value=wait_fixed(0)),
        patch("charm.Patroni.is_patroni_running", return_value=True),
    ):
        _get.return_value.json.return_value = {"state": "stopped"}

        assert not patroni.member_started

        _get.assert_called_once_with(
            "https://1.1.1.1:8008/health",
            verify=patroni.verify,
            timeout=5,
            auth=patroni._patroni_auth,
        )


def test_member_started_error(peers_ips, patroni):
    with (
        patch("cluster.requests.get") as _get,
        patch("cluster.stop_after_delay", return_value=stop_after_delay(0)),
        patch("cluster.wait_fixed", return_value=wait_fixed(0)),
        patch("charm.Patroni.is_patroni_running", return_value=True),
    ):
        _get.side_effect = Exception

        assert not patroni.member_started

        _get.assert_called_once_with(
            "https://1.1.1.1:8008/health",
            verify=patroni.verify,
            timeout=5,
            auth=patroni._patroni_auth,
        )


def test_member_inactive_true(peers_ips, patroni):
    with (
        patch("cluster.requests.get") as _get,
        patch("cluster.stop_after_delay", return_value=stop_after_delay(0)),
        patch("cluster.wait_fixed", return_value=wait_fixed(0)),
    ):
        _get.return_value.json.return_value = {"state": "stopped"}

        assert patroni.member_inactive

        _get.assert_called_once_with(
            "https://1.1.1.1:8008/health",
            verify=patroni.verify,
            timeout=5,
            auth=patroni._patroni_auth,
        )


def test_member_inactive_false(peers_ips, patroni):
    with (
        patch("cluster.requests.get") as _get,
        patch("cluster.stop_after_delay", return_value=stop_after_delay(0)),
        patch("cluster.wait_fixed", return_value=wait_fixed(0)),
    ):
        _get.return_value.json.return_value = {"state": "starting"}

        assert not patroni.member_inactive

        _get.assert_called_once_with(
            "https://1.1.1.1:8008/health",
            verify=patroni.verify,
            timeout=5,
            auth=patroni._patroni_auth,
        )


def test_member_inactive_error(peers_ips, patroni):
    with (
        patch("cluster.requests.get") as _get,
        patch("cluster.stop_after_delay", return_value=stop_after_delay(0)),
        patch("cluster.wait_fixed", return_value=wait_fixed(0)),
    ):
        _get.side_effect = Exception

        assert patroni.member_inactive

        _get.assert_called_once_with(
            "https://1.1.1.1:8008/health",
            verify=patroni.verify,
            timeout=5,
            auth=patroni._patroni_auth,
        )


def test_patroni_logs(patroni):
    with patch("charm.snap.SnapCache") as _snap_cache:
        # Test when the logs are returned successfully.
        logs = _snap_cache.return_value.__getitem__.return_value.logs
        logs.return_value = "fake-logs"
        assert patroni.patroni_logs() == "fake-logs"

        # Test the charm fails to get the logs.
        logs.side_effect = snap.SnapError
        assert patroni.patroni_logs() == ""


def test_last_postgresql_logs(patroni):
    with (
        patch("glob.glob") as _glob,
        patch("builtins.open", mock_open(read_data="fake-logs")) as _open,
    ):
        # Test when there are no files to read.
        assert patroni.last_postgresql_logs() == ""
        _open.assert_not_called()

        # Test when there are multiple files in the logs directory.
        _glob.return_value = [
            "/var/snap/charmed-postgresql/common/var/log/postgresql/postgresql.log.1",
            "/var/snap/charmed-postgresql/common/var/log/postgresql/postgresql.log.2",
            "/var/snap/charmed-postgresql/common/var/log/postgresql/postgresql.log.3",
        ]
        assert patroni.last_postgresql_logs() == "fake-logs"
        _open.assert_called_once_with(
            "/var/snap/charmed-postgresql/common/var/log/postgresql/postgresql.log.3"
        )

        # Test when the charm fails to read the logs.
        _open.reset_mock()
        _open.side_effect = OSError
        assert patroni.last_postgresql_logs() == ""
        _open.assert_called_with(
            "/var/snap/charmed-postgresql/common/var/log/postgresql/postgresql.log.3"
        )


def test_get_patroni_restart_condition(patroni):
    mock = mock_open()
    with patch("builtins.open", mock) as _open:
        # Test when there is a restart condition set.
        _open.return_value.__enter__.return_value.read.return_value = "Restart=always"
        assert patroni.get_patroni_restart_condition() == "always"

        # Test when there is no restart condition set.
        _open.return_value.__enter__.return_value.read.return_value = ""
        with pytest.raises(RuntimeError):
            patroni.get_patroni_restart_condition()
            assert False


@pytest.mark.parametrize("new_restart_condition", ["on-success", "on-failure"])
def test_update_patroni_restart_condition(patroni, new_restart_condition):
    with (
        patch("builtins.open", mock_open(read_data="Restart=always")) as _open,
        patch("subprocess.run") as _run,
    ):
        _open.return_value.__enter__.return_value.read.return_value = "Restart=always"
        patroni.update_patroni_restart_condition(new_restart_condition)
        _open.return_value.__enter__.return_value.write.assert_called_once_with(
            f"Restart={new_restart_condition}"
        )
        _run.assert_called_once_with(["/bin/systemctl", "daemon-reload"])


def test_cleanup_raft_cluster(patroni):
    with (
        patch("cluster.TcpUtility") as _tcp_utility,
        patch("cluster.Patroni.remove_raft_member", return_value=True) as _remove_raft_member,
        patch(
            "charm.PostgresqlOperatorCharm._units_ips",
            new_callable=PropertyMock,
            return_value={"1.1.1.1"},
        ),
        patch(
            "charm.PostgresqlOperatorCharm._remove_from_members_ips"
        ) as _remove_from_members_ips,
        patch(
            "charm.PostgresqlOperatorCharm._is_workload_running",
            new_callable=PropertyMock,
            return_value=True,
        ) as _is_workload_running,
    ):
        # Error connecting to raft
        _tcp_utility.side_effect = Exception

        assert not patroni.cleanup_raft_cluster()

        _tcp_utility.assert_called_once_with(password="fake-raft-password", timeout=3)
        _tcp_utility.reset_mock()

        # No status
        _tcp_utility.side_effect = None
        _tcp_utility.return_value.executeCommand.return_value = {}

        assert not patroni.cleanup_raft_cluster()

        _tcp_utility.return_value.executeCommand.assert_called_once_with(
            "127.0.0.1:2222", ["status"]
        )

        # All members active
        _tcp_utility.return_value.executeCommand.return_value = {
            f"{RAFT_PARTNER_PREFIX}1.1.1.1:2222": 2
        }

        assert patroni.cleanup_raft_cluster()

        assert not _remove_raft_member.called

        # Filter by unit ips
        _tcp_utility.return_value.executeCommand.return_value = {
            f"{RAFT_PARTNER_PREFIX}1.1.1.1:2222": 0,
            f"{RAFT_PARTNER_PREFIX}2.2.2.2:2222": 0,
        }

        assert patroni.cleanup_raft_cluster()

        _remove_raft_member.assert_called_once_with("2.2.2.2:2222")
        _remove_from_members_ips.assert_called_once_with("2.2.2.2")


def test_remove_raft_member(patroni):
    with patch("cluster.TcpUtility") as _tcp_utility:
        # Member already removed
        _tcp_utility.return_value.executeCommand.return_value = "Response message"

        patroni.remove_raft_member("1.2.3.4:2222")

        _tcp_utility.assert_called_once_with(password="fake-raft-password", timeout=3)
        _tcp_utility.return_value.executeCommand.assert_called_once_with(
            "127.0.0.1:2222", ["status"]
        )
        _tcp_utility.reset_mock()

        # Removing member
        _tcp_utility.return_value.executeCommand.side_effect = [
            {
                "partner_node_status_server_1.2.3.4:2222": 0,
                "has_quorum": True,
                "leader": sentinel.raft_leader,
            },
            "SUCCESS",
        ]

        patroni.remove_raft_member("1.2.3.4:2222")

        _tcp_utility.assert_called_once_with(password="fake-raft-password", timeout=3)
        assert _tcp_utility.return_value.executeCommand.call_count == 2
        _tcp_utility.return_value.executeCommand.assert_any_call("127.0.0.1:2222", ["status"])
        _tcp_utility.return_value.executeCommand.assert_any_call(
            "127.0.0.1:2222", ["remove", "1.2.3.4:2222"]
        )
        _tcp_utility.reset_mock()

        # Raises on failed status
        _tcp_utility.return_value.executeCommand.side_effect = [
            {
                "partner_node_status_server_1.2.3.4:2222": 0,
                "has_quorum": True,
                "leader": sentinel.raft_leader,
            },
            "FAIL",
        ]

        with pytest.raises(RemoveRaftMemberFailedError):
            patroni.remove_raft_member("1.2.3.4:2222")
            assert False

        # Raises on remove error
        _tcp_utility.return_value.executeCommand.side_effect = [
            {
                "partner_node_status_server_1.2.3.4:2222": 0,
                "has_quorum": True,
                "leader": sentinel.raft_leader,
            },
            UtilityException,
        ]

        with pytest.raises(RemoveRaftMemberFailedError):
            patroni.remove_raft_member("1.2.3.4:2222")
            assert False

        # Raises on status error
        _tcp_utility.return_value.executeCommand.side_effect = [UtilityException]

        with pytest.raises(RemoveRaftMemberFailedError):
            patroni.remove_raft_member("1.2.3.4:2222")
            assert False


def test_remove_raft_member_no_quorum(patroni, harness):
    with (
        patch("cluster.TcpUtility") as _tcp_utility,
        patch("cluster.requests.get") as _get,
        patch(
            "charm.PostgresqlOperatorCharm.unit_peer_data", new_callable=PropertyMock
        ) as _unit_peer_data,
    ):
        # Async replica
        _unit_peer_data.return_value = {}
        _tcp_utility.return_value.executeCommand.return_value = {
            "partner_node_status_server_1.2.3.4:2222": 0,
            "has_quorum": False,
            "leader": None,
        }
        _get.return_value.json.return_value = {
            "members": [{"role": "async_replica", "name": "postgresql-0"}]
        }

        patroni.remove_raft_member("1.2.3.4:2222")
        assert harness.charm.unit_peer_data == {"raft_stuck": "True"}

        # No health
        _unit_peer_data.return_value = {}
        _tcp_utility.return_value.executeCommand.return_value = {
            "partner_node_status_server_1.2.3.4:2222": 0,
            "has_quorum": False,
            "leader": None,
        }
        _get.side_effect = Exception

        patroni.remove_raft_member("1.2.3.4:2222")

        assert harness.charm.unit_peer_data == {"raft_stuck": "True"}

        # Sync replica
        _unit_peer_data.return_value = {}
        leader_mock = Mock()
        leader_mock.address = "1.2.3.4:2222"
        _tcp_utility.return_value.executeCommand.return_value = {
            "partner_node_status_server_1.2.3.4:2222": 0,
            "has_quorum": False,
            "leader": leader_mock,
        }
        _get.side_effect = None
        _get.return_value.json.return_value = {
            "members": [{"role": "sync_standby", "name": "postgresql-0"}]
        }

        patroni.remove_raft_member("1.2.3.4:2222")

        assert harness.charm.unit_peer_data == {"raft_stuck": "True"}


def test_remove_raft_data(patroni):
    with (
        patch("cluster.Patroni.stop_patroni") as _stop_patroni,
        patch("cluster.psutil") as _psutil,
        patch("cluster.wait_fixed", return_value=wait_fixed(0)),
        patch("shutil.rmtree") as _rmtree,
        patch("pathlib.Path.is_dir") as _is_dir,
        patch("pathlib.Path.exists") as _exists,
    ):
        mock_proc_pg = Mock()
        mock_proc_not_pg = Mock()
        mock_proc_pg.name.return_value = "postgres"
        mock_proc_not_pg.name.return_value = "something_else"
        _psutil.process_iter.side_effect = [[mock_proc_not_pg, mock_proc_pg], [mock_proc_not_pg]]

        patroni.remove_raft_data()

        _stop_patroni.assert_called_once_with()
        assert _psutil.process_iter.call_count == 2
        _psutil.process_iter.assert_any_call(["name"])
        _rmtree.assert_called_once_with(Path(f"{PATRONI_CONF_PATH}/raft"))


def test_remove_raft_journals(patroni):
    with (
        patch("cluster.Patroni.stop_patroni") as _stop_patroni,
        patch("shutil.rmtree") as _rmtree,
        patch("pathlib.Path.is_dir", return_value=True),
        patch("pathlib.Path.exists", return_value=True),
    ):
        patroni.remove_raft_journals()

        # Journals are removed but Patroni is left running (migration-commit path).
        _rmtree.assert_called_once_with(Path(f"{PATRONI_CONF_PATH}/raft"))
        _stop_patroni.assert_not_called()


def test_reinitialise_raft_data(patroni):
    with (
        patch("cluster.Patroni.get_patroni_health") as _get_patroni_health,
        patch("charm.PostgresqlOperatorCharm.update_config") as _update_config,
        patch("cluster.Patroni.start_patroni") as _start_patroni,
        patch("cluster.Patroni.restart_patroni") as _restart_patroni,
        patch("cluster.psutil") as _psutil,
        patch("cluster.wait_fixed", return_value=wait_fixed(0)),
    ):
        mock_proc_pg = Mock()
        mock_proc_not_pg = Mock()
        mock_proc_pg.name.return_value = "postgres"
        mock_proc_not_pg.name.return_value = "something_else"
        _psutil.process_iter.side_effect = [[mock_proc_not_pg], [mock_proc_not_pg, mock_proc_pg]]
        _get_patroni_health.side_effect = [
            {"role": "replica", "state": "streaming"},
            {"role": "leader", "state": "running"},
        ]

        patroni.reinitialise_raft_data()

        _update_config.assert_called_once_with(no_peers=True)
        _start_patroni.assert_called_once_with()
        _restart_patroni.assert_called_once_with()
        assert _psutil.process_iter.call_count == 2
        _psutil.process_iter.assert_any_call(["name"])


def test_are_replicas_up(patroni):
    with patch("charm.Patroni.cluster_status") as _cluster_status:
        _cluster_status.return_value = [
            {"host": "1.1.1.1", "state": "running"},
            {"host": "2.2.2.2", "state": "streaming"},
            {"host": "3.3.3.3", "state": "other state"},
        ]
        assert patroni.are_replicas_up() == {"1.1.1.1": True, "2.2.2.2": True, "3.3.3.3": False}

        # Return None on error
        _cluster_status.side_effect = Exception
        assert patroni.are_replicas_up() is None
