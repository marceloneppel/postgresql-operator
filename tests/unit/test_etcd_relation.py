# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Unit tests for the PostgreSQL etcd DCS relation handler."""

from unittest.mock import MagicMock, PropertyMock, patch

from src.relations.etcd import EtcdEndpointsChangedEvent, PostgreSQLEtcd, parse_etcd_hosts


class TestParseEtcdHosts:
    """Tests for the parse_etcd_hosts helper."""

    def test_none_returns_empty(self):
        assert parse_etcd_hosts(None) == []

    def test_empty_string_returns_empty(self):
        assert parse_etcd_hosts("") == []

    def test_single_endpoint(self):
        assert parse_etcd_hosts("10.0.0.1:2379") == ["10.0.0.1:2379"]

    def test_multiple_endpoints_sorted_and_stripped(self):
        assert parse_etcd_hosts(" 10.0.0.2:2379 , 10.0.0.1:2379 ") == [
            "10.0.0.1:2379",
            "10.0.0.2:2379",
        ]

    def test_strips_scheme_if_present(self):
        # `uris` form (scheme://ip:port) reduces to ip:port for Patroni hosts.
        assert parse_etcd_hosts("https://10.0.0.1:2379,https://10.0.0.2:2379") == [
            "10.0.0.1:2379",
            "10.0.0.2:2379",
        ]


def make_etcd_charm():
    charm = MagicMock()
    charm.app.name = "postgresql"
    charm.unit.is_leader.return_value = True
    return charm


def build_relation(patches):
    """Construct PostgreSQLEtcd with the heavy libs patched out."""
    with (
        patch("src.relations.etcd.EtcdRequires") as mock_requires,
        patch("src.relations.etcd.TLSCertificatesRequiresV4") as mock_tls,
    ):
        rel = PostgreSQLEtcd(make_etcd_charm())
        rel._etcd = mock_requires.return_value
        rel.certificate = mock_tls.return_value
        for attr, value in patches.items():
            setattr(rel._etcd, attr, value)
        return rel


class TestPostgreSQLEtcd:
    def test_prefix_is_app_scoped(self):
        rel = build_relation({})
        assert rel.prefix.startswith("/postgresql/")

    def test_hosts_prefers_endpoints_field(self):
        rel = build_relation({})
        with patch.object(
            PostgreSQLEtcd, "_relation", new_callable=PropertyMock, return_value=MagicMock(id=7)
        ):
            rel._etcd.fetch_relation_field.side_effect = lambda rid, field: {
                "endpoints": "10.0.0.2:2379,10.0.0.1:2379",
            }.get(field)
            assert rel.hosts == ["10.0.0.1:2379", "10.0.0.2:2379"]

    def test_is_ready_true_when_all_present(self):
        rel = build_relation({})
        with patch.object(
            PostgreSQLEtcd, "_relation", new_callable=PropertyMock, return_value=MagicMock(id=7)
        ):
            rel._etcd.fetch_relation_field.side_effect = lambda rid, field: {
                "endpoints": "10.0.0.1:2379",
                "tls-ca": "CAPEM",
                "username": "postgresql",
                "password": "s3cr3t",
            }.get(field)
            assert rel.is_ready is True

    def test_is_ready_false_when_no_relation(self):
        rel = build_relation({})
        with patch.object(
            PostgreSQLEtcd, "_relation", new_callable=PropertyMock, return_value=None
        ):
            assert rel.is_ready is False

    def test_is_ready_false_when_username_missing(self):
        rel = build_relation({})
        with patch.object(
            PostgreSQLEtcd, "_relation", new_callable=PropertyMock, return_value=MagicMock(id=7)
        ):
            rel._etcd.fetch_relation_field.side_effect = lambda rid, field: {
                "endpoints": "10.0.0.1:2379",
                "tls-ca": "CAPEM",
                "password": "s3cr3t",
            }.get(field)
            assert rel.is_ready is False

    def test_is_ready_false_when_password_missing(self):
        # Token auth through the gRPC gateway needs the password; without it we are
        # not ready even though username/endpoints/CA are present.
        rel = build_relation({})
        with patch.object(
            PostgreSQLEtcd, "_relation", new_callable=PropertyMock, return_value=MagicMock(id=7)
        ):
            rel._etcd.fetch_relation_field.side_effect = lambda rid, field: {
                "endpoints": "10.0.0.1:2379",
                "tls-ca": "CAPEM",
                "username": "postgresql",
            }.get(field)
            assert rel.is_ready is False

    def test_password_reads_relation_field(self):
        rel = build_relation({})
        with patch.object(
            PostgreSQLEtcd, "_relation", new_callable=PropertyMock, return_value=MagicMock(id=7)
        ):
            rel._etcd.fetch_relation_field.side_effect = lambda rid, field: {
                "password": "s3cr3t",
            }.get(field)
            assert rel.password == "s3cr3t"

    def test_certificate_request_is_cn_less(self):
        # etcd's gRPC gateway rejects client certs that carry a CommonName, so the
        # request must be CN-less and instead identify via a SAN.
        with (
            patch("src.relations.etcd.EtcdRequires"),
            patch("src.relations.etcd.TLSCertificatesRequiresV4") as mock_tls,
        ):
            PostgreSQLEtcd(make_etcd_charm())
        request = mock_tls.call_args.kwargs["certificate_requests"][0]
        assert request.common_name == ""
        assert request.sans_dns == {"postgresql"}

    def test_publish_mtls_cert_calls_set_mtls_cert(self):
        rel = build_relation({})
        rel.certificate.get_assigned_certificates.return_value = (
            [MagicMock(certificate="LEAFPEM")],
            MagicMock(),
        )
        with patch.object(
            PostgreSQLEtcd, "_relation", new_callable=PropertyMock, return_value=MagicMock(id=7)
        ):
            rel.publish_mtls_cert()
            rel._etcd.set_mtls_cert.assert_called_once_with(7, "LEAFPEM")

    def test_template_vars_exposes_hosts_and_prefix(self):
        rel = build_relation({})
        with (
            patch.object(
                PostgreSQLEtcd,
                "_relation",
                new_callable=PropertyMock,
                return_value=MagicMock(id=7),
            ),
            patch.object(
                PostgreSQLEtcd, "hosts", new_callable=PropertyMock, return_value=["10.0.0.1:2379"]
            ),
        ):
            v = rel.template_vars()
            assert v["etcd_hosts"] == ["10.0.0.1:2379"]
            assert v["etcd_prefix"].startswith("/postgresql/")

    def test_hosts_falls_back_to_uris_when_no_endpoints(self):
        rel = build_relation({})
        with patch.object(
            PostgreSQLEtcd, "_relation", new_callable=PropertyMock, return_value=MagicMock(id=7)
        ):
            rel._etcd.fetch_relation_field.side_effect = lambda rid, field: {
                "uris": "https://10.0.0.9:2379",
            }.get(field)
            assert rel.endpoints == "https://10.0.0.9:2379"
            assert rel.hosts == ["10.0.0.9:2379"]

    def test_publish_mtls_cert_noop_when_not_leader(self):
        rel = build_relation({})
        rel.charm.unit.is_leader.return_value = False
        rel.certificate.get_assigned_certificates.return_value = (
            [MagicMock(certificate="LEAFPEM")],
            MagicMock(),
        )
        with patch.object(
            PostgreSQLEtcd, "_relation", new_callable=PropertyMock, return_value=MagicMock(id=7)
        ):
            rel.publish_mtls_cert()
            rel._etcd.set_mtls_cert.assert_not_called()

    def test_write_tls_files_writes_all_three_when_present(self):
        rel = build_relation({})
        rel.certificate.get_assigned_certificates.return_value = (
            [MagicMock(certificate="LEAFPEM")],
            "KEYPEM",
        )
        with (
            patch.object(
                PostgreSQLEtcd, "tls_ca", new_callable=PropertyMock, return_value="CAPEM"
            ),
            patch("src.relations.etcd.render_file") as mock_render,
        ):
            assert rel.write_tls_files() is True
            assert mock_render.call_count == 3

    def test_write_tls_files_returns_false_when_ca_missing(self):
        rel = build_relation({})
        rel.certificate.get_assigned_certificates.return_value = (
            [MagicMock(certificate="LEAFPEM")],
            "KEYPEM",
        )
        with (
            patch.object(PostgreSQLEtcd, "tls_ca", new_callable=PropertyMock, return_value=None),
            patch("src.relations.etcd.render_file") as mock_render,
        ):
            assert rel.write_tls_files() is False
            mock_render.assert_not_called()


class TestEtcdLifecycleEdges:
    def test_endpoints_changed_emits_signal(self):
        rel = build_relation({})
        with patch.object(rel.framework, "_emit") as mock_emit:
            rel._on_endpoints_changed(MagicMock())
            mock_emit.assert_called_once()
            assert isinstance(mock_emit.call_args[0][0], EtcdEndpointsChangedEvent)

    def test_certificate_available_rewrites_files_and_reloads(self):
        rel = build_relation({})
        with (
            patch.object(rel, "publish_mtls_cert"),
            patch.object(rel, "write_tls_files", return_value=True),
            patch.object(PostgreSQLEtcd, "is_ready", new_callable=PropertyMock, return_value=True),
        ):
            rel._on_certificate_available(MagicMock())
            rel.charm.update_config.assert_called_once()

    def test_relation_broken_blocks_when_on_etcd(self):
        rel = build_relation({})
        rel.charm.dcs.is_etcd = True
        rel._on_relation_broken(MagicMock())
        rel.charm._set_dcs_blocked_status.assert_called_once()
