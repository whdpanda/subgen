from __future__ import annotations

from typing import Any

from subgen.service.rq.worker_identity import build_worker_name


def test_build_worker_name_includes_namespace_timestamp_and_pod_uid(monkeypatch: Any) -> None:
    monkeypatch.setenv("POD_NAME", "subgen-pod")
    monkeypatch.setenv("POD_NAMESPACE", "prod")
    monkeypatch.setenv("POD_UID", "c0ffee12-3456-7890-abcd-1234567890ef")
    monkeypatch.setattr("subgen.service.rq.worker_identity.time.time_ns", lambda: 0xABCD)
    monkeypatch.setattr("subgen.service.rq.worker_identity.secrets.token_hex", lambda _n: "deadbeef")

    out = build_worker_name()

    assert out == "subgen-pod-prod-abcd-c0ffee12-deadbeef"


def test_build_worker_name_falls_back_to_hostname_when_pod_data_absent(monkeypatch: Any) -> None:
    monkeypatch.delenv("POD_NAME", raising=False)
    monkeypatch.delenv("POD_NAMESPACE", raising=False)
    monkeypatch.delenv("K8S_NAMESPACE", raising=False)
    monkeypatch.delenv("POD_UID", raising=False)
    monkeypatch.delenv("K8S_POD_UID", raising=False)
    monkeypatch.setenv("HOSTNAME", "subgen-host")
    monkeypatch.setattr("subgen.service.rq.worker_identity.time.time_ns", lambda: 0x1234)
    monkeypatch.setattr("subgen.service.rq.worker_identity.secrets.token_hex", lambda _n: "beadfeed")

    out = build_worker_name()

    assert out == "subgen-host-1234-beadfeed"
