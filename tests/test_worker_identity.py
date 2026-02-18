from __future__ import annotations

from typing import Any

from subgen.service.rq.worker_identity import build_worker_name


def test_build_worker_name_prefers_pod_uid(monkeypatch: Any) -> None:
    monkeypatch.setenv("POD_NAME", "subgen-pod")
    monkeypatch.setenv("POD_UID", "c0ffee12-3456-7890-abcd-1234567890ef")

    out = build_worker_name()

    assert out == "subgen-pod-c0ffee12"


def test_build_worker_name_falls_back_to_random(monkeypatch: Any) -> None:
    monkeypatch.delenv("POD_UID", raising=False)
    monkeypatch.delenv("K8S_POD_UID", raising=False)
    monkeypatch.setenv("HOSTNAME", "subgen-host")
    monkeypatch.setattr("subgen.service.rq.worker_identity.secrets.token_hex", lambda _n: "deadbeef")

    out = build_worker_name()

    assert out == "subgen-host-deadbeef"
