# tests/test_worker_run_job_generate.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import subgen.service.rq.tasks as tasks_mod
from subgen.core.jobs.spec_store import JobSpecStore


def test_run_job_generate_happy_path(tmp_path: Path, monkeypatch: Any) -> None:
    """
    Unit test for rq task entrypoint:
    - no real Redis
    - no real tools / pipeline
    - asserts: status/result written on disk
    """
    job_root = tmp_path / "jobs"
    store = JobSpecStore(str(job_root))
    job_id = "job_test_002"

    # --- Write spec/status as dicts (store supports dict) ---
    created_at = "2026-02-15T00:00:00Z"
    spec = {
        "job_id": job_id,
        "kind": "SUBTITLES_GENERATE",
        "created_at": created_at,
        "inputs": {"video_path": "x.webm", "out_dir": str(tmp_path / "out")},
        "meta": {},
    }
    status = {
        "job_id": job_id,
        "kind": "SUBTITLES_GENERATE",
        "state": "QUEUED",
        "progress": None,
        "timestamps": {"created_at": created_at, "started_at": None, "finished_at": None},
        "worker": {"worker_id": None, "attempt": 1},
        "message": "queued",
        "error": None,
        "artifacts": None,
    }
    store.write_spec(job_id, spec)
    store.write_status(job_id, status)

    # --- Monkeypatch config + store path ---
    monkeypatch.setattr(tasks_mod, "load_config", lambda: SimpleNamespace(job_root=str(job_root)))

    # --- Avoid tool wiring; run_pr4c_closed_loop returns a stable dict ---
    monkeypatch.setattr(tasks_mod, "_tool_map", lambda: {})  # bypass required tools check
    monkeypatch.setattr(
        tasks_mod,
        "run_pr4c_closed_loop",
        lambda *args, **kwargs: {
            "ok": True,
            "primary_path": str(tmp_path / "out" / "x.zh.srt"),
            "srt_paths": [str(tmp_path / "out" / "x.zh.srt")],
            "outputs": {},
            "artifacts": {},
            "meta": {},
        },
    )

    # --- Avoid request debug logging context ---
    from tests._helpers import nullctx

    monkeypatch.setattr(tasks_mod, "request_debug_logging", lambda *args, **kwargs: nullctx())

    # --- Run ---
    out = tasks_mod.run_job(job_id)

    assert out["ok"] is True
    assert out["job_id"] == job_id

    # --- Disk state ---
    st = store.read_status_dict(job_id)
    assert st["state"] in ("SUCCEEDED", "SUCCEEDED".lower(), "succeeded") or st["message"] == "succeeded"

    res = store.read_result_dict(job_id)
    assert res["ok"] is True
    assert res["job_id"] == job_id