# tests/test_spec_store.py
from __future__ import annotations

from pathlib import Path

from subgen.core.jobs.spec_store import JobSpecStore


def test_spec_store_write_read_dict(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    store = JobSpecStore(str(root))

    job_id = "job_test_001"
    spec = {"job_id": job_id, "kind": "SUBTITLES_GENERATE", "inputs": {"video_path": "a", "out_dir": "b"}}
    status = {"job_id": job_id, "state": "QUEUED"}
    result = {"job_id": job_id, "ok": True}

    store.write_spec(job_id, spec)
    store.write_status(job_id, status)
    store.write_result(job_id, result)

    assert store.read_spec_dict(job_id)["job_id"] == job_id
    assert store.read_status_dict(job_id)["state"] == "QUEUED"
    assert store.read_result_dict(job_id)["ok"] is True