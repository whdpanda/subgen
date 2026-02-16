from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

from subgen.api.schemas.jobs import JobKind
from subgen.api.services.jobs_service import JobsService
from subgen.core.jobs import JobSpecStore


class _FakeJob:
    def __init__(self) -> None:
        self.meta: Dict[str, Any] = {}
        self.meta_saved = False

    def save_meta(self) -> None:
        self.meta_saved = True


class _FakeQueue:
    def __init__(self) -> None:
        self.enqueues: list[dict[str, Any]] = []
        self.connection = object()

    def enqueue(self, func: str, *args: Any, **kwargs: Any) -> _FakeJob:
        self.enqueues.append({"func": func, "args": args, "kwargs": kwargs})
        return _FakeJob()


def _build_service(tmp_path: Any) -> tuple[JobsService, _FakeQueue, JobSpecStore]:
    cfg = SimpleNamespace(
        job_root=str(tmp_path / "jobs"),
        allowed_roots=[str(tmp_path)],
        allow_create_output_dir=True,
        rq_job_timeout_sec=5400,
    )
    store = JobSpecStore(str(tmp_path / "jobs"))
    queue = _FakeQueue()
    svc = JobsService(cfg=cfg, store=store, queue=queue)
    svc._rq_job_exists = lambda _job_id: False  # type: ignore[method-assign]
    return svc, queue, store


def test_timeout_moved_from_pipeline_args_to_rq_job_timeout(tmp_path: Any) -> None:
    svc, queue, store = _build_service(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"stub")

    job_id = "jobtimeout01"
    svc.create_and_enqueue(
        kind=JobKind.SUBTITLES_GENERATE,
        job_id=job_id,
        inputs={
            "video_path": str(video_path),
            "out_dir": str(out_dir),
            "pipeline_args": {"timeout": 120, "foo": "bar"},
        },
    )

    enq = queue.enqueues[0]
    assert enq["kwargs"]["timeout"] == 120

    spec = store.read_spec_dict(job_id)
    assert spec["inputs"]["pipeline_args"] == {"foo": "bar"}


def test_invalid_pipeline_timeout_uses_default_rq_timeout(tmp_path: Any) -> None:
    svc, queue, store = _build_service(tmp_path)
    out_dir = tmp_path / "out2"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_path / "video2.mp4"
    video_path.write_bytes(b"stub")

    job_id = "jobtimeout02"
    svc.create_and_enqueue(
        kind=JobKind.SUBTITLES_GENERATE,
        job_id=job_id,
        inputs={
            "video_path": str(video_path),
            "out_dir": str(out_dir),
            "pipeline_args": {"timeout": "bad", "foo": "bar"},
        },
    )

    enq = queue.enqueues[0]
    assert enq["kwargs"]["timeout"] == 5400

    spec = store.read_spec_dict(job_id)
    assert spec["inputs"]["pipeline_args"] == {"foo": "bar"}
