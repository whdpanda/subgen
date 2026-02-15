from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rq import Queue
    from rq.job import Job
else:
    Queue = Any  # type: ignore[misc,assignment]
    Job = Any  # type: ignore[misc,assignment]

from subgen.api.config import ApiConfig
from subgen.api.schemas.jobs import (
    JobArtifacts,
    JobError,
    JobKind,
    JobResult,
    JobState,
    JobStatus,
    JobTimestamps,
    JobWorkerInfo,
    JobSpec,
)
from subgen.core.jobs import JobSpecStore
from subgen.utils.logger import get_logger

logger = get_logger("subgen.jobs_service")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _job_fetch(job_id: str, connection: Any) -> Any:
    try:
        from rq.job import Job as RQJob
    except ModuleNotFoundError as e:
        raise RuntimeError("rq package is required for job operations; install subgen runtime dependencies") from e
    return RQJob.fetch(job_id, connection=connection)


def _norm_abs(p: str | Path) -> Path:
    return Path(os.path.normpath(str(p))).resolve(strict=False)


def _is_under_roots(p: Path, roots: list[str]) -> bool:
    """
    Check if p is under any of roots (prefix match on resolved paths).
    """
    try:
        rp = p.resolve(strict=False)
    except Exception:
        rp = p

    for r in roots:
        rr = _norm_abs(r)
        try:
            rr_resolved = rr.resolve(strict=False)
        except Exception:
            rr_resolved = rr

        try:
            rp.relative_to(rr_resolved)
            return True
        except Exception:
            continue
    return False


def _require_abs_under_roots(path_str: str, roots: list[str], field_name: str) -> Path:
    if not path_str:
        raise ValueError(f"{field_name} is empty")
    p = Path(path_str)
    if not p.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path inside the container: {path_str!r}")

    rp = _norm_abs(p)
    if not _is_under_roots(rp, roots):
        raise ValueError(f"{field_name} must be under allowed_roots={roots}, got={path_str!r}")
    return rp


@dataclass(frozen=True)
class EnqueueResult:
    spec: JobSpec
    status: JobStatus


class JobsService:
    """
    Create job + persist spec/status + enqueue to RQ + query status/result.

    Design:
    - Use job_id as RQ job id (stable mapping)
    - Disk (spec/status/result) is the source of truth
    - RQ is execution + hints only
    """

    def __init__(self, cfg: ApiConfig, store: JobSpecStore, queue: Queue) -> None:
        self.cfg = cfg
        self.store = store
        self.queue = queue

        # Validate global job_root early
        _require_abs_under_roots(str(self.store.job_root), self.cfg.allowed_roots, "JOB_ROOT")

    def new_job_id(self) -> str:
        return uuid.uuid4().hex

    def create_and_enqueue(
        self,
        *,
        kind: JobKind,
        inputs: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> EnqueueResult:
        jid = job_id or self.new_job_id()

        if self.store.job_dir(jid).exists() or self._rq_job_exists(jid):
            raise ValueError(f"job_id already exists: {jid}")

        job_dir = self.store.ensure_job_dir(jid, create_out=True, create_tmp=True)

        self._validate_common_paths(inputs)

        spec = JobSpec(
            job_id=jid,
            kind=kind,
            created_at=_utc_now(),
            job_root=str(self.store.job_dir(jid)),
            inputs=inputs or {},
            options=options or {},
            meta=meta or {},
        )

        artifacts = JobArtifacts(
            job_dir=str(job_dir),
            spec_path=str(self.store.spec_path(jid)),
            status_path=str(self.store.status_path(jid)),
            result_path=str(self.store.result_path(jid)),
            log_path=str(self.store.debug_log_path(jid)),
            report_path=str(self.store.report_path(jid)),
        )

        status = JobStatus(
            job_id=jid,
            kind=kind,
            state=JobState.QUEUED,
            progress=None,
            timestamps=JobTimestamps(
                created_at=spec.created_at,
                enqueued_at=_utc_now(),
                started_at=None,
                finished_at=None,
            ),
            worker=JobWorkerInfo(worker_id=None, attempt=1),
            message="queued",
            error=None,
            artifacts=artifacts,
        )

        # Persist before enqueue (polling works immediately)
        self.store.write_spec(jid, spec)
        self.store.write_status(jid, status)

        # Enqueue (job_id == rq job id) and stamp meta.kind for rq-only fallback.
        job = self.queue.enqueue(
            "subgen.service.rq.tasks.run_job",
            jid,
            job_id=jid,
            timeout=self.cfg.rq_job_timeout_sec,
            result_ttl=24 * 3600,
            failure_ttl=24 * 3600,
        )

        try:
            if isinstance(job.meta, dict):
                job.meta["kind"] = kind.value
                job.save_meta()
        except Exception:
            # Non-fatal; disk is the truth
            pass

        logger.info("JOB_ENQUEUED job_id=%s kind=%s job_dir=%s", jid, kind.value, str(job_dir))
        return EnqueueResult(spec=spec, status=status)

    def get_status(self, job_id: str) -> JobStatus:
        if self.store.has_status(job_id):
            status = self.store.read_as_model(job_id, JobStatus, "status")

            try:
                rq_job = self._fetch_rq_job(job_id)
            except Exception:
                rq_job = None

            if rq_job is not None:
                refreshed = self._refresh_status_from_rq(status, rq_job)
                if refreshed is not None:
                    status = refreshed
                    self.store.write_status(job_id, status)

            return status

        rq_job = self._fetch_rq_job(job_id)
        return self._status_from_rq_only(job_id, rq_job)

    def get_result(self, job_id: str) -> JobResult:
        if self.store.has_result(job_id):
            return self.store.read_as_model(job_id, JobResult, "result")
        raise FileNotFoundError(f"result.json not found for job_id={job_id}")

    # -------------------------
    # Internal helpers
    # -------------------------

    def _rq_job_exists(self, job_id: str) -> bool:
        try:
            _job_fetch(job_id, connection=self.queue.connection)  # type: ignore[arg-type]
            return True
        except Exception:
            return False

    def _fetch_rq_job(self, job_id: str) -> Job:
        try:
            return _job_fetch(job_id, connection=self.queue.connection)  # type: ignore[arg-type]
        except Exception as e:
            raise FileNotFoundError(f"RQ job not found: {job_id}") from e

    def _validate_common_paths(self, inputs: Dict[str, Any]) -> None:
        roots = self.cfg.allowed_roots

        if "video_path" in inputs and inputs["video_path"]:
            _require_abs_under_roots(str(inputs["video_path"]), roots, "video_path")

        if "srt_path" in inputs and inputs["srt_path"]:
            _require_abs_under_roots(str(inputs["srt_path"]), roots, "srt_path")

        if "out_path" in inputs and inputs["out_path"]:
            _require_abs_under_roots(str(inputs["out_path"]), roots, "out_path")

        if "out_dir" in inputs and inputs["out_dir"]:
            out_dir = _require_abs_under_roots(str(inputs["out_dir"]), roots, "out_dir")
            if not out_dir.exists():
                if self.cfg.allow_create_output_dir:
                    out_dir.mkdir(parents=True, exist_ok=True)
                else:
                    raise ValueError(f"out_dir does not exist and creation is disabled: {str(out_dir)!r}")

    def _refresh_status_from_rq(self, status: JobStatus, rq_job: Job) -> Optional[JobStatus]:
        if status.state in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED):
            return None

        rq_status = rq_job.get_status()

        if rq_status == "queued":
            desired = JobState.QUEUED
        elif rq_status == "started":
            desired = JobState.STARTED if status.state == JobState.QUEUED else status.state
        elif rq_status == "finished":
            desired = status.state
        elif rq_status == "failed":
            desired = JobState.FAILED
        elif rq_status == "canceled":
            desired = JobState.CANCELED
        else:
            desired = status.state

        if desired == status.state:
            return None

        status.state = desired  # type: ignore[misc]
        status.message = f"rq:{rq_status}"  # type: ignore[misc]

        if desired in (JobState.STARTED, JobState.RUNNING) and status.timestamps.started_at is None:
            status.timestamps.started_at = _utc_now()  # type: ignore[misc]

        if desired in (JobState.FAILED, JobState.CANCELED):
            status.timestamps.finished_at = status.timestamps.finished_at or _utc_now()  # type: ignore[misc]
            if desired == JobState.FAILED and status.error is None:
                detail = getattr(rq_job, "exc_info", None) or "rq job failed"
                status.error = JobError(code="RQ_FAILED", detail=str(detail))  # type: ignore[misc]

        return status

    def _status_from_rq_only(self, job_id: str, rq_job: Job) -> JobStatus:
        rq_status = rq_job.get_status()

        if rq_status == "queued":
            state = JobState.QUEUED
        elif rq_status == "started":
            state = JobState.STARTED
        elif rq_status == "finished":
            state = JobState.SUCCEEDED
        elif rq_status == "failed":
            state = JobState.FAILED
        elif rq_status == "canceled":
            state = JobState.CANCELED
        else:
            state = JobState.QUEUED

        ts = JobTimestamps(created_at=_utc_now())
        if rq_job.enqueued_at:
            ts.enqueued_at = rq_job.enqueued_at  # type: ignore[assignment]
        if rq_job.started_at:
            ts.started_at = rq_job.started_at  # type: ignore[assignment]
        if rq_job.ended_at:
            ts.finished_at = rq_job.ended_at  # type: ignore[assignment]

        err = None
        if state == JobState.FAILED:
            detail = getattr(rq_job, "exc_info", None) or "rq job failed"
            err = JobError(code="RQ_FAILED", detail=str(detail))

        kind = JobKind.SUBTITLES_GENERATE
        try:
            if isinstance(rq_job.meta, dict) and rq_job.meta.get("kind"):
                kind = JobKind(str(rq_job.meta["kind"]))
        except Exception:
            kind = JobKind.SUBTITLES_GENERATE

        return JobStatus(
            job_id=job_id,
            kind=kind,
            state=state,
            progress=None,
            timestamps=ts,
            worker=JobWorkerInfo(worker_id=None, attempt=1),
            message=f"rq:{rq_status}",
            error=err,
            artifacts=None,
        )