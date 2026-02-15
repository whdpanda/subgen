from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# -----------------------------
# Versioning & enums
# -----------------------------

JOB_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class JobKind(str, Enum):
    SUBTITLES_GENERATE = "subtitles.generate"
    SUBTITLES_FIX = "subtitles.fix"
    SUBTITLES_BURN = "subtitles.burn"


class JobState(str, Enum):
    QUEUED = "queued"
    STARTED = "started"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


# -----------------------------
# Shared primitives
# -----------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobTimestamps(BaseModel):
    created_at: datetime = Field(default_factory=utc_now)
    enqueued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    model_config = {"extra": "allow"}


class JobWorkerInfo(BaseModel):
    worker_id: Optional[str] = None  # e.g., pod name / hostname
    attempt: int = 1  # PR#6: no retry yet, keep field for forward compat

    model_config = {"extra": "allow"}


class JobError(BaseModel):
    code: str  # e.g. INVALID_INPUT / PIPELINE_FAILED / FFMPEG_FAILED / TIMEOUT
    detail: str
    trace_id: Optional[str] = None

    model_config = {"extra": "allow"}


class JobArtifacts(BaseModel):
    job_dir: str
    spec_path: Optional[str] = None
    status_path: Optional[str] = None
    result_path: Optional[str] = None
    log_path: Optional[str] = None
    report_path: Optional[str] = None

    model_config = {"extra": "allow"}


# -----------------------------
# Core contracts (disk truth)
# -----------------------------


class JobSpec(BaseModel):
    """
    Immutable "intent" of a job.
    This is the single source of truth for the worker.
    """

    schema_version: Literal["1.0"] = Field(default=JOB_SCHEMA_VERSION)

    job_id: str = Field(min_length=8, max_length=64)
    kind: JobKind

    created_at: datetime = Field(default_factory=utc_now)

    # Absolute path of the job directory root: /data/jobs/<job_id>
    job_root: str = Field(min_length=1)

    # Free-form inputs per kind (forward compatible)
    inputs: Dict[str, Any] = Field(default_factory=dict)

    # Common execution options
    options: Dict[str, Any] = Field(default_factory=dict)

    # Non-execution metadata (client_request_id, api_version, etc.)
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "extra": "allow",  # forward-compat: allow new fields
    }

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return v

    @field_validator("job_root")
    @classmethod
    def _job_root_basic_sanity(cls, v: str) -> str:
        # Policy check belongs to API/config; schema only enforces non-empty.
        if not v.strip():
            raise ValueError("job_root must not be empty")
        return v


class JobStatus(BaseModel):
    """
    Mutable state; updated by API (enqueue) and worker (start/finish/fail).
    """

    schema_version: Literal["1.0"] = Field(default=JOB_SCHEMA_VERSION)

    job_id: str = Field(min_length=8, max_length=64)
    kind: JobKind

    state: JobState = JobState.QUEUED
    progress: Optional[float] = None  # 0..1, optional

    timestamps: JobTimestamps = Field(default_factory=JobTimestamps)
    worker: JobWorkerInfo = Field(default_factory=JobWorkerInfo)

    message: Optional[str] = None
    error: Optional[JobError] = None

    artifacts: Optional[JobArtifacts] = None

    model_config = {"extra": "allow"}

    @field_validator("progress")
    @classmethod
    def _progress_range(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if v < 0.0 or v > 1.0:
            raise ValueError("progress must be within [0, 1]")
        return v


class JobResult(BaseModel):
    """
    Successful output payload; written by worker on success.
    Keep it aligned with your pipeline tool return keys so callers can reuse.
    """

    schema_version: Literal["1.0"] = Field(default=JOB_SCHEMA_VERSION)

    job_id: str = Field(min_length=8, max_length=64)
    kind: JobKind

    ok: bool = True

    primary_path: Optional[str] = None
    srt_paths: Optional[List[str]] = None
    outputs: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=utc_now)

    model_config = {"extra": "allow"}

    @field_validator("created_at")
    @classmethod
    def _result_created_at_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return v


# -----------------------------
# Step 6: API request/response models
# -----------------------------


class GenerateJobRequest(BaseModel):
    """
    POST /v1/jobs/subtitles/generate
    These fields are copied into JobSpec.inputs.
    """
    video_path: str
    out_dir: str

    max_passes: Optional[int] = None
    pipeline_args: Optional[Dict[str, Any]] = None
    quality_args: Optional[Dict[str, Any]] = None
    fix_args: Optional[Dict[str, Any]] = None

    model_config = {"extra": "forbid"}


class FixJobRequest(BaseModel):
    """
    POST /v1/jobs/subtitles/fix
    """
    srt_path: str
    out_dir: Optional[str] = None

    fix_args: Optional[Dict[str, Any]] = None
    quality_args: Optional[Dict[str, Any]] = None

    model_config = {"extra": "forbid"}


class BurnJobRequest(BaseModel):
    """
    POST /v1/jobs/subtitles/burn
    """
    video_path: str
    srt_path: str
    out_path: Optional[str] = None

    burn_args: Optional[Dict[str, Any]] = None

    model_config = {"extra": "forbid"}


class JobCreateResponse(BaseModel):
    job_id: str
    status_url: str
    result_url: str

    model_config = {"extra": "forbid"}


class JobStatusResponse(BaseModel):
    job: JobStatus
    status_url: str
    result_url: str

    model_config = {"extra": "forbid"}


class JobResultResponse(BaseModel):
    job_id: str
    kind: JobKind
    result: JobResult

    model_config = {"extra": "forbid"}