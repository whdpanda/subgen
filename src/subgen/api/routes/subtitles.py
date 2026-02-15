# src/subgen/api/routes/subtitles.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, status

from subgen.api.errors import InvalidRequestError
from subgen.api.schemas.jobs import JobKind, JobStatus
from subgen.api.schemas.subtitles import BurnRequest, FixRequest, GenerateRequest
from subgen.api.services.jobs_service import get_jobs_service
from subgen.api.utils.path_policy import get_path_policy

router = APIRouter(prefix="/v1/subtitles", tags=["subtitles"])


def _get_opt(obj: Any, name: str, default: Any = None) -> Any:
    # Pydantic model -> attribute access; tolerate missing fields for backward-compat
    return getattr(obj, name, default)


@router.post(
    "/generate",
    response_model=JobStatus,
    status_code=status.HTTP_202_ACCEPTED,
)
def generate(req: GenerateRequest) -> JobStatus:
    """
    Step 7: async job
    - Validate paths
    - Create job spec + enqueue
    - Return JobStatus (queued/enqueued)
    """
    pp = get_path_policy()
    video_path = pp.ensure_file_exists(req.video_path)
    out_dir = pp.ensure_dir(req.out_dir, create=True)

    inputs: Dict[str, Any] = {
        "video_path": video_path,
        "out_dir": out_dir,
    }

    max_passes = _get_opt(req, "max_passes", None)
    if max_passes is not None:
        inputs["max_passes"] = max_passes

    pipeline_args = _get_opt(req, "pipeline_args", None)
    if pipeline_args:
        inputs["pipeline_args"] = pipeline_args

    # Forward-compatible knobs (Step78 worker supports these)
    quality_args = _get_opt(req, "quality_args", None)
    if quality_args:
        inputs["quality_args"] = quality_args

    fix_args = _get_opt(req, "fix_args", None)
    if fix_args:
        inputs["fix_args"] = fix_args

    js = get_jobs_service()
    st = js.create_job(
        kind=JobKind.SUBTITLES_GENERATE,
        inputs=inputs,
        options={},
        meta={"api": "subtitles.generate"},
    )
    st = js.enqueue_job(st.job_id)
    return st


@router.post(
    "/fix",
    response_model=JobStatus,
    status_code=status.HTTP_202_ACCEPTED,
)
def fix(req: FixRequest) -> JobStatus:
    """
    Step 7: async job
    """
    pp = get_path_policy()
    srt_path = pp.ensure_file_exists(req.srt_path)

    out_dir_in: Optional[str] = _get_opt(req, "out_dir", None)
    out_dir: Optional[str] = None
    if out_dir_in:
        out_dir = pp.ensure_dir(out_dir_in, create=True)

    inputs: Dict[str, Any] = {"srt_path": srt_path}
    if out_dir:
        inputs["out_dir"] = out_dir

    fix_args = _get_opt(req, "fix_args", None)
    if fix_args:
        inputs["fix_args"] = fix_args

    quality_args = _get_opt(req, "quality_args", None)
    if quality_args:
        inputs["quality_args"] = quality_args

    js = get_jobs_service()
    st = js.create_job(
        kind=JobKind.SUBTITLES_FIX,
        inputs=inputs,
        options={},
        meta={"api": "subtitles.fix"},
    )
    st = js.enqueue_job(st.job_id)
    return st


@router.post(
    "/burn",
    response_model=JobStatus,
    status_code=status.HTTP_202_ACCEPTED,
)
def burn(req: BurnRequest) -> JobStatus:
    """
    Step 7: async job
    """
    pp = get_path_policy()
    video_path = pp.ensure_file_exists(req.video_path)
    srt_path = pp.ensure_file_exists(req.srt_path)

    out_path_in: Optional[str] = _get_opt(req, "out_path", None)
    out_dir_in: Optional[str] = _get_opt(req, "out_dir", None)

    out_path: Optional[str] = None
    if out_path_in:
        # parent directory must also be allowed; create if enabled
        out_p = Path(pp.ensure_allowed(out_path_in))
        parent = pp.ensure_dir(str(out_p.parent), create=True)
        out_path = str(Path(parent) / out_p.name)
    elif out_dir_in:
        out_dir = pp.ensure_dir(out_dir_in, create=True)
        out_path = str(Path(out_dir) / f"{Path(video_path).stem}.burned.mp4")
    else:
        # Keep it explicit: avoid silently writing next to input video unless user provided out_path/out_dir
        raise InvalidRequestError(
            "burn requires out_path or out_dir",
            details={"hint": "provide out_path or out_dir in request body"},
        )

    inputs: Dict[str, Any] = {
        "video_path": video_path,
        "srt_path": srt_path,
        "out_path": out_path,
    }

    burn_args = _get_opt(req, "burn_args", None)
    if burn_args:
        inputs["burn_args"] = burn_args

    js = get_jobs_service()
    st = js.create_job(
        kind=JobKind.SUBTITLES_BURN,
        inputs=inputs,
        options={},
        meta={"api": "subtitles.burn"},
    )
    st = js.enqueue_job(st.job_id)
    return st