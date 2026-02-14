# src/subgen/api/routes/subtitles.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter

from subgen.api.errors import InvalidPathError
from subgen.api.schemas.subtitles import (
    BurnRequest,
    BurnResponse,
    ErrorResponse,
    FixRequest,
    FixResponse,
    GenerateRequest,
    GenerateResponse,
)
from subgen.api.services.subtitles_service import (
    burn_subtitles,
    fix_subtitles,
    generate_subtitles,
)
from subgen.api.utils.path_policy import get_path_policy

router = APIRouter(prefix="/v1/subtitles", tags=["subtitles"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    501: {"model": ErrorResponse},
}


@router.post("/generate", response_model=GenerateResponse, responses=_ERROR_RESPONSES)
def generate(req: GenerateRequest) -> GenerateResponse:
    policy = get_path_policy()

    video_path = policy.ensure_file_exists(req.video_path)

    out_dir = None
    if req.out_dir:
        out_dir = policy.ensure_dir(req.out_dir, create=True)

    if not out_dir:
        # In this repo, generate path should always be deterministic; require out_dir to avoid ambiguity
        # (prevents future refactor around log/output placement).
        raise InvalidPathError("out_dir is required for /generate", details={"field": "out_dir"})

    res = generate_subtitles(
        video_path=video_path,
        out_dir=out_dir,
        max_passes=req.max_passes,
        pipeline_args=req.pipeline_args,  # NEW: pass-through
    )
    return GenerateResponse(**res)


@router.post("/fix", response_model=FixResponse, responses=_ERROR_RESPONSES)
def fix(req: FixRequest) -> FixResponse:
    policy = get_path_policy()

    srt_path = policy.ensure_file_exists(req.srt_path)

    out_dir = None
    if req.out_dir:
        out_dir = policy.ensure_dir(req.out_dir, create=True)

    res = fix_subtitles(
        srt_path=srt_path,
        out_dir=out_dir,
    )
    return FixResponse(**res)


def _derive_burn_out_path(*, video_path: str, out_dir: str) -> str:
    """
    Stable default naming to avoid future contract changes:
      <out_dir>/<video_stem>.burned.mp4
    """
    vp = Path(video_path)
    stem = vp.stem or "output"
    return str((Path(out_dir) / f"{stem}.burned.mp4").resolve())


@router.post("/burn", response_model=BurnResponse, responses=_ERROR_RESPONSES)
def burn(req: BurnRequest) -> BurnResponse:
    policy = get_path_policy()

    video_path = policy.ensure_file_exists(req.video_path)
    srt_path = policy.ensure_file_exists(req.srt_path)

    # Support out_path (preferred) + out_dir (compat).
    out_dir: Optional[str] = None
    if req.out_dir:
        out_dir = policy.ensure_dir(req.out_dir, create=True)

    out_path: Optional[str] = None
    if req.out_path:
        # Validate allowed roots + parent dir existence (or create)
        out_path_allowed = policy.ensure_allowed(req.out_path)
        parent = str(Path(out_path_allowed).parent)
        policy.ensure_dir(parent, create=True)
        out_path = out_path_allowed
    elif out_dir:
        out_path = _derive_burn_out_path(video_path=video_path, out_dir=out_dir)

        # Validate derived out_path under allowed roots (defense-in-depth)
        out_path = policy.ensure_allowed(out_path)
        parent = str(Path(out_path).parent)
        policy.ensure_dir(parent, create=True)
    else:
        raise InvalidPathError(
            "either out_path or out_dir is required for /burn",
            details={"fields": ["out_path", "out_dir"]},
        )

    res = burn_subtitles(
        video_path=video_path,
        srt_path=srt_path,
        out_path=out_path,
    )
    return BurnResponse(**res)