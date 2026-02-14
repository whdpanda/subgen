# src/subgen/api/schemas/subtitles.py
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    request_id: str = ""


class ErrorResponse(BaseModel):
    error: ErrorBody


class GenerateRequest(BaseModel):
    video_path: str = Field(..., description="Container path to the input video file")
    out_dir: Optional[str] = Field(None, description="Output directory (container path). Required for /generate.")
    max_passes: Optional[int] = Field(None, description="Override quality loop max passes (Step 3)")

    # NEW: pass-through knobs for pipeline (PR6-ready)
    # Example: {"preprocess":"demucs","demucs_model":"htdemucs","demucs_device":"cpu","demucs_stems":"vocals"}
    pipeline_args: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional pipeline args pass-through (e.g. preprocess/demucs).",
    )


class GenerateResponse(BaseModel):
    ok: bool
    primary_path: Optional[str] = None
    srt_path: Optional[str] = None
    report_path: Optional[str] = None
    out_video_path: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class FixRequest(BaseModel):
    srt_path: str = Field(..., description="Container path to the input .srt to fix")
    out_dir: Optional[str] = Field(None, description="Optional output directory (container path)")


class FixResponse(BaseModel):
    ok: bool
    srt_path: Optional[str] = None
    report_path: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class BurnRequest(BaseModel):
    video_path: str = Field(..., description="Container path to the input video file")
    srt_path: str = Field(..., description="Container path to the subtitle .srt file")

    # Preferred: explicit output file path
    out_path: Optional[str] = Field(None, description="Optional output file path (container path)")

    # Compatibility: output directory (API will derive out_path as <out_dir>/<video_stem>.burned.mp4)
    out_dir: Optional[str] = Field(None, description="Optional output directory (container path)")


class BurnResponse(BaseModel):
    ok: bool
    out_video_path: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)