from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from subgen.core.video.burn import burn_subtitles


class BurnToolArgs(BaseModel):
    video_path: Path = Field(..., description="Input video path")
    srt_path: Path = Field(..., description="Subtitle SRT path to burn-in")
    out_path: Optional[Path] = Field(None, description="Output video path (mp4). If omitted, auto-generate.")

    ffmpeg_bin: str = Field("ffmpeg", description="ffmpeg binary name/path")
    force_style: Optional[str] = Field(None, description="ASS force_style string")

    crf: int = Field(18, description="x264 CRF")
    preset: str = Field("veryfast", description="x264 preset")

    copy_audio: bool = Field(False, description="Copy audio stream if possible")
    audio_codec: str = Field("aac", description="Audio codec when not copying")
    audio_bitrate: str = Field("192k", description="Audio bitrate when encoding")


def _parse_tool_args(args_or_kwargs: Union[BurnToolArgs, dict[str, Any]]) -> BurnToolArgs:
    if isinstance(args_or_kwargs, BurnToolArgs):
        return args_or_kwargs

    if hasattr(BurnToolArgs, "model_validate"):
        return BurnToolArgs.model_validate(args_or_kwargs)  # type: ignore[attr-defined]
    return BurnToolArgs(**args_or_kwargs)


def _resolve_path(p: Path) -> Path:
    """
    Resolve relative paths robustly for CLI usage:
    1) treat as relative to cwd
    2) if not exists, try ./out/<name>
    """
    if p.is_absolute():
        return p

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    out_candidate = (Path.cwd() / "out" / p.name).resolve()
    if out_candidate.exists():
        return out_candidate

    # give best-effort resolved path for error messages
    return cwd_candidate


def burn_subtitles_tool(**kwargs: Any) -> dict[str, Any]:
    args = _parse_tool_args(kwargs)

    video_path = _resolve_path(args.video_path)
    srt_path = _resolve_path(args.srt_path)

    if args.out_path is None:
        stem = video_path.stem
        out_path = video_path.with_name(f"{stem}.burned.mp4")
    else:
        out_path = args.out_path if args.out_path.is_absolute() else (Path.cwd() / args.out_path).resolve()

    p = burn_subtitles(
        video_path=video_path,
        srt_path=srt_path,
        out_path=out_path,
        ffmpeg_bin=args.ffmpeg_bin,
        force_style=args.force_style,
        crf=args.crf,
        preset=args.preset,
        copy_audio=args.copy_audio,
        audio_codec=args.audio_codec,
        audio_bitrate=args.audio_bitrate,
    )

    return {
        "ok": True,
        "out_video_path": str(p),
        "video_path": str(video_path),
        "srt_path": str(srt_path),
        "out_path": str(out_path),
    }
