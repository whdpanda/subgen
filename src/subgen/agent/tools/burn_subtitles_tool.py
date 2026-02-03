from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from subgen.core.video.burn import burn_subtitles


class BurnToolArgs(BaseModel):
    video_path: Path = Field(..., description="Input video path")
    srt_path: Path = Field(..., description="Subtitle SRT path to burn-in")
    out_path: Optional[Path] = Field(
        None, description="Output video path (mp4). If omitted, auto-generate."
    )

    ffmpeg_bin: str = Field("ffmpeg", description="ffmpeg binary name/path")
    force_style: Optional[str] = Field(None, description="ASS force_style string")

    crf: int = Field(18, description="x264 CRF")
    preset: str = Field("veryfast", description="x264 preset")

    copy_audio: bool = Field(False, description="Copy audio stream if possible")
    audio_codec: str = Field("aac", description="Audio codec when not copying")
    audio_bitrate: str = Field("192k", description="Audio bitrate when encoding")


def _ok(data: Any) -> dict[str, Any]:
    """Success envelope."""
    return {"ok": True, "data": data, "error": None}


def _fail(err_type: str, message: str, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Failure envelope."""
    return {
        "ok": False,
        "data": None,
        "error": {
            "type": err_type,
            "message": message,
            "details": details or {},
        },
    }


def _safe_str_dict(d: dict[str, Any]) -> dict[str, Any]:
    """
    Make kwargs safe-ish for JSON serialization:
    - Path -> str
    - everything else unchanged (best-effort)
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


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
    """
    LangChain tool entrypoint.

    Returns unified envelope:
      - success: {"ok": True, "data": {...}, "error": None}
      - failure: {"ok": False, "data": None, "error": {...}}

    Never raises to runtime.
    """
    # 1) Parse / validate args
    try:
        args = _parse_tool_args(kwargs)
    except ValidationError as e:
        return _fail(
            err_type="burn.validation_error",
            message="invalid tool arguments",
            details={
                "errors": e.errors(),
                "input": _safe_str_dict(kwargs),
            },
        )
    except Exception as e:
        return _fail(
            err_type="burn.args_parse_error",
            message=str(e) or e.__class__.__name__,
            details={
                "exception_class": e.__class__.__name__,
                "traceback": traceback.format_exc(),
                "input": _safe_str_dict(kwargs),
            },
        )

    # 2) Resolve paths
    video_path = _resolve_path(args.video_path)
    srt_path = _resolve_path(args.srt_path)

    # 3) Pre-check inputs (更利于 PR#4 自愈：不用等 ffmpeg 报错才知道路径不对)
    if not video_path.exists():
        return _fail(
            err_type="burn.input_not_found",
            message="video_path does not exist",
            details={"video_path": str(video_path), "given": str(args.video_path)},
        )
    if not srt_path.exists():
        return _fail(
            err_type="burn.input_not_found",
            message="srt_path does not exist",
            details={"srt_path": str(srt_path), "given": str(args.srt_path)},
        )

    # 4) Resolve out_path
    try:
        if args.out_path is None:
            stem = video_path.stem
            out_path = video_path.with_name(f"{stem}.burned.mp4")
        else:
            out_path = args.out_path if args.out_path.is_absolute() else (Path.cwd() / args.out_path).resolve()
    except Exception as e:
        return _fail(
            err_type="burn.output_path_error",
            message=str(e) or e.__class__.__name__,
            details={
                "exception_class": e.__class__.__name__,
                "traceback": traceback.format_exc(),
                "video_path": str(video_path),
                "out_path": str(args.out_path) if args.out_path is not None else None,
            },
        )

    # 5) Burn (never raise)
    try:
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
    except FileNotFoundError as e:
        # 常见：ffmpeg_bin 找不到 或输入路径问题
        return _fail(
            err_type="burn.not_found",
            message=str(e) or "file not found",
            details={
                "video_path": str(video_path),
                "srt_path": str(srt_path),
                "out_path": str(out_path),
                "ffmpeg_bin": args.ffmpeg_bin,
                "traceback": traceback.format_exc(),
            },
        )
    except Exception as e:
        # 如果你的 burn_subtitles 内部用 subprocess.run(capture_output=True)，
        # 强烈建议它把 stderr 附在异常里（或返回结构里），这里就能塞进 details，利于自愈。
        return _fail(
            err_type="burn.runtime_error",
            message=str(e) or e.__class__.__name__,
            details={
                "exception_class": e.__class__.__name__,
                "traceback": traceback.format_exc(),
                "video_path": str(video_path),
                "srt_path": str(srt_path),
                "out_path": str(out_path),
                "ffmpeg_bin": args.ffmpeg_bin,
                "force_style": args.force_style,
                "crf": args.crf,
                "preset": args.preset,
                "copy_audio": args.copy_audio,
                "audio_codec": args.audio_codec,
                "audio_bitrate": args.audio_bitrate,
            },
        )

    # 6) Success
    data = {
        "out_video_path": str(p),
        "video_path": str(video_path),
        "srt_path": str(srt_path),
        "out_path": str(out_path),
    }
    return _ok(data)
