# src/subgen/agent/tools/burn_subtitles_tool.py
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import ValidationError

from subgen.agent.tools.schemas import BurnToolArgs, parse_tool_args
from subgen.core.video.burn import burn_subtitles


# -------------------------
# JSON-safe helpers
# -------------------------
def _safe_str_dict(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


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

    return cwd_candidate


# -------------------------
# Flat schema helpers (TOOL-side)
# -------------------------
def _ok_flat(*, out_video_path: Union[str, Path], artifacts: Optional[dict[str, Any]] = None, meta: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return {
        "ok": True,
        "out_video_path": str(out_video_path),
        "artifacts": _safe_str_dict(artifacts or {}),
        "meta": _safe_str_dict(meta or {}),
    }


def _fail_flat(
    *,
    err_type: str,
    message: str,
    details: Optional[dict[str, Any]] = None,
    # keep schema stable
    out_video_path: Optional[Union[str, Path]] = None,
    artifacts: Optional[dict[str, Any]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    m = dict(meta or {})
    m["error"] = {
        "type": err_type,
        "message": message,
        "details": _safe_str_dict(details or {}),
    }
    return {
        "ok": False,
        "out_video_path": str(out_video_path) if out_video_path is not None else None,
        "artifacts": _safe_str_dict(artifacts or {}),
        "meta": _safe_str_dict(m),
    }


def burn_subtitles_tool(**kwargs: Any) -> dict[str, Any]:
    """
    LangChain tool entrypoint.

    Returns stable FLAT schema (NO envelope):
      - success/failure keys are always:
        {"ok","out_video_path","artifacts","meta"}
      - failure details are stored in meta["error"]

    Never raises to runtime.
    """
    # 1) Parse / validate args
    try:
        args = parse_tool_args(BurnToolArgs, kwargs)
    except ValidationError as e:
        return _fail_flat(
            err_type="burn.validation_error",
            message="invalid tool arguments",
            details={"errors": e.errors(), "input": _safe_str_dict(kwargs)},
        )
    except Exception as e:
        return _fail_flat(
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

    # 3) Pre-check inputs
    if not video_path.exists():
        return _fail_flat(
            err_type="burn.input_not_found",
            message="video_path does not exist",
            details={"video_path": str(video_path), "given": str(args.video_path)},
            meta={"video_path": str(video_path), "srt_path": str(srt_path)},
        )
    if not srt_path.exists():
        return _fail_flat(
            err_type="burn.input_not_found",
            message="srt_path does not exist",
            details={"srt_path": str(srt_path), "given": str(args.srt_path)},
            meta={"video_path": str(video_path), "srt_path": str(srt_path)},
        )

    # 4) Resolve out_path
    try:
        if args.out_path is None:
            stem = video_path.stem
            out_path = video_path.with_name(f"{stem}.burned.mp4")
        else:
            out_path = args.out_path if args.out_path.is_absolute() else (Path.cwd() / args.out_path).resolve()
    except Exception as e:
        return _fail_flat(
            err_type="burn.output_path_error",
            message=str(e) or e.__class__.__name__,
            details={
                "exception_class": e.__class__.__name__,
                "traceback": traceback.format_exc(),
                "video_path": str(video_path),
                "out_path": str(args.out_path) if args.out_path is not None else None,
            },
            meta={"video_path": str(video_path), "srt_path": str(srt_path)},
        )

    # 5) Burn
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
        return _fail_flat(
            err_type="burn.not_found",
            message=str(e) or "file not found",
            details={
                "video_path": str(video_path),
                "srt_path": str(srt_path),
                "out_path": str(out_path),
                "ffmpeg_bin": args.ffmpeg_bin,
                "traceback": traceback.format_exc(),
            },
            meta={
                "video_path": str(video_path),
                "srt_path": str(srt_path),
                "out_path": str(out_path),
                "ffmpeg_bin": args.ffmpeg_bin,
            },
        )
    except Exception as e:
        return _fail_flat(
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
            meta={
                "video_path": str(video_path),
                "srt_path": str(srt_path),
                "out_path": str(out_path),
                "ffmpeg_bin": args.ffmpeg_bin,
            },
        )

    # 6) Success
    return _ok_flat(
        out_video_path=p,
        meta={
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
