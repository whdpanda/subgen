# src/subgen/agent/tools/quality_tool.py
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from subgen.core.quality.report import QualityProfile, quality_check_srt


class QualityToolArgs(BaseModel):
    srt_path: Path = Field(..., description="Input SRT path")
    profile: str = Field("default", description="Quality profile name")

    # Optional overrides (PR#4a minimal profile knobs)
    max_cps: float = Field(17.0, description="Max chars-per-second")
    max_line_len: int = Field(42, description="Max line length")
    max_lines: int = Field(2, description="Max number of lines")
    min_dur_ms: int = Field(700, description="Min cue duration ms")
    max_dur_ms: int = Field(7000, description="Max cue duration ms")
    max_overlap_ms: int = Field(0, description="Max allowed overlap ms")

    out_dir: Optional[Path] = Field(
        None,
        description="Directory to write quality_report.json (default: alongside srt)",
    )


def _parse_tool_args(args_or_kwargs: Union[QualityToolArgs, dict[str, Any]]) -> QualityToolArgs:
    if isinstance(args_or_kwargs, QualityToolArgs):
        return args_or_kwargs
    if hasattr(QualityToolArgs, "model_validate"):
        return QualityToolArgs.model_validate(args_or_kwargs)  # type: ignore[attr-defined]
    return QualityToolArgs(**args_or_kwargs)


def _mk_profile_from_args(name: str, args: QualityToolArgs) -> QualityProfile:
    return QualityProfile(
        name=name,
        max_cps=float(args.max_cps),
        max_line_len=int(args.max_line_len),
        max_lines=int(args.max_lines),
        min_dur_ms=int(args.min_dur_ms),
        max_dur_ms=int(args.max_dur_ms),
        max_overlap_ms=int(args.max_overlap_ms),
    )


def _fail_flat(
    *,
    err_type: str,
    message: str,
    details: Optional[dict[str, Any]] = None,
    report_path: Optional[Union[str, Path]] = None,
    report: Optional[dict[str, Any]] = None,
    summary: Optional[dict[str, Any]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    m = dict(meta or {})
    m["error"] = {
        "type": err_type,
        "message": message,
        "details": details or {},
    }
    return {
        "ok": False,
        "report_path": str(report_path) if report_path is not None else None,
        "report": report,
        "summary": summary,
        "meta": m,
    }


def quality_check_subtitles_tool(**kwargs: Any) -> dict[str, Any]:
    """
    Flat schema (NO envelope):
      {"ok","report_path","report","summary","meta"}

    - ok: True if no violations
    - report_path: written JSON path (quality_report.json)
    - report: dict
    - summary: dict
    - meta: includes srt_path/profile and on failure meta["error"]
    """
    # 1) Parse / validate args (never raise)
    try:
        args = _parse_tool_args(kwargs)
    except ValidationError as e:
        return _fail_flat(
            err_type="quality.validation_error",
            message="invalid tool arguments",
            details={"errors": e.errors(), "input": {k: str(v) for k, v in kwargs.items()}},
        )
    except Exception as e:
        return _fail_flat(
            err_type="quality.args_parse_error",
            message=str(e) or e.__class__.__name__,
            details={
                "exception_class": e.__class__.__name__,
                "traceback": traceback.format_exc(),
                "input": {k: str(v) for k, v in kwargs.items()},
            },
        )

    # 2) Resolve paths
    srt_path = args.srt_path if args.srt_path.is_absolute() else (Path.cwd() / args.srt_path).resolve()
    if not srt_path.exists():
        return _fail_flat(
            err_type="quality.input_not_found",
            message="srt_path does not exist",
            details={"srt_path": str(srt_path), "given": str(args.srt_path)},
            meta={"srt_path": str(srt_path), "profile": args.profile},
        )

    # 3) Build profile
    try:
        profile = _mk_profile_from_args(args.profile, args)
    except Exception as e:
        return _fail_flat(
            err_type="quality.profile_error",
            message=str(e) or e.__class__.__name__,
            details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
            meta={"srt_path": str(srt_path), "profile": args.profile},
        )

    # 4) Run quality + write report
    try:
        rep = quality_check_srt(str(srt_path), profile)

        out_dir = args.out_dir if args.out_dir is not None else srt_path.parent
        out_dir = out_dir if out_dir.is_absolute() else (Path.cwd() / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        report_path = (out_dir / "quality_report.json").resolve()
        report_dict = rep.to_dict()
        report_path.write_text(json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "ok": rep.ok(),
            "report_path": str(report_path),
            "report": report_dict,
            "summary": rep.summary(),
            "meta": {"srt_path": str(srt_path), "profile": profile.to_dict()},
        }
    except Exception as e:
        return _fail_flat(
            err_type="quality.runtime_error",
            message=str(e) or e.__class__.__name__,
            details={
                "exception_class": e.__class__.__name__,
                "traceback": traceback.format_exc(),
                "srt_path": str(srt_path),
            },
            meta={"srt_path": str(srt_path), "profile": profile.to_dict()},
        )
