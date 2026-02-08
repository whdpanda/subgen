# src/subgen/agent/tools/quality_check_subtitles_tool.py
from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from subgen.core.quality.report import QualityProfile, quality_check_srt


class QualityToolArgs(BaseModel):
    srt_path: Path = Field(..., description="Input SRT path")
    profile: str = Field("default", description="Quality profile name")

    # Optional overrides (PR#4a minimal profile knobs)
    max_cps: float = Field(16.0, description="Max chars-per-second")
    max_line_len: int = Field(18, description="Max line length")
    max_lines: int = Field(1, description="Max number of lines")
    min_dur_ms: int = Field(900, description="Min cue duration ms")
    max_dur_ms: int = Field(6500, description="Max cue duration ms")
    max_overlap_ms: int = Field(0, description="Max allowed overlap ms")

    out_dir: Optional[Path] = Field(
        None,
        description="Directory to write quality_report*.json (default: alongside srt; fallback: cwd/out)",
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


def _resolve_out_dir(
    *,
    requested_out_dir: Optional[Path],
    srt_path_resolved: Optional[Path],
) -> Path:
    """
    Best-effort:
      1) requested_out_dir if provided
      2) srt parent if known
      3) cwd/out
    Always mkdir(parents=True, exist_ok=True).
    """
    if requested_out_dir is not None:
        out_dir = requested_out_dir
        out_dir = out_dir if out_dir.is_absolute() else (Path.cwd() / out_dir).resolve()
    elif srt_path_resolved is not None:
        out_dir = srt_path_resolved.parent.resolve()
    else:
        out_dir = (Path.cwd() / "out").resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _report_filename(srt_path_resolved: Optional[Path]) -> str:
    stem = srt_path_resolved.stem if (srt_path_resolved is not None) else "unknown"
    return f"quality_report.{stem}.json"


def _write_report(out_dir: Path, payload: dict[str, Any]) -> Path:
    report_path = (out_dir / _report_filename(payload.get("_srt_path_resolved"))).resolve()  # type: ignore[arg-type]
    # remove helper key before writing
    payload.pop("_srt_path_resolved", None)

    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    PR#4c hard constraint:
      - Even on failure, MUST return best-effort + report_path (tool-written file path).
    """
    # We want to write a report even when parsing fails, so we keep best-effort context.
    srt_path_resolved: Optional[Path] = None
    out_dir: Optional[Path] = None
    profile_dict: Optional[dict[str, Any]] = None

    # 1) Parse / validate args (never raise)
    try:
        args = _parse_tool_args(kwargs)
    except ValidationError as e:
        # best-effort report location: kwargs.out_dir if any else cwd/out
        try:
            req_out_dir = kwargs.get("out_dir")
            if isinstance(req_out_dir, (str, Path)):
                out_dir = _resolve_out_dir(
                    requested_out_dir=Path(req_out_dir),
                    srt_path_resolved=None,
                )
            else:
                out_dir = _resolve_out_dir(requested_out_dir=None, srt_path_resolved=None)

            err_report = {
                "ts": _now_iso(),
                "ok": False,
                "kind": "quality_check_error",
                "error": {
                    "type": "quality.validation_error",
                    "message": "invalid tool arguments",
                    "details": {"errors": e.errors()},
                },
                "input": {k: str(v) for k, v in kwargs.items()},
                "_srt_path_resolved": None,
            }
            rp = _write_report(out_dir, err_report)
            return _fail_flat(
                err_type="quality.validation_error",
                message="invalid tool arguments",
                details={"errors": e.errors(), "input": {k: str(v) for k, v in kwargs.items()}},
                report_path=rp,
                report=err_report,
                meta={"profile": kwargs.get("profile", "default")},
            )
        except Exception as e2:
            return _fail_flat(
                err_type="quality.validation_error",
                message="invalid tool arguments (and failed to write report)",
                details={
                    "errors": e.errors(),
                    "report_write_error": str(e2) or e2.__class__.__name__,
                    "traceback": traceback.format_exc(),
                },
            )
    except Exception as e:
        try:
            out_dir = _resolve_out_dir(requested_out_dir=None, srt_path_resolved=None)
            err_report = {
                "ts": _now_iso(),
                "ok": False,
                "kind": "quality_check_error",
                "error": {
                    "type": "quality.args_parse_error",
                    "message": str(e) or e.__class__.__name__,
                    "details": {"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
                },
                "input": {k: str(v) for k, v in kwargs.items()},
                "_srt_path_resolved": None,
            }
            rp = _write_report(out_dir, err_report)
            return _fail_flat(
                err_type="quality.args_parse_error",
                message=str(e) or e.__class__.__name__,
                details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
                report_path=rp,
                report=err_report,
            )
        except Exception as e2:
            return _fail_flat(
                err_type="quality.args_parse_error",
                message="args parse error (and failed to write report)",
                details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc(), "report_write_error": str(e2)},
            )

    # 2) Resolve input path
    try:
        srt_path_resolved = args.srt_path if args.srt_path.is_absolute() else (Path.cwd() / args.srt_path).resolve()
        out_dir = _resolve_out_dir(requested_out_dir=args.out_dir, srt_path_resolved=srt_path_resolved)
    except Exception as e:
        # still must write report
        try:
            out_dir = _resolve_out_dir(requested_out_dir=args.out_dir, srt_path_resolved=None)
            err_report = {
                "ts": _now_iso(),
                "ok": False,
                "kind": "quality_check_error",
                "error": {
                    "type": "quality.path_resolve_error",
                    "message": str(e) or e.__class__.__name__,
                    "details": {"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
                },
                "input": {"srt_path": str(args.srt_path), "out_dir": str(args.out_dir) if args.out_dir else None},
                "_srt_path_resolved": None,
            }
            rp = _write_report(out_dir, err_report)
            return _fail_flat(
                err_type="quality.path_resolve_error",
                message=str(e) or e.__class__.__name__,
                details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
                report_path=rp,
                report=err_report,
                meta={"profile": args.profile},
            )
        except Exception as e2:
            return _fail_flat(
                err_type="quality.path_resolve_error",
                message="path resolve error (and failed to write report)",
                details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc(), "report_write_error": str(e2)},
            )

    if not srt_path_resolved.exists():
        # must write report_path even if missing input
        err_report = {
            "ts": _now_iso(),
            "ok": False,
            "kind": "quality_check_error",
            "error": {
                "type": "quality.input_not_found",
                "message": "srt_path does not exist",
                "details": {"srt_path": str(srt_path_resolved), "given": str(args.srt_path)},
            },
            "input": {"srt_path": str(args.srt_path), "profile": args.profile},
            "_srt_path_resolved": srt_path_resolved,
        }
        rp = _write_report(out_dir, err_report)
        return _fail_flat(
            err_type="quality.input_not_found",
            message="srt_path does not exist",
            details={"srt_path": str(srt_path_resolved), "given": str(args.srt_path)},
            report_path=rp,
            report=err_report,
            meta={"srt_path": str(srt_path_resolved), "profile": args.profile},
        )

    # 3) Build profile
    try:
        profile = _mk_profile_from_args(args.profile, args)
        profile_dict = profile.to_dict()
    except Exception as e:
        err_report = {
            "ts": _now_iso(),
            "ok": False,
            "kind": "quality_check_error",
            "error": {
                "type": "quality.profile_error",
                "message": str(e) or e.__class__.__name__,
                "details": {"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
            },
            "input": {"srt_path": str(srt_path_resolved), "profile": args.profile},
            "_srt_path_resolved": srt_path_resolved,
        }
        rp = _write_report(out_dir, err_report)
        return _fail_flat(
            err_type="quality.profile_error",
            message=str(e) or e.__class__.__name__,
            details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
            report_path=rp,
            report=err_report,
            meta={"srt_path": str(srt_path_resolved), "profile": args.profile},
        )

    # 4) Run quality + write report
    try:
        rep = quality_check_srt(str(srt_path_resolved), profile)

        report_dict = rep.to_dict()
        ok = rep.ok()
        summary = rep.summary()

        payload = {
            "ts": _now_iso(),
            "ok": ok,
            "kind": "quality_check",
            "srt_path": str(srt_path_resolved),
            "profile": profile_dict,
            "report": report_dict,
            "summary": summary,
            "_srt_path_resolved": srt_path_resolved,
        }
        rp = _write_report(out_dir, payload)

        return {
            "ok": ok,
            "report_path": str(rp),
            "report": report_dict,
            "summary": summary,
            "meta": {"srt_path": str(srt_path_resolved), "profile": profile_dict},
        }
    except Exception as e:
        # still must write error report
        err_report = {
            "ts": _now_iso(),
            "ok": False,
            "kind": "quality_check_error",
            "error": {
                "type": "quality.runtime_error",
                "message": str(e) or e.__class__.__name__,
                "details": {"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
            },
            "input": {"srt_path": str(srt_path_resolved), "profile": profile_dict or args.profile},
            "_srt_path_resolved": srt_path_resolved,
        }
        try:
            rp = _write_report(out_dir, err_report)
        except Exception:
            rp = None

        return _fail_flat(
            err_type="quality.runtime_error",
            message=str(e) or e.__class__.__name__,
            details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc(), "srt_path": str(srt_path_resolved)},
            report_path=rp,
            report=err_report if rp is not None else None,
            meta={"srt_path": str(srt_path_resolved), "profile": profile_dict or args.profile},
        )
