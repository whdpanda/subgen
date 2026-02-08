# src/subgen/agent/tools/fix_subtitles_tool.py
from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from subgen.core.quality.pipeline import FixBudget, apply_fixes
from subgen.core.quality.report import QualityProfile, quality_check_srt
from subgen.core.subtitle.srt_io import read_srt, write_srt


class FixToolArgs(BaseModel):
    srt_path: Path = Field(..., description="Input SRT path")

    profile: str = Field("default", description="Quality profile name")
    max_cps: float = Field(16.0, description="Max chars-per-second")
    max_line_len: int = Field(18, description="Max line length")
    max_lines: int = Field(1, description="Max number of lines")
    min_dur_ms: int = Field(900, description="Min cue duration ms")
    max_dur_ms: int = Field(6500, description="Max cue duration ms")
    max_overlap_ms: int = Field(0, description="Max allowed overlap ms")

    max_passes: int = Field(2, description="Max fix passes (core deterministic)")
    out_path: Optional[Path] = Field(
        None,
        description="Output fixed SRT path (default: <stem>.fixed.srt alongside input)",
    )
    out_dir: Optional[Path] = Field(
        None,
        description="Directory for fixed outputs (default: alongside srt)",
    )


def _parse_tool_args(args_or_kwargs: Union[FixToolArgs, dict[str, Any]]) -> FixToolArgs:
    if isinstance(args_or_kwargs, FixToolArgs):
        return args_or_kwargs
    if hasattr(FixToolArgs, "model_validate"):
        return FixToolArgs.model_validate(args_or_kwargs)  # type: ignore[attr-defined]
    return FixToolArgs(**args_or_kwargs)


def _mk_profile_from_args(name: str, args: FixToolArgs) -> QualityProfile:
    return QualityProfile(
        name=name,
        max_cps=float(args.max_cps),
        max_line_len=int(args.max_line_len),
        max_lines=int(args.max_lines),
        min_dur_ms=int(args.min_dur_ms),
        max_dur_ms=int(args.max_dur_ms),
        max_overlap_ms=int(args.max_overlap_ms),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_out_dir(srt_path_resolved: Path, requested_out_dir: Optional[Path]) -> Path:
    if requested_out_dir is not None:
        out_dir = requested_out_dir
        out_dir = out_dir if out_dir.is_absolute() else (Path.cwd() / out_dir).resolve()
    else:
        out_dir = srt_path_resolved.parent.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _default_fixed_path(out_dir: Path, srt_path_resolved: Path) -> Path:
    return (out_dir / f"{srt_path_resolved.stem}.fixed.srt").resolve()


def _before_json_path(out_dir: Path, srt_path_resolved: Path) -> Path:
    return (out_dir / f"{srt_path_resolved.stem}.quality_before.json").resolve()


def _after_json_path(out_dir: Path, srt_path_resolved: Path) -> Path:
    return (out_dir / f"{srt_path_resolved.stem}.quality_after.json").resolve()


def _after_error_json_path(out_dir: Path, srt_path_resolved: Path) -> Path:
    return (out_dir / f"{srt_path_resolved.stem}.quality_after_error.json").resolve()


def _fail_flat(
    *,
    err_type: str,
    message: str,
    details: Optional[dict[str, Any]] = None,
    fixed_srt_path: Optional[Union[str, Path]] = None,
    quality_before_path: Optional[Union[str, Path]] = None,
    quality_after_path: Optional[Union[str, Path]] = None,
    changed: bool = False,
    actions: Optional[list[dict[str, Any]]] = None,
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
        "fixed_srt_path": str(fixed_srt_path) if fixed_srt_path is not None else None,
        "quality_before_path": str(quality_before_path) if quality_before_path is not None else None,
        "quality_after_path": str(quality_after_path) if quality_after_path is not None else None,
        "changed": bool(changed),
        "actions": actions or [],
        "meta": m,
    }


def fix_subtitles_tool(**kwargs: Any) -> dict[str, Any]:
    """
    Flat schema (NO envelope):
      {"ok","fixed_srt_path","quality_before_path","quality_after_path","changed","actions","meta"}

    PR#4c best-effort:
      - If input exists but fixing fails, return fixed_srt_path=input path (real path, not hallucinated)
      - Try to write before/after quality json; if after cannot be computed, write an error-json and return its path.
    """
    # 1) Parse / validate args (never raise)
    try:
        args = _parse_tool_args(kwargs)
    except ValidationError as e:
        return _fail_flat(
            err_type="fix.validation_error",
            message="invalid tool arguments",
            details={"errors": e.errors(), "input": {k: str(v) for k, v in kwargs.items()}},
        )
    except Exception as e:
        return _fail_flat(
            err_type="fix.args_parse_error",
            message=str(e) or e.__class__.__name__,
            details={
                "exception_class": e.__class__.__name__,
                "traceback": traceback.format_exc(),
                "input": {k: str(v) for k, v in kwargs.items()},
            },
        )

    # 2) Resolve input path
    srt_path = args.srt_path if args.srt_path.is_absolute() else (Path.cwd() / args.srt_path).resolve()
    if not srt_path.exists():
        return _fail_flat(
            err_type="fix.input_not_found",
            message="srt_path does not exist",
            details={"srt_path": str(srt_path), "given": str(args.srt_path)},
            meta={"input_srt_path": str(srt_path), "profile": args.profile},
        )

    # 3) Build profile
    try:
        profile = _mk_profile_from_args(args.profile, args)
        profile_dict = profile.to_dict()
    except Exception as e:
        return _fail_flat(
            err_type="fix.profile_error",
            message=str(e) or e.__class__.__name__,
            details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
            fixed_srt_path=str(srt_path),  # best-effort: return existing input
            meta={"input_srt_path": str(srt_path), "profile": args.profile},
        )

    # 4) Resolve output dir/path
    try:
        out_dir = _resolve_out_dir(srt_path, args.out_dir)

        if args.out_path is None:
            fixed_path = _default_fixed_path(out_dir, srt_path)
        else:
            fixed_path = args.out_path if args.out_path.is_absolute() else (Path.cwd() / args.out_path).resolve()

        before_path = _before_json_path(out_dir, srt_path)
        after_path = _after_json_path(out_dir, srt_path)
        after_err_path = _after_error_json_path(out_dir, srt_path)
    except Exception as e:
        return _fail_flat(
            err_type="fix.output_path_error",
            message=str(e) or e.__class__.__name__,
            details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
            fixed_srt_path=str(srt_path),  # best-effort
            meta={"input_srt_path": str(srt_path), "profile": profile_dict},
        )

    # 5) Fix (deterministic) + before/after reports
    actions: list[dict[str, Any]] = []
    changed = False

    try:
        # Before report (best-effort write)
        before_rep = quality_check_srt(str(srt_path), profile)
        before_path.write_text(json.dumps(before_rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        # Apply fixes
        doc = read_srt(srt_path).sorted()
        budget = FixBudget(max_passes=int(max(0, args.max_passes)))
        res = apply_fixes(doc, profile, budget=budget)

        # Write fixed srt
        write_srt(res.fixed_doc, fixed_path)

        # Actions
        actions = [a.to_dict() for a in res.actions]
        changed = len(actions) > 0

        # After report
        after_rep = quality_check_srt(str(fixed_path), profile)
        after_path.write_text(json.dumps(after_rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "ok": after_rep.ok(),
            "fixed_srt_path": str(fixed_path),
            "quality_before_path": str(before_path),
            "quality_after_path": str(after_path),
            "changed": changed,
            "actions": actions,
            "meta": {
                "input_srt_path": str(srt_path),
                "profile": profile_dict,
                "passes": res.passes,
                "before_summary": before_rep.summary(),
                "after_summary": after_rep.summary(),
            },
        }

    except Exception as e:
        # best-effort: if fixed_path exists, use it; else fallback to input
        fallback_fixed = fixed_path if (fixed_path is not None and fixed_path.exists()) else srt_path

        # best-effort: write after_error report so caller still gets a real path
        try:
            after_error_payload = {
                "ts": _now_iso(),
                "ok": False,
                "kind": "fix_after_error",
                "input_srt_path": str(srt_path),
                "fixed_srt_path": str(fallback_fixed),
                "profile": profile_dict,
                "error": {
                    "type": "fix.runtime_error",
                    "message": str(e) or e.__class__.__name__,
                    "details": {"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
                },
            }
            after_err_path.write_text(json.dumps(after_error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            qa_path: Union[str, Path] = after_err_path
        except Exception:
            qa_path = None  # last resort

        return _fail_flat(
            err_type="fix.runtime_error",
            message=str(e) or e.__class__.__name__,
            details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
            fixed_srt_path=str(fallback_fixed),
            quality_before_path=str(before_path) if before_path.exists() else None,
            quality_after_path=str(qa_path) if qa_path is not None else None,
            changed=changed,
            actions=actions,
            meta={"input_srt_path": str(srt_path), "fixed_srt_path": str(fallback_fixed), "profile": profile_dict},
        )
