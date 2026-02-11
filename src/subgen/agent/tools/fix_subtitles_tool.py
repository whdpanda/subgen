# src/subgen/agent/tools/fix_subtitles_tool.py
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import ValidationError

from subgen.agent.tools.quality_profiles import _mk_profile_from_args
from subgen.agent.tools.schemas import FixToolArgs, parse_tool_args
from subgen.agent.tools.tool_utils import abs_path, fail_flat, now_iso, resolve_out_dir
from subgen.core.quality.pipeline import FixBudget, apply_fixes
from subgen.core.quality.report import quality_check_srt
from subgen.core.subtitle.srt_io import read_srt, write_srt


_FIXED_SUFFIX = ".fixed"


def _stable_base_stem(srt_path_resolved: Path) -> str:
    """
    Make the output basename stable across repeated fixing.
    Example:
      - input:  a.srt        -> base: a
      - input:  a.fixed.srt  -> base: a
      - input:  a.fixed.fixed.srt -> base: a
    """
    stem = srt_path_resolved.stem
    while stem.endswith(_FIXED_SUFFIX):
        stem = stem[: -len(_FIXED_SUFFIX)]
    return stem


def _default_fixed_path(out_dir: Path, srt_path_resolved: Path) -> Path:
    base = _stable_base_stem(srt_path_resolved)
    return (out_dir / f"{base}.fixed.srt").resolve()


def _before_json_path(out_dir: Path, srt_path_resolved: Path) -> Path:
    base = _stable_base_stem(srt_path_resolved)
    return (out_dir / f"{base}.quality_before.json").resolve()


def _after_json_path(out_dir: Path, srt_path_resolved: Path) -> Path:
    base = _stable_base_stem(srt_path_resolved)
    return (out_dir / f"{base}.quality_after.json").resolve()


def _after_error_json_path(out_dir: Path, srt_path_resolved: Path) -> Path:
    base = _stable_base_stem(srt_path_resolved)
    return (out_dir / f"{base}.quality_after_error.json").resolve()


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
        args = parse_tool_args(FixToolArgs, kwargs)
    except ValidationError as e:
        return fail_flat(
            base={
                "fixed_srt_path": None,
                "quality_before_path": None,
                "quality_after_path": None,
                "changed": False,
                "actions": [],
            },
            err_type="fix.validation_error",
            message="invalid tool arguments",
            details={"errors": e.errors(), "input": {k: str(v) for k, v in kwargs.items()}},
        )
    except Exception as e:
        return fail_flat(
            base={
                "fixed_srt_path": None,
                "quality_before_path": None,
                "quality_after_path": None,
                "changed": False,
                "actions": [],
            },
            err_type="fix.args_parse_error",
            message=str(e) or e.__class__.__name__,
            details={
                "exception_class": e.__class__.__name__,
                "traceback": traceback.format_exc(),
                "input": {k: str(v) for k, v in kwargs.items()},
            },
        )

    # 2) Resolve input path
    srt_path = abs_path(args.srt_path)
    if not srt_path.exists():
        return fail_flat(
            base={
                "fixed_srt_path": None,
                "quality_before_path": None,
                "quality_after_path": None,
                "changed": False,
                "actions": [],
            },
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
        return fail_flat(
            base={
                "fixed_srt_path": str(srt_path),  # best-effort: return existing input
                "quality_before_path": None,
                "quality_after_path": None,
                "changed": False,
                "actions": [],
            },
            err_type="fix.profile_error",
            message=str(e) or e.__class__.__name__,
            details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
            meta={"input_srt_path": str(srt_path), "profile": args.profile},
        )

    # 4) Resolve output dir/path
    try:
        out_dir = resolve_out_dir(requested_out_dir=args.out_dir, fallback_parent=srt_path.parent)

        # ✅ Stable target path: never stack suffix on repeated fixes (overwrite)
        if args.out_path is None:
            fixed_path = _default_fixed_path(out_dir, srt_path)
        else:
            fixed_path = abs_path(args.out_path)

        before_path = _before_json_path(out_dir, srt_path)
        after_path = _after_json_path(out_dir, srt_path)
        after_err_path = _after_error_json_path(out_dir, srt_path)

        # ensure dirs exist (in case out_dir is new)
        out_dir.mkdir(parents=True, exist_ok=True)
        fixed_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return fail_flat(
            base={
                "fixed_srt_path": str(srt_path),  # best-effort
                "quality_before_path": None,
                "quality_after_path": None,
                "changed": False,
                "actions": [],
            },
            err_type="fix.output_path_error",
            message=str(e) or e.__class__.__name__,
            details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
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

        # Write fixed srt (overwrite)
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
                "ts": now_iso(),
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

        return fail_flat(
            base={
                "fixed_srt_path": str(fallback_fixed),
                "quality_before_path": str(before_path) if before_path.exists() else None,
                "quality_after_path": str(qa_path) if qa_path is not None else None,
                "changed": bool(changed),
                "actions": actions or [],
            },
            err_type="fix.runtime_error",
            message=str(e) or e.__class__.__name__,
            details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
            meta={"input_srt_path": str(srt_path), "fixed_srt_path": str(fallback_fixed), "profile": profile_dict},
        )
