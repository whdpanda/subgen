# src/subgen/agent/tools/quality_check_subtitles_tool.py
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from subgen.agent.tools.quality_profiles import _mk_profile_from_args
from subgen.agent.tools.schemas import QualityToolArgs, parse_tool_args
from subgen.agent.tools.tool_utils import abs_path, fail_flat, now_iso, resolve_out_dir
from subgen.core.quality.report import quality_check_srt


def _report_filename(srt_path_resolved: Optional[Path]) -> str:
    stem = srt_path_resolved.stem if (srt_path_resolved is not None) else "unknown"
    return f"quality_report.{stem}.json"


def _write_report(out_dir: Path, payload: dict[str, Any]) -> Path:
    report_path = (out_dir / _report_filename(payload.get("_srt_path_resolved"))).resolve()  # type: ignore[arg-type]
    # remove helper key before writing
    payload.pop("_srt_path_resolved", None)

    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


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
        args = parse_tool_args(QualityToolArgs, kwargs)
    except ValidationError as e:
        # best-effort report location: kwargs.out_dir if any else cwd/out
        try:
            req_out_dir = kwargs.get("out_dir")
            if isinstance(req_out_dir, (str, Path)):
                out_dir = resolve_out_dir(requested_out_dir=Path(req_out_dir), fallback_parent=None)
            else:
                out_dir = resolve_out_dir(requested_out_dir=None, fallback_parent=None)

            err_report = {
                "ts": now_iso(),
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
            return fail_flat(
                base={
                    "report_path": str(rp),
                    "report": err_report,
                    "summary": None,
                },
                err_type="quality.validation_error",
                message="invalid tool arguments",
                details={"errors": e.errors(), "input": {k: str(v) for k, v in kwargs.items()}},
                meta={"profile": kwargs.get("profile", "default")},
            )
        except Exception as e2:
            return fail_flat(
                base={
                    "report_path": None,
                    "report": None,
                    "summary": None,
                },
                err_type="quality.validation_error",
                message="invalid tool arguments (and failed to write report)",
                details={
                    "errors": e.errors(),
                    "report_write_error": str(e2) or e2.__class__.__name__,
                    "traceback": traceback.format_exc(),
                },
                meta={"profile": kwargs.get("profile", "default")},
            )
    except Exception as e:
        try:
            out_dir = resolve_out_dir(requested_out_dir=None, fallback_parent=None)
            err_report = {
                "ts": now_iso(),
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
            return fail_flat(
                base={
                    "report_path": str(rp),
                    "report": err_report,
                    "summary": None,
                },
                err_type="quality.args_parse_error",
                message=str(e) or e.__class__.__name__,
                details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
            )
        except Exception as e2:
            return fail_flat(
                base={
                    "report_path": None,
                    "report": None,
                    "summary": None,
                },
                err_type="quality.args_parse_error",
                message="args parse error (and failed to write report)",
                details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc(), "report_write_error": str(e2)},
            )

    # 2) Resolve input path
    try:
        srt_path_resolved = abs_path(args.srt_path)
        out_dir = resolve_out_dir(
            requested_out_dir=args.out_dir,
            fallback_parent=srt_path_resolved.parent if srt_path_resolved is not None else None,
        )
    except Exception as e:
        # still must write report
        try:
            out_dir = resolve_out_dir(requested_out_dir=args.out_dir, fallback_parent=None)
            err_report = {
                "ts": now_iso(),
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
            return fail_flat(
                base={
                    "report_path": str(rp),
                    "report": err_report,
                    "summary": None,
                },
                err_type="quality.path_resolve_error",
                message=str(e) or e.__class__.__name__,
                details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
                meta={"profile": args.profile},
            )
        except Exception as e2:
            return fail_flat(
                base={
                    "report_path": None,
                    "report": None,
                    "summary": None,
                },
                err_type="quality.path_resolve_error",
                message="path resolve error (and failed to write report)",
                details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc(), "report_write_error": str(e2)},
                meta={"profile": args.profile},
            )

    if not srt_path_resolved.exists():
        # must write report_path even if missing input
        err_report = {
            "ts": now_iso(),
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
        return fail_flat(
            base={
                "report_path": str(rp),
                "report": err_report,
                "summary": None,
            },
            err_type="quality.input_not_found",
            message="srt_path does not exist",
            details={"srt_path": str(srt_path_resolved), "given": str(args.srt_path)},
            meta={"srt_path": str(srt_path_resolved), "profile": args.profile},
        )

    # 3) Build profile
    try:
        profile = _mk_profile_from_args(args.profile, args)
        profile_dict = profile.to_dict()
    except Exception as e:
        err_report = {
            "ts": now_iso(),
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
        return fail_flat(
            base={
                "report_path": str(rp),
                "report": err_report,
                "summary": None,
            },
            err_type="quality.profile_error",
            message=str(e) or e.__class__.__name__,
            details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc()},
            meta={"srt_path": str(srt_path_resolved), "profile": args.profile},
        )

    # 4) Run quality + write report
    try:
        rep = quality_check_srt(str(srt_path_resolved), profile)

        report_dict = rep.to_dict()
        ok = rep.ok()
        summary = rep.summary()

        payload = {
            "ts": now_iso(),
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
            "ts": now_iso(),
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

        return fail_flat(
            base={
                "report_path": str(rp) if rp is not None else None,
                "report": err_report if rp is not None else None,
                "summary": None,
            },
            err_type="quality.runtime_error",
            message=str(e) or e.__class__.__name__,
            details={"exception_class": e.__class__.__name__, "traceback": traceback.format_exc(), "srt_path": str(srt_path_resolved)},
            meta={"srt_path": str(srt_path_resolved), "profile": profile_dict or args.profile},
        )
