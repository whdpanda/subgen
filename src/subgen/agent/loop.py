# src/subgen/agent/loop.py
from __future__ import annotations

import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from subgen.agent.tools.tool_names import (
    RUN_SUBGEN_PIPELINE,
    QUALITY_CHECK_SUBTITLES,
    FIX_SUBTITLES,
    BURN_SUBTITLES,
)
from subgen.utils.logger import get_logger

logger = get_logger()


_FIXED_SUFFIX = ".fixed"


def _stable_base_stem_from_path(p: Path) -> str:
    stem = p.stem
    while stem.endswith(_FIXED_SUFFIX):
        stem = stem[: -len(_FIXED_SUFFIX)]
    return stem


def _stable_fixed_out_path(initial_srt_path: str) -> str:
    """
    Always write fixes to ONE stable target path (overwrite), derived from the initial SRT:
      initial: a.srt           -> a.fixed.srt
      initial: a.fixed.srt     -> a.fixed.srt
      initial: a.fixed.fixed.srt -> a.fixed.srt
    """
    p = Path(initial_srt_path).expanduser().resolve()
    base = _stable_base_stem_from_path(p)
    return str((p.parent / f"{base}.fixed.srt").resolve())


@dataclass
class QualityPass:
    """One check/fix iteration record (best-effort)."""

    pass_index: int
    checked_srt_path: str
    quality_ok: bool
    report_path: Optional[str] = None
    fixed_srt_path: Optional[str] = None
    errors: List[dict] = field(default_factory=list)


@dataclass
class LoopResult:
    """Agent-facing result (paths must be tool-produced)."""

    ok: bool
    primary_path: Optional[str]
    srt_path: Optional[str]
    report_path: Optional[str]
    out_video_path: Optional[str]
    passes_used: int
    history: List[QualityPass] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopConfig:
    """Runtime policy knobs for PR#4c."""

    max_passes: int = 3
    emit_default: str = "zh-only"
    target_lang_default: str = "zh"
    zh_layout_default: bool = True


def _is_envelope(x: Any) -> bool:
    return isinstance(x, dict) and set(x.keys()) >= {"ok", "data", "error"}


def _flatten_envelope(tool_name: str, env: Dict[str, Any]) -> Dict[str, Any]:
    ok = bool(env.get("ok"))
    data = env.get("data")
    err = env.get("error")

    if isinstance(data, dict):
        out = dict(data)
        out["ok"] = ok
        if err:
            meta = out.get("meta") if isinstance(out.get("meta"), dict) else {}
            meta["error"] = err
            out["meta"] = meta
        return out

    return {
        "ok": ok,
        "meta": {
            "error": err
            or {
                "type": f"{tool_name}.legacy_envelope_shape",
                "message": "non-dict envelope data",
            }
        },
    }


def safe_invoke_flat(tool: Any, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        res = tool.invoke(tool_args)

        if _is_envelope(res):
            return _flatten_envelope(tool_name, res)

        if isinstance(res, dict):
            return res

        return {"ok": True, "result": res}

    except ValidationError as e:
        return {
            "ok": False,
            "meta": {
                "error": {
                    "type": f"{tool_name}.validation_error",
                    "message": "tool args validation failed",
                    "details": {"errors": e.errors(), "args": tool_args},
                }
            },
        }
    except Exception as e:
        return {
            "ok": False,
            "meta": {
                "error": {
                    "type": f"{tool_name}.invoke_error",
                    "message": str(e) or e.__class__.__name__,
                    "details": {
                        "exception_class": e.__class__.__name__,
                        "traceback": traceback.format_exc(),
                        "args": tool_args,
                    },
                }
            },
        }


def inject_default_zh_only_pipeline_args(args: Dict[str, Any], cfg: LoopConfig) -> Dict[str, Any]:
    out = dict(args or {})
    out.setdefault("target_lang", cfg.target_lang_default)
    out.setdefault("emit", cfg.emit_default)
    out.setdefault("zh_layout", cfg.zh_layout_default)
    return out


def pick_srt_path_from_pipeline_result(pipeline_res: Dict[str, Any]) -> Optional[str]:
    primary = pipeline_res.get("primary_path")
    if isinstance(primary, str) and primary:
        return primary

    srt_paths = pipeline_res.get("srt_paths")
    if isinstance(srt_paths, list) and srt_paths:
        first = srt_paths[0]
        if isinstance(first, str) and first:
            return first

    outputs = pipeline_res.get("outputs")
    if isinstance(outputs, dict):
        for _, v in outputs.items():
            if isinstance(v, str) and v.lower().endswith(".srt"):
                return v

    return None


def _extract_quality_counts(q_res: Dict[str, Any]) -> tuple[Optional[int], Optional[int], Optional[int]]:
    summary = q_res.get("summary")
    if not isinstance(summary, dict):
        return None, None, None
    major = summary.get("major_count") or summary.get("majors") or summary.get("major")
    minor = summary.get("minor_count") or summary.get("minors") or summary.get("minor")
    total = summary.get("violation_count") or summary.get("total")

    def _to_int(x: Any) -> Optional[int]:
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        return None

    return _to_int(major), _to_int(minor), _to_int(total)


def _extract_fix_stats(f_res: Dict[str, Any]) -> tuple[Optional[bool], int]:
    changed = f_res.get("changed")
    ch = changed if isinstance(changed, bool) else None
    actions = f_res.get("actions")
    cnt = len(actions) if isinstance(actions, list) else 0
    return ch, cnt


def run_quality_fix_loop(
    tool_map: Dict[str, Any],
    *,
    srt_path: str,
    cfg: LoopConfig,
    quality_args: Optional[Dict[str, Any]] = None,
    fix_args: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Optional[str], List[QualityPass], List[dict]]:
    errors: List[dict] = []
    history: List[QualityPass] = []
    current_srt = srt_path
    last_report_path: Optional[str] = None

    # ✅ Pin a stable out_path for all fix passes (overwrite), derived from the initial SRT.
    pinned_fix_args = dict(fix_args or {})
    if not pinned_fix_args.get("out_path"):
        pinned_fix_args["out_path"] = _stable_fixed_out_path(srt_path)

    q_tool = tool_map[QUALITY_CHECK_SUBTITLES]
    q_payload = {"srt_path": current_srt, **(quality_args or {})}
    q_res = safe_invoke_flat(q_tool, QUALITY_CHECK_SUBTITLES, q_payload)

    q_ok = bool(q_res.get("ok"))
    rp = q_res.get("report_path")
    if isinstance(rp, str) and rp:
        last_report_path = rp

    major, minor, total = _extract_quality_counts(q_res)
    logger.info(f"PASS 0: quality_ok={q_ok} major={major} minor={minor} total={total} report={last_report_path}")

    p0 = QualityPass(
        pass_index=0,
        checked_srt_path=current_srt,
        quality_ok=q_ok,
        report_path=last_report_path,
    )
    if not q_ok:
        err = (q_res.get("meta") or {}).get("error")
        if err:
            p0.errors.append(err)
            errors.append(err)
    history.append(p0)

    if q_ok:
        return True, current_srt, last_report_path, history, errors

    fix_tool = tool_map[FIX_SUBTITLES]
    for i in range(1, cfg.max_passes + 1):
        logger.info(f"RETRY {i}/{cfg.max_passes}: reason=quality_fail -> fix_subtitles")

        # ✅ Always pass the pinned out_path so filename never stacks suffix.
        f_payload = {"srt_path": current_srt, **pinned_fix_args}
        f_res = safe_invoke_flat(fix_tool, FIX_SUBTITLES, f_payload)

        fixed_path = f_res.get("fixed_srt_path")
        if isinstance(fixed_path, str) and fixed_path:
            current_srt = fixed_path

        changed, action_cnt = _extract_fix_stats(f_res)
        if not f_res.get("ok"):
            logger.info(f"FIX failed: ok=False changed={changed} actions={action_cnt}")
        else:
            reason = "fix_no_change" if changed is False else "fix_applied"
            logger.info(f"FIX ok: reason={reason} changed={changed} actions={action_cnt} fixed_srt={current_srt}")

        q_payload = {"srt_path": current_srt, **(quality_args or {})}
        q_res = safe_invoke_flat(q_tool, QUALITY_CHECK_SUBTITLES, q_payload)

        q_ok = bool(q_res.get("ok"))
        rp = q_res.get("report_path")
        if isinstance(rp, str) and rp:
            last_report_path = rp

        major, minor, total = _extract_quality_counts(q_res)
        logger.info(f"PASS {i}: quality_ok={q_ok} major={major} minor={minor} total={total} report={last_report_path}")

        step = QualityPass(
            pass_index=i,
            checked_srt_path=current_srt,
            quality_ok=q_ok,
            report_path=last_report_path,
            fixed_srt_path=fixed_path if isinstance(fixed_path, str) and fixed_path else None,
        )

        if not f_res.get("ok"):
            err = (f_res.get("meta") or {}).get("error")
            if err:
                step.errors.append(err)
                errors.append(err)

        if not q_ok:
            err = (q_res.get("meta") or {}).get("error")
            if err:
                step.errors.append(err)
                errors.append(err)

        history.append(step)

        if q_ok:
            return True, current_srt, last_report_path, history, errors

        if changed is False:
            logger.info(f"RETRY {i}/{cfg.max_passes}: reason=fix_no_change (still failing)")

    return False, current_srt, last_report_path, history, errors


def run_pr4c_closed_loop(
    tool_map: Dict[str, Any],
    *,
    pipeline_args: Dict[str, Any],
    cfg: Optional[LoopConfig] = None,
    max_passes: Optional[int] = None,
    quality_args: Optional[Dict[str, Any]] = None,
    fix_args: Optional[Dict[str, Any]] = None,
    burn_args: Optional[Dict[str, Any]] = None,
) -> LoopResult:
    cfg = cfg or LoopConfig()
    if max_passes is not None:
        cfg.max_passes = max_passes

    meta: Dict[str, Any] = {"trace_id": uuid.uuid4().hex}

    logger.info("PIPELINE: run_subgen_pipeline")
    p_tool = tool_map[RUN_SUBGEN_PIPELINE]
    p_args = inject_default_zh_only_pipeline_args(pipeline_args, cfg)
    p_res = safe_invoke_flat(p_tool, RUN_SUBGEN_PIPELINE, p_args)

    primary_path = p_res.get("primary_path") if isinstance(p_res.get("primary_path"), str) else None
    srt_path = pick_srt_path_from_pipeline_result(p_res)

    if not p_res.get("ok") or not srt_path:
        err = (p_res.get("meta") or {}).get("error")
        if err:
            meta["error"] = err
        meta["pipeline_ok"] = bool(p_res.get("ok"))
        meta["note"] = "Pipeline failed or did not produce an SRT path; cannot run quality loop."
        logger.info("PIPELINE failed or no SRT path; stop.")
        return LoopResult(
            ok=False,
            primary_path=primary_path,
            srt_path=srt_path,
            report_path=None,
            out_video_path=None,
            passes_used=0,
            history=[],
            meta=meta,
        )

    ok, final_srt, report_path, history, loop_errors = run_quality_fix_loop(
        tool_map,
        srt_path=srt_path,
        cfg=cfg,
        quality_args=quality_args,
        fix_args=fix_args,
    )
    if loop_errors:
        meta["errors"] = loop_errors
    meta["passes_used"] = len(history) - 1 if history else 0

    out_video_path: Optional[str] = None

    if burn_args:
        video_path = burn_args.get("video_path")
        if isinstance(video_path, str) and video_path:
            logger.info("BURN: burn_subtitles")
            b_tool = tool_map[BURN_SUBTITLES]
            b_payload = dict(burn_args)
            b_payload["video_path"] = video_path
            b_payload["srt_path"] = final_srt
            b_res = safe_invoke_flat(b_tool, BURN_SUBTITLES, b_payload)
            ov = b_res.get("out_video_path")
            if isinstance(ov, str) and ov:
                out_video_path = ov
            if not b_res.get("ok"):
                err = (b_res.get("meta") or {}).get("error")
                if err:
                    meta.setdefault("burn_errors", []).append(err)
            logger.info(f"BURN done: ok={bool(b_res.get('ok'))} out_video={out_video_path}")
        else:
            meta["burn_skipped"] = True
            meta["burn_skip_reason"] = "burn_args provided but missing/invalid video_path"
            logger.info("BURN skipped: missing/invalid video_path")

    logger.info(
        f"DONE: ok={bool(ok) and (report_path is not None)} "
        f"passes_used={len(history) - 1 if history else 0} final_srt={final_srt} report={report_path}"
    )

    return LoopResult(
        ok=bool(ok) and (report_path is not None),
        primary_path=primary_path,
        srt_path=final_srt,
        report_path=report_path,
        out_video_path=out_video_path,
        passes_used=len(history) - 1 if history else 0,
        history=history,
        meta=meta,
    )
