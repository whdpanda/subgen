# src/subgen/api/services/subtitles_service.py
from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from subgen.agent.loop import LoopConfig, run_pr4c_closed_loop, safe_invoke_flat
from subgen.agent.tools import build_agent_tools
from subgen.agent.tools.tool_names import (
    BURN_SUBTITLES,
    FIX_SUBTITLES,
    QUALITY_CHECK_SUBTITLES,
    RUN_SUBGEN_PIPELINE,
)
from subgen.api.utils.logging_context import request_debug_logging
from subgen.utils.logger import get_logger

logger = get_logger("subgen.api")


def _as_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    return x


@lru_cache(maxsize=1)
def _tool_map() -> Dict[str, Any]:
    tools = build_agent_tools()
    tool_map = {t.name: t for t in tools}

    missing = [n for n in (RUN_SUBGEN_PIPELINE, QUALITY_CHECK_SUBTITLES, FIX_SUBTITLES, BURN_SUBTITLES) if n not in tool_map]
    if missing:
        raise RuntimeError(f"Missing required tools: {missing}")

    return tool_map


def _loop_cfg_from_env() -> LoopConfig:
    cfg = LoopConfig()

    mp = os.getenv("SUBGEN_QUALITY_MAX_PASSES")
    if mp:
        try:
            cfg.max_passes = int(mp)
        except Exception:
            logger.debug(f"Invalid SUBGEN_QUALITY_MAX_PASSES={mp}; keep default.")

    emit = os.getenv("SUBGEN_EMIT_DEFAULT")
    if emit:
        cfg.emit_default = emit

    tgt = os.getenv("SUBGEN_TARGET_LANG_DEFAULT")
    if tgt:
        cfg.target_lang_default = tgt

    zhl = os.getenv("SUBGEN_ZH_LAYOUT_DEFAULT")
    if zhl:
        cfg.zh_layout_default = zhl.strip().lower() in ("1", "true", "yes", "y", "on")

    return cfg


def generate_subtitles(
    *,
    video_path: str,
    out_dir: str,
    max_passes: Optional[int] = None,
    pipeline_args: Optional[Dict[str, Any]] = None,
    quality_args: Optional[Dict[str, Any]] = None,
    fix_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tool_map = _tool_map()
    cfg = _loop_cfg_from_env()

    p_args: Dict[str, Any] = {"video_path": video_path, "out_dir": out_dir}
    if pipeline_args:
        p_args.update(pipeline_args)

    # Request-scoped debug.log + stdout capture
    with request_debug_logging(out_dir=out_dir, capture_stdout=True, logger_name="subgen"):
        res = run_pr4c_closed_loop(
            tool_map,
            pipeline_args=p_args,
            cfg=cfg,
            max_passes=max_passes,
            quality_args=quality_args,
            fix_args=fix_args,
            burn_args=None,
        )

    out = _as_jsonable(res)
    assert isinstance(out, dict)
    return out


def fix_subtitles(
    *,
    srt_path: str,
    out_dir: Optional[str] = None,
    fix_args: Optional[Dict[str, Any]] = None,
    quality_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tool_map = _tool_map()

    # debug.log location: explicit out_dir else srt parent
    effective_out_dir = out_dir
    if not effective_out_dir:
        try:
            effective_out_dir = str(Path(srt_path).expanduser().resolve().parent)
        except Exception:
            effective_out_dir = None

    payload: Dict[str, Any] = {"srt_path": srt_path}
    if out_dir:
        payload["out_dir"] = out_dir
    if fix_args:
        payload.update(fix_args)

    with request_debug_logging(out_dir=effective_out_dir, capture_stdout=True, logger_name="subgen"):
        fix_flat = safe_invoke_flat(tool_map[FIX_SUBTITLES], FIX_SUBTITLES, payload)

        fixed_path = fix_flat.get("fixed_srt_path")
        final_srt = fixed_path if isinstance(fixed_path, str) and fixed_path else srt_path

        # best-effort quality report
        q_flat: Optional[Dict[str, Any]] = None
        report_path: Optional[str] = None
        try:
            q_payload: Dict[str, Any] = {"srt_path": final_srt}
            if out_dir:
                q_payload["out_dir"] = out_dir
            if quality_args:
                q_payload.update(quality_args)

            q_flat = safe_invoke_flat(tool_map[QUALITY_CHECK_SUBTITLES], QUALITY_CHECK_SUBTITLES, q_payload)
            rp = q_flat.get("report_path")
            report_path = rp if isinstance(rp, str) and rp else None
        except Exception:
            logger.exception("fix_subtitles: post-quality check failed (best-effort).")
            q_flat = None
            report_path = None

    ok_fix = bool(fix_flat.get("ok"))
    ok_quality = bool(q_flat.get("ok")) if isinstance(q_flat, dict) else True

    meta: Dict[str, Any] = {
        "input_srt_path": srt_path,
        "final_srt_path": final_srt,
    }
    if isinstance(fix_flat.get("meta"), dict) and fix_flat["meta"].get("error"):
        meta["error"] = fix_flat["meta"]["error"]
    if isinstance(q_flat, dict) and isinstance(q_flat.get("meta"), dict) and q_flat["meta"].get("error"):
        meta.setdefault("errors", []).append(q_flat["meta"]["error"])

    # NOTE: response_model will drop extra fields if any; we keep fix/quality for debugging.
    return {
        "ok": bool(ok_fix and ok_quality),
        "srt_path": final_srt,
        "report_path": report_path,
        "meta": meta,
        "fix": fix_flat,
        "quality": q_flat,
    }


def burn_subtitles(
    *,
    video_path: str,
    srt_path: str,
    out_path: Optional[str] = None,
    burn_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tool_map = _tool_map()

    # debug.log location: out_path parent else video parent
    log_dir: Optional[str] = None
    try:
        if out_path:
            log_dir = str(Path(out_path).expanduser().resolve().parent)
        else:
            log_dir = str(Path(video_path).expanduser().resolve().parent)
    except Exception:
        log_dir = None

    payload: Dict[str, Any] = {"video_path": video_path, "srt_path": srt_path}
    if out_path:
        payload["out_path"] = out_path
    if burn_args:
        payload.update(burn_args)

    with request_debug_logging(out_dir=log_dir, capture_stdout=True, logger_name="subgen"):
        burn_flat = safe_invoke_flat(tool_map[BURN_SUBTITLES], BURN_SUBTITLES, payload)

    return burn_flat