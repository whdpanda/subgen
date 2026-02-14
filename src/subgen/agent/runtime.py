# src/subgen/agent/runtime.py
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import uuid
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from subgen.utils.logger import configure_logging, get_logger, set_trace_id

# --- Optional deps (extras: agent / rag) ---
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import StructuredTool

    # Canonical tool names (must be stable)
    from subgen.agent.tools.tool_names import (
        RUN_SUBGEN_PIPELINE,
        QUALITY_CHECK_SUBTITLES,
        FIX_SUBTITLES,
        BURN_SUBTITLES,
        TOOL_NAMES,
    )

    # Registry: single source of truth for tool registration
    from subgen.agent.tools import build_agent_tools

except ImportError as e:
    raise SystemExit(
        "[subgen] Missing optional dependencies for Agent/RAG runtime.\n"
        'Install: pip install -e ".[agent,rag]"  (or only: ".[agent]" / ".[rag]")\n'
        f"ImportError: {e}"
    ) from e


SYSTEM = """You are SubGen Agent.

You can call tools to:
- Search project knowledge base for tool usage/policies (kb_search).
- Generate subtitles with pipeline (run_subgen_pipeline).
- Check subtitle quality (quality_check_subtitles).
- Fix subtitles deterministically (fix_subtitles).
- Burn subtitles into video (burn_subtitles).

STRICT rules:
- If uncertain about args/schema/how-to, call kb_search first.
- For subtitle generation: call run_subgen_pipeline.
- After generation, ALWAYS perform: quality_check_subtitles -> fix_subtitles loop until pass or max_passes.
- burn_subtitles is OPTIONAL and must only happen after the loop (pass or best-effort on max_passes).

Output rule (MOST IMPORTANT):
- When finishing, output a single JSON object with ONLY the following fields:
  ok, primary_path, srt_path, report_path, out_video_path
- Paths must come ONLY from tool outputs. Do NOT hallucinate paths.

Default behavior (unless user overrides explicitly in args):
- Output Chinese mono SRT (target_lang=zh, emit=zh-only)
- Apply proper Chinese segmentation/layout for readability.
"""


def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {"_args": obj}
    except Exception:
        return {"_raw": s}


def _normalize_one_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    call_id = call.get("id") or call.get("tool_call_id") or f"call_{uuid.uuid4().hex}"

    if "name" in call:
        name = call.get("name")
        args = call.get("args") or {}
        if isinstance(args, str):
            args = _safe_json_loads(args)
        if not isinstance(args, dict):
            args = {"_args": args}
        return {"id": call_id, "name": name, "args": args}

    fn = call.get("function") or {}
    name = fn.get("name")
    raw_args = fn.get("arguments") or "{}"

    args = raw_args if isinstance(raw_args, str) else raw_args
    if isinstance(args, str):
        args = _safe_json_loads(args)
    if not isinstance(args, dict):
        args = {"_args": args}
    return {"id": call_id, "name": name, "args": args}


def _extract_tool_calls(ai_msg: Any) -> List[Dict[str, Any]]:
    calls = getattr(ai_msg, "tool_calls", None)
    if calls:
        return [_normalize_one_tool_call(dict(c)) for c in list(calls)]

    ak = getattr(ai_msg, "additional_kwargs", None) or {}
    calls2 = ak.get("tool_calls")
    if calls2:
        return [_normalize_one_tool_call(dict(c)) for c in list(calls2)]

    return []


def _is_envelope(x: Any) -> bool:
    return isinstance(x, dict) and set(x.keys()) >= {"ok", "data", "error"}


def _flatten_envelope_for_llm(tool_name: str, env: dict[str, Any]) -> dict[str, Any]:
    ok = bool(env.get("ok"))
    data = env.get("data")
    err = env.get("error")

    if isinstance(data, dict):
        out = dict(data)
        out["ok"] = ok
        if err:
            meta = out.get("meta") or {}
            if not isinstance(meta, dict):
                meta = {"_meta": meta}
            meta["error"] = err
            out["meta"] = meta
        return out

    return {
        "ok": ok,
        "meta": {
            "error": err or {"type": f"{tool_name}.legacy_envelope_shape", "message": "non-dict envelope data"},
            "data_type": str(type(data)),
        },
    }


def _safe_tool_invoke_flat(tool: StructuredTool, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    logger = get_logger("subgen")
    try:
        logger.debug(f"TOOL_CALL -> {tool_name} args={tool_args}")
        result = tool.invoke(tool_args)

        if _is_envelope(result):
            flat = _flatten_envelope_for_llm(tool_name, result)
            logger.debug(f"TOOL_RET  <- {tool_name} keys={list(flat.keys())} ok={bool(flat.get('ok', True))}")
            return flat

        if isinstance(result, dict):
            logger.debug(f"TOOL_RET  <- {tool_name} keys={list(result.keys())} ok={bool(result.get('ok', True))}")
            return result

        logger.debug(f"TOOL_RET  <- {tool_name} (non-dict) type={type(result)}")
        return {"ok": True, "result": result}

    except ValidationError as e:
        logger.debug(f"TOOL_ERR  !! {tool_name} validation_error errors={e.errors()} args={tool_args}")
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
        logger.exception(f"TOOL_ERR  !! {tool_name} invoke_error args={tool_args}")
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


def _tool_error_brief(flat_result: Dict[str, Any]) -> Optional[str]:
    """Extract short `type: message` text from flat tool error payload."""
    meta = flat_result.get("meta")
    if not isinstance(meta, dict):
        return None

    err = meta.get("error")
    if not isinstance(err, dict):
        return None

    err_type = err.get("type")
    message = err.get("message")

    if isinstance(err_type, str) and isinstance(message, str):
        return f"{err_type}: {message}"
    if isinstance(message, str):
        return message
    if isinstance(err_type, str):
        return err_type
    return None


def _inject_default_zh_only_pipeline_args(tool_args: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(tool_args or {})
    out.setdefault("target_lang", "zh")
    out.setdefault("emit", "zh-only")
    out.setdefault("zh_layout", True)
    return out


def _pick_srt_path_from_pipeline_result(pipeline_res: Dict[str, Any]) -> Optional[str]:
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


def _quality_check(
    tool_map: Dict[str, StructuredTool],
    srt_path: str,
    extra_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    args = {"srt_path": srt_path}
    if extra_args:
        args.update(extra_args)
    tool = tool_map[QUALITY_CHECK_SUBTITLES]
    return _safe_tool_invoke_flat(tool, QUALITY_CHECK_SUBTITLES, args)


def _fix_subtitles(
    tool_map: Dict[str, StructuredTool],
    srt_path: str,
    extra_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    args = {"srt_path": srt_path}
    if extra_args:
        args.update(extra_args)
    tool = tool_map[FIX_SUBTITLES]
    return _safe_tool_invoke_flat(tool, FIX_SUBTITLES, args)


def _burn(
    tool_map: Dict[str, StructuredTool],
    video_path: str,
    srt_path: str,
    extra_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    args = {"video_path": video_path, "srt_path": srt_path}
    if extra_args:
        args.update(extra_args)
    tool = tool_map[BURN_SUBTITLES]
    return _safe_tool_invoke_flat(tool, BURN_SUBTITLES, args)


def _extract_quality_counts(q_res: Dict[str, Any]) -> tuple[Optional[int], Optional[int], Optional[int]]:
    summary = q_res.get("summary")
    if not isinstance(summary, dict):
        return None, None, None

    major = summary.get("major_count")
    minor = summary.get("minor_count")

    if major is None:
        major = summary.get("majors") or summary.get("major")
    if minor is None:
        minor = summary.get("minors") or summary.get("minor")

    total = summary.get("violation_count") or summary.get("total") or None

    def _to_int(x: Any) -> Optional[int]:
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        return None

    return _to_int(major), _to_int(minor), _to_int(total)


def _extract_fix_stats(f_res: Dict[str, Any]) -> tuple[Optional[bool], int, List[str]]:
    changed = f_res.get("changed")
    ch = changed if isinstance(changed, bool) else None

    actions = f_res.get("actions")
    names: List[str] = []
    if isinstance(actions, list):
        for a in actions:
            if isinstance(a, str):
                names.append(a)
            elif isinstance(a, dict) and isinstance(a.get("name"), str):
                names.append(a["name"])
    return ch, len(names), names[:3]


def _final_stdout_only_paths(
    ok: bool,
    primary_path: Optional[str],
    srt_path: Optional[str],
    report_path: Optional[str],
    out_video_path: Optional[str],
) -> None:
    payload = {
        "ok": bool(ok),
        "primary_path": primary_path,
        "srt_path": srt_path,
        "report_path": report_path,
        "out_video_path": out_video_path,
    }
    print(json.dumps(payload, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="SubGen Agent (PR#4c runtime-enforced quality loop)")
    parser.add_argument("--query", "-q", required=True, help="User query to the agent")
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("SUBGEN_AGENT_MAX_STEPS", "16")))
    parser.add_argument("--max-passes", type=int, default=int(os.getenv("SUBGEN_QUALITY_MAX_PASSES", "3")))
    parser.add_argument(
        "--log-path",
        type=str,
        default=os.getenv("SUBGEN_LOG_PATH", ""),
        help="Optional debug log file path (default: resolved to <out_dir>/debug.log after pipeline starts)",
    )
    args = parser.parse_args()

    # Trace id for this run (bind to logger via filter).
    trace_id = uuid.uuid4().hex
    set_trace_id(trace_id)

    # Console logging first (stderr). File logging may be added later once out_dir is known.
    logger = configure_logging(log_path=(args.log_path or None))
    logger.debug(f"BOOT trace_id={trace_id} argv={sys.argv}")

    model_name = os.getenv("SUBGEN_AGENT_MODEL", "gpt-5.2")
    llm = ChatOpenAI(model=model_name, temperature=0)

    tools = build_agent_tools()
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages: List[Any] = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=args.query),
    ]

    primary_path: Optional[str] = None
    current_srt_path: Optional[str] = None
    report_path: Optional[str] = None
    out_video_path: Optional[str] = None
    pending_burn_args: Optional[Dict[str, Any]] = None

    did_generate = False
    loop_completed = False

    # console acceptance summary
    def _note(msg: str) -> None:
        logger.info(msg)

    _note(f"START: trace={trace_id} model={model_name} max_steps={args.max_steps} max_passes={args.max_passes}")

    for step in range(args.max_steps):
        ai_msg = llm_with_tools.invoke(messages)
        tool_calls = _extract_tool_calls(ai_msg)

        logger.debug(f"LLM_STEP {step}: tool_calls={len(tool_calls)}")
        if tool_calls:
            logger.debug(f"LLM_STEP {step}: calls={tool_calls}")

        if not tool_calls:
            if loop_completed:
                _note("DONE: model finished with no further tool calls.")
                _final_stdout_only_paths(
                    ok=bool(current_srt_path) and (report_path is not None),
                    primary_path=primary_path,
                    srt_path=current_srt_path,
                    report_path=report_path,
                    out_video_path=out_video_path,
                )
                return

            messages.append(
                HumanMessage(
                    content=(
                        "You must call tools. Do not answer in text. "
                        "Proceed with the required tool sequence to produce final paths JSON."
                    )
                )
            )
            continue

        messages.append(ai_msg)

        for call in tool_calls:
            name = call.get("name")
            call_id = call.get("id", "")
            raw_args = call.get("args") or {}
            if not isinstance(raw_args, dict):
                raw_args = {"_args": raw_args}

            logger.debug(f"DISPATCH step={step} tool={name} call_id={call_id} raw_args={raw_args}")

            if name == BURN_SUBTITLES and not loop_completed:
                pending_burn_args = dict(raw_args)
                _note("BURN deferred until after quality loop completes.")
                flat = {"ok": True, "deferred": True, "meta": {"note": "burn_subtitles deferred until after quality loop."}}
                messages.append(ToolMessage(tool_call_id=call_id, content=json.dumps(flat, ensure_ascii=False)))
                continue

            if name == RUN_SUBGEN_PIPELINE:
                pipeline_args = _inject_default_zh_only_pipeline_args(raw_args)

                out_dir = pipeline_args.get("out_dir")
                if isinstance(out_dir, str) and out_dir:
                    debug_path = os.path.join(out_dir, "debug.log")
                    configure_logging(log_path=debug_path)
                    _note(f"LOG: debug file -> {debug_path}")
                    logger.debug(f"LOG_CONFIG file_handler={debug_path}")

                _note(f"STEP {step}: run_subgen_pipeline")
                flat_pipeline = _safe_tool_invoke_flat(tool_map[RUN_SUBGEN_PIPELINE], RUN_SUBGEN_PIPELINE, pipeline_args)

                logger.debug(f"PIPELINE_RES ok={bool(flat_pipeline.get('ok', True))} keys={list(flat_pipeline.keys())}")
                messages.append(ToolMessage(tool_call_id=call_id, content=json.dumps(flat_pipeline, ensure_ascii=False)))

                did_generate = True

                if isinstance(flat_pipeline.get("primary_path"), str):
                    primary_path = flat_pipeline.get("primary_path")

                current_srt_path = _pick_srt_path_from_pipeline_result(flat_pipeline) or current_srt_path
                logger.debug(f"PIPELINE_PATHS primary_path={primary_path} current_srt_path={current_srt_path}")

                if not flat_pipeline.get("ok") or not current_srt_path:
                    err_brief = _tool_error_brief(flat_pipeline)
                    if err_brief:
                        _note(f"PIPELINE failed or did not return an SRT path (best-effort stop). reason={err_brief}")
                    else:
                        _note("PIPELINE failed or did not return an SRT path (best-effort stop).")
                    _final_stdout_only_paths(
                        ok=False,
                        primary_path=primary_path,
                        srt_path=current_srt_path,
                        report_path=report_path,
                        out_video_path=out_video_path,
                    )
                    return

                passes_used = 0

                q_res = _quality_check(tool_map, current_srt_path)
                logger.debug(f"QUALITY_RES pass=0 full={q_res}")

                if isinstance(q_res.get("report_path"), str):
                    report_path = q_res.get("report_path")

                major, minor, total = _extract_quality_counts(q_res)
                _note(
                    f"PASS 0: quality_ok={bool(q_res.get('ok'))} "
                    f"major={major} minor={minor} total={total} report={report_path}"
                )

                messages.append(
                    ToolMessage(tool_call_id=f"runtime_{uuid.uuid4().hex}", content=json.dumps(q_res, ensure_ascii=False))
                )

                while (not q_res.get("ok")) and passes_used < args.max_passes:
                    passes_used += 1

                    _note(f"RETRY {passes_used}/{args.max_passes}: reason=quality_fail -> fix_subtitles")
                    f_res = _fix_subtitles(tool_map, current_srt_path, extra_args={"max_passes": 1})
                    logger.debug(f"FIX_RES pass={passes_used} full={f_res}")

                    messages.append(
                        ToolMessage(tool_call_id=f"runtime_{uuid.uuid4().hex}", content=json.dumps(f_res, ensure_ascii=False))
                    )

                    fixed_path = f_res.get("fixed_srt_path")
                    if isinstance(fixed_path, str) and fixed_path:
                        current_srt_path = fixed_path

                    changed, action_cnt, top_actions = _extract_fix_stats(f_res)
                    if not f_res.get("ok"):
                        _note(f"FIX failed: ok=False changed={changed} actions={action_cnt} top={top_actions}")
                    else:
                        reason = "fix_no_change" if changed is False else "fix_applied"
                        _note(
                            f"FIX ok: reason={reason} changed={changed} actions={action_cnt} top={top_actions} "
                            f"fixed_srt={current_srt_path}"
                        )

                    q_res = _quality_check(tool_map, current_srt_path)
                    logger.debug(f"QUALITY_RES pass={passes_used} full={q_res}")

                    if isinstance(q_res.get("report_path"), str):
                        report_path = q_res.get("report_path")

                    major, minor, total = _extract_quality_counts(q_res)
                    _note(
                        f"PASS {passes_used}: quality_ok={bool(q_res.get('ok'))} "
                        f"major={major} minor={minor} total={total} report={report_path}"
                    )

                    messages.append(
                        ToolMessage(tool_call_id=f"runtime_{uuid.uuid4().hex}", content=json.dumps(q_res, ensure_ascii=False))
                    )

                    if changed is False and not q_res.get("ok"):
                        _note(f"RETRY {passes_used}/{args.max_passes}: reason=fix_no_change (still failing)")

                loop_completed = True

                if pending_burn_args:
                    video_path = None
                    if isinstance(pending_burn_args.get("video_path"), str):
                        video_path = pending_burn_args.get("video_path")
                    if not video_path and isinstance(pipeline_args.get("video_path"), str):
                        video_path = pipeline_args.get("video_path")

                    if video_path and current_srt_path:
                        _note("BURN: executing deferred burn_subtitles")
                        burn_extra = dict(pending_burn_args)
                        burn_extra.pop("video_path", None)
                        burn_extra.pop("srt_path", None)
                        b_res = _burn(tool_map, video_path, current_srt_path, extra_args=burn_extra)
                        logger.debug(f"BURN_RES full={b_res}")

                        if isinstance(b_res.get("out_video_path"), str):
                            out_video_path = b_res.get("out_video_path")

                        messages.append(
                            ToolMessage(tool_call_id=f"runtime_{uuid.uuid4().hex}", content=json.dumps(b_res, ensure_ascii=False))
                        )
                        _note(f"BURN done: ok={bool(b_res.get('ok'))} out_video={out_video_path}")
                    else:
                        _note("BURN skipped: missing video_path or current_srt_path")

                    pending_burn_args = None

                _note(
                    f"DONE: ok={bool(current_srt_path) and (report_path is not None) and bool(q_res.get('ok'))} "
                    f"passes_used={passes_used} final_srt={current_srt_path} report={report_path} burn={out_video_path}"
                )

                _final_stdout_only_paths(
                    ok=bool(current_srt_path) and (report_path is not None),
                    primary_path=primary_path,
                    srt_path=current_srt_path,
                    report_path=report_path,
                    out_video_path=out_video_path,
                )
                return

            if not isinstance(name, str) or name not in tool_map:
                _note(f"ERROR: unknown tool called: {name}")
                flat = {
                    "ok": False,
                    "meta": {"error": {"type": "runtime.unknown_tool", "message": f"Unknown tool called: {name}", "details": {"expected": TOOL_NAMES}}},
                }
                messages.append(ToolMessage(tool_call_id=call_id, content=json.dumps(flat, ensure_ascii=False)))
                continue

            flat = _safe_tool_invoke_flat(tool_map[name], name, raw_args if isinstance(raw_args, dict) else {})
            logger.debug(f"TOOL_GENERIC_RES tool={name} ok={bool(flat.get('ok', True))} keys={list(flat.keys())}")
            messages.append(ToolMessage(tool_call_id=call_id, content=json.dumps(flat, ensure_ascii=False)))

        if not did_generate:
            continue

    _note("DONE: max_steps reached (best-effort).")
    _final_stdout_only_paths(
        ok=False,
        primary_path=primary_path,
        srt_path=current_srt_path,
        report_path=report_path,
        out_video_path=out_video_path,
    )


if __name__ == "__main__":
    main()
