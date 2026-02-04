from __future__ import annotations

import argparse
import json
import os
import traceback
import uuid
from typing import Any, Dict, List, Optional, Callable

from pydantic import ValidationError

# --- Optional deps (extras: agent / rag) ---
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import StructuredTool

    from subgen.agent.tools.tool_names import RUN_SUBGEN_PIPELINE, BURN_SUBTITLES, KB_SEARCH
    from subgen.agent.tools.pipeline_tool import run_subgen_pipeline_tool, PipelineToolArgs
    from subgen.agent.tools.burn_subtitles_tool import burn_subtitles_tool, BurnToolArgs
    from subgen.agent.tools.kb_search_tool import make_kb_search_tool
except ImportError as e:
    raise SystemExit(
        "[subgen] Missing optional dependencies for Agent/RAG runtime.\n"
        'Install: pip install -e ".[agent,rag]"  (or only: ".[agent]" / ".[rag]")\n'
        f"ImportError: {e}"
    ) from e


SYSTEM = """You are SubGen Agent.

Primary goals:
1) Generate subtitles (SRT) for a given video.
2) Optionally burn subtitles into the video (hard-sub) when the user asks.

Rules:
- If you are uncertain about tool args, output schema, defaults, or how to call tools,
  you MUST call kb_search first.
- If the user asks to generate subtitles, you MUST call run_subgen_pipeline.
- If the user asks to burn subtitles into video, you MUST call burn_subtitles.
- Always return output file path(s) from tool outputs ONLY.
- Do NOT hallucinate file paths.

Tool best practice:
- Always call kb_search before deciding which tool to use, unless the user request is trivial and unambiguous.
"""


# -------------------------
# Envelope helpers (runtime-side ONLY: printing/logging/API response)
# -------------------------
def _ok(data: Any) -> Dict[str, Any]:
    return {"ok": True, "data": data, "error": None}


def _fail(err_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": {"type": err_type, "message": message, "details": details or {}},
    }


def _is_envelope(x: Any) -> bool:
    return isinstance(x, dict) and set(x.keys()) >= {"ok", "data", "error"}


def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        return {"_args": obj}
    except Exception:
        return {"_raw": s}


def _normalize_one_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    call_id = call.get("id") or call.get("tool_call_id") or f"call_{uuid.uuid4().hex}"

    # LangChain typically: {"name":..., "args":...}
    if "name" in call:
        name = call.get("name")
        args = call.get("args") or {}
        if isinstance(args, str):
            args = _safe_json_loads(args)
        if not isinstance(args, dict):
            args = {"_args": args}
        return {"id": call_id, "name": name, "args": args}

    # OpenAI-ish shape: {"function": {"name":..., "arguments": "..."}}
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
        out: List[Dict[str, Any]] = []
        for c in list(calls):
            out.append(_normalize_one_tool_call(dict(c)))
        return out

    ak = getattr(ai_msg, "additional_kwargs", None) or {}
    calls2 = ak.get("tool_calls")
    if calls2:
        out2: List[Dict[str, Any]] = []
        for c in list(calls2):
            out2.append(_normalize_one_tool_call(dict(c)))
        return out2

    return []


# -------------------------
# Tool output normalization for LLM:
# - LLM should see FLAT schemas
# - runtime may still envelope for printing/logging, but NOT in ToolMessage
# -------------------------
def _flat_fail_pipeline(err_type: str, message: str, details: dict[str, Any], tool_args: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": False,
        "primary_path": None,
        "srt_paths": [],
        "outputs": {},
        "artifacts": {},
        "meta": {
            "error": {"type": err_type, "message": message, "details": details},
            "input": tool_args,
        },
    }


def _flat_fail_burn(err_type: str, message: str, details: dict[str, Any], tool_args: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": False,
        "out_video_path": None,
        "artifacts": {},
        "meta": {
            "error": {"type": err_type, "message": message, "details": details},
            "input": tool_args,
        },
    }


def _flat_fail_kb(err_type: str, message: str, details: dict[str, Any], tool_args: dict[str, Any]) -> dict[str, Any]:
    # best-effort; your kb tool通常返回 {"query","k","kb","results"} 之类
    return {
        "ok": False,
        "query": tool_args.get("query"),
        "k": tool_args.get("k"),
        "kb": None,
        "results": [],
        "meta": {"error": {"type": err_type, "message": message, "details": details}},
    }


FAIL_BUILDERS: dict[str, Callable[[str, str, dict[str, Any], dict[str, Any]], dict[str, Any]]] = {
    RUN_SUBGEN_PIPELINE: _flat_fail_pipeline,
    BURN_SUBTITLES: _flat_fail_burn,
    KB_SEARCH: _flat_fail_kb,
}


def _flatten_envelope_for_llm(tool_name: str, env: dict[str, Any]) -> dict[str, Any]:
    """
    Convert legacy envelope {"ok","data","error"} into flat schema for LLM, best-effort.
    Rule:
      - if data is a dict => return data plus ok; if error exists, put into meta.error
      - else => fallback to tool-specific fail builder
    """
    ok = bool(env.get("ok"))
    data = env.get("data")
    err = env.get("error")

    if isinstance(data, dict):
        # Ensure ok is present at top-level (some legacy data may not contain it)
        out = dict(data)
        out["ok"] = ok

        if err:
            meta = out.get("meta") or {}
            if not isinstance(meta, dict):
                meta = {"_meta": meta}
            meta["error"] = err
            out["meta"] = meta
        return out

    # If not dict, fallback
    builder = FAIL_BUILDERS.get(tool_name)
    if builder:
        return builder(
            f"{tool_name}.legacy_envelope_shape",
            "legacy tool returned non-dict data in envelope",
            {"data_type": str(type(data)), "error": err},
            {},
        )
    return {"ok": ok, "meta": {"error": err, "data": data}}


def _safe_tool_invoke_flat_for_llm(tool: StructuredTool, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure:
    - runtime never crashes due to tool.invoke errors (incl. args_schema validation)
    - ALWAYS return a FLAT schema dict to feed back to LLM via ToolMessage
    """
    try:
        result = tool.invoke(tool_args)

        # Legacy compat: if some tool still returns envelope, flatten it for LLM
        if _is_envelope(result):
            return _flatten_envelope_for_llm(tool_name, result)

        # New world: tool returns flat schema already
        if isinstance(result, dict):
            return result

        # Non-dict result is unexpected for tools; convert to minimal dict
        return {"ok": True, "result": result}

    except ValidationError as e:
        builder = FAIL_BUILDERS.get(tool_name)
        if builder:
            return builder(
                f"{tool_name}.validation_error",
                "tool args validation failed (raised by StructuredTool args_schema before tool ran)",
                {"errors": e.errors()},
                tool_args,
            )
        return {"ok": False, "meta": {"error": {"type": f"{tool_name}.validation_error", "message": "validation failed", "details": {"errors": e.errors(), "args": tool_args}}}}

    except Exception as e:
        builder = FAIL_BUILDERS.get(tool_name)
        details = {
            "exception_class": e.__class__.__name__,
            "traceback": traceback.format_exc(),
            "args": tool_args,
        }
        if builder:
            return builder(
                f"{tool_name}.invoke_error",
                str(e) or e.__class__.__name__,
                details,
                tool_args,
            )
        return {"ok": False, "meta": {"error": {"type": f"{tool_name}.invoke_error", "message": str(e) or e.__class__.__name__, "details": details}}}


def _assert_tool_name(tool: Any, expected: str, label: str) -> None:
    actual = getattr(tool, "name", None)
    if actual != expected:
        raise RuntimeError(f"{label} tool name must be {expected!r}, got: {actual!r}")


def build_tools() -> List[StructuredTool]:
    """
    Build tools with STRICT name pinning to avoid docs/prompt/KB drifting.

    Required external tool names:
      - kb_search
      - run_subgen_pipeline
      - burn_subtitles
    """
    kb_tool = make_kb_search_tool()
    _assert_tool_name(kb_tool, KB_SEARCH, "kb_search")

    pipeline_tool = StructuredTool.from_function(
        name=RUN_SUBGEN_PIPELINE,
        description=(
            "Run subgen pipeline to generate subtitles. "
            "Inputs: video_path, out_dir, language, target_lang, preprocess, segmenter, "
            "openai_segment_model, translator_name, emit, use_cache, dump_intermediates, etc. "
            "Returns: primary_path, srt_paths, outputs, artifacts, meta (flat schema)."
        ),
        func=run_subgen_pipeline_tool,
        args_schema=PipelineToolArgs,
    )
    _assert_tool_name(pipeline_tool, RUN_SUBGEN_PIPELINE, "run_subgen_pipeline")

    burn_tool = StructuredTool.from_function(
        name=BURN_SUBTITLES,
        description=(
            "Burn an SRT subtitle file into the video (hard-sub) using ffmpeg. "
            "Inputs: video_path, srt_path, out_path(optional), force_style(optional), crf, preset, copy_audio. "
            "Returns: out_video_path, artifacts, meta (flat schema)."
        ),
        func=burn_subtitles_tool,
        args_schema=BurnToolArgs,
    )
    _assert_tool_name(burn_tool, BURN_SUBTITLES, "burn_subtitles")

    tools = [kb_tool, pipeline_tool, burn_tool]

    names = [t.name for t in tools]
    if len(set(names)) != len(names):
        raise RuntimeError(f"Duplicate tool names found: {names}")

    return tools


def main() -> None:
    parser = argparse.ArgumentParser(description="SubGen Agent (LangChain minimal runtime)")
    parser.add_argument("--query", "-q", required=True, help="User query to the agent")
    parser.add_argument("--debug-envelope", action="store_true", help="Print runtime envelope for debugging (NOT fed to LLM)")
    args = parser.parse_args()

    model_name = os.getenv("SUBGEN_AGENT_MODEL", "gpt-5.2")
    llm = ChatOpenAI(model=model_name, temperature=0)

    tools = build_tools()
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages: List[Any] = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=args.query),
    ]

    for _ in range(8):
        ai_msg = llm_with_tools.invoke(messages)

        tool_calls = _extract_tool_calls(ai_msg)
        if tool_calls:
            print("[DEBUG] tool_calls:", [(c.get("name"), c.get("args")) for c in tool_calls])

            messages.append(ai_msg)

            for call in tool_calls:
                name = call.get("name")
                call_id = call.get("id", "")

                tool = tool_map.get(name)
                if tool is None:
                    # Unknown tool call should NOT crash the whole runtime
                    flat = {"ok": False, "meta": {"error": {"type": "runtime.unknown_tool", "message": f"Unknown tool called: {name}", "details": {"available": list(tool_map.keys())}}}}
                else:
                    tool_args: Dict[str, Any] = call.get("args") or {}
                    if not isinstance(tool_args, dict):
                        tool_args = {"_args": tool_args}

                    flat = _safe_tool_invoke_flat_for_llm(tool, tool_name=name or "unknown", tool_args=tool_args)

                # (Optional) runtime-side envelope for printing/logging only
                if args.debug_envelope:
                    print("[DEBUG] tool_result_enveloped_for_logs:", json.dumps(_ok(flat) if flat.get("ok") else _fail("tool.failed", "see data", {"data": flat}), ensure_ascii=False))

                # Feed FLAT result to LLM
                messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=json.dumps(flat, ensure_ascii=False),
                    )
                )
            continue

        # No tool call => final answer
        content = getattr(ai_msg, "content", "")
        if isinstance(content, list):
            content = "\n".join([str(x) for x in content])
        print(content)
        return

    raise RuntimeError("Agent did not finish after max steps.")


if __name__ == "__main__":
    main()
