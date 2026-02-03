from __future__ import annotations

import argparse
import json
import os
import traceback
import uuid
from typing import Any, Dict, List, Optional

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
# Envelope helpers (runtime-side)
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
        # some tool calls might pass a JSON list; normalize to dict
        return {"_args": obj}
    except Exception:
        # If arguments aren't valid JSON, keep as raw string
        return {"_raw": s}


def _normalize_one_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize tool call from multiple possible shapes into:
      {"id": "...", "name": "...", "args": {...}}

    Supported shapes:
    1) {"id": "...", "name": "...", "args": {...}}
    2) {"id": "...", "function": {"name": "...", "arguments": "<json string>"}}
    3) {"function": {"name": "...", "arguments": "<json string>"}}
    """
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
    """
    Normalize tool calls across langchain versions.
    Returns: [{"id": "...", "name": "...", "args": {...}}, ...]
    """
    calls = getattr(ai_msg, "tool_calls", None)
    if calls:
        # calls could be list[dict] or list[ToolCall]
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


def _safe_tool_invoke(tool: StructuredTool, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure:
    - runtime never crashes due to tool.invoke errors (incl. args_schema validation)
    - always returns envelope
    """
    try:
        result = tool.invoke(tool_args)

        # Backward compat: if tool not yet migrated, wrap it
        if not _is_envelope(result):
            return _ok(result)
        return result

    except ValidationError as e:
        # IMPORTANT: this happens BEFORE your tool function runs if args_schema is set
        return _fail(
            err_type=f"{tool_name}.validation_error",
            message="tool args validation failed",
            details={"errors": e.errors(), "args": tool_args},
        )
    except Exception as e:
        return _fail(
            err_type=f"{tool_name}.invoke_error",
            message=str(e) or e.__class__.__name__,
            details={
                "exception_class": e.__class__.__name__,
                "traceback": traceback.format_exc(),
                "args": tool_args,
            },
        )


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
        name=RUN_SUBGEN_PIPELINE,  # ✅ 固定对外工具名
        description=(
            "Run subgen pipeline to generate subtitles. "
            "Inputs: video_path, out_dir, language, target_lang, preprocess, segmenter, "
            "openai_segment_model, translator_name, emit, use_cache, dump_intermediates, etc. "
            "Returns (enveloped): data.primary_path, data.srt_paths, data.outputs, data.artifacts, data.meta."
        ),
        func=run_subgen_pipeline_tool,
        args_schema=PipelineToolArgs,
    )
    _assert_tool_name(pipeline_tool, RUN_SUBGEN_PIPELINE, "run_subgen_pipeline")

    burn_tool = StructuredTool.from_function(
        name=BURN_SUBTITLES,  # ✅ 固定对外工具名
        description=(
            "Burn an SRT subtitle file into the video (hard-sub) using ffmpeg. "
            "Inputs: video_path, srt_path, out_path(optional), force_style(optional), crf, preset, copy_audio. "
            "Returns (enveloped): data.out_video_path."
        ),
        func=burn_subtitles_tool,
        args_schema=BurnToolArgs,
    )
    _assert_tool_name(burn_tool, BURN_SUBTITLES, "burn_subtitles")

    tools = [kb_tool, pipeline_tool, burn_tool]

    # --- 防止重复 name（会覆盖 tool_map）---
    names = [t.name for t in tools]
    if len(set(names)) != len(names):
        raise RuntimeError(f"Duplicate tool names found: {names}")

    return tools


def main() -> None:
    parser = argparse.ArgumentParser(description="SubGen Agent (LangChain minimal runtime)")
    parser.add_argument("--query", "-q", required=True, help="User query to the agent")
    args = parser.parse_args()

    # NOTE: model name and key are resolved by langchain_openai / env
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

            # Append assistant message containing tool calls
            messages.append(ai_msg)

            for call in tool_calls:
                name = call.get("name")
                call_id = call.get("id", "")

                tool = tool_map.get(name)
                if tool is None:
                    # Unknown tool call should NOT crash the whole runtime in PR#4 context
                    result = _fail(
                        err_type="runtime.unknown_tool",
                        message=f"Unknown tool called: {name}",
                        details={"name": name, "available": list(tool_map.keys())},
                    )
                else:
                    tool_args: Dict[str, Any] = call.get("args") or {}
                    if not isinstance(tool_args, dict):
                        tool_args = {"_args": tool_args}
                    result = _safe_tool_invoke(tool, tool_name=name or "unknown", tool_args=tool_args)

                # Append tool result (must include tool_call_id)
                messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=json.dumps(result, ensure_ascii=False),
                    )
                )
            continue

        # No tool call => final answer
        content = getattr(ai_msg, "content", "")
        # Some LC versions may return list/blocks; normalize to string.
        if isinstance(content, list):
            content = "\n".join([str(x) for x in content])
        print(content)
        return

    raise RuntimeError("Agent did not finish after max steps.")


if __name__ == "__main__":
    main()
