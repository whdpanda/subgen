from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from subgen.agent.tools.pipeline_tool import run_subgen_pipeline_tool, PipelineToolArgs
from subgen.agent.tools.burn_tool import burn_subtitles_tool, BurnToolArgs


SYSTEM = """You are SubGen Agent.

Primary goals:
1) Generate subtitles (SRT) for a given video.
2) Optionally burn subtitles into the video (hard-sub) when the user asks.

Rules:
- If the user asks to generate subtitles, you MUST call run_subgen_pipeline.
- If the user asks to burn subtitles into video, you MUST call burn_subtitles.
- Always return output file path(s) from tool outputs ONLY.
- Do NOT hallucinate file paths.
"""


def build_tools() -> List[StructuredTool]:
    pipeline_tool = StructuredTool.from_function(
        name="run_subgen_pipeline",
        description=(
            "Run subgen pipeline to generate subtitles. "
            "Inputs: video_path, out_dir, language, target_lang, preprocess, segmenter, "
            "openai_segment_model, translator_name, emit, use_cache, dump_intermediates, etc. "
            "Returns: primary_path, srt_paths, outputs, artifacts, meta."
        ),
        func=run_subgen_pipeline_tool,
        args_schema=PipelineToolArgs,
    )

    burn_tool = StructuredTool.from_function(
        name="burn_subtitles",
        description=(
            "Burn an SRT subtitle file into the video (hard-sub) using ffmpeg. "
            "Inputs: video_path, srt_path, out_path(optional), force_style(optional), crf, preset, copy_audio. "
            "Returns: out_video_path."
        ),
        func=burn_subtitles_tool,
        args_schema=BurnToolArgs,
    )

    return [pipeline_tool, burn_tool]


def _extract_tool_calls(ai_msg: Any) -> List[Dict[str, Any]]:
    """
    Normalize tool calls across langchain versions.
    Expected shape: [{"id": "...", "name": "...", "args": {...}}, ...]
    """
    calls = getattr(ai_msg, "tool_calls", None)
    if calls:
        return list(calls)

    ak = getattr(ai_msg, "additional_kwargs", None) or {}
    calls2 = ak.get("tool_calls")
    if calls2:
        return list(calls2)

    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="SubGen Agent (LangChain minimal runtime)")
    parser.add_argument("--query", "-q", required=True, help="User query to the agent")
    args = parser.parse_args()

    model_name = os.getenv("SUBGEN_AGENT_MODEL", "gpt-5.2")
    llm = ChatOpenAI(model=model_name, temperature=0)

    tools = build_tools()
    tool_map = {t.name: t for t in tools}

    llm_with_tools = llm.bind_tools(tools)

    # ✅ 关键：用 LangChain Message 对象，不要用 dict
    messages: List[Any] = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=args.query),
    ]

    for _ in range(8):
        ai_msg = llm_with_tools.invoke(messages)

        tool_calls = _extract_tool_calls(ai_msg)
        if tool_calls:
            # ✅ 关键：append ai_msg 本体（它包含 tool_calls）
            messages.append(ai_msg)

            for call in tool_calls:
                name = call.get("name")
                call_id = call.get("id", "")
                tool = tool_map.get(name)
                if tool is None:
                    raise RuntimeError(f"Unknown tool called: {name}")

                tool_args: Dict[str, Any] = call.get("args") or {}
                result = tool.invoke(tool_args)

                # ✅ 关键：用 ToolMessage（并带 tool_call_id）
                messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=json.dumps(result, ensure_ascii=False),
                    )
                )
            continue

        # No tool call => final answer
        print(getattr(ai_msg, "content", ""))
        return

    raise RuntimeError("Agent did not finish after max steps.")


if __name__ == "__main__":
    main()
