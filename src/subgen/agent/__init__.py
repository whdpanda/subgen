# src/subgen/agent/tools/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from subgen.agent.tools.tool_names import TOOL_NAMES, TOOL_NAME_SET, assert_tool_name

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool


def _guard_tool_obj(tool: Any) -> None:
    """Guard: tool must have a canonical name and belong to TOOL_NAME_SET."""
    name = getattr(tool, "name", None)
    if not isinstance(name, str):
        raise TypeError(f"Tool object missing .name str attribute: {tool!r}")
    assert_tool_name(name)


def build_agent_tools() -> List["StructuredTool"]:
    """
    Build and return all agent tools.

    Design:
    - Lazy imports to avoid optional-deps errors at package import time.
    - Enforce canonical tool names + exact tool set to prevent drift.
    - Return tools in TOOL_NAMES order for prompt/KB/docs consistency.
    """
    # Lazy import: only needed when actually building tools
    from langchain_core.tools import StructuredTool

    # --- Lazy imports of each tool module (match your actual module names) ---
    # If your module filenames differ, change ONLY these imports.
    from subgen.agent.tools.kb_search_tool import make_kb_search_tool
    from subgen.agent.tools.run_subgen_pipeline_tool import run_subgen_pipeline_tool, PipelineToolArgs
    from subgen.agent.tools.quality_check_subtitles_tool import quality_check_subtitles_tool, QualityToolArgs
    from subgen.agent.tools.fix_subtitles_tool import fix_subtitles_tool, FixToolArgs
    from subgen.agent.tools.burn_subtitles_tool import burn_subtitles_tool, BurnToolArgs

    # --- Build tool objects ---
    kb_tool = make_kb_search_tool()
    _guard_tool_obj(kb_tool)

    pipeline_tool = StructuredTool.from_function(
        name="run_subgen_pipeline",
        description="Run subgen pipeline to generate subtitles. Return primary_path and outputs/artifacts/meta.",
        func=run_subgen_pipeline_tool,
        args_schema=PipelineToolArgs,
    )
    _guard_tool_obj(pipeline_tool)

    quality_tool = StructuredTool.from_function(
        name="quality_check_subtitles",
        description="Check subtitle quality for an SRT and write a report.",
        func=quality_check_subtitles_tool,
        args_schema=QualityToolArgs,
    )
    _guard_tool_obj(quality_tool)

    fix_tool = StructuredTool.from_function(
        name="fix_subtitles",
        description="Fix subtitles based on deterministic rules / report hints.",
        func=fix_subtitles_tool,
        args_schema=FixToolArgs,
    )
    _guard_tool_obj(fix_tool)

    burn_tool = StructuredTool.from_function(
        name="burn_subtitles",
        description="Burn an SRT into a video using ffmpeg (hard-sub).",
        func=burn_subtitles_tool,
        args_schema=BurnToolArgs,
    )
    _guard_tool_obj(burn_tool)

    tools = [kb_tool, pipeline_tool, quality_tool, fix_tool, burn_tool]

    # --- Enforce exact tool set ---
    name_to_tool: Dict[str, StructuredTool] = {t.name: t for t in tools}

    # 1) No duplicates
    if len(name_to_tool) != len(tools):
        names = [t.name for t in tools]
        raise RuntimeError(f"Duplicate tool names detected: {names}")

    # 2) Must match canonical set exactly
    got = set(name_to_tool.keys())
    expected = set(TOOL_NAMES)

    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        raise RuntimeError(f"Tool set mismatch. missing={missing}, extra={extra}, expected={TOOL_NAMES}")

    # 3) Return in canonical order
    return [name_to_tool[name] for name in TOOL_NAMES]


__all__ = ["build_agent_tools"]
