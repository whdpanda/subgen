from __future__ import annotations

from typing import TYPE_CHECKING, Any

from subgen.agent.tools.tool_names import TOOL_NAMES, assert_tool_name

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool


def _guard_tool_obj(tool: Any) -> None:
    name = getattr(tool, "name", None)
    if not isinstance(name, str):
        raise TypeError(f"Tool object missing .name str attribute: {tool!r}")
    assert_tool_name(name)


def build_agent_tools() -> list["StructuredTool"]:
    """Build and return all agent tools in the canonical TOOL_NAMES order."""
    from langchain_core.tools import StructuredTool

    from subgen.agent.tools.burn_subtitles_tool import BurnToolArgs, burn_subtitles_tool
    from subgen.agent.tools.fix_subtitles_tool import FixToolArgs, fix_subtitles_tool
    from subgen.agent.tools.kb_search_tool import make_kb_search_tool
    from subgen.agent.tools.quality_check_subtitles_tool import (
        QualityToolArgs,
        quality_check_subtitles_tool,
    )
    from subgen.agent.tools.run_subgen_pipeline_tool import (
        PipelineToolArgs,
        run_subgen_pipeline_tool,
    )

    kb_tool = make_kb_search_tool()
    _guard_tool_obj(kb_tool)

    pipeline_tool = StructuredTool.from_function(
        name="run_subgen_pipeline",
        description="Run subgen pipeline to generate subtitles.",
        func=run_subgen_pipeline_tool,
        args_schema=PipelineToolArgs,
    )
    _guard_tool_obj(pipeline_tool)

    quality_tool = StructuredTool.from_function(
        name="quality_check_subtitles",
        description="Check subtitle quality and output a report.",
        func=quality_check_subtitles_tool,
        args_schema=QualityToolArgs,
    )
    _guard_tool_obj(quality_tool)

    fix_tool = StructuredTool.from_function(
        name="fix_subtitles",
        description="Fix subtitles based on deterministic rules / hints.",
        func=fix_subtitles_tool,
        args_schema=FixToolArgs,
    )
    _guard_tool_obj(fix_tool)

    burn_tool = StructuredTool.from_function(
        name="burn_subtitles",
        description="Burn subtitles into a video using ffmpeg.",
        func=burn_subtitles_tool,
        args_schema=BurnToolArgs,
    )
    _guard_tool_obj(burn_tool)

    tools = [kb_tool, pipeline_tool, quality_tool, fix_tool, burn_tool]
    name_to_tool: dict[str, StructuredTool] = {t.name: t for t in tools}

    if len(name_to_tool) != len(tools):
        names = [t.name for t in tools]
        raise RuntimeError(f"Duplicate tool names detected: {names}")

    got = set(name_to_tool.keys())
    expected = set(TOOL_NAMES)
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        raise RuntimeError(f"Tool set mismatch. missing={missing}, extra={extra}, expected={TOOL_NAMES}")

    return [name_to_tool[n] for n in TOOL_NAMES]


__all__ = ["build_agent_tools"]
