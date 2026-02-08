# src/subgen/agent/tools/tool_names.py

# Canonical external tool names (MUST NOT CHANGE)
KB_SEARCH = "kb_search"
RUN_SUBGEN_PIPELINE = "run_subgen_pipeline"
QUALITY_CHECK_SUBTITLES = "quality_check_subtitles"
FIX_SUBTITLES = "fix_subtitles"
BURN_SUBTITLES = "burn_subtitles"

# Ordered for docs/KB/prompt consistency
TOOL_NAMES = [
    KB_SEARCH,
    RUN_SUBGEN_PIPELINE,
    QUALITY_CHECK_SUBTITLES,
    FIX_SUBTITLES,
    BURN_SUBTITLES,
]

TOOL_NAME_SET = set(TOOL_NAMES)


def assert_tool_name(name: str) -> None:
    """Runtime-side guard: avoid tool name drift."""
    if name not in TOOL_NAME_SET:
        raise RuntimeError(f"Unknown tool name: {name!r}. Expected one of: {TOOL_NAMES}")
