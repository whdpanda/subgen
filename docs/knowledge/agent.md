# Agent

## Runtime
- `python -m subgen.agent.runtime -q "..."`

## Tools
- `run_subgen_pipeline`: run subtitle pipeline
- `burn_subtitles`: burn SRT into video with ffmpeg
- `kb_search`: retrieve relevant internal docs chunks

## Execution pattern
1) `kb_search` to confirm schemas/flags/defaults
2) call tools with correct params
3) if tool fails, inspect error, retry with adjusted params
