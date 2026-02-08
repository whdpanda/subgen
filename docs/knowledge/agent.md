# Agent

SubGen is an agentized subtitle pipeline with a runtime-enforced quality loop.

## Runtime
Run the agent:
- `python -m subgen.agent.runtime -q "..."`

### Output contract (PR#4c)
The runtime prints **ONLY** one JSON object with **ONLY** these fields:

- `ok`
- `primary_path`
- `srt_path`
- `report_path`
- `out_video_path`

**Paths MUST come from tool outputs. Never hallucinate paths.**

## Tools (canonical names)
These tool names are fixed and must not drift:

- `kb_search`: retrieve relevant internal docs chunks (RAG)
- `run_subgen_pipeline`: generate subtitles via pipeline
- `quality_check_subtitles`: check subtitle quality and write report (always returns report_path best-effort)
- `fix_subtitles`: deterministically fix subtitle issues (best-effort returns fixed_srt_path)
- `burn_subtitles`: hard-sub into video with ffmpeg (optional)

## Required execution pattern (PR#4c)
The loop is **enforced by runtime**:

1) (Optional) `kb_search` to confirm schemas/flags/defaults
2) `run_subgen_pipeline`
3) `quality_check_subtitles`
4) If not ok → repeat:
   - `fix_subtitles`
   - `quality_check_subtitles`
   until:
   - pass, or
   - hit max passes N
5) (Optional) `burn_subtitles` (only after the loop ends)

### Defaults (PR#4c)
Unless user overrides explicitly:
- target language: `zh`
- emit mode: `zh-only` (Chinese mono SRT)
- Chinese layout enabled: `zh_layout=true` (adds line breaks for readability)

## Failure semantics (PR#4c)
- If tools fail: runtime should still return best-effort paths from tool outputs.
- `quality_check_subtitles` must always try to write a report and return `report_path` even on failure.
- The runtime may stop after max passes or max steps and still return a best-effort report path.
