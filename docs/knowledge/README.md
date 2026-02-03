# SubGen Knowledge Base (RAG)

This folder contains curated internal documentation for **SubGen Agent**.
It is indexed into a local **Chroma** vector store and queried at runtime via the `kb_search` tool.

## What goes here

Put **stable**, **project-specific** knowledge that the agent should rely on instead of guessing:

- Tool schemas and examples (`run_subgen_pipeline`, `burn_subtitles`, `kb_search`)
- Pipeline configuration fields (`PipelineConfig`) and output (`PipelineResult`)
- Default behaviors (cache, emit modes, audio settings, ffmpeg flags)
- Troubleshooting / FAQs (common ffmpeg errors, subtitle encoding, path quoting)

Recommended files:
- `pipeline.md` – pipeline entry/config/output schema
- `burn.md` – hard-sub behavior, ffmpeg notes
- `agent.md` – tool orchestration patterns (kb_search → pipeline → burn)
- `config.md` – env vars and conventions

## Build / Update the KB index

From repo root:

```bash
python scripts/build_kb.py \
  --src docs/knowledge \
  --persist ./.subgen_kb \
  --collection subgen-kb
