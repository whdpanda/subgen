# SubGen Knowledge Base (RAG)

This folder contains curated internal documentation for **SubGen Agent**.  
It is indexed into a local **Chroma** vector store and queried at runtime via the `kb_search` tool.

---

## What goes here

Put **stable**, **project-specific** knowledge that the agent should rely on instead of guessing:

- Tool schemas and examples:
  - `kb_search`
  - `run_subgen_pipeline`
  - `quality_check_subtitles`
  - `fix_subtitles`
  - `burn_subtitles`
- Pipeline configuration fields (`PipelineConfig`) and output (`PipelineResult`)
- Default behaviors (cache, emit modes, Chinese layout rules, ffmpeg flags)
- Troubleshooting / FAQs (common ffmpeg errors, subtitle encoding, path quoting)

### Recommended files

- `pipeline.md` – pipeline entry/config/output schema
- `burn.md` – hard-sub behavior, ffmpeg notes (**burn runs only after quality loop**)
- `agent.md` – tool orchestration patterns (**PR#4c quality loop**)
- `config.md` – env vars and conventions

---

## PR#4c execution pattern (IMPORTANT)

The runtime enforces a multi-step loop:

1) (Optional) `kb_search`  
2) `run_subgen_pipeline`  
3) `quality_check_subtitles`  
4) If not ok: `fix_subtitles` → `quality_check_subtitles` (repeat until pass or max passes)  
5) (Optional) `burn_subtitles`

### PR#4c defaults

Unless user overrides explicitly:

- target language: `zh`
- emit: `zh-only` (Chinese mono SRT)
- Chinese layout: `zh_layout=true` (adds `\n` for readability)

---

## Output contract (IMPORTANT)

The runtime must return **ONLY paths produced by tools**.

- Never guess or hallucinate paths.
- On failure, return best-effort outputs and a `report_path` whenever available  
  (the `quality_check_subtitles` tool always attempts to write a report).

---

## Build / Update the KB index

From repo root:

```bash
python scripts/build_kb.py \
  --src docs/knowledge \
  --persist ./.subgen_kb \
  --collection subgen-kb
```

### Rebuild from scratch

```bash
python scripts/build_kb.py \
  --src docs/knowledge \
  --persist ./.subgen_kb \
  --collection subgen-kb \
  --reset
```
