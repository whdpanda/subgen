# Pipeline

## Entry
- `subgen.core.pipeline.run_pipeline(cfg: PipelineConfig) -> PipelineResult`

## Inputs
- `PipelineConfig` in `subgen.core.contracts`

Key fields (high-level):
- I/O: `video_path`, `out_dir`, optional `output_basename`
- language: `language` (src), `target_lang` (default `zh`)
- preprocess: `none/demucs/...`
- ASR: whisper backend/model/device
- segmenter: `rule/openai`
- translator: `auto_non_en/openai/nllb`
- emit mode: output selection
- cache: `use_cache`
- dumps: `dump_intermediates`

### Emit modes (PR#4c)
- `zh-only` (default): output **Chinese mono SRT** (`<basename>.<target_lang>.srt`)
- `literal`: output literal SRT (legacy name: `<basename>.<target_lang>.literal.srt`)
- `bilingual-only` / `bilingual`: output bilingual SRT (`<basename>.<target_lang>.bilingual.srt`)
- `all`: output all supported SRTs
- `none`: no SRT, keep JSON/debug outputs

### Chinese layout (PR#4c)
If:
- `target_lang` starts with `zh`, and
- `zh_layout=true`

then the pipeline applies Chinese layout before exporting SRT:
- inserts `\n` for better readability
- respects:
  - `zh_max_line_len`
  - `zh_max_lines`
  - `zh_line_len_cap`

(Only text layout; does not change timestamps.)

## Outputs
`PipelineResult` contains:
- `primary_path`: main output path (depends on emit)
- `srt_paths`: list of produced SRT paths
- `outputs`: stable mapping of known output paths (srt/json/etc)
- `artifacts`: additional debugging/paths
- `meta`: metadata

### Agent tool
- `run_subgen_pipeline` wraps `run_pipeline` and returns:
  `{ok, primary_path, srt_paths, outputs, artifacts, meta}`

Downstream:
- `quality_check_subtitles` consumes an SRT path and writes `report_path`
- `fix_subtitles` writes a fixed SRT and returns `fixed_srt_path`
