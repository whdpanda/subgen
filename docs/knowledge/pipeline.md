# Pipeline

## Entry
- `subgen.core.pipeline.run_pipeline(cfg: PipelineConfig) -> PipelineResult`

## Inputs
- `PipelineConfig` in `subgen.core.contracts`

Key fields (high-level):
- I/O: input video path, output directory
- language: `lang`
- preprocess: demucs/speech_filter/none
- ASR: whisper backend/model
- segmenter, translator
- emit mode, cache

## Outputs
`PipelineResult` contains:
- `primary_path`: main subtitle output path
- `outputs`: mapping (e.g., srt, json, artifacts)
- `artifacts`: additional files (logs, debug)
- `meta`: metadata

Agent tool:
- `run_subgen_pipeline` wraps `run_pipeline` and returns
  `{primary_path, srt_paths, outputs, artifacts, meta}`
