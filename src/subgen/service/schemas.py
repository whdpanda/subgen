from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from subgen.core.audio.extract import AudioPreprocess
from subgen.core.contracts import EmitMode, PipelineConfig, SegmenterName, TranslatorName


class RunPipelineRequest(BaseModel):
    video_path: Path = Field(..., description="Input video path")
    out_dir: Path = Field(..., description="Output directory")

    job_id: Optional[str] = None
    output_basename: Optional[str] = None

    language: str = "auto"
    target_lang: str = "zh"
    glossary_path: Optional[Path] = None

    preprocess: AudioPreprocess = "none"
    demucs_model: str = "htdemucs"

    asr_model: str = "large-v3"
    asr_device: str = "cuda"
    asr_compute_type: Optional[str] = "float16"
    asr_beam_size: int = 1
    asr_best_of: int = 1
    asr_vad_filter: bool = False

    segmenter: SegmenterName = "openai"
    openai_segment_model: str = "gpt-5.2"
    soft_max: float = 7.0
    hard_max: float = 15.0

    suspect_dur: float = 10.0
    suspect_cps: float = 6.0

    translator_name: TranslatorName = "auto_non_en"
    translator_model: str = "facebook/nllb-200-distilled-600M"
    translator_device: str = "cuda"
    openai_translate_model: str = "gpt-5.2"

    emit: EmitMode = "zh-only"

    zh_layout: bool = True
    zh_max_line_len: int = 18
    zh_max_lines: int = 2
    zh_line_len_cap: int = 42

    use_cache: bool = True
    dump_intermediates: bool = True

    def to_config(self) -> PipelineConfig:
        return PipelineConfig(**self.model_dump())


class RunPipelineResponse(BaseModel):
    status: Literal["ok"] = "ok"
    primary_path: Path
    srt_paths: list[Path]
    outputs: dict[str, Path]
    artifacts: dict[str, str]
    meta: dict[str, Any]

    @classmethod
    def from_result(cls, result) -> "RunPipelineResponse":
        return cls(
            primary_path=result.primary_path,
            srt_paths=result.srt_paths,
            outputs=result.outputs,
            artifacts={k: str(v) for k, v in result.artifacts.items()},
            meta={k: v for k, v in result.meta.items()},
        )
