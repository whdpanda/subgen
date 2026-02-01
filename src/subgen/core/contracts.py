from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Literal

from subgen.core.audio.extract import AudioPreprocess

SegmenterName = Literal["rule", "openai"]
TranslatorName = Literal["auto_non_en", "nllb", "openai"]
EmitMode = Literal["all", "literal", "bilingual-only", "bilingual", "none"]


@dataclass(frozen=True)
class PipelineConfig:
    # I/O
    video_path: Path
    out_dir: Path

    # Optional orchestration fields (for Agent/Service/Queue)
    job_id: Optional[str] = None
    output_basename: Optional[str] = None  # default: video_path.stem if None

    # language
    language: str = "auto"
    target_lang: str = "zh"
    glossary_path: Optional[Path] = None

    # Audio preprocess
    preprocess: AudioPreprocess = "none"
    demucs_model: str = "htdemucs"

    # ASR
    asr_model: str = "large-v3"
    asr_device: str = "cuda"
    asr_compute_type: Optional[str] = "float16"
    asr_beam_size: int = 1
    asr_best_of: int = 1
    asr_vad_filter: bool = False

    # Segmenter
    segmenter: SegmenterName = "openai"
    openai_segment_model: str = "gpt-5.2"
    soft_max: float = 7.0
    hard_max: float = 15.0

    # Suspect tail-listen
    suspect_dur: float = 10.0
    suspect_cps: float = 6.0

    # Translator
    translator_name: TranslatorName = "auto_non_en"
    translator_model: str = "facebook/nllb-200-distilled-600M"
    translator_device: str = "cuda"
    openai_translate_model: str = "gpt-5.2"

    # Output
    emit: EmitMode = "bilingual-only"

    # Cache & dumps
    use_cache: bool = True
    dump_intermediates: bool = True


@dataclass(frozen=True)
class PipelineResult:
    # Core identifiers
    video_path: Path
    out_dir: Path

    # Primary outputs
    primary_path: Path
    srt_paths: list[Path]

    # Strongly-typed known outputs (stable keys for downstream tools)
    # Example keys:
    # - audio, asr_cache, src_json, literal_json, literal_srt, bilingual_srt, primary
    outputs: dict[str, Path]

    # Flexible extension fields for experiments/metrics
    artifacts: dict[str, Any]
    meta: dict[str, Any]


__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "SegmenterName",
    "TranslatorName",
    "EmitMode",
]
