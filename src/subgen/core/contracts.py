from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Literal

from subgen.core.audio.extract import AudioPreprocess

SegmenterName = Literal["rule", "openai"]
TranslatorName = Literal["auto_non_en", "nllb", "openai"]

# PR#4c:
# - add "zh-only" as the new default emit mode (Chinese mono SRT)
EmitMode = Literal[
    "all",
    "literal",
    "zh-only",
    "bilingual-only",
    "bilingual",
    "none",
]


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

    # Audio preprocess (legacy knobs kept)
    preprocess: AudioPreprocess = "none"
    demucs_model: str = "htdemucs"

    # Audio preprocess (PR6-ready structured knobs)
    # - Keep these optional and safe defaults so PR5 remains stable.
    demucs_device: str = "cpu"  # cpu/cuda (future)
    demucs_stems: str = "vocals"  # usually vocals
    preprocess_cache_dir: Optional[str] = None  # e.g. "/cache"
    demucs_params: dict[str, Any] = field(default_factory=dict)  # extra demucs flags

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
    # PR#4c: default -> zh-only (Chinese mono SRT)
    emit: EmitMode = "zh-only"

    # PR#4c: apply Chinese layout/wrapping before SRT export when target_lang startswith "zh"
    zh_layout: bool = True
    zh_max_line_len: int = 18
    zh_max_lines: int = 2
    zh_line_len_cap: int = 42  # hard cap to avoid pathological lines

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