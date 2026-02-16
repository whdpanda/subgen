# src/subgen/agent/tools/schemas.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Type, TypeVar, Union, Dict

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


def parse_tool_args(model_cls: Type[T], args_or_kwargs: Union[T, dict[str, Any]]) -> T:
    """
    Unified Pydantic args parser compatible with:
      - Pydantic v2: model_validate
      - Pydantic v1: constructor
    Also supports passing an already-instantiated model object.
    """
    if isinstance(args_or_kwargs, model_cls):
        return args_or_kwargs

    if hasattr(model_cls, "model_validate"):
        # pydantic v2
        return model_cls.model_validate(args_or_kwargs)
    # pydantic v1
    if not isinstance(args_or_kwargs, dict):
        raise TypeError("args_or_kwargs must be dict when using pydantic v1 constructor")
    return model_cls(**args_or_kwargs)


# -------------------------
# Quality Check Tool
# -------------------------
class QualityProfileOverrides(BaseModel):
    """Shared quality profile override knobs used by quality/fix tools."""

    max_cps: float = Field(16.0, description="Max chars-per-second")
    max_line_len: int = Field(18, description="Max line length")
    max_lines: int = Field(1, description="Max number of lines")
    min_dur_ms: int = Field(900, description="Min cue duration ms")
    max_dur_ms: int = Field(6500, description="Max cue duration ms")
    max_overlap_ms: int = Field(0, description="Max allowed overlap ms")


class QualityToolArgs(QualityProfileOverrides):
    srt_path: Path = Field(..., description="Input SRT path")
    profile: str = Field("default", description="Quality profile name")

    out_dir: Optional[Path] = Field(
        None,
        description="Directory to write quality_report*.json (default: alongside srt; fallback: cwd/out)",
    )


# -------------------------
# Fix Subtitles Tool
# -------------------------
class FixToolArgs(QualityProfileOverrides):
    srt_path: Path = Field(..., description="Input SRT path")

    profile: str = Field("default", description="Quality profile name")

    max_passes: int = Field(2, description="Max fix passes (core deterministic)")
    out_path: Optional[Path] = Field(
        None,
        description="Output fixed SRT path (default: <stem>.fixed.srt alongside input)",
    )
    out_dir: Optional[Path] = Field(
        None,
        description="Directory for fixed outputs (default: alongside srt)",
    )


# -------------------------
# Burn Subtitles Tool
# -------------------------
class BurnToolArgs(BaseModel):
    video_path: Path = Field(..., description="Input video path")
    srt_path: Path = Field(..., description="Subtitle SRT path to burn-in")
    out_path: Optional[Path] = Field(
        None, description="Output video path (mp4). If omitted, auto-generate."
    )

    ffmpeg_bin: str = Field("ffmpeg", description="ffmpeg binary name/path")
    force_style: Optional[str] = Field(None, description="ASS force_style string")

    crf: int = Field(18, description="x264 CRF")
    preset: str = Field("veryfast", description="x264 preset")

    copy_audio: bool = Field(False, description="Copy audio stream if possible")
    audio_codec: str = Field("aac", description="Audio codec when not copying")
    audio_bitrate: str = Field("192k", description="Audio bitrate when encoding")


# -------------------------
# Run Pipeline Tool
# -------------------------
class PipelineToolArgs(BaseModel):
    # I/O
    video_path: Path = Field(..., description="Input video file path")
    out_dir: Path = Field(Path("./out"), description="Output directory")

    # language
    language: str = Field("auto", description="Source language or 'auto'")
    target_lang: str = Field("zh", description="Target language (default: zh)")
    glossary_path: Optional[Path] = Field(None, description="Optional glossary json path")

    # Audio preprocess (legacy)
    preprocess: str = Field("none", description="none/speech_filter/demucs")
    demucs_model: str = Field("htdemucs", description="Demucs model name when preprocess=demucs")

    # Audio preprocess (PR6-ready)
    demucs_device: str = Field("cpu", description="Demucs device cpu/cuda (future)")
    demucs_stems: str = Field("vocals", description="Demucs stems (typically vocals)")
    preprocess_cache_dir: Optional[str] = Field(None, description="Cache dir for torch/demucs (e.g. /cache)")
    demucs_params: Optional[Dict[str, Any]] = Field(
        None,
        description="Extra demucs flags (e.g. {shifts:1, overlap:0.25})",
    )

    # ASR
    asr_model: str = Field("large-v3", description="ASR model name")
    asr_device: str = Field("auto", description="cuda/cpu/auto (default: auto-select)")
    asr_compute_type: Optional[str] = Field(
        None,
        description="ASR compute type; None => cuda->float16, cpu->int8",
    )
    asr_beam_size: int = Field(5, description="beam size")
    asr_best_of: int = Field(5, description="best_of")
    asr_vad_filter: bool = Field(True, description="Enable VAD filter")

    # Segmenter
    segmenter: str = Field("openai", description="openai/rule")
    openai_segment_model: str = Field("gpt-5.2", description="OpenAI model for segmentation")
    soft_max: float = Field(7.0, description="Soft max seconds per subtitle")
    hard_max: float = Field(20.0, description="Hard max seconds per subtitle")

    # Suspect tail-listen
    suspect_dur: float = Field(10.0, description="Suspect duration threshold (seconds)")
    suspect_cps: float = Field(2.0, description="Suspect chars-per-second threshold")

    # Translator
    translator_name: str = Field("auto_non_en", description="auto_non_en/openai/nllb")
    translator_model: str = Field("facebook/nllb-200-distilled-600M", description="NLLB model")
    translator_device: str = Field("cuda", description="cuda/cpu/auto")
    openai_translate_model: str = Field("gpt-5.2", description="OpenAI model for translation")

    # Output (PR#4c default: Chinese mono SRT)
    emit: str = Field("zh-only", description="zh-only/all/literal/bilingual-only/bilingual/none")

    # PR#4c: Chinese layout/segmentation (best-effort; only passed if PipelineConfig supports)
    zh_layout: bool = Field(True, description="Apply Chinese layout/line-wrapping postprocess when supported")
    zh_max_line_len: int = Field(18, description="Preferred max line length for Chinese layout when supported")
    zh_max_lines: int = Field(2, description="Max lines per cue for Chinese layout when supported")
    zh_line_len_cap: int = Field(42, description="Hard cap line length for Chinese layout when supported")

    # Cache & dumps
    use_cache: bool = Field(True, description="Enable ASR cache")
    dump_intermediates: bool = Field(True, description="Dump intermediates json")

    # Orchestration (future-facing)
    job_id: Optional[str] = Field(None, description="Optional job id")
    output_basename: Optional[str] = Field(None, description="Optional output basename")
