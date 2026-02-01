from __future__ import annotations

from pathlib import Path
from typing import Optional, cast, Any

import typer

from subgen.core.contracts import (
    PipelineConfig,
    SegmenterName,
    TranslatorName,
    EmitMode,
)
from subgen.core.pipeline import run_pipeline
from subgen.core.audio.extract import AudioPreprocess

app = typer.Typer(help="High-accuracy-first subtitle generator (V1.2: OpenAI semantic segmentation)")


# ---- Defaults: single source of truth = PipelineConfig (contracts.py) ----
_CFG_FIELDS = PipelineConfig.__dataclass_fields__


def _cfg_default(name: str) -> Any:
    """
    Read default value from PipelineConfig dataclass field.
    This eliminates drift between CLI defaults and contract defaults.
    """
    return _CFG_FIELDS[name].default


# ---- Normalizers (CLI string -> strongly-typed contract enums) ----
def _normalize_preprocess(p: Optional[str]) -> AudioPreprocess:
    """
    CLI input is Optional[str], but pipeline expects a concrete enum-like value.
    - None / "" / "none" -> "none"
    - maps to speech_filter / demucs
    """
    if p is None:
        return "none"
    p2 = p.strip().lower()
    if p2 in ("", "none"):
        return "none"
    if p2 in ("speech_filter", "speech", "filter"):
        return "speech_filter"
    if p2 in ("demucs", "vocal", "vocals"):
        return "demucs"
    raise typer.BadParameter("preprocess must be one of: none, speech_filter, demucs")


def _normalize_segmenter(s: str) -> SegmenterName:
    s2 = s.strip().lower()
    if s2 in ("openai",):
        return "openai"
    if s2 in ("rule",):
        return "rule"
    raise typer.BadParameter("segmenter must be one of: openai, rule")


def _normalize_translator(s: str) -> TranslatorName:
    s2 = s.strip().lower()
    if s2 in ("auto_non_en", "auto"):
        return "auto_non_en"
    if s2 in ("openai",):
        return "openai"
    if s2 in ("nllb",):
        return "nllb"
    raise typer.BadParameter("translator must be one of: auto_non_en, openai, nllb")


def _normalize_emit(s: str) -> EmitMode:
    s2 = s.strip().lower()
    if s2 in ("all",):
        return "all"
    if s2 in ("literal",):
        return "literal"
    if s2 in ("bilingual-only", "bilingual_only", "bilingualonly"):
        return "bilingual-only"
    if s2 in ("bilingual",):
        return "bilingual"
    if s2 in ("none",):
        return "none"
    raise typer.BadParameter("emit must be one of: all, literal, bilingual-only, bilingual, none")


@app.command()
def gen(
    input: Path = typer.Argument(..., exists=True, help="Input video file"),
    out: Path = typer.Option(Path("./out"), help="Output directory"),
    lang: str = typer.Option(cast(str, _cfg_default("language")), help="Source language or auto (e.g., ja/bg/hy/auto)"),
    to: str = typer.Option(cast(str, _cfg_default("target_lang")), help="Target language"),
    glossary: Optional[Path] = typer.Option(None, help="Glossary json path"),

    # Audio preprocess
    preprocess: Optional[str] = typer.Option(
        None,
        help="Audio preprocess: none/speech_filter/demucs (demucs=best for background music)",
    ),
    demucs_model: str = typer.Option(cast(str, _cfg_default("demucs_model")), help="Demucs model"),

    # ASR (defaults from PipelineConfig)
    asr_model: str = typer.Option(cast(str, _cfg_default("asr_model")), help="ASR model name"),
    asr_device: str = typer.Option(cast(str, _cfg_default("asr_device")), help="ASR device: cuda/cpu"),
    asr_compute_type: Optional[str] = typer.Option(
        cast(Optional[str], _cfg_default("asr_compute_type")),
        help="ASR compute type: float16/int8/...",
    ),
    asr_beam_size: int = typer.Option(cast(int, _cfg_default("asr_beam_size")), help="ASR beam size"),
    asr_best_of: int = typer.Option(cast(int, _cfg_default("asr_best_of")), help="ASR best_of"),
    asr_vad_filter: bool = typer.Option(cast(bool, _cfg_default("asr_vad_filter")), help="Enable VAD filter"),

    # Segmenter (defaults from PipelineConfig)
    segmenter: str = typer.Option(cast(str, _cfg_default("segmenter")), help="Segmenter: openai/rule"),
    openai_segment_model: str = typer.Option(
        cast(str, _cfg_default("openai_segment_model")),
        help="OpenAI model for segmentation",
    ),
    soft_max: float = typer.Option(cast(float, _cfg_default("soft_max")), help="Soft max seconds per subtitle (beauty)"),
    hard_max: float = typer.Option(cast(float, _cfg_default("hard_max")), help="Hard max seconds per subtitle"),

    suspect_dur: float = typer.Option(cast(float, _cfg_default("suspect_dur")), help="Suspect duration threshold (seconds)"),
    suspect_cps: float = typer.Option(cast(float, _cfg_default("suspect_cps")), help="Suspect chars-per-second threshold"),

    # Translator (defaults from PipelineConfig)
    translator: str = typer.Option(cast(str, _cfg_default("translator_name")), help="Translator: auto_non_en/openai/nllb"),
    openai_translate_model: str = typer.Option(
        cast(str, _cfg_default("openai_translate_model")),
        help="OpenAI translate model",
    ),
    translator_model: str = typer.Option(
        cast(str, _cfg_default("translator_model")),
        help="NLLB model",
    ),
    translator_device: str = typer.Option(
        cast(str, _cfg_default("translator_device")),
        help="Translator device: cuda/cpu",
    ),

    # Output (defaults from PipelineConfig)
    emit: str = typer.Option(cast(str, _cfg_default("emit")), help="Emit: all/literal/bilingual-only/bilingual/none"),

    # Cache & dumps (defaults from PipelineConfig; keep CLI flag for disabling cache)
    no_use_cache: bool = typer.Option(False, "--no-use-cache", help="Disable ASR cache read/write"),
    dump_intermediates: bool = typer.Option(cast(bool, _cfg_default("dump_intermediates")), help="Dump src/literal JSON"),

    # Orchestration (optional)
    job_id: Optional[str] = typer.Option(None, help="Optional job id for orchestration/logging"),
    output_basename: Optional[str] = typer.Option(
        None,
        help="Optional output basename (default: input stem). Useful to avoid overwriting.",
    ),
):
    pp = _normalize_preprocess(preprocess)
    seg = _normalize_segmenter(segmenter)
    tr = _normalize_translator(translator)
    em = _normalize_emit(emit)

    cfg = PipelineConfig(
        video_path=input,
        out_dir=out,
        job_id=job_id,
        output_basename=output_basename,

        language=lang,
        target_lang=to,
        glossary_path=glossary,

        preprocess=pp,
        demucs_model=demucs_model,

        asr_model=asr_model,
        asr_device=asr_device,
        asr_compute_type=asr_compute_type,
        asr_beam_size=asr_beam_size,
        asr_best_of=asr_best_of,
        asr_vad_filter=asr_vad_filter,

        segmenter=seg,
        openai_segment_model=openai_segment_model,
        soft_max=soft_max,
        hard_max=hard_max,
        suspect_dur=suspect_dur,
        suspect_cps=suspect_cps,

        translator_name=tr,
        translator_model=translator_model,
        translator_device=translator_device,
        openai_translate_model=openai_translate_model,

        emit=em,
        use_cache=(not no_use_cache),
        dump_intermediates=dump_intermediates,
    )

    res = run_pipeline(cfg)

    primary = res.outputs.get("primary", res.primary_path)
    typer.echo(f"OK: {primary}")

    # Print key outputs (agent/service-friendly, consistent)
    if "bilingual_srt" in res.outputs:
        typer.echo(f"BILINGUAL_SRT: {res.outputs['bilingual_srt']}")
    if "literal_srt" in res.outputs:
        typer.echo(f"LITERAL_SRT: {res.outputs['literal_srt']}")
    if "src_json" in res.outputs:
        typer.echo(f"SRC_JSON: {res.outputs['src_json']}")
    if "literal_json" in res.outputs:
        typer.echo(f"LITERAL_JSON: {res.outputs['literal_json']}")

    for p in res.srt_paths:
        typer.echo(f"SRT: {p}")


if __name__ == "__main__":
    app()
