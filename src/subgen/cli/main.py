from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from subgen.core.pipeline import run_pipeline
from subgen.core.audio.extract import AudioPreprocess

app = typer.Typer(help="High-accuracy-first subtitle generator (V1.2: OpenAI semantic segmentation)")


def _normalize_preprocess(p: Optional[str]) -> Optional[AudioPreprocess]:
    if p is None:
        return None
    p2 = p.strip().lower()
    if p2 in ("", "none"):
        return "none"
    if p2 in ("speech_filter", "speech", "filter"):
        return "speech_filter"
    if p2 in ("demucs", "vocal", "vocals"):
        return "demucs"
    raise typer.BadParameter("preprocess must be one of: none, speech_filter, demucs")


@app.command()
def gen(
    input: Path = typer.Argument(..., exists=True, help="Input video file"),
    out: Path = typer.Option(Path("./out"), help="Output directory"),
    lang: str = typer.Option("auto", help="Source language or auto (e.g., ja/bg/hy/auto)"),
    to: str = typer.Option("zh", help="Target language"),
    glossary: Optional[Path] = typer.Option(None, help="Glossary json path"),

    # Audio preprocess (NEW)
    preprocess: Optional[str] = typer.Option(
        None,
        help="Audio preprocess: none/speech_filter/demucs (demucs=best for background music)",
    ),

    # ASR
    asr_model: str = typer.Option("large-v3", help="ASR model name"),
    asr_device: str = typer.Option("cuda", help="ASR device: auto/cuda/cpu"),
    asr_compute_type: Optional[str] = typer.Option("float16", help="ASR compute type: float16/int8/..."),
    asr_beam_size: int = typer.Option(5, help="ASR beam size"),
    asr_best_of: int = typer.Option(5, help="ASR best_of"),
    asr_vad_filter: bool = typer.Option(True, help="Enable VAD filter"),

    # Segmenter (V1.2)
    segmenter: str = typer.Option("openai", help="Segmenter: openai/rule"),
    openai_segment_model: str = typer.Option("gpt-5-mini", help="OpenAI model for segmentation"),
    soft_max: float = typer.Option(7.0, help="Soft max seconds per subtitle (beauty)"),
    hard_max: float = typer.Option(20.0, help="Hard max seconds per subtitle"),

    suspect_dur: float = typer.Option(10.0, help="Suspect duration threshold (seconds)"),
    suspect_cps: float = typer.Option(2.0, help="Suspect chars-per-second threshold"),

    # Translator
    translator: str = typer.Option("auto_non_en", help="Translator: auto_non_en/openai/nllb"),
    openai_translate_model: str = typer.Option("gpt-5-mini", help="OpenAI translate model"),
    translator_model: str = typer.Option("facebook/nllb-200-distilled-600M", help="NLLB model"),
    translator_device: str = typer.Option("cuda", help="Translator device: auto/cuda/cpu"),

    # Output
    emit: str = typer.Option("bilingual-only", help="Emit: all/literal/bilingual-only/none"),

    # Cache & dumps
    no_use_cache: bool = typer.Option(False, "--no-use-cache", help="Disable ASR cache read/write"),
    dump_intermediates: bool = typer.Option(True, help="Dump src/literal JSON"),
):
    pp = _normalize_preprocess(preprocess)

    run_pipeline(
        video_path=input,
        out_dir=out,
        language=lang,
        target_lang=to,
        glossary_path=glossary,

        preprocess=pp,

        asr_model=asr_model,
        asr_device=asr_device,
        asr_compute_type=asr_compute_type,
        asr_beam_size=asr_beam_size,
        asr_best_of=asr_best_of,
        asr_vad_filter=asr_vad_filter,

        segmenter=segmenter,
        openai_segment_model=openai_segment_model,
        soft_max=soft_max,
        hard_max=hard_max,
        suspect_dur=suspect_dur,
        suspect_cps=suspect_cps,

        translator_name=translator,
        openai_translate_model=openai_translate_model,
        translator_model=translator_model,
        translator_device=translator_device,

        emit=emit,
        use_cache=(not no_use_cache),
        dump_intermediates=dump_intermediates,
    )


if __name__ == "__main__":
    app()
