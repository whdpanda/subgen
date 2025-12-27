from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from subgen.core.pipeline import run_pipeline

app = typer.Typer(help="High-accuracy-first local subtitle generator")


@app.command()
def gen(
    # ===== basic =====
    input: Path = typer.Argument(..., exists=True, help="Input video file"),
    out: Path = typer.Option(Path("./out"), help="Output directory"),
    lang: str = typer.Option("auto", help="Source language or auto"),
    to: str = typer.Option("zh", help="Target language"),
    glossary: Optional[Path] = typer.Option(None, help="Glossary json path"),
    mode: str = typer.Option("accurate", help="Mode: accurate/fast"),

    # ===== ASR options =====
    asr_model: str = typer.Option("large-v3", help="ASR model name"),
    asr_device: str = typer.Option("cuda", help="ASR device: auto/cuda/cpu"),
    asr_compute_type: Optional[str] = typer.Option("float16", help="ASR compute type: float16/int8/..."),
    asr_beam_size: int = typer.Option(5, help="ASR beam size"),
    asr_best_of: int = typer.Option(5, help="ASR best_of"),
    asr_vad_filter: bool = typer.Option(True, help="Enable VAD filter"),

    # ===== Translator options =====
    translator: str = typer.Option("nllb", help="Translator engine: nllb/openai"),
    openai_translate_model: str = typer.Option("gpt-5-mini", help="OpenAI translate model"),
    translator_model: str = typer.Option("facebook/nllb-200-distilled-600M", help="Translator model"),
    translator_device: str = typer.Option("cuda", help="Translator device: auto/cuda/cpu"),
    emit: str = typer.Option("all",help="What to export: all/literal/final/bilingual/bilingual-only/none"),


    # ===== LLM rewrite =====
    use_llm_rewrite: bool = typer.Option(True, help="Use LLM rewrite for high-risk segments"),

    # ===== Cache & intermediates =====
    use_cache: bool = typer.Option(True, help="Reuse cached ASR transcript if available"),
    dump_intermediates: bool = typer.Option(True, help="Dump literal/final JSON intermediates"),
):
    run_pipeline(
        video_path=input,
        out_dir=out,
        language=lang,
        target_lang=to,
        glossary_path=glossary,
        mode=mode,

        # ASR
        asr_model=asr_model,
        asr_device=asr_device,
        asr_compute_type=asr_compute_type,
        asr_beam_size=asr_beam_size,
        asr_best_of=asr_best_of,
        asr_vad_filter=asr_vad_filter,

        # Translator
        translator_name=translator,
        translator_model=translator_model,
        translator_device=translator_device,
        openai_translate_model=openai_translate_model,
        emit=emit,


        # LLM rewrite
        use_llm_rewrite=use_llm_rewrite,

        # Cache & dumps
        use_cache=use_cache,
        dump_intermediates=dump_intermediates,
    )


if __name__ == "__main__":
    app()
