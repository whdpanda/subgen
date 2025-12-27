from __future__ import annotations

import json
from pathlib import Path

from subgen.core.audio.extract import extract_audio
from subgen.core.asr.local_whisper import LocalWhisperASR
from subgen.core.align.noop import NoopAlign
from subgen.core.translate.engine_nllb import NLLBTranslator
from subgen.core.translate.engine_openai import OpenAITranslator
from subgen.core.refine.glossary import load_glossary, apply_glossary
from subgen.core.export.srt import to_srt,to_srt_bilingual
from subgen.utils.io import ensure_dir
from subgen.utils.logger import get_logger

from subgen.core.refine.zh_rewrite_smart import SmartSelectiveZhRewriter
from subgen.core.refine.risk import RiskConfig
from subgen.core.refine.llm_openai import openai_rewrite_text

from subgen.core_types import Transcript, Segment

# cache helpers (需要你已添加 core/cache.py)
from subgen.core.cache import (
    make_asr_cache_key, asr_cache_path,
    save_transcript_json, load_transcript_json
)

logger = get_logger()

def merge_repeated_segments(transcript: Transcript) -> Transcript:
    merged: list[Segment] = []
    for seg in transcript.segments:
        text = (seg.text or "").strip()
        if not text:
            continue

        if merged and text == (merged[-1].text or "").strip():
            # 如果和上一条文字完全一样，只延长上一条的结束时间
            merged[-1].end = seg.end
        else:
            merged.append(seg)

    return Transcript(language=transcript.language, segments=merged)

def run_pipeline(
    video_path: Path,
    out_dir: Path,
    language: str = "auto",
    target_lang: str = "zh",
    glossary_path: Path | None = None,
    mode: str = "accurate",

    # ===== ASR options (默认=你的GPU个人版) =====
    asr_model: str = "large-v3",
    asr_device: str = "cuda",
    asr_compute_type: str | None = "float16",
    asr_beam_size: int = 5,
    asr_best_of: int = 5,
    asr_vad_filter: bool = True,

    # ===== Translator options (默认=你的GPU个人版) =====
    translator_name: str = "auto_non_en",
    translator_model: str = "facebook/nllb-200-distilled-600M",
    translator_device: str = "cuda",
    openai_translate_model: str = "gpt-5-mini",
    emit: str = "all",  # all/literal/final/bilingual/bilingual-only/none

    # ===== LLM rewrite =====
    use_llm_rewrite: bool = True,

    # ===== Cache =====
    use_cache: bool = True,
    dump_intermediates: bool = True,
) -> Path:
    ensure_dir(out_dir)

    # 1) audio
    audio = extract_audio(video_path, out_dir)

    # 2) ASR with cache
    key = make_asr_cache_key(audio, asr_model, language)
    t_path = asr_cache_path(out_dir, video_path.stem, key)

    if use_cache and t_path.exists():
        logger.info(f"ASR cache hit -> {t_path.name}")
        t_dict = load_transcript_json(t_path)
        transcript = Transcript.model_validate(t_dict)
    else:
        asr = LocalWhisperASR(
            model_name=asr_model,
            device=asr_device,
            compute_type=asr_compute_type,
            beam_size=asr_beam_size,
            best_of=asr_best_of,
            vad_filter=asr_vad_filter,
        )
        transcript = asr.transcribe(audio, language=language)
        save_transcript_json(t_path, transcript.model_dump())
        logger.info(f"ASR cached -> {t_path.name}")

    # 3) align (noop for now)
    aligner = NoopAlign()
    transcript = aligner.align(audio, transcript)

    # 3.5) 去掉相邻重复句子
    transcript = merge_repeated_segments(transcript)

    # 4) translate -> zh_literal
    src_lang = language if language != "auto" else (transcript.language or "unknown")

    if translator_name == "auto_non_en":
        if src_lang.lower().startswith("en"):
            translator = NLLBTranslator(
                model_name=translator_model,
                device=translator_device,
            )
            zh_literal = translator.translate(transcript, target_lang=target_lang)
        else:
            translator = OpenAITranslator(model=openai_translate_model)
            zh_literal = translator.translate(transcript, target_lang=target_lang)

    elif translator_name == "openai":
        translator = OpenAITranslator(model=openai_translate_model)
        zh_literal = translator.translate(transcript, target_lang=target_lang)

    elif translator_name == "nllb":
        translator = NLLBTranslator(
            model_name=translator_model,
            device=translator_device,
        )
        zh_literal = translator.translate(transcript, target_lang=target_lang)

    else:
        raise ValueError(f"Unknown translator: {translator_name}")

    # 5) glossary refine on zh_literal
    glossary = {}
    if glossary_path and glossary_path.exists():
        glossary = load_glossary(glossary_path)
        zh_literal = apply_glossary(zh_literal, glossary)

    # 5.5) zh_final (only compute if needed)
    emit = (emit or "all").lower()

    need_final = emit in ("all", "final")
    need_literal = emit in ("all", "literal")
    need_bilingual = emit in ("all", "bilingual", "bilingual-only")

    # bilingual-only 的典型场景：不需要 final
    if emit == "bilingual-only":
        need_final = False

    # 只在需要 final 且开启润色时才跑 rewriter
    if need_final and use_llm_rewrite:
        rewriter = SmartSelectiveZhRewriter(
            glossary=glossary,
            risk_cfg=RiskConfig(
                low_conf_threshold=0.35,
                number_count_threshold=2,
                glossary_hit_threshold=1,
                enable_mt_stiff_detector=True,
            ),
            llm_rewrite_fn=openai_rewrite_text
        )
        zh_final = rewriter.rewrite(zh_literal)
    else:
        # 不需要 final 或关闭润色时，直接复用 literal
        zh_final = zh_literal

    # 5.6) dump intermediates json (only when relevant)
    if dump_intermediates and emit in ("all", "literal", "final"):
        (out_dir / f"{video_path.stem}.{target_lang}.literal.json").write_text(
            json.dumps(zh_literal.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        (out_dir / f"{video_path.stem}.{target_lang}.final.json").write_text(
            json.dumps(zh_final.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    # 6) export SRT (controlled by emit)
    outputs: list[Path] = []

    if emit == "none":
        logger.info("Emit=none: no SRT exported.")
        return out_dir

    # literal
    if need_literal:
        literal_text = to_srt(zh_literal)
        literal_path = out_dir / f"{video_path.stem}.{target_lang}.literal.srt"
        literal_path.write_text(literal_text, encoding="utf-8")
        logger.info(f"Generated -> {literal_path.name}")
        outputs.append(literal_path)

    # final
    if need_final:
        final_text = to_srt(zh_final)
        final_path = out_dir / f"{video_path.stem}.{target_lang}.final.srt"
        final_path.write_text(final_text, encoding="utf-8")
        logger.info(f"Generated -> {final_path.name}")
        outputs.append(final_path)

    # bilingual (src + zh_literal)  ← 你要 literal
    if need_bilingual:
        bilingual_text = to_srt_bilingual(transcript, zh_literal)
        bilingual_path = out_dir / f"{video_path.stem}.{target_lang}.bilingual.srt"
        bilingual_path.write_text(bilingual_text, encoding="utf-8")
        logger.info(f"Generated -> {bilingual_path.name}")
        outputs.append(bilingual_path)

    # 返回“本次最主要产物”
    if outputs:
        return outputs[-1]
    return out_dir

