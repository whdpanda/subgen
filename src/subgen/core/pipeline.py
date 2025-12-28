from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from subgen.core.audio.extract import extract_audio, AudioPreprocess
from subgen.core.asr.local_whisper import LocalWhisperASR
from subgen.core.align.noop import NoopAlign
from subgen.core.translate.engine_nllb import NLLBTranslator
from subgen.core.translate.engine_openai import OpenAITranslator
from subgen.core.refine.glossary import load_glossary, apply_glossary
from subgen.core.export.srt import to_srt, to_srt_bilingual
from subgen.utils.io import ensure_dir
from subgen.utils.logger import get_logger

from subgen.core.segment.rule import RuleSegmenter
from subgen.core.segment.openai_segmenter import OpenAISegmenter
from subgen.core.postprocess.repair import repair_segments_with_tail_listen
from subgen.core.postprocess.coalesce import coalesce_segments

from subgen.core_types import Transcript

from subgen.core.cache import (
    make_asr_cache_key, asr_cache_path,
    save_transcript_json, load_transcript_json
)

logger = get_logger()


def run_pipeline(
    video_path: Path,
    out_dir: Path,
    language: str = "auto",
    target_lang: str = "zh",
    glossary_path: Optional[Path] = None,

    # Audio preprocess (NEW)
    preprocess: AudioPreprocess = "none",
    demucs_model: str = "htdemucs",

    # ASR
    asr_model: str = "large-v3",
    asr_device: str = "cuda",
    asr_compute_type: Optional[str] = "float16",
    asr_beam_size: int = 5,
    asr_best_of: int = 5,
    asr_vad_filter: bool = True,

    # Segmenter
    segmenter: str = "openai",  # rule/openai
    openai_segment_model: str = "gpt-5-mini",
    soft_max: float = 7.0,
    hard_max: float = 20.0,

    # Suspect tail-listen
    suspect_dur: float = 10.0,
    suspect_cps: float = 2.0,

    # Translator
    translator_name: str = "auto_non_en",
    translator_model: str = "facebook/nllb-200-distilled-600M",
    translator_device: str = "cuda",
    openai_translate_model: str = "gpt-5-mini",

    # Output
    emit: str = "bilingual-only",  # all/literal/bilingual-only/none

    # Cache & dumps
    use_cache: bool = True,
    dump_intermediates: bool = True,
) -> Path:
    ensure_dir(out_dir)

    logger.info(f"Pipeline config: preprocess={preprocess}, use_cache={use_cache}")

    # 1) audio
    audio = extract_audio(video_path, out_dir, preprocess=preprocess, demucs_model=demucs_model)
    logger.info(f"Audio for ASR -> {audio.name}")

    # 2) ASR (cache)
    key = make_asr_cache_key(
        audio,
        asr_model=asr_model,
        language=language,
        preprocess=preprocess,
        asr_device=asr_device,
        asr_compute_type=asr_compute_type,
        asr_beam_size=asr_beam_size,
        asr_best_of=asr_best_of,
        asr_vad_filter=asr_vad_filter,
    )
    t_path = asr_cache_path(out_dir, video_path.stem, key)

    if use_cache and t_path.exists():
        logger.info(f"ASR cache hit -> {t_path.name}")
        t_dict = load_transcript_json(t_path)
        transcript = Transcript.model_validate(t_dict)
    else:
        if not use_cache:
            logger.info("ASR cache disabled (use_cache=False) -> forcing fresh ASR")
        else:
            logger.info("ASR cache miss -> running ASR")

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

    # 3) align (noop)
    aligner = NoopAlign()
    transcript = aligner.align(audio, transcript)

    if not transcript.words:
        raise RuntimeError("V1.2 requires word timestamps. transcript.words is empty.")

    # 4) segmentation
    rule_seg = RuleSegmenter(soft_max=soft_max, hard_max=hard_max)

    if segmenter == "openai":
        segger = OpenAISegmenter(
            model=openai_segment_model,
            soft_max=soft_max,
            hard_max=hard_max,
            min_seg=2.5,
            window_s=60.0,
            overlap_s=2.5,
        )
        raw_segments = segger.segment(transcript.words)
    elif segmenter == "rule":
        raw_segments = rule_seg.segment(transcript.words)
    else:
        raise ValueError(f"Unknown segmenter: {segmenter}")

    # 5) repairs
    src_lang = language if language != "auto" else (transcript.language or "unknown")
    repair_lang = None if src_lang == "unknown" else src_lang

    asr_for_retry = LocalWhisperASR(
        model_name=asr_model,
        device=asr_device,
        compute_type=asr_compute_type,
        beam_size=asr_beam_size,
        best_of=asr_best_of,
        vad_filter=asr_vad_filter,
    )

    segments = repair_segments_with_tail_listen(
        audio_path=audio,
        segments=raw_segments,
        asr=asr_for_retry,
        language=repair_lang,
        segmenter_rule=rule_seg,
        soft_max=soft_max,
        hard_max=hard_max,
        suspect_dur=suspect_dur,
        suspect_cps=suspect_cps,
    )

    # FINAL GUARD: coalesce tiny fragments
    segments = coalesce_segments(
        segments,
        min_dur=2.5,
        min_chars=8,
        target_dur=soft_max,
        hard_max=hard_max,
    )

    transcript_src = Transcript(language=src_lang, segments=segments)

    # 6) translate
    if translator_name == "auto_non_en":
        if src_lang.lower().startswith("en"):
            translator = NLLBTranslator(model_name=translator_model, device=translator_device)
        else:
            translator = OpenAITranslator(model=openai_translate_model)
    elif translator_name == "openai":
        translator = OpenAITranslator(model=openai_translate_model)
    elif translator_name == "nllb":
        translator = NLLBTranslator(model_name=translator_model, device=translator_device)
    else:
        raise ValueError(f"Unknown translator: {translator_name}")

    zh_literal = translator.translate(transcript_src, target_lang=target_lang)

    # 7) glossary
    if glossary_path and glossary_path.exists():
        glossary = load_glossary(glossary_path)
        zh_literal = apply_glossary(zh_literal, glossary)

    # 8) dumps
    if dump_intermediates:
        (out_dir / f"{video_path.stem}.src.json").write_text(
            json.dumps(transcript_src.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / f"{video_path.stem}.{target_lang}.literal.json").write_text(
            json.dumps(zh_literal.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 9) export
    out_last: Optional[Path] = None

    if emit in ("all", "literal"):
        srt_text = to_srt(zh_literal)
        p = out_dir / f"{video_path.stem}.{target_lang}.literal.srt"
        p.write_text(srt_text, encoding="utf-8")
        logger.info(f"Generated -> {p.name}")
        out_last = p

    if emit in ("all", "bilingual", "bilingual-only"):
        bi_text = to_srt_bilingual(transcript_src, zh_literal)
        p = out_dir / f"{video_path.stem}.{target_lang}.bilingual.srt"
        p.write_text(bi_text, encoding="utf-8")
        logger.info(f"Generated -> {p.name}")
        out_last = p

    if emit == "none":
        out_last = out_dir / f"{video_path.stem}.{target_lang}.literal.json"

    return out_last or (out_dir / f"{video_path.stem}.{target_lang}.bilingual.srt")
