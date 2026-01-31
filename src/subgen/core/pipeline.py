from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Literal

from subgen.core.audio.extract import extract_audio, AudioPreprocess
from subgen.core.asr.local_whisper import LocalWhisperASR
from subgen.core.align.noop import NoopAlign
from subgen.core.postprocess.punct_split import split_segments_on_sentence_end_punct
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
    make_asr_cache_key,
    asr_cache_path,
    save_transcript_json,
    load_transcript_json,
)

logger = get_logger()

SegmenterName = Literal["rule", "openai"]
TranslatorName = Literal["auto_non_en", "nllb", "openai"]
EmitMode = Literal["all", "literal", "bilingual-only", "bilingual", "none"]


@dataclass(frozen=True)
class PipelineConfig:
    # I/O
    video_path: Path
    out_dir: Path

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
    out_dir: Path
    primary_path: Path
    srt_paths: list[Path]
    artifacts: dict[str, Any]
    meta: dict[str, Any]


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    """
    Best practice: stable core entrypoint.
    - CLI / Agent / API all call this
    - Structured output for orchestration + observability
    """
    ensure_dir(cfg.out_dir)

    logger.info(
        "Pipeline config: preprocess=%s use_cache=%s segmenter=%s translator=%s emit=%s",
        cfg.preprocess,
        cfg.use_cache,
        cfg.segmenter,
        cfg.translator_name,
        cfg.emit,
    )

    artifacts: dict[str, Any] = {}
    meta: dict[str, Any] = {}

    # 1) audio
    audio = extract_audio(
        cfg.video_path,
        cfg.out_dir,
        preprocess=cfg.preprocess,
        demucs_model=cfg.demucs_model,
    )
    logger.info("Audio for ASR -> %s", audio.name)
    artifacts["audio_path"] = str(audio)

    # 2) ASR (cache)
    key = make_asr_cache_key(
        audio,
        asr_model=cfg.asr_model,
        language=cfg.language,
        preprocess=cfg.preprocess,
        asr_device=cfg.asr_device,
        asr_compute_type=cfg.asr_compute_type,
        asr_beam_size=cfg.asr_beam_size,
        asr_best_of=cfg.asr_best_of,
        asr_vad_filter=cfg.asr_vad_filter,
    )
    t_path = asr_cache_path(cfg.out_dir, cfg.video_path.stem, key)
    artifacts["asr_cache_key"] = key
    artifacts["asr_cache_path"] = str(t_path)

    asr_cache_hit = False
    if cfg.use_cache and t_path.exists():
        asr_cache_hit = True
        logger.info("ASR cache hit -> %s", t_path.name)
        t_dict = load_transcript_json(t_path)
        transcript = Transcript.model_validate(t_dict)
    else:
        if not cfg.use_cache:
            logger.info("ASR cache disabled -> forcing fresh ASR")
        else:
            logger.info("ASR cache miss -> running ASR")

        asr = LocalWhisperASR(
            model_name=cfg.asr_model,
            device=cfg.asr_device,
            compute_type=cfg.asr_compute_type,
            beam_size=cfg.asr_beam_size,
            best_of=cfg.asr_best_of,
            vad_filter=cfg.asr_vad_filter,
        )
        transcript = asr.transcribe(audio, language=cfg.language)

        save_transcript_json(t_path, transcript.model_dump())
        logger.info("ASR cached -> %s", t_path.name)

    # 3) align (noop)
    transcript = NoopAlign().align(audio, transcript)
    if not transcript.words:
        raise RuntimeError("Word timestamps required. transcript.words is empty.")

    # 4) segmentation
    rule_seg = RuleSegmenter(soft_max=cfg.soft_max, hard_max=cfg.hard_max)

    if cfg.segmenter == "openai":
        segger = OpenAISegmenter(
            model=cfg.openai_segment_model,
            soft_max=cfg.soft_max,
            hard_max=cfg.hard_max,
            min_seg=2.5,
            window_s=60.0,
            overlap_s=2.5,
        )
        raw_segments = segger.segment(transcript.words)
    elif cfg.segmenter == "rule":
        raw_segments = rule_seg.segment(transcript.words)
    else:
        raise ValueError(f"Unknown segmenter: {cfg.segmenter}")

    # 5) repairs
    src_lang = cfg.language if cfg.language != "auto" else (transcript.language or "unknown")
    repair_lang = None if src_lang == "unknown" else src_lang

    asr_for_retry = LocalWhisperASR(
        model_name=cfg.asr_model,
        device=cfg.asr_device,
        compute_type=cfg.asr_compute_type,
        beam_size=cfg.asr_beam_size,
        best_of=cfg.asr_best_of,
        vad_filter=cfg.asr_vad_filter,
    )

    segments = repair_segments_with_tail_listen(
        audio_path=audio,
        segments=raw_segments,
        asr=asr_for_retry,
        language=repair_lang,
        segmenter_rule=rule_seg,
        soft_max=cfg.soft_max,
        hard_max=cfg.hard_max,
        suspect_dur=cfg.suspect_dur,
        suspect_cps=cfg.suspect_cps,
    )

    segments = split_segments_on_sentence_end_punct(
        words=list(transcript.words or []),
        segments=segments,
        min_seg=2.5,
        hard_max=20.0,
    )

    segments = coalesce_segments(
        segments,
        min_dur=2.5,
        min_chars=8,
        target_dur=cfg.soft_max,
        hard_max=cfg.hard_max,
    )

    transcript_src = Transcript(language=src_lang, segments=segments)

    # 6) translate
    if cfg.translator_name == "auto_non_en":
        # NOTE: 保留你当前策略：en -> NLLB, others -> OpenAI
        if src_lang.lower().startswith("en"):
            translator = NLLBTranslator(model_name=cfg.translator_model, device=cfg.translator_device)
        else:
            translator = OpenAITranslator(model=cfg.openai_translate_model)
    elif cfg.translator_name == "openai":
        translator = OpenAITranslator(model=cfg.openai_translate_model)
    elif cfg.translator_name == "nllb":
        translator = NLLBTranslator(model_name=cfg.translator_model, device=cfg.translator_device)
    else:
        raise ValueError(f"Unknown translator: {cfg.translator_name}")

    zh_literal = translator.translate(transcript_src, target_lang=cfg.target_lang)

    # 7) glossary
    if cfg.glossary_path and cfg.glossary_path.exists():
        glossary = load_glossary(cfg.glossary_path)
        zh_literal = apply_glossary(zh_literal, glossary)
        artifacts["glossary_path"] = str(cfg.glossary_path)

    # 8) dumps
    src_json_path = cfg.out_dir / f"{cfg.video_path.stem}.src.json"
    literal_json_path = cfg.out_dir / f"{cfg.video_path.stem}.{cfg.target_lang}.literal.json"

    if cfg.dump_intermediates:
        src_json_path.write_text(
            json.dumps(transcript_src.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        literal_json_path.write_text(
            json.dumps(zh_literal.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    artifacts["src_json_path"] = str(src_json_path)
    artifacts["literal_json_path"] = str(literal_json_path)

    # 9) export
    srt_paths: list[Path] = []
    primary_path: Optional[Path] = None

    if cfg.emit in ("all", "literal"):
        p = cfg.out_dir / f"{cfg.video_path.stem}.{cfg.target_lang}.literal.srt"
        p.write_text(to_srt(zh_literal), encoding="utf-8")
        logger.info("Generated -> %s", p.name)
        srt_paths.append(p)
        primary_path = p

    if cfg.emit in ("all", "bilingual", "bilingual-only"):
        p = cfg.out_dir / f"{cfg.video_path.stem}.{cfg.target_lang}.bilingual.srt"
        p.write_text(to_srt_bilingual(transcript_src, zh_literal), encoding="utf-8")
        logger.info("Generated -> %s", p.name)
        srt_paths.append(p)
        primary_path = p

    if cfg.emit == "none":
        primary_path = literal_json_path

    if primary_path is None:
        # 保底：与你原逻辑一致
        primary_path = cfg.out_dir / f"{cfg.video_path.stem}.{cfg.target_lang}.bilingual.srt"

    meta.update(
        {
            "video_stem": cfg.video_path.stem,
            "src_lang": src_lang,
            "target_lang": cfg.target_lang,
            "segmenter": cfg.segmenter,
            "translator": cfg.translator_name,
            "emit": cfg.emit,
            "asr_cache_hit": asr_cache_hit,
        }
    )

    return PipelineResult(
        out_dir=cfg.out_dir,
        primary_path=primary_path,
        srt_paths=srt_paths,
        artifacts=artifacts,
        meta=meta,
    )
