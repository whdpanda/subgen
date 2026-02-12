from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from subgen.core.contracts import PipelineConfig, PipelineResult

from subgen.core.audio.extract import extract_audio
from subgen.core.asr.local_whisper import LocalWhisperASR
from subgen.core.align.noop import NoopAlign
from subgen.core.postprocess.pipeline import apply_postprocess_pipeline
from subgen.core.postprocess.zh_layout import apply_zh_layout_split_to_cues  # UPDATED: split-to-cues
from subgen.core.translate.engine_nllb import NLLBTranslator
from subgen.core.translate.engine_openai import OpenAITranslator
from subgen.core.refine.glossary import load_glossary, apply_glossary
from subgen.core.export.srt import to_srt, to_srt_bilingual
from subgen.utils.io import ensure_dir
from subgen.utils.logger import get_logger

from subgen.core.segment.rule import RuleSegmenter
from subgen.core.segment.openai_segmenter import OpenAISegmenter
from subgen.core.quality.fixers import repair_segments_with_tail_listen

from subgen.core_types import Transcript

from subgen.core.cache import (
    make_asr_cache_key,
    asr_cache_path,
    save_transcript_json,
    load_transcript_json,
)

logger = get_logger()


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
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
    outputs: dict[str, Path] = {}

    # ---- fixed identifiers ----
    artifacts["video_path"] = str(cfg.video_path)
    artifacts["out_dir"] = str(cfg.out_dir)

    outputs["video"] = cfg.video_path
    outputs["out_dir"] = cfg.out_dir  # note: Path, though "dir" is not a file

    # IMPORTANT: honor output_basename to avoid overwriting and keep contract consistent
    basename = cfg.output_basename or cfg.video_path.stem
    artifacts["output_basename"] = basename

    # Pre-compute output paths (even if not emitted), so downstream tools don't guess filenames.
    src_json_path = cfg.out_dir / f"{basename}.src.json"
    literal_json_path = cfg.out_dir / f"{basename}.{cfg.target_lang}.literal.json"

    # Legacy name kept for compatibility/tests
    literal_srt_path = cfg.out_dir / f"{basename}.{cfg.target_lang}.literal.srt"
    bilingual_srt_path = cfg.out_dir / f"{basename}.{cfg.target_lang}.bilingual.srt"

    # PR#4c: new canonical mono SRT name (no ".literal" suffix)
    mono_srt_path = cfg.out_dir / f"{basename}.{cfg.target_lang}.srt"

    artifacts["src_json_path"] = str(src_json_path)
    artifacts["literal_json_path"] = str(literal_json_path)
    artifacts["literal_srt_path"] = str(literal_srt_path)
    artifacts["bilingual_srt_path"] = str(bilingual_srt_path)
    artifacts["mono_srt_path"] = str(mono_srt_path)

    outputs["src_json"] = src_json_path
    outputs["literal_json"] = literal_json_path
    outputs["literal_srt"] = literal_srt_path
    outputs["bilingual_srt"] = bilingual_srt_path
    outputs["mono_srt"] = mono_srt_path

    # 1) audio
    audio = extract_audio(
        cfg.video_path,
        cfg.out_dir,
        preprocess=cfg.preprocess,
        demucs_model=cfg.demucs_model,
    )
    logger.info("Audio for ASR -> %s", audio.name)
    artifacts["audio_path"] = str(audio)
    outputs["audio"] = audio

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
    # Keep cache keyed by original video stem (stable identity); output_basename only affects final artifacts.
    t_path = asr_cache_path(cfg.out_dir, cfg.video_path.stem, key)
    artifacts["asr_cache_key"] = key
    artifacts["asr_cache_path"] = str(t_path)
    outputs["asr_cache"] = t_path

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

    segments = apply_postprocess_pipeline(
        words=list(transcript.words or []),
        segments=segments,
        min_seg=2.5,
        soft_max=cfg.soft_max,
        hard_max=cfg.hard_max,
        min_chars=8,
    )

    transcript_src = Transcript(language=src_lang, segments=segments)

    # 6) translate
    if cfg.translator_name == "auto_non_en":
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

    # UPDATED: Chinese layout -> split into multiple cues and re-allocate timestamps
    if cfg.zh_layout and str(cfg.target_lang).lower().startswith("zh"):
        zh_literal = apply_zh_layout_split_to_cues(
            zh_literal,
            max_line_len=int(cfg.zh_max_line_len),
            max_lines=int(cfg.zh_max_lines),
            line_len_cap=int(cfg.zh_line_len_cap),
        )
        artifacts["zh_layout"] = True
        artifacts["zh_layout_mode"] = "split_to_cues"
        artifacts["zh_layout_params"] = {
            "max_line_len": int(cfg.zh_max_line_len),
            "max_lines": int(cfg.zh_max_lines),
            "line_len_cap": int(cfg.zh_line_len_cap),
        }
    else:
        artifacts["zh_layout"] = False

    # 8) dumps
    if cfg.dump_intermediates:
        src_json_path.write_text(
            json.dumps(transcript_src.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        literal_json_path.write_text(
            json.dumps(zh_literal.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 9) export
    srt_paths: list[Path] = []
    primary_path: Optional[Path] = None

    # PR#4c: zh-only -> write mono_srt_path
    if cfg.emit in ("zh-only", "all"):
        mono_srt_path.write_text(to_srt(zh_literal), encoding="utf-8")
        logger.info("Generated -> %s", mono_srt_path.name)
        srt_paths.append(mono_srt_path)
        primary_path = mono_srt_path

    # Legacy literal mode retained: writes ".literal.srt"
    if cfg.emit == "literal":
        literal_srt_path.write_text(to_srt(zh_literal), encoding="utf-8")
        logger.info("Generated -> %s", literal_srt_path.name)
        srt_paths.append(literal_srt_path)
        primary_path = literal_srt_path

    if cfg.emit in ("all", "bilingual", "bilingual-only"):
        bilingual_srt_path.write_text(to_srt_bilingual(transcript_src, zh_literal), encoding="utf-8")
        logger.info("Generated -> %s", bilingual_srt_path.name)
        srt_paths.append(bilingual_srt_path)
        primary_path = bilingual_srt_path

    if cfg.emit == "none":
        primary_path = literal_json_path

    # Safe fallback: choose something that would have been generated for the mode
    if primary_path is None:
        if cfg.emit in ("bilingual", "bilingual-only"):
            primary_path = bilingual_srt_path
        elif cfg.emit in ("literal",):
            primary_path = literal_srt_path
        elif cfg.emit in ("zh-only", "all"):
            primary_path = mono_srt_path
        else:
            primary_path = literal_json_path

    outputs["primary"] = primary_path

    artifacts["primary_path"] = str(primary_path)
    artifacts["srt_paths"] = [str(p) for p in srt_paths]

    meta.update(
        {
            "basename": basename,
            "video_stem": cfg.video_path.stem,
            "src_lang": src_lang,
            "target_lang": cfg.target_lang,
            "segmenter": cfg.segmenter,
            "translator": cfg.translator_name,
            "emit": cfg.emit,
            "asr_cache_hit": asr_cache_hit,
            "job_id": cfg.job_id,
        }
    )

    return PipelineResult(
        video_path=cfg.video_path,
        out_dir=cfg.out_dir,
        primary_path=primary_path,
        srt_paths=srt_paths,
        outputs=outputs,
        artifacts=artifacts,
        meta=meta,
    )
