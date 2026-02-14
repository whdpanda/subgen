# src/subgen/agent/tools/run_subgen_pipeline_tool.py
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import ValidationError

from subgen.agent.tools.schemas import PipelineToolArgs, parse_tool_args
from subgen.agent.tools.tool_utils import path_values_to_str
from subgen.core.contracts import PipelineConfig
from subgen.core.pipeline import run_pipeline


def _ok_flat(
    *,
    primary_path: Optional[Union[str, Path]],
    srt_paths: list[Union[str, Path]],
    outputs: dict[str, Any],
    artifacts: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": True,
        "primary_path": str(primary_path) if primary_path is not None else None,
        "srt_paths": [str(p) for p in (srt_paths or [])],
        "outputs": path_values_to_str(outputs or {}),
        "artifacts": path_values_to_str(artifacts or {}),
        "meta": path_values_to_str(meta or {}),
    }


def _fail_flat(
    *,
    err_type: str,
    message: str,
    details: Optional[dict[str, Any]] = None,
    # keep schema stable
    primary_path: Optional[Union[str, Path]] = None,
    srt_paths: Optional[list[Union[str, Path]]] = None,
    outputs: Optional[dict[str, Any]] = None,
    artifacts: Optional[dict[str, Any]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    m = dict(meta or {})
    m["error"] = {
        "type": err_type,
        "message": message,
        "details": path_values_to_str(details or {}),
    }
    return {
        "ok": False,
        "primary_path": str(primary_path) if primary_path is not None else None,
        "srt_paths": [str(p) for p in (srt_paths or [])],
        "outputs": path_values_to_str(outputs or {}),
        "artifacts": path_values_to_str(artifacts or {}),
        "meta": path_values_to_str(m),
    }


def _pipelineconfig_supports(field_name: str) -> bool:
    """
    Best-effort: only pass new args when PipelineConfig defines them.
    Works for dataclass and pydantic models.
    """
    fields = getattr(PipelineConfig, "__dataclass_fields__", None)
    if isinstance(fields, dict) and field_name in fields:
        return True

    model_fields = getattr(PipelineConfig, "model_fields", None)
    if isinstance(model_fields, dict) and field_name in model_fields:
        return True

    __fields__ = getattr(PipelineConfig, "__fields__", None)
    if isinstance(__fields__, dict) and field_name in __fields__:
        return True

    return False


def run_subgen_pipeline_tool(**kwargs: Any) -> dict[str, Any]:
    """
    Tool entrypoint for LangChain StructuredTool.

    Returns stable flat schema (NO envelope):
      - success/failure keys are always:
        {"ok","primary_path","srt_paths","outputs","artifacts","meta"}
      - failure details are stored in meta["error"]
    """
    # 1) Parse / validate args (never raise)
    try:
        args = parse_tool_args(PipelineToolArgs, kwargs)
    except ValidationError as e:
        return _fail_flat(
            err_type="pipeline.validation_error",
            message="invalid tool arguments",
            details={"errors": e.errors(), "input": path_values_to_str(kwargs)},
        )
    except Exception as e:
        return _fail_flat(
            err_type="pipeline.args_parse_error",
            message=str(e) or e.__class__.__name__,
            details={
                "exception_class": e.__class__.__name__,
                "traceback": traceback.format_exc(),
                "input": path_values_to_str(kwargs),
            },
        )

    # 2) Build config (never raise)
    try:
        cfg_kwargs: dict[str, Any] = dict(
            video_path=args.video_path,
            out_dir=args.out_dir,
            job_id=args.job_id,
            output_basename=args.output_basename,
            language=args.language,
            target_lang=args.target_lang,
            glossary_path=args.glossary_path,
            preprocess=args.preprocess,
            demucs_model=args.demucs_model,
            asr_model=args.asr_model,
            asr_device=args.asr_device,
            asr_compute_type=args.asr_compute_type,
            asr_beam_size=args.asr_beam_size,
            asr_best_of=args.asr_best_of,
            asr_vad_filter=args.asr_vad_filter,
            segmenter=args.segmenter,
            openai_segment_model=args.openai_segment_model,
            soft_max=args.soft_max,
            hard_max=args.hard_max,
            suspect_dur=args.suspect_dur,
            suspect_cps=args.suspect_cps,
            translator_name=args.translator_name,
            translator_model=args.translator_model,
            translator_device=args.translator_device,
            openai_translate_model=args.openai_translate_model,
            emit=args.emit,
            use_cache=args.use_cache,
            dump_intermediates=args.dump_intermediates,
        )

        # PR#4c: only pass zh layout knobs if PipelineConfig supports them
        if _pipelineconfig_supports("zh_layout"):
            cfg_kwargs["zh_layout"] = args.zh_layout
        if _pipelineconfig_supports("zh_max_line_len"):
            cfg_kwargs["zh_max_line_len"] = args.zh_max_line_len
        if _pipelineconfig_supports("zh_max_lines"):
            cfg_kwargs["zh_max_lines"] = args.zh_max_lines
        if _pipelineconfig_supports("zh_line_len_cap"):
            cfg_kwargs["zh_line_len_cap"] = args.zh_line_len_cap

        # PR6-ready: demucs structured knobs (only if PipelineConfig supports them)
        if _pipelineconfig_supports("demucs_device"):
            cfg_kwargs["demucs_device"] = args.demucs_device
        if _pipelineconfig_supports("demucs_stems"):
            cfg_kwargs["demucs_stems"] = args.demucs_stems
        if _pipelineconfig_supports("preprocess_cache_dir"):
            cfg_kwargs["preprocess_cache_dir"] = args.preprocess_cache_dir
        if _pipelineconfig_supports("demucs_params"):
            cfg_kwargs["demucs_params"] = args.demucs_params or {}

        cfg = PipelineConfig(**cfg_kwargs)
    except Exception as e:
        return _fail_flat(
            err_type="pipeline.config_build_error",
            message=str(e) or e.__class__.__name__,
            details={
                "exception_class": e.__class__.__name__,
                "traceback": traceback.format_exc(),
                "video_path": str(args.video_path),
                "out_dir": str(args.out_dir),
            },
            meta={"video_path": str(args.video_path), "out_dir": str(args.out_dir)},
        )

    # 3) Run pipeline (never raise)
    try:
        res = run_pipeline(cfg)
    except Exception as e:
        return _fail_flat(
            err_type="pipeline.runtime_error",
            message=str(e) or e.__class__.__name__,
            details={
                "exception_class": e.__class__.__name__,
                "traceback": traceback.format_exc(),
                "video_path": str(args.video_path),
                "out_dir": str(args.out_dir),
                "language": args.language,
                "target_lang": args.target_lang,
                "segmenter": args.segmenter,
                "translator_name": args.translator_name,
                "emit": args.emit,
            },
            meta={
                "video_path": str(args.video_path),
                "out_dir": str(args.out_dir),
                "language": args.language,
                "target_lang": args.target_lang,
                "segmenter": args.segmenter,
                "translator_name": args.translator_name,
                "emit": args.emit,
            },
        )

    # 4) Build outputs payload (prefer res.outputs; fallback to artifacts)
    outputs: dict[str, str] = {}
    if hasattr(res, "outputs") and getattr(res, "outputs") is not None:
        for k, v in res.outputs.items():
            outputs[k] = str(v)

    artifacts = res.artifacts or {}

    if not outputs:
        for k in (
            "primary_path",
            "mono_srt_path",
            "bilingual_srt_path",
            "literal_srt_path",
            "src_json_path",
            "literal_json_path",
            "asr_cache_path",
            "audio_path",
        ):
            if k in artifacts:
                outputs[k] = str(artifacts[k])

    return _ok_flat(
        primary_path=getattr(res, "primary_path", None),
        srt_paths=getattr(res, "srt_paths", []) or [],
        outputs=outputs,
        artifacts=artifacts,
        meta=getattr(res, "meta", {}) or {},
    )