from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from subgen.core.contracts import PipelineConfig
from subgen.core.pipeline import run_pipeline


class PipelineToolArgs(BaseModel):
    # I/O
    video_path: Path = Field(..., description="Input video file path")
    out_dir: Path = Field(Path("./out"), description="Output directory")

    # language
    language: str = Field("auto", description="Source language or 'auto'")
    target_lang: str = Field("zh", description="Target language (default: zh)")
    glossary_path: Optional[Path] = Field(None, description="Optional glossary json path")

    # Audio preprocess
    preprocess: str = Field("none", description="none/speech_filter/demucs")
    demucs_model: str = Field("htdemucs", description="Demucs model name when preprocess=demucs")

    # ASR
    asr_model: str = Field("large-v3", description="ASR model name")
    asr_device: str = Field("cuda", description="cuda/cpu/auto")
    asr_compute_type: Optional[str] = Field("float16", description="float16/int8/...")
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


def _safe_str_value(v: Any) -> Any:
    return str(v) if isinstance(v, Path) else v


def _safe_str_dict(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        out[k] = _safe_str_value(v)
    return out


def _parse_tool_args(args_or_kwargs: Union[PipelineToolArgs, dict[str, Any]]) -> PipelineToolArgs:
    if isinstance(args_or_kwargs, PipelineToolArgs):
        return args_or_kwargs

    # Pydantic v2: model_validate; v1: constructor
    if hasattr(PipelineToolArgs, "model_validate"):
        # type: ignore[attr-defined]
        return PipelineToolArgs.model_validate(args_or_kwargs)  # pyright: ignore
    return PipelineToolArgs(**args_or_kwargs)


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
        "outputs": _safe_str_dict(outputs or {}),
        "artifacts": _safe_str_dict(artifacts or {}),
        "meta": _safe_str_dict(meta or {}),
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
        "details": _safe_str_dict(details or {}),
    }
    return {
        "ok": False,
        "primary_path": str(primary_path) if primary_path is not None else None,
        "srt_paths": [str(p) for p in (srt_paths or [])],
        "outputs": _safe_str_dict(outputs or {}),
        "artifacts": _safe_str_dict(artifacts or {}),
        "meta": _safe_str_dict(m),
    }


def _pipelineconfig_supports(field_name: str) -> bool:
    """
    Best-effort: only pass new args (e.g., zh_layout) when PipelineConfig defines them.
    Works for dataclass and pydantic models.
    """
    # dataclass
    fields = getattr(PipelineConfig, "__dataclass_fields__", None)
    if isinstance(fields, dict) and field_name in fields:
        return True

    # pydantic v2
    model_fields = getattr(PipelineConfig, "model_fields", None)
    if isinstance(model_fields, dict) and field_name in model_fields:
        return True

    # pydantic v1
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
        args = _parse_tool_args(kwargs)
    except ValidationError as e:
        return _fail_flat(
            err_type="pipeline.validation_error",
            message="invalid tool arguments",
            details={"errors": e.errors(), "input": _safe_str_dict(kwargs)},
        )
    except Exception as e:
        return _fail_flat(
            err_type="pipeline.args_parse_error",
            message=str(e) or e.__class__.__name__,
            details={
                "exception_class": e.__class__.__name__,
                "traceback": traceback.format_exc(),
                "input": _safe_str_dict(kwargs),
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

        cfg = PipelineConfig(**cfg_kwargs)  # type: ignore[arg-type]
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
        for k, v in res.outputs.items():  # type: ignore[attr-defined]
            outputs[k] = str(v)

    artifacts = res.artifacts or {}

    # Fallback keys (only if tool really returned them in artifacts)
    # NOTE: include PR#4c mono keys as well.
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
