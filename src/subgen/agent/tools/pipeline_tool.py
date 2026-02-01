from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from subgen.core.contracts import PipelineConfig
from subgen.core.pipeline import run_pipeline


class PipelineToolArgs(BaseModel):
    # I/O
    video_path: Path = Field(..., description="Input video file path")
    out_dir: Path = Field(Path("./out"), description="Output directory")

    # language
    language: str = Field("auto", description="Source language or 'auto'")
    target_lang: str = Field("zh", description="Target language")
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

    # Output
    emit: str = Field("bilingual-only", description="all/literal/bilingual-only/bilingual/none")

    # Cache & dumps
    use_cache: bool = Field(True, description="Enable ASR cache")
    dump_intermediates: bool = Field(True, description="Dump intermediates json")

    # Orchestration (future-facing)
    job_id: Optional[str] = Field(None, description="Optional job id")
    output_basename: Optional[str] = Field(None, description="Optional output basename")


def _parse_tool_args(
    args_or_kwargs: Union[PipelineToolArgs, dict[str, Any]]
) -> PipelineToolArgs:
    """
    Normalize tool inputs into PipelineToolArgs.

    LangChain StructuredTool will call tools with **kwargs (dict-like).
    Older direct calls may pass PipelineToolArgs instance.
    """
    if isinstance(args_or_kwargs, PipelineToolArgs):
        return args_or_kwargs

    # Pydantic v2: model_validate; v1: constructor
    if hasattr(PipelineToolArgs, "model_validate"):
        # type: ignore[attr-defined]
        return PipelineToolArgs.model_validate(args_or_kwargs)  # pyright: ignore
    return PipelineToolArgs(**args_or_kwargs)


def run_subgen_pipeline_tool(**kwargs: Any) -> dict[str, Any]:
    """
    LangChain tool entrypoint.

    IMPORTANT:
    - StructuredTool(args_schema=PipelineToolArgs) will pass inputs as **kwargs.
    - Therefore this function MUST accept **kwargs (not a single `args` parameter).

    Returns a dict (NOT json string). runtime will json.dumps once.
    """
    args = _parse_tool_args(kwargs)

    cfg = PipelineConfig(
        video_path=args.video_path,
        out_dir=args.out_dir,
        job_id=args.job_id,
        output_basename=args.output_basename,
        language=args.language,
        target_lang=args.target_lang,
        glossary_path=args.glossary_path,
        preprocess=args.preprocess,  # AudioPreprocess is Literal; str ok in runtime
        demucs_model=args.demucs_model,
        asr_model=args.asr_model,
        asr_device=args.asr_device,
        asr_compute_type=args.asr_compute_type,
        asr_beam_size=args.asr_beam_size,
        asr_best_of=args.asr_best_of,
        asr_vad_filter=args.asr_vad_filter,
        segmenter=args.segmenter,  # SegmenterName is Literal; str ok in runtime
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

    res = run_pipeline(cfg)

    # Compatibility: res.outputs may or may not exist depending on your contracts version
    outputs: dict[str, str] = {}
    if hasattr(res, "outputs") and getattr(res, "outputs") is not None:
        for k, v in res.outputs.items():  # type: ignore[attr-defined]
            outputs[k] = str(v)

    # Fallback: derive from artifacts if outputs missing
    artifacts = res.artifacts or {}
    if not outputs:
        for k in (
            "primary_path",
            "bilingual_srt_path",
            "literal_srt_path",
            "src_json_path",
            "literal_json_path",
            "asr_cache_path",
            "audio_path",
        ):
            if k in artifacts:
                outputs[k] = str(artifacts[k])

    return {
        "ok": True,
        "primary_path": str(res.primary_path),
        "srt_paths": [str(p) for p in res.srt_paths],
        "outputs": outputs,
        "artifacts": artifacts,
        "meta": res.meta or {},
    }
