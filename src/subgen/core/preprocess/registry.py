from __future__ import annotations

from typing import Any, Dict, Optional

from subgen.core.preprocess.base import PreprocessResult, PreprocessSpec, Preprocessor
from subgen.core.preprocess.demucs import DemucsPreprocessor
from subgen.core.preprocess.noop import NoopPreprocessor


_PREPROCESSORS: Dict[str, Preprocessor] = {
    "none": NoopPreprocessor(),
    "noop": NoopPreprocessor(),
    "demucs": DemucsPreprocessor(),
}


def build_preprocess_spec(
    *,
    preprocess: Optional[str],
    demucs_model: Optional[str] = None,
    device: Optional[str] = None,
    stems: Optional[str] = None,
    cache_dir: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> PreprocessSpec:
    """
    Backward-compatible spec builder.

    - preprocess: existing field in PipelineConfig / tool args (e.g. "demucs" or "none")
    - demucs_model: existing field in PipelineConfig / tool args (e.g. "htdemucs")
    """
    name = (preprocess or "none").strip().lower()

    if name in ("", "0", "false", "off", "no"):
        name = "none"

    if name not in _PREPROCESSORS:
        # Future-proof: unknown preprocess -> treat as none but keep note in params
        p = dict(params or {})
        p["unknown_preprocess"] = name
        return PreprocessSpec(name="none", params=p)

    if name == "demucs":
        return PreprocessSpec(
            name="demucs",
            model=(demucs_model or "htdemucs"),
            stems=(stems or "vocals"),
            device=(device or "cpu"),
            cache_dir=cache_dir,
            params=dict(params or {}),
        )

    return PreprocessSpec(name="none", params=dict(params or {}))


def get_preprocessor(name: str) -> Preprocessor:
    key = (name or "none").strip().lower()
    return _PREPROCESSORS.get(key) or _PREPROCESSORS["none"]


def run_preprocess(*, audio_in_path: str, out_dir: str, spec: PreprocessSpec) -> PreprocessResult:
    """
    Single entrypoint for pipeline to call.
    Stable for PR6: worker只要复用这个函数即可。
    """
    proc = get_preprocessor(spec.name)
    return proc.run(audio_in_path=audio_in_path, out_dir=out_dir, spec=spec)