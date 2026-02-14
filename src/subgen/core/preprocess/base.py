from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True)
class PreprocessSpec:
    """
    Stable, serializable preprocess spec. PR6 job化时可以直接dump进job.json。
    设计目标：兼容当前PR5的 preprocess/demucs_model 字段，同时为未来扩展预留空间。
    """

    name: str = "none"  # "none" | "demucs" | future
    model: str = "htdemucs"  # demucs model name
    stems: str = "vocals"  # demucs stems: typically "vocals"
    device: str = "cpu"  # "cpu" | "cuda" (future)
    cache_dir: Optional[str] = None  # e.g. "/cache"
    params: Dict[str, Any] = field(default_factory=dict)  # extra knobs


@dataclass
class PreprocessResult:
    """
    Output contract of preprocess stage.
    audio_path_for_asr: the audio file pipeline should feed into ASR.
    artifacts: any extra files/dirs created (for debugging/inspection).
    meta: timing, cache hit, commandline, etc.
    """

    ok: bool
    audio_path_for_asr: str
    artifacts: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


class Preprocessor(Protocol):
    """
    Interface for preprocess implementations.
    Keep signature stable so PR6 job runner can reuse it unchanged.
    """

    def run(self, *, audio_in_path: str, out_dir: str, spec: PreprocessSpec) -> PreprocessResult: ...