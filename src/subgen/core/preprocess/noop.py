from __future__ import annotations

from pathlib import Path

from subgen.core.preprocess.base import PreprocessResult, PreprocessSpec


class NoopPreprocessor:
    """
    Default preprocess: do nothing, return original audio.
    """

    name = "none"

    def run(self, *, audio_in_path: str, out_dir: str, spec: PreprocessSpec) -> PreprocessResult:
        p = Path(audio_in_path).expanduser()
        return PreprocessResult(
            ok=True,
            audio_path_for_asr=str(p),
            artifacts={},
            meta={
                "preprocess": "none",
                "note": "noop preprocessor",
            },
        )
