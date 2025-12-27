from pathlib import Path
from subgen.core_types import Transcript
from subgen.core.align.base import AlignProvider

class NoopAlign(AlignProvider):
    def align(self, audio_path: Path, transcript: Transcript) -> Transcript:
        return transcript
