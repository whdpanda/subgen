from abc import ABC, abstractmethod
from pathlib import Path
from subgen.core_types import Transcript

class AlignProvider(ABC):
    @abstractmethod
    def align(self, audio_path: Path, transcript: Transcript) -> Transcript:
        ...
