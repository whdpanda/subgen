from abc import ABC, abstractmethod
from pathlib import Path
from subgen.core_types import Transcript

class ASRProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path, language: str = "auto") -> Transcript:
        ...
