from abc import ABC, abstractmethod
from subgen.core_types import Transcript

class TranslatorProvider(ABC):
    @abstractmethod
    def translate(self, transcript: Transcript, target_lang: str = "zh") -> Transcript:
        ...
