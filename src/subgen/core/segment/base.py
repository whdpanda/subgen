from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from subgen.core_types import Word, Segment


class SegmenterProvider(ABC):
    @abstractmethod
    def segment(self, words: List[Word]) -> List[Segment]:
        raise NotImplementedError
