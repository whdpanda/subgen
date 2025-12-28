from __future__ import annotations

from pydantic import BaseModel
from typing import List, Optional


class Word(BaseModel):
    start: float
    end: float
    word: str


class Segment(BaseModel):
    start: float
    end: float
    text: str
    confidence: Optional[float] = None


class Transcript(BaseModel):
    language: str = "unknown"
    segments: List[Segment]
    # V1.2: 可选的 word-level 时间轴（用于 OpenAI 语义切分）
    words: Optional[List[Word]] = None
