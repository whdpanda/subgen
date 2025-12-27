from pydantic import BaseModel
from typing import List, Optional

class Segment(BaseModel):
    start: float
    end: float
    text: str
    confidence: Optional[float] = None

class Transcript(BaseModel):
    language: str = "unknown"
    segments: List[Segment]
