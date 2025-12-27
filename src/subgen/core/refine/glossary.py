from pathlib import Path
from typing import Dict
from subgen.core_types import Transcript, Segment
from subgen.utils.io import read_json

def load_glossary(path: Path) -> Dict[str, str]:
    data = read_json(path)
    # expected: {"term": "指定译法", ...}
    return {str(k): str(v) for k, v in data.items()}

def apply_glossary(transcript: Transcript, glossary: Dict[str, str]) -> Transcript:
    if not glossary:
        return transcript

    new_segments = []
    for s in transcript.segments:
        text = s.text
        for src, tgt in glossary.items():
            text = text.replace(src, tgt)
        new_segments.append(Segment(
            start=s.start, end=s.end,
            text=text,
            confidence=s.confidence
        ))
    return Transcript(language=transcript.language, segments=new_segments)
