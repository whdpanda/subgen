from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class ViolationType(str, Enum):
    OVERLAP = "OVERLAP"
    TOO_FAST_CPS = "TOO_FAST_CPS"
    LINE_TOO_LONG = "LINE_TOO_LONG"
    TOO_MANY_LINES = "TOO_MANY_LINES"
    TOO_SHORT_DURATION = "TOO_SHORT_DURATION"
    TOO_LONG_DURATION = "TOO_LONG_DURATION"
    EMPTY_TEXT = "EMPTY_TEXT"
    BAD_TIMECODE_ORDER = "BAD_TIMECODE_ORDER"


class Severity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"


@dataclass(frozen=True)
class Violation:
    """
    A single quality violation.

    cue_index:
      - 0-based index into SubtitleDoc.cues
    span_ms:
      - [start_ms, end_ms] span for where issue happens (often cue time)
    data:
      - additional structured info (e.g. cps value, max line length, overlap ms)
    """
    type: ViolationType
    severity: Severity
    cue_index: int
    start_ms: int
    end_ms: int
    message: str
    data: Dict[str, Any]

    @property
    def span_ms(self) -> tuple[int, int]:
        return (self.start_ms, self.end_ms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "cue_index": self.cue_index,
            "time": {"start_ms": self.start_ms, "end_ms": self.end_ms},
            "message": self.message,
            "data": self.data,
        }
