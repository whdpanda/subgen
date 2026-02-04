from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class SubtitleCue:
    """
    A single subtitle cue.

    Notes:
    - start_ms/end_ms are absolute timeline times in milliseconds.
    - text preserves line breaks with '\n'.
    """
    start_ms: int
    end_ms: int
    text: str

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms < 0:
            raise ValueError("start_ms/end_ms must be non-negative")
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def lines(self) -> List[str]:
        # Keep empty lines out (SRT usually doesn't want them inside a cue).
        return [ln for ln in self.text.split("\n") if ln.strip() != ""]

    def with_text(self, new_text: str) -> "SubtitleCue":
        return SubtitleCue(start_ms=self.start_ms, end_ms=self.end_ms, text=new_text)

    def with_times(self, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> "SubtitleCue":
        return SubtitleCue(
            start_ms=self.start_ms if start_ms is None else start_ms,
            end_ms=self.end_ms if end_ms is None else end_ms,
            text=self.text,
        )


@dataclass(frozen=True)
class SubtitleDoc:
    cues: List[SubtitleCue] = field(default_factory=list)
    source_path: Optional[str] = None  # optional, for trace/debug

    def __iter__(self) -> Iterable[SubtitleCue]:
        return iter(self.cues)

    def sorted(self) -> "SubtitleDoc":
        return SubtitleDoc(cues=sorted(self.cues, key=lambda c: (c.start_ms, c.end_ms)), source_path=self.source_path)

    def validate_monotonic(self) -> None:
        """
        Validate that cues are in non-decreasing time order (start_ms monotonic),
        and each cue has end_ms >= start_ms (already checked).
        """
        prev_start = -1
        for i, cue in enumerate(self.cues):
            if cue.start_ms < prev_start:
                raise ValueError(f"Cue time order is not monotonic at index={i}")
            prev_start = cue.start_ms

    @property
    def total_cues(self) -> int:
        return len(self.cues)
