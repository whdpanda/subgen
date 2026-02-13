from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from subgen.core.subtitle.models import SubtitleDoc


def _count_reading_chars(text: str) -> int:
    """
    Minimal CPS char counting:
    - counts non-whitespace chars
    - ignores newline and spaces/tabs
    """
    return sum(1 for ch in text if not ch.isspace())


@dataclass(frozen=True)
class CueMetrics:
    cue_index: int
    start_ms: int
    end_ms: int
    duration_ms: int
    char_count: int
    cps: float
    max_line_len: int
    line_count: int

    def to_dict(self) -> Dict:
        return {
            "cue_index": self.cue_index,
            "time": {"start_ms": self.start_ms, "end_ms": self.end_ms},
            "duration_ms": self.duration_ms,
            "char_count": self.char_count,
            "cps": self.cps,
            "max_line_len": self.max_line_len,
            "line_count": self.line_count,
        }


@dataclass(frozen=True)
class PairMetrics:
    """
    Metrics between consecutive cues i-1 and i.
    """
    cue_index: int  # current cue index i
    prev_end_ms: int
    cur_start_ms: int
    gap_ms: int
    overlap_ms: int

    def to_dict(self) -> Dict:
        return {
            "cue_index": self.cue_index,
            "prev_end_ms": self.prev_end_ms,
            "cur_start_ms": self.cur_start_ms,
            "gap_ms": self.gap_ms,
            "overlap_ms": self.overlap_ms,
        }


def compute_cue_metrics(doc: SubtitleDoc) -> List[CueMetrics]:
    out: List[CueMetrics] = []
    for i, cue in enumerate(doc.cues):
        duration_ms = cue.duration_ms
        duration_s = max(duration_ms / 1000.0, 1e-6)

        char_count = _count_reading_chars(cue.text)
        cps = char_count / duration_s

        lines = cue.lines
        max_line_len = max((len(ln) for ln in lines), default=0)
        line_count = len(lines) if lines else (0 if cue.text.strip() == "" else 1)

        out.append(
            CueMetrics(
                cue_index=i,
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                duration_ms=duration_ms,
                char_count=char_count,
                cps=cps,
                max_line_len=max_line_len,
                line_count=line_count,
            )
        )
    return out


def compute_pair_metrics(doc: SubtitleDoc) -> List[PairMetrics]:
    out: List[PairMetrics] = []
    for i in range(1, len(doc.cues)):
        prev = doc.cues[i - 1]
        cur = doc.cues[i]
        gap_ms = cur.start_ms - prev.end_ms
        overlap_ms = max(0, prev.end_ms - cur.start_ms)

        out.append(
            PairMetrics(
                cue_index=i,
                prev_end_ms=prev.end_ms,
                cur_start_ms=cur.start_ms,
                gap_ms=gap_ms,
                overlap_ms=overlap_ms,
            )
        )
    return out
