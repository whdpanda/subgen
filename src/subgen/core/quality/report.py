from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from subgen.core.subtitle.srt_io import read_srt
from subgen.core.subtitle.models import SubtitleDoc
from .metrics import CueMetrics, PairMetrics, compute_cue_metrics, compute_pair_metrics
from .violations import Severity, Violation, ViolationType


@dataclass(frozen=True)
class QualityProfile:
    """
    Threshold configuration.

    Notes:
    - max_overlap_ms = 0 means no overlap allowed.
    - max_line_len/max_lines: rendering constraints.
    - cps uses "reading chars" (non-whitespace) per second.
    """
    name: str = "default"
    max_cps: float = 17.0
    max_line_len: int = 42
    max_lines: int = 2
    min_dur_ms: int = 700
    max_dur_ms: int = 7000
    max_overlap_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_cps": self.max_cps,
            "max_line_len": self.max_line_len,
            "max_lines": self.max_lines,
            "min_dur_ms": self.min_dur_ms,
            "max_dur_ms": self.max_dur_ms,
            "max_overlap_ms": self.max_overlap_ms,
        }


@dataclass
class QualityReport:
    version: str = "1.0"
    srt_path: str = ""
    profile: QualityProfile = field(default_factory=QualityProfile)
    total_cues: int = 0
    violations: List[Violation] = field(default_factory=list)
    cue_metrics: List[CueMetrics] = field(default_factory=list)
    pair_metrics: List[PairMetrics] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        major = 0
        for v in self.violations:
            by_type[v.type.value] = by_type.get(v.type.value, 0) + 1
            if v.severity == Severity.MAJOR:
                major += 1
        return {
            "total_cues": self.total_cues,
            "violation_count": len(self.violations),
            "major_count": major,
            "by_type": by_type,
        }

    def ok(self) -> bool:
        # For MVP: no major violations means pass.
        return all(v.severity != Severity.MAJOR for v in self.violations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "input": {
                "srt_path": self.srt_path,
                "profile": self.profile.name,
                "thresholds": self.profile.to_dict(),
            },
            "summary": self.summary(),
            "violations": [v.to_dict() for v in self.violations],
            "per_cue": [m.to_dict() for m in self.cue_metrics],
            "pairs": [m.to_dict() for m in self.pair_metrics],
        }


def _severity_for(vtype: ViolationType) -> Severity:
    # MVP: treat these as MAJOR; adjust later if you want minor/major gates.
    if vtype in {
        ViolationType.OVERLAP,
        ViolationType.TOO_FAST_CPS,
        ViolationType.BAD_TIMECODE_ORDER,
        ViolationType.EMPTY_TEXT,
    }:
        return Severity.MAJOR
    # Layout constraints can be MAJOR too; keep consistent for now.
    return Severity.MAJOR


def quality_check_doc(doc: SubtitleDoc, profile: QualityProfile) -> QualityReport:
    cue_m = compute_cue_metrics(doc)
    pair_m = compute_pair_metrics(doc)

    violations: List[Violation] = []

    # BAD_TIMECODE_ORDER (monotonic start times)
    prev_start = -1
    for i, cue in enumerate(doc.cues):
        if cue.start_ms < prev_start:
            violations.append(
                Violation(
                    type=ViolationType.BAD_TIMECODE_ORDER,
                    severity=_severity_for(ViolationType.BAD_TIMECODE_ORDER),
                    cue_index=i,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    message=f"Timecode order not monotonic at cue {i}",
                    data={"prev_start_ms": prev_start, "cur_start_ms": cue.start_ms},
                )
            )
        prev_start = cue.start_ms

    # Per-cue checks
    for m in cue_m:
        cue = doc.cues[m.cue_index]
        if cue.text.strip() == "":
            violations.append(
                Violation(
                    type=ViolationType.EMPTY_TEXT,
                    severity=_severity_for(ViolationType.EMPTY_TEXT),
                    cue_index=m.cue_index,
                    start_ms=m.start_ms,
                    end_ms=m.end_ms,
                    message="Empty subtitle text",
                    data={},
                )
            )

        if m.duration_ms < profile.min_dur_ms:
            violations.append(
                Violation(
                    type=ViolationType.TOO_SHORT_DURATION,
                    severity=_severity_for(ViolationType.TOO_SHORT_DURATION),
                    cue_index=m.cue_index,
                    start_ms=m.start_ms,
                    end_ms=m.end_ms,
                    message=f"Duration {m.duration_ms}ms < min {profile.min_dur_ms}ms",
                    data={"dur_ms": m.duration_ms, "min_dur_ms": profile.min_dur_ms},
                )
            )

        if m.duration_ms > profile.max_dur_ms:
            violations.append(
                Violation(
                    type=ViolationType.TOO_LONG_DURATION,
                    severity=_severity_for(ViolationType.TOO_LONG_DURATION),
                    cue_index=m.cue_index,
                    start_ms=m.start_ms,
                    end_ms=m.end_ms,
                    message=f"Duration {m.duration_ms}ms > max {profile.max_dur_ms}ms",
                    data={"dur_ms": m.duration_ms, "max_dur_ms": profile.max_dur_ms},
                )
            )

        if m.cps > profile.max_cps:
            violations.append(
                Violation(
                    type=ViolationType.TOO_FAST_CPS,
                    severity=_severity_for(ViolationType.TOO_FAST_CPS),
                    cue_index=m.cue_index,
                    start_ms=m.start_ms,
                    end_ms=m.end_ms,
                    message=f"CPS {m.cps:.2f} > max {profile.max_cps}",
                    data={"cps": m.cps, "max_cps": profile.max_cps, "char_count": m.char_count, "dur_ms": m.duration_ms},
                )
            )

        if m.max_line_len > profile.max_line_len:
            violations.append(
                Violation(
                    type=ViolationType.LINE_TOO_LONG,
                    severity=_severity_for(ViolationType.LINE_TOO_LONG),
                    cue_index=m.cue_index,
                    start_ms=m.start_ms,
                    end_ms=m.end_ms,
                    message=f"Line length {m.max_line_len} > max {profile.max_line_len}",
                    data={"max_line_len": m.max_line_len, "threshold": profile.max_line_len, "lines": cue.lines},
                )
            )

        if m.line_count > profile.max_lines:
            violations.append(
                Violation(
                    type=ViolationType.TOO_MANY_LINES,
                    severity=_severity_for(ViolationType.TOO_MANY_LINES),
                    cue_index=m.cue_index,
                    start_ms=m.start_ms,
                    end_ms=m.end_ms,
                    message=f"Line count {m.line_count} > max {profile.max_lines}",
                    data={"line_count": m.line_count, "threshold": profile.max_lines, "lines": cue.lines},
                )
            )

    # Pair checks (overlap)
    for pm in pair_m:
        if pm.overlap_ms > profile.max_overlap_ms:
            violations.append(
                Violation(
                    type=ViolationType.OVERLAP,
                    severity=_severity_for(ViolationType.OVERLAP),
                    cue_index=pm.cue_index,
                    start_ms=doc.cues[pm.cue_index].start_ms,
                    end_ms=doc.cues[pm.cue_index].end_ms,
                    message=f"Overlap {pm.overlap_ms}ms > max {profile.max_overlap_ms}ms (with previous cue)",
                    data={
                        "overlap_ms": pm.overlap_ms,
                        "max_overlap_ms": profile.max_overlap_ms,
                        "prev_end_ms": pm.prev_end_ms,
                        "cur_start_ms": pm.cur_start_ms,
                    },
                )
            )

    rep = QualityReport(
        srt_path=doc.source_path or "",
        profile=profile,
        total_cues=doc.total_cues,
        violations=violations,
        cue_metrics=cue_m,
        pair_metrics=pair_m,
    )
    return rep


def quality_check_srt(path: str, profile: QualityProfile) -> QualityReport:
    doc = read_srt(path)
    # Keep path for report.
    doc = SubtitleDoc(cues=doc.cues, source_path=str(path))
    return quality_check_doc(doc, profile)
