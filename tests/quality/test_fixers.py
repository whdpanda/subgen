from pathlib import Path

from subgen.core.subtitle.srt_io import read_srt
from subgen.core.quality.fixers import fix_overlaps, wrap_lines
from subgen.core.quality.report import QualityProfile, quality_check_doc
from subgen.core.quality.violations import ViolationType


def test_fix_overlaps_resolves_overlap() -> None:
    doc = read_srt("tests/fixtures/bad_overlap.srt")
    profile = QualityProfile(max_overlap_ms=0)

    rep0 = quality_check_doc(doc, profile)
    assert any(v.type == ViolationType.OVERLAP for v in rep0.violations)

    fixed_doc, actions = fix_overlaps(doc, max_overlap_ms=0)
    assert actions, "Expected overlap fix actions"

    rep1 = quality_check_doc(fixed_doc, profile)
    assert not any(v.type == ViolationType.OVERLAP for v in rep1.violations)


def test_wrap_lines_reduces_long_lines() -> None:
    doc = read_srt("tests/fixtures/bad_wrap.srt")
    profile = QualityProfile(max_line_len=20, max_lines=2)

    rep0 = quality_check_doc(doc, profile)
    assert any(v.type in {ViolationType.LINE_TOO_LONG, ViolationType.TOO_MANY_LINES} for v in rep0.violations)

    fixed_doc, actions = wrap_lines(doc, max_line_len=20, max_lines=2)
    assert actions, "Expected wrap actions"

    rep1 = quality_check_doc(fixed_doc, profile)
    assert not any(v.type in {ViolationType.LINE_TOO_LONG, ViolationType.TOO_MANY_LINES} for v in rep1.violations)
