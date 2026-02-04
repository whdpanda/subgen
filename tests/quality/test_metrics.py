from pathlib import Path

from subgen.core.subtitle.srt_io import read_srt
from subgen.core.quality.metrics import compute_cue_metrics, compute_pair_metrics
from subgen.core.quality.report import QualityProfile, quality_check_doc
from subgen.core.quality.violations import ViolationType


def test_overlap_detected() -> None:
    srt = Path("tests/fixtures/bad_overlap.srt")
    doc = read_srt(srt)

    profile = QualityProfile(max_overlap_ms=0)
    rep = quality_check_doc(doc, profile)

    types = [v.type for v in rep.violations]
    assert ViolationType.OVERLAP in types


def test_cps_metric_basic(tmp_path: Path) -> None:
    srt = tmp_path / "cps.srt"
    # 10 chars in 1 sec => cps ~ 10
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nabcdefghij\n\n", encoding="utf-8")
    doc = read_srt(srt)

    cm = compute_cue_metrics(doc)
    assert len(cm) == 1
    assert cm[0].char_count == 10
    assert 9.9 <= cm[0].cps <= 10.1

    pm = compute_pair_metrics(doc)
    assert pm == []
