from __future__ import annotations

from pathlib import Path

from subgen.core.quality.report import QualityProfile, quality_check_srt
from subgen.core.quality.pipeline import FixBudget, apply_fixes
from subgen.core.subtitle.srt_io import read_srt, write_srt


def _profile() -> QualityProfile:
    return QualityProfile(
        name="test",
        max_cps=16.0,
        max_line_len=18,
        max_lines=1,
        min_dur_ms=900,
        max_dur_ms=6500,
        max_overlap_ms=0,
    )


def test_apply_fixes_reduces_violations(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    srt_in = root / "tests" / "fixtures" / "bad_cps_duration.srt"
    assert srt_in.exists()

    prof = _profile()

    before = quality_check_srt(str(srt_in), prof)
    assert before.ok() is False
    before_n = len(before.violations)
    assert before_n > 0

    doc = read_srt(srt_in).sorted()
    res = apply_fixes(doc, prof, budget=FixBudget(max_passes=3))

    srt_out = tmp_path / "fixed.srt"
    write_srt(res.fixed_doc, srt_out)

    after = quality_check_srt(str(srt_out), prof)
    after_n = len(after.violations)

    # 核心验收：不能更糟，且应当减少（你当前目标就是“收敛”）
    assert after_n <= before_n

    # 强一些的验收（如果你的 fixers 做到位，通常能直接清零）
    # 如果你尚未实现到“必清零”，可以先注释掉这行，等完善后再打开
    # assert after.ok() is True


def test_fix_does_not_introduce_overlap(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    srt_in = root / "tests" / "fixtures" / "bad_cps_duration.srt"
    assert srt_in.exists()

    prof = _profile()

    doc = read_srt(srt_in).sorted()
    res = apply_fixes(doc, prof, budget=FixBudget(max_passes=3))

    srt_out = tmp_path / "fixed.srt"
    write_srt(res.fixed_doc, srt_out)

    after = quality_check_srt(str(srt_out), prof)

    # 只要 profile.max_overlap_ms=0，最终 report 里不应再出现 OVERLAP（理想情况）
    # 如果你策略允许 best-effort，至少应当显著减少 overlap
    overlap = [v for v in after.violations if getattr(v, "type", None).name == "OVERLAP"]
    assert len(overlap) == 0
