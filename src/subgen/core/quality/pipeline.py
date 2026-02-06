from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from subgen.core.subtitle.models import SubtitleDoc

from .fixers import (
    FixAction,
    fix_overlaps,
    wrap_lines,
    fix_too_fast_cps,
    fix_too_short_duration,
)
from .report import QualityProfile, QualityReport, quality_check_doc


@dataclass
class FixBudget:
    """
    Fix pass budget for core-level deterministic fixing.

    max_passes: how many times apply fixes + recheck in this core pipeline.
    """
    max_passes: int = 3  # ✅ 建议从 2 提到 3：因为 CPS/Duration 可能互相影响，需要多 1 次收敛机会


@dataclass
class FixResult:
    ok: bool
    fixed_doc: SubtitleDoc
    before_report: QualityReport
    after_report: QualityReport
    actions: List[FixAction] = field(default_factory=list)
    passes: int = 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "passes": self.passes,
            "actions": [a.to_dict() for a in self.actions],
            "before": self.before_report.to_dict(),
            "after": self.after_report.to_dict(),
        }


def apply_fixes(doc: SubtitleDoc, profile: QualityProfile, budget: Optional[FixBudget] = None) -> FixResult:
    """
    Deterministic fix loop.

    Key ordering (important for convergence):
      1) overlaps (time monotonicity)
      2) too_fast_cps (may shift NEXT cue start -> can create TOO_SHORT_DURATION on next cue)
      3) too_short_duration (repair what CPS shifting might have broken)
      4) wrap_lines (pure text formatting)

    Repeat until:
      - report.ok() is True, OR
      - no actions in a pass, OR
      - reach budget.max_passes
    """
    if budget is None:
        budget = FixBudget()

    before = quality_check_doc(doc, profile)
    cur_doc = doc
    all_actions: List[FixAction] = []
    passes = 0

    cur_report = before
    while passes < budget.max_passes and not cur_report.ok():
        pass_actions: List[FixAction] = []

        # 1) overlap fix (low risk)
        cur_doc, a1 = fix_overlaps(cur_doc, max_overlap_ms=profile.max_overlap_ms)
        pass_actions.extend(a1)

        # 2) CPS fix (may shift NEXT start, thus can cause TOO_SHORT_DURATION downstream)
        cur_doc, a2 = fix_too_fast_cps(
            cur_doc,
            max_cps=profile.max_cps,
            max_overlap_ms=profile.max_overlap_ms,
        )
        pass_actions.extend(a2)

        # 3) duration fix AFTER cps fix (to repair any cues shortened by CPS shifts)
        cur_doc, a3 = fix_too_short_duration(
            cur_doc,
            min_dur_ms=profile.min_dur_ms,
            max_overlap_ms=profile.max_overlap_ms,
        )
        pass_actions.extend(a3)

        # 4) wrap lines (pure text)
        cur_doc, a4 = wrap_lines(
            cur_doc,
            max_line_len=profile.max_line_len,
            max_lines=profile.max_lines,
        )
        pass_actions.extend(a4)

        all_actions.extend(pass_actions)
        passes += 1

        cur_report = quality_check_doc(cur_doc, profile)

        # No progress => stop to avoid infinite loop
        if not pass_actions:
            break

    after = cur_report
    return FixResult(
        ok=after.ok(),
        fixed_doc=cur_doc,
        before_report=before,
        after_report=after,
        actions=all_actions,
        passes=passes,
    )
