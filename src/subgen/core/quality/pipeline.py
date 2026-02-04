from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from subgen.core.subtitle.models import SubtitleDoc
from .fixers import FixAction, fix_overlaps, wrap_lines
from .report import QualityProfile, QualityReport, quality_check_doc


@dataclass
class FixBudget:
    """
    Fix pass budget for core-level deterministic fixing.

    max_passes: how many times apply fixes + recheck in this core pipeline.
    """
    max_passes: int = 2


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
    Minimal deterministic fix pipeline:
    - check -> (fix overlaps, wrap lines) -> recheck (repeat up to budget.max_passes)
    """
    if budget is None:
        budget = FixBudget()

    before = quality_check_doc(doc, profile)
    cur_doc = doc
    all_actions: List[FixAction] = []
    passes = 0

    cur_report = before
    while passes < budget.max_passes and not cur_report.ok():
        # Order matters (low-risk first):
        # 1) overlaps (time)
        # 2) wrap (text)
        cur_doc, actions1 = fix_overlaps(cur_doc, max_overlap_ms=profile.max_overlap_ms)
        cur_doc, actions2 = wrap_lines(cur_doc, max_line_len=profile.max_line_len, max_lines=profile.max_lines)
        all_actions.extend(actions1)
        all_actions.extend(actions2)

        passes += 1
        cur_report = quality_check_doc(cur_doc, profile)

        # If no changes were applied in this pass, break to avoid infinite loop.
        if not actions1 and not actions2:
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
