# src/subgen/core/quality/quality_profiles.py
from __future__ import annotations

from typing import Any

from subgen.core.quality.report import QualityProfile


def _mk_profile_from_args(name: str, args: Any) -> QualityProfile:
    """
    Build QualityProfile from a tool-args object that has:
      max_cps, max_line_len, max_lines, min_dur_ms, max_dur_ms, max_overlap_ms
    """
    return QualityProfile(
        name=name,
        max_cps=float(args.max_cps),
        max_line_len=int(args.max_line_len),
        max_lines=int(args.max_lines),
        min_dur_ms=int(args.min_dur_ms),
        max_dur_ms=int(args.max_dur_ms),
        max_overlap_ms=int(args.max_overlap_ms),
    )
