# subgen/core/segment/coalesce.py
from __future__ import annotations

from typing import List, Optional, Any, cast

from subgen.core_types import Segment


def _text_len(text: str) -> int:
    return len((text or "").strip())


def _merge(a: Segment, b: Segment) -> Segment:
    """
    Merge b into a (in-place):
    - a.end becomes b.end
    - a.text becomes "a.text + b.text" with a single space if needed
    """
    a_text = (a.text or "").rstrip()
    b_text = (b.text or "").lstrip()
    if a_text and b_text:
        joined = (a_text + " " + b_text).strip()
    else:
        joined = (a_text + b_text).strip()
    a.end = b.end
    a.text = joined
    return a


def coalesce_segments(
    segments: List[Segment],
    *,
    min_dur: float = 2.5,
    min_chars: int = 8,
    target_dur: float = 7.0,
    hard_max: float = 15.0,
    # ---- Final-pass de-duplication (recommended) ----
    enable_dedupe: bool = True,
    dedupe_time_window_s: float = 30.0,
    dedupe_similarity: float = 0.90,
    dedupe_short_chars: int = 10,
    logger: Optional[Any] = None,
) -> List[Segment]:
    """
    Coalesce short / low-information segments to approach target_dur without exceeding hard_max,
    and (optionally) run a final-pass dedupe to remove near-duplicates caused by overlapping
    candidate windows (e.g., repair/relisten).

    Parameters
    ----------
    segments:
        Input segments (may contain duplicates or weak segments).
    min_dur:
        Segments shorter than this are considered weak.
    min_chars:
        Segments with fewer characters than this are considered weak.
    target_dur:
        Target duration for typical segments; used as a soft guideline in a second pass.
    hard_max:
        Never merge past this max duration.
    enable_dedupe:
        If True, run a final de-duplication pass after coalescing.
    dedupe_time_window_s / dedupe_similarity / dedupe_short_chars:
        Dedupe configuration knobs.
    logger:
        Optional logger to pass into dedupe (expected methods: info/debug/warning).
    """
    if not segments:
        return []

    # Sort by time for stable coalescing
    segs = sorted(segments, key=lambda s: (float(s.start), float(s.end)))

    # Pass 1: absorb weak segments into previous when possible
    out: List[Segment] = []
    for s in segs:
        txt = (s.text or "").strip()
        dur = float(s.end) - float(s.start)
        weak = (dur < float(min_dur)) or (_text_len(txt) < int(min_chars))

        if not out:
            if txt:
                out.append(s)
            continue

        prev = out[-1]
        prev_dur = float(prev.end) - float(prev.start)

        # If current is weak, try merging into previous
        if weak and (prev_dur + dur) <= float(hard_max):
            out[-1] = _merge(prev, s)
            continue

        # If previous is weak-ish, try merging current into it
        prev_weakish = (prev_dur < float(min_dur)) or (_text_len(prev.text) < int(min_chars))
        if prev_weakish and (prev_dur + dur) <= float(hard_max):
            out[-1] = _merge(prev, s)
            continue

        if txt:
            out.append(s)

    # Pass 2: make segments closer to target duration (soft), still respecting hard_max
    merged: List[Segment] = []
    i = 0
    while i < len(out):
        cur = out[i]
        cur_dur = float(cur.end) - float(cur.start)

        if i + 1 < len(out):
            nxt = out[i + 1]
            nxt_dur = float(nxt.end) - float(nxt.start)

            # If current is too short compared to target, and merging is safe, merge with next
            if cur_dur < float(target_dur) * 0.6 and (cur_dur + nxt_dur) <= float(hard_max):
                cur = _merge(cur, nxt)
                merged.append(cur)
                i += 2
                continue

        merged.append(cur)
        i += 1

    # Drop empty-text segments
    merged = [s for s in merged if (s.text or "").strip()]

    # Final-pass: de-duplication / conflict resolution (post-coalesce is the best place)
    if enable_dedupe and merged:
        # Local import to avoid circular dependency and keep coalesce focused.
        from subgen.core.postprocess.dedupe import DedupeConfig, dedupe_segments

        merged = cast(List[Segment], dedupe_segments(
            merged,
            cfg=DedupeConfig(
                time_window_s=float(dedupe_time_window_s),
                similarity_threshold=float(dedupe_similarity),
                short_text_chars=int(dedupe_short_chars),
            ),
            logger=logger,
        ))

    return merged
