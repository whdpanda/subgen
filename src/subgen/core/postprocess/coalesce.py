from __future__ import annotations

from typing import List

from subgen.core_types import Segment


def _text_len(text: str) -> int:
    return len((text or "").strip())


def _merge(a: Segment, b: Segment) -> Segment:
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
    hard_max: float = 20.0,
) -> List[Segment]:
    """
    合并过短/低信息段，尽量靠近 target_dur，同时不超过 hard_max。
    """
    if not segments:
        return []

    segs = sorted(segments, key=lambda s: (float(s.start), float(s.end)))
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

        if weak and (prev_dur + dur) <= float(hard_max):
            out[-1] = _merge(prev, s)
            continue

        prev_weakish = (prev_dur < float(min_dur)) or (_text_len(prev.text) < int(min_chars))
        if prev_weakish and (prev_dur + dur) <= float(hard_max):
            out[-1] = _merge(prev, s)
            continue

        if txt:
            out.append(s)

    merged: List[Segment] = []
    i = 0
    while i < len(out):
        cur = out[i]
        cur_dur = float(cur.end) - float(cur.start)

        if i + 1 < len(out):
            nxt = out[i + 1]
            nxt_dur = float(nxt.end) - float(nxt.start)
            if cur_dur < float(target_dur) * 0.6 and (cur_dur + nxt_dur) <= float(hard_max):
                cur = _merge(cur, nxt)
                merged.append(cur)
                i += 2
                continue

        merged.append(cur)
        i += 1

    return [s for s in merged if (s.text or "").strip()]
