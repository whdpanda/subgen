from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

from subgen.core_types import Transcript

# Sentence-ish punctuation where we prefer to break.
_SENT_PUNCT = set("。！？!?；;…")
# Avoid starting a line with these punctuations.
_LEADING_PUNCT = set("，。、！？!?；;：:）)]】》」』”’…")
# Avoid ending a line with an opening bracket/quote-like.
_TRAILING_BAD = set("（([【《「『“‘")


def _normalize_zh_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # Remove spaces around common Chinese punctuations
    s = re.sub(r"\s*([，。！？；：、])\s*", r"\1", s)
    # Keep a single space around ASCII words if they exist
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    return s.strip()


def _split_by_punct_keep(p: str) -> List[str]:
    """
    Split by sentence-ish punctuation but keep punctuation attached to the left chunk.
    """
    if not p:
        return []
    out: List[str] = []
    buf: List[str] = []
    for ch in p:
        buf.append(ch)
        if ch in _SENT_PUNCT:
            out.append("".join(buf).strip())
            buf = []
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return [x for x in out if x]


def _display_len(s: str) -> int:
    """
    Rough display length:
    - Count every non-newline char as 1.
    (Good enough for CJK; quality tool caps line length anyway.)
    """
    return len(s.replace("\n", ""))


def _can_break_here(prev: str, nxt: str) -> bool:
    if not prev:
        return False
    if prev[-1] in _TRAILING_BAD:
        return False
    if nxt and nxt[0] in _LEADING_PUNCT:
        return False
    return True


def _wrap_greedy(text: str, max_len: int) -> List[str]:
    """
    Greedy wrap: prefer breaking at spaces; otherwise break anywhere safe.
    """
    text = text.strip()
    if not text:
        return []

    # If already contains newlines, respect them as hard breaks first.
    hard_parts = [p.strip() for p in text.split("\n") if p.strip()]
    parts: List[str] = []
    for hp in hard_parts:
        parts.extend(_split_by_punct_keep(hp))

    lines: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        i = 0
        while i < len(part):
            remaining = part[i:]
            if _display_len(remaining) <= max_len:
                lines.append(remaining.strip())
                break

            # candidate cut window
            cut = min(len(part), i + max_len)

            # prefer last space within window
            window = part[i:cut]
            last_space = window.rfind(" ")
            best_cut = None

            if last_space != -1 and last_space > 0:
                cand_left = part[i : i + last_space]
                cand_right = part[i + last_space + 1 :]
                if _can_break_here(cand_left, cand_right):
                    best_cut = i + last_space

            # else: try punctuation boundary inside window (closest to end)
            if best_cut is None:
                for j in range(cut - 1, i, -1):
                    if part[j] in _SENT_PUNCT:
                        cand_left = part[i : j + 1]
                        cand_right = part[j + 1 :]
                        if _can_break_here(cand_left, cand_right):
                            best_cut = j + 1
                            break

            # else: fallback cut but avoid leading punct on next line
            if best_cut is None:
                best_cut = cut
                # if next char would be leading punct, shift left by 1 if possible
                if best_cut < len(part) and part[best_cut] in _LEADING_PUNCT and best_cut > i + 1:
                    best_cut -= 1

            left = part[i:best_cut].strip()
            if left:
                lines.append(left)
            i = best_cut
            # skip spaces at the beginning of next
            while i < len(part) and part[i] == " ":
                i += 1

    # Fix: avoid 1-char last line (common ugly case)
    if len(lines) >= 2 and _display_len(lines[-1]) == 1:
        lines[-2] = (lines[-2] + lines[-1]).strip()
        lines.pop()

    return [ln.strip() for ln in lines if ln.strip()]


def wrap_zh(text: str, *, max_line_len: int = 18, max_lines: int = 2, line_len_cap: int = 42) -> str:
    """
    Return text with '\n' inserted for Chinese subtitle layout.

    Strategy:
    - Normalize whitespace/punct spacing
    - Prefer sentence punctuation breaks
    - Greedy wrap
    - If exceeds max_lines, increase max_len (up to cap) and rewrap
    """
    text = _normalize_zh_text(text)
    if not text:
        return text

    # Base wrap
    lines = _wrap_greedy(text, max_line_len)

    if max_lines <= 0:
        return "\n".join(lines)

    if len(lines) <= max_lines:
        return "\n".join(lines)

    # Too many lines -> increase max_len and rewrap (bounded)
    # Estimate needed length to fit in max_lines
    needed = int(math.ceil(_display_len(text) / max_lines))
    new_len = min(line_len_cap, max(max_line_len, needed))

    lines2 = _wrap_greedy(text, new_len)
    if len(lines2) <= max_lines:
        return "\n".join(lines2)

    # Final fallback: force exactly max_lines by merging from the end
    lines3 = lines2[:]
    while len(lines3) > max_lines:
        # merge the last two
        b = lines3.pop()
        a = lines3.pop()
        lines3.append((a + b).strip())
    return "\n".join(lines3)


def _get_text_field(seg_dict: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return (field_name, text_value) from a segment dict.
    Supports common fields: text / t
    """
    if "text" in seg_dict and isinstance(seg_dict["text"], str):
        return "text", seg_dict["text"]
    if "t" in seg_dict and isinstance(seg_dict["t"], str):
        return "t", seg_dict["t"]
    # If unknown, don't change.
    return "", ""


def _count_hanzi(text: str) -> int:
    """
    Count Chinese Hanzi characters for time allocation.
    If none found (e.g., mixed content), fall back to non-whitespace length.
    """
    if not text:
        return 0
    han = re.findall(r"[\u4e00-\u9fff]", text)
    if han:
        return len(han)
    # fallback: count non-whitespace chars
    return sum(1 for ch in text if not ch.isspace())


def _alloc_times_by_hanzi(start: float, end: float, parts: List[str]) -> List[Tuple[float, float]]:
    """
    Allocate [start,end] into len(parts) consecutive ranges.
    Weight by Hanzi counts (or non-whitespace if no Hanzi), proportional allocation.
    """
    n = len(parts)
    if n <= 0:
        return []

    dur = float(end) - float(start)
    if dur <= 0:
        # degenerate; still return monotonic tiny spans
        eps = 0.001
        out = []
        cur = float(start)
        for _ in range(n):
            out.append((cur, cur + eps))
            cur += eps
        return out

    weights = [_count_hanzi(p) for p in parts]
    total = sum(weights)

    if total <= 0:
        # equal split
        step = dur / n
        out = []
        for i in range(n):
            s = float(start) + step * i
            e = float(start) + step * (i + 1)
            out.append((s, e))
        # force last end exact
        out[-1] = (out[-1][0], float(end))
        return out

    # proportional split
    out: List[Tuple[float, float]] = []
    acc = 0.0
    for i, w in enumerate(weights):
        s = float(start) + dur * (acc / total)
        acc += float(w)
        e = float(start) + dur * (acc / total)
        out.append((s, e))

    # numeric safety: enforce monotonic & force last end exact
    fixed: List[Tuple[float, float]] = []
    last_s = float(start)
    for i, (s, e) in enumerate(out):
        s = max(s, last_s)
        if e <= s:
            e = s + 0.001
        if i == len(out) - 1:
            e = float(end)
            if e <= s:
                e = s + 0.001
        fixed.append((s, e))
        last_s = e
    return fixed


def apply_zh_layout(
    transcript: Transcript,
    *,
    max_line_len: int = 18,
    max_lines: int = 2,
    line_len_cap: int = 42,
) -> Transcript:
    """
    Apply Chinese layout to each segment's text inside a Transcript.
    Does NOT modify timestamps.
    (Keeps '\n' inside each segment text.)
    """
    d = transcript.model_dump()
    segs = d.get("segments")
    if not isinstance(segs, list):
        return transcript

    out_segs: List[Dict[str, Any]] = []
    for seg in segs:
        if not isinstance(seg, dict):
            out_segs.append(seg)  # type: ignore[list-item]
            continue

        field, txt = _get_text_field(seg)
        if not field or not txt:
            out_segs.append(seg)
            continue

        seg2 = dict(seg)
        seg2[field] = wrap_zh(
            txt,
            max_line_len=max_line_len,
            max_lines=max_lines,
            line_len_cap=line_len_cap,
        )
        out_segs.append(seg2)

    d["segments"] = out_segs
    return Transcript.model_validate(d)


def apply_zh_layout_split_to_cues(
    transcript: Transcript,
    *,
    max_line_len: int = 18,
    max_lines: int = 2,
    line_len_cap: int = 42,
) -> Transcript:
    """
    NEW behavior requested:

    - First do Chinese layout wrapping (wrap_zh).
    - Then split the wrapped parts (by '\n') into multiple NEW segments (cues).
    - Re-allocate start/end times within the original segment by Hanzi-average (weighted by Hanzi count).

    Result:
      Each split piece becomes a new cue with its own start/end.
      Text in each new cue is single-line (no '\n').
    """
    d = transcript.model_dump()
    segs = d.get("segments")
    if not isinstance(segs, list):
        return transcript

    out_segs: List[Dict[str, Any]] = []

    for seg in segs:
        if not isinstance(seg, dict):
            out_segs.append(seg)  # type: ignore[list-item]
            continue

        # require timestamps
        if "start" not in seg or "end" not in seg:
            out_segs.append(seg)
            continue

        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except Exception:
            out_segs.append(seg)
            continue

        field, txt = _get_text_field(seg)
        if not field or not txt:
            out_segs.append(seg)
            continue

        wrapped = wrap_zh(
            txt,
            max_line_len=max_line_len,
            max_lines=max_lines,
            line_len_cap=line_len_cap,
        )

        parts = [p.strip() for p in wrapped.split("\n") if p.strip()]
        if len(parts) <= 1:
            # keep as-is, but ensure normalized single-line text
            seg2 = dict(seg)
            seg2[field] = parts[0] if parts else _normalize_zh_text(txt)
            out_segs.append(seg2)
            continue

        # allocate times across parts
        spans = _alloc_times_by_hanzi(start, end, parts)

        for (s, e), part in zip(spans, parts):
            seg2 = dict(seg)
            seg2["start"] = float(s)
            seg2["end"] = float(e)
            seg2[field] = part  # single line per cue
            out_segs.append(seg2)

    d["segments"] = out_segs
    return Transcript.model_validate(d)
