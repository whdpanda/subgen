from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import List, Optional, Tuple

from subgen.core.subtitle.models import SubtitleCue, SubtitleDoc


@dataclass(frozen=True)
class FixAction:
    name: str
    cue_index: int
    before: dict
    after: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "cue_index": self.cue_index, "before": self.before, "after": self.after}


# ============================================================
# 1) Overlap fixing
# ============================================================
def fix_overlaps(doc: SubtitleDoc, max_overlap_ms: int = 0) -> tuple[SubtitleDoc, List[FixAction]]:
    """
    Deterministic overlap fix:
    - If cue[i].start < cue[i-1].end - max_overlap_ms, push cue[i].start to cue[i-1].end - max_overlap_ms.
    - Ensure start < end; if not possible, clamp start to end-1.
    """
    cues = list(doc.cues)
    actions: List[FixAction] = []

    for i in range(1, len(cues)):
        prev = cues[i - 1]
        cur = cues[i]

        allowed_start = prev.end_ms - max_overlap_ms
        if cur.start_ms < allowed_start:
            new_start = allowed_start
            # keep at least 1ms duration if needed
            if new_start >= cur.end_ms:
                new_start = max(cur.end_ms - 1, 0)

            actions.append(
                FixAction(
                    name="fix_overlaps",
                    cue_index=i,
                    before={"start_ms": cur.start_ms, "end_ms": cur.end_ms},
                    after={"start_ms": new_start, "end_ms": cur.end_ms},
                )
            )
            cues[i] = cur.with_times(start_ms=new_start)

    return SubtitleDoc(cues=cues, source_path=doc.source_path), actions


# ============================================================
# 2) Duration/CPS fixing helpers
# ============================================================
def _prev_end_bound(cues: List[SubtitleCue], i: int, max_overlap_ms: int) -> int:
    """
    Earliest allowed start to avoid overlapping prev cue beyond max_overlap_ms.
    With max_overlap_ms=0 => start >= prev.end
    """
    if i <= 0:
        return 0
    return cues[i - 1].end_ms - max_overlap_ms


def _next_start_bound(cues: List[SubtitleCue], i: int, max_overlap_ms: int) -> Optional[int]:
    """
    Latest allowed end to avoid overlapping next cue beyond max_overlap_ms.
    With max_overlap_ms=0 => end <= next.start
    """
    if i >= len(cues) - 1:
        return None
    return cues[i + 1].start_ms + max_overlap_ms


def _ensure_positive_span(start_ms: int, end_ms: int) -> Tuple[int, int]:
    if start_ms < 0:
        start_ms = 0
    if end_ms <= start_ms:
        end_ms = start_ms + 1
    return start_ms, end_ms


def _text_char_count_for_cps(text: str) -> int:
    """
    CPS char count heuristic:
    - count characters excluding newlines
    - keep spaces (reading cost in latin languages)
    """
    return len(text.replace("\n", ""))


def fix_too_short_duration(
    doc: SubtitleDoc,
    min_dur_ms: int,
    max_overlap_ms: int = 0,
    *,
    shift_next_max_ms: int = 1000,
    min_next_keep_ms: int = 1,
) -> tuple[SubtitleDoc, List[FixAction]]:
    """
    Deterministic duration fix (min duration):
    - If duration < min_dur_ms, try to extend end_ms.
    - If end would collide with next cue, try to shift start earlier (without colliding prev).
    - If still short and the remaining gap is small, shift NEXT cue.start later by delta_ms (<= shift_next_max_ms),
      then extend current end to new_next_start - max_overlap_ms.
    - Best-effort: clamp within available window; never create invalid timecode.
    """
    cues = list(doc.cues)
    actions: List[FixAction] = []

    if min_dur_ms <= 0:
        return SubtitleDoc(cues=cues, source_path=doc.source_path), actions

    for i, cur in enumerate(cues):
        dur = cur.end_ms - cur.start_ms
        if dur >= min_dur_ms:
            continue

        before = {"start_ms": cur.start_ms, "end_ms": cur.end_ms, "duration_ms": dur}

        prev_bound = _prev_end_bound(cues, i, max_overlap_ms)
        next_bound = _next_start_bound(cues, i, max_overlap_ms)

        new_start = cur.start_ms
        new_end = cur.end_ms
        note = "extended_end"

        # 1) extend end up to bound (end <= next.start + max_overlap_ms)
        need_end = new_start + min_dur_ms
        candidate_end = need_end
        if next_bound is not None:
            candidate_end = min(candidate_end, next_bound)
        new_end = max(new_end, candidate_end)
        new_start, new_end = _ensure_positive_span(new_start, new_end)

        # 2) if still short, shift start earlier (down to prev bound)
        if (new_end - new_start) < min_dur_ms:
            note = "shifted_start_earlier"
            want_start = new_end - min_dur_ms
            new_start = max(want_start, prev_bound, 0)
            new_start, new_end = _ensure_positive_span(new_start, new_end)

        # 3) if STILL short and we have a next cue, try to shift next.start later by small delta
        cur_dur = new_end - new_start
        if cur_dur < min_dur_ms and i < len(cues) - 1:
            delta = min_dur_ms - cur_dur  # ms we still need
            if delta > 0 and delta <= shift_next_max_ms:
                nxt = cues[i + 1]
                proposed_next_start = nxt.start_ms + delta
                max_next_start = nxt.end_ms - min_next_keep_ms

                if proposed_next_start <= max_next_start:
                    # shift next.start
                    actions.append(
                        FixAction(
                            name="shift_next_start_for_min_duration",
                            cue_index=i + 1,
                            before={"start_ms": nxt.start_ms, "end_ms": nxt.end_ms},
                            after={"start_ms": proposed_next_start, "end_ms": nxt.end_ms, "delta_ms": delta},
                        )
                    )
                    cues[i + 1] = nxt.with_times(start_ms=proposed_next_start)

                    # now extend current end deterministically WITHOUT overlap
                    cap_end = proposed_next_start - max_overlap_ms
                    new_end = min(new_start + min_dur_ms, cap_end)
                    new_start, new_end = _ensure_positive_span(new_start, new_end)
                    note = "shifted_next_start_then_extended_end"

        # 4) final best-effort note
        if (new_end - new_start) < min_dur_ms:
            note = "best_effort_insufficient_room"

        if new_start != cur.start_ms or new_end != cur.end_ms:
            actions.append(
                FixAction(
                    name="fix_too_short_duration",
                    cue_index=i,
                    before=before,
                    after={
                        "start_ms": new_start,
                        "end_ms": new_end,
                        "duration_ms": new_end - new_start,
                        "min_dur_ms": min_dur_ms,
                        "note": note,
                    },
                )
            )
            cues[i] = cur.with_times(start_ms=new_start, end_ms=new_end)

    return SubtitleDoc(cues=cues, source_path=doc.source_path), actions

def fix_too_fast_cps(
    doc: SubtitleDoc,
    max_cps: float,
    max_overlap_ms: int = 0,
    *,
    shift_next_max_ms: int = 1000,
    min_next_keep_ms: int = 1,
) -> tuple[SubtitleDoc, List[FixAction]]:
    """
    Deterministic CPS fix (max reading speed), with a small extra capability:

    Base strategy:
      1) extend end_ms up to next boundary
      2) if capped, shift start earlier down to prev boundary
      3) if still capped AND the remaining gap to pass is small,
         shift NEXT cue.start later by delta_ms (<= shift_next_max_ms),
         then extend current end to the new next.start - max_overlap_ms.

    Notes:
      - This does NOT split/merge cues (keeps deterministic and minimal).
      - The "shift next" path is guarded (delta small) to avoid large rhythm changes.
    """
    cues = list(doc.cues)
    actions: List[FixAction] = []

    if max_cps <= 0:
        return SubtitleDoc(cues=cues, source_path=doc.source_path), actions

    for i, cur in enumerate(cues):
        text = cur.text.strip()
        chars = _text_char_count_for_cps(text)
        if chars <= 0:
            continue

        dur_ms = cur.end_ms - cur.start_ms
        if dur_ms <= 0:
            continue

        cps = chars / (dur_ms / 1000.0)
        if cps <= max_cps:
            continue

        required_ms = int(ceil((chars / max_cps) * 1000.0))

        before = {
            "start_ms": cur.start_ms,
            "end_ms": cur.end_ms,
            "duration_ms": dur_ms,
            "chars": chars,
            "cps": round(cps, 2),
        }

        prev_bound = _prev_end_bound(cues, i, max_overlap_ms)
        next_bound = _next_start_bound(cues, i, max_overlap_ms)

        new_start = cur.start_ms
        new_end = cur.end_ms
        note = "extended_end"

        # 1) extend end as much as allowed (end <= next.start + max_overlap_ms)
        want_end = new_start + required_ms
        candidate_end = want_end
        if next_bound is not None:
            candidate_end = min(candidate_end, next_bound)
        new_end = max(new_end, candidate_end)
        new_start, new_end = _ensure_positive_span(new_start, new_end)

        new_dur = new_end - new_start
        new_cps = chars / (new_dur / 1000.0) if new_dur > 0 else float("inf")

        # 2) if still too fast, shift start earlier as much as allowed
        if new_cps > max_cps:
            note = "shifted_start_earlier"
            want_start = new_end - required_ms
            new_start = max(want_start, prev_bound, 0)
            new_start, new_end = _ensure_positive_span(new_start, new_end)

            new_dur = new_end - new_start
            new_cps = chars / (new_dur / 1000.0) if new_dur > 0 else float("inf")

        # 3) If still too fast due to next cue cap, and delta is small -> shift NEXT cue.start later
        if new_cps > max_cps and i < len(cues) - 1:
            # delta is the missing ms we need for current cue to reach required_ms
            delta = (new_start + required_ms) - new_end
            if delta > 0 and delta <= shift_next_max_ms:
                nxt = cues[i + 1]

                proposed_next_start = nxt.start_ms + delta
                max_next_start = nxt.end_ms - min_next_keep_ms

                if proposed_next_start <= max_next_start:
                    # 3.1) shift next cue start
                    actions.append(
                        FixAction(
                            name="shift_next_start_for_cps",
                            cue_index=i + 1,
                            before={"start_ms": nxt.start_ms, "end_ms": nxt.end_ms},
                            after={"start_ms": proposed_next_start, "end_ms": nxt.end_ms, "delta_ms": delta},
                        )
                    )
                    cues[i + 1] = nxt.with_times(start_ms=proposed_next_start)

                    # 3.2) extend current end without overlap:
                    # end <= next.start - max_overlap_ms
                    cap_end = proposed_next_start - max_overlap_ms
                    new_end = min(new_start + required_ms, cap_end)
                    new_start, new_end = _ensure_positive_span(new_start, new_end)

                    new_dur = new_end - new_start
                    new_cps = chars / (new_dur / 1000.0) if new_dur > 0 else float("inf")
                    note = "shifted_next_start_then_extended_end"

        # 4) best-effort note if still failing
        if new_cps > max_cps:
            note = "best_effort_insufficient_room"

        if new_start != cur.start_ms or new_end != cur.end_ms:
            actions.append(
                FixAction(
                    name="fix_too_fast_cps",
                    cue_index=i,
                    before=before,
                    after={
                        "start_ms": new_start,
                        "end_ms": new_end,
                        "duration_ms": new_end - new_start,
                        "required_ms": required_ms,
                        "max_cps": max_cps,
                        "new_cps": round(new_cps, 2) if new_dur > 0 else None,
                        "note": note,
                    },
                )
            )
            cues[i] = cur.with_times(start_ms=new_start, end_ms=new_end)

    return SubtitleDoc(cues=cues, source_path=doc.source_path), actions


# ============================================================
# 3) Wrapping (your original)
# ============================================================
_PUNCT = set("，。！？；：、,.!?;:")


def _normalize_ws(text: str) -> str:
    return " ".join(text.strip().split())


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0x3000 <= code <= 0x303F  # CJK Symbols and Punctuation
        or 0xFF00 <= code <= 0xFFEF  # Fullwidth forms
    )


def _looks_like_cjk(text: str) -> bool:
    cjk = sum(1 for ch in text if _is_cjk_char(ch))
    letters = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    return cjk >= 4 and cjk >= letters


def _best_split_2_lines(words: List[str], max_line_len: int) -> List[str] | None:
    """
    Find a split of words into 2 lines where both lines <= max_line_len.
    Choose the split that minimizes the maximum line length (more balanced).
    """
    best: tuple[int, str, str] | None = None

    for i in range(1, len(words)):
        l1 = " ".join(words[:i])
        l2 = " ".join(words[i:])
        if len(l1) <= max_line_len and len(l2) <= max_line_len:
            score = max(len(l1), len(l2))
            cand = (score, l1, l2)
            if best is None or cand < best:
                best = cand

    if best is None:
        return None
    return [best[1], best[2]]


def _greedy_word_wrap(words: List[str], max_line_len: int, max_lines: int) -> List[str]:
    """
    Greedy word wrap into <= max_lines, WITHOUT forcing leftover into last line,
    because that can create LINE_TOO_LONG.
    """
    lines: List[str] = []
    cur = ""

    for w in words:
        candidate = w if cur == "" else f"{cur} {w}"
        if len(candidate) <= max_line_len:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = w  # may itself exceed max_line_len; best-effort
            if len(lines) >= max_lines - 1:
                break

    if cur:
        lines.append(cur)

    return lines[:max_lines]


def _wrap_cjk_best_effort(text: str, max_line_len: int, max_lines: int) -> List[str]:
    """
    CJK-like char wrapping with punctuation preference.
    IMPORTANT: do NOT merge leftover into last line (would exceed max_line_len).
    """
    text = _normalize_ws(text)
    if text == "":
        return []

    remaining = text
    lines: List[str] = []

    while remaining and len(lines) < max_lines:
        remaining = remaining.strip()
        if remaining == "":
            break

        if len(remaining) <= max_line_len:
            lines.append(remaining)
            remaining = ""
            break

        cut = max_line_len

        # Prefer cut at punctuation within last 1/3 segment
        search_start = max(0, max_line_len - max_line_len // 3)
        upper = min(len(remaining), max_line_len)
        best = -1
        for j in range(search_start, upper):
            if remaining[j] in _PUNCT:
                best = j + 1  # include punct

        if best != -1:
            cut = best

        piece = remaining[:cut].strip()
        if piece:
            lines.append(piece)
        remaining = remaining[cut:].strip()

    return lines[:max_lines]


def _wrap_text_best_effort(text: str, max_line_len: int, max_lines: int) -> List[str]:
    """
    Wrap text to lines with constraints.

    - If text looks like whitespace-language (latin) -> word wrap (don't split words).
      * max_lines==2: optimal split for stability.
      * else: greedy wrap up to max_lines.
    - Otherwise (CJK-like) -> char wrap with punctuation preference.
      * Even if it contains spaces, treat as CJK if heuristic says so.
    - Never force leftover by merging into last line (often recreates LINE_TOO_LONG).
    """
    text = text.strip()
    if text == "":
        return []

    if max_lines <= 0:
        return []

    has_spaces = any(ch.isspace() for ch in text)

    # Treat Chinese-with-spaces as CJK.
    if _looks_like_cjk(text):
        return _wrap_cjk_best_effort(text, max_line_len=max_line_len, max_lines=max_lines)

    # Whitespace-language path (latin-ish)
    if has_spaces:
        text = _normalize_ws(text)
        if text == "":
            return []

        if len(text) <= max_line_len:
            return [text]

        words = text.split(" ")

        if max_lines == 1:
            return [" ".join(words).strip()]

        if max_lines == 2:
            best = _best_split_2_lines(words, max_line_len=max_line_len)
            if best is not None:
                return best
            # Not splittable into 2 lines within limit (e.g., a single huge token)
            return _greedy_word_wrap(words, max_line_len=max_line_len, max_lines=max_lines)

        return _greedy_word_wrap(words, max_line_len=max_line_len, max_lines=max_lines)

    # No spaces and not CJK-looking: fallback to char wrap (same as CJK wrap)
    return _wrap_cjk_best_effort(text, max_line_len=max_line_len, max_lines=max_lines)


def wrap_lines(doc: SubtitleDoc, max_line_len: int = 42, max_lines: int = 2) -> tuple[SubtitleDoc, List[FixAction]]:
    """
    Deterministic wrapping:
    - Reflow each cue's text into <= max_lines lines
    - Attempts to keep words intact for whitespace languages
    """
    cues = list(doc.cues)
    actions: List[FixAction] = []

    for i, cue in enumerate(cues):
        if cue.text.strip() == "":
            continue

        # Flatten current text for reflow
        original_lines = cue.lines
        flat = " ".join([ln.strip() for ln in original_lines]) if original_lines else cue.text.strip()
        flat = _normalize_ws(flat)

        new_lines = _wrap_text_best_effort(flat, max_line_len=max_line_len, max_lines=max_lines)
        new_text = "\n".join(new_lines)

        if new_text != cue.text:
            actions.append(
                FixAction(
                    name="wrap_lines",
                    cue_index=i,
                    before={"text": cue.text},
                    after={"text": new_text},
                )
            )
            cues[i] = cue.with_text(new_text)

    return SubtitleDoc(cues=cues, source_path=doc.source_path), actions
