from __future__ import annotations

from dataclasses import dataclass
from typing import List

from subgen.core.subtitle.models import SubtitleCue, SubtitleDoc


@dataclass(frozen=True)
class FixAction:
    name: str
    cue_index: int
    before: dict
    after: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "cue_index": self.cue_index, "before": self.before, "after": self.after}


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
    # Heuristic: treat as CJK if it contains enough CJK chars and they are not dominated by latin letters.
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
    # For CJK, normalize whitespace aggressively (multiple spaces -> single)
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
      * Even if it contains spaces (your fixture's Chinese line), treat as CJK.
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
