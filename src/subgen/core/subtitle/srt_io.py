from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from .models import SubtitleCue, SubtitleDoc

_TIMECODE_RE = re.compile(
    r"^(?P<h1>\d{2}):(?P<m1>\d{2}):(?P<s1>\d{2}),(?P<ms1>\d{3})\s*-->\s*"
    r"(?P<h2>\d{2}):(?P<m2>\d{2}):(?P<s2>\d{2}),(?P<ms2>\d{3})(?:\s+.*)?$"
)

# SRT index line is usually an integer, but we allow non-integer and ignore it.
_INDEX_RE = re.compile(r"^\s*\d+\s*$")


def _tc_to_ms(h: int, m: int, s: int, ms: int) -> int:
    return (((h * 60) + m) * 60 + s) * 1000 + ms


def _parse_timecode_line(line: str) -> Tuple[int, int]:
    m = _TIMECODE_RE.match(line.strip())
    if not m:
        raise ValueError(f"Invalid SRT timecode line: {line!r}")

    h1, m1, s1, ms1 = int(m.group("h1")), int(m.group("m1")), int(m.group("s1")), int(m.group("ms1"))
    h2, m2, s2, ms2 = int(m.group("h2")), int(m.group("m2")), int(m.group("s2")), int(m.group("ms2"))

    start_ms = _tc_to_ms(h1, m1, s1, ms1)
    end_ms = _tc_to_ms(h2, m2, s2, ms2)
    if end_ms < start_ms:
        raise ValueError(f"SRT cue end before start: {line!r}")
    return start_ms, end_ms


def _format_timecode(ms: int) -> str:
    if ms < 0:
        ms = 0
    total_s, milli = divmod(ms, 1000)
    total_m, sec = divmod(total_s, 60)
    hour, minute = divmod(total_m, 60)
    return f"{hour:02d}:{minute:02d}:{sec:02d},{milli:03d}"


def read_srt(path: str | Path, encoding: str = "utf-8") -> SubtitleDoc:
    """
    Read a .srt file into SubtitleDoc.

    - Handles UTF-8 BOM.
    - Splits blocks by blank lines.
    - Ignores cue index line if present.
    - Preserves cue text line breaks.
    """
    p = Path(path)
    raw = p.read_text(encoding=encoding, errors="replace")

    # Strip BOM if any
    raw = raw.lstrip("\ufeff")

    # Normalize newlines to '\n'
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    blocks = [blk for blk in raw.split("\n\n") if blk.strip() != ""]
    cues: List[SubtitleCue] = []

    for blk in blocks:
        lines = [ln.rstrip("\n") for ln in blk.split("\n")]

        if not lines:
            continue

        # Optional index line
        idx = 0
        if _INDEX_RE.match(lines[0] or ""):
            idx = 1

        if idx >= len(lines):
            continue

        # Timecode line
        start_ms, end_ms = _parse_timecode_line(lines[idx])
        text_lines = lines[idx + 1 :] if idx + 1 < len(lines) else []
        # Keep original line structure, but remove trailing empty lines inside block
        while text_lines and text_lines[-1].strip() == "":
            text_lines.pop()

        text = "\n".join(text_lines).strip("\n")
        cues.append(SubtitleCue(start_ms=start_ms, end_ms=end_ms, text=text))

    doc = SubtitleDoc(cues=cues, source_path=str(p))
    return doc


def write_srt(doc: SubtitleDoc, path: str | Path, encoding: str = "utf-8") -> Path:
    """
    Write SubtitleDoc into a .srt file.

    - Re-numbers cues from 1.
    - Writes normalized timecodes.
    - Ensures file ends with newline.
    """
    p = Path(path)

    out_lines: List[str] = []
    for i, cue in enumerate(doc.cues, start=1):
        out_lines.append(str(i))
        out_lines.append(f"{_format_timecode(cue.start_ms)} --> {_format_timecode(cue.end_ms)}")
        if cue.text.strip() != "":
            out_lines.extend(cue.text.split("\n"))
        out_lines.append("")  # blank line between cues

    content = "\n".join(out_lines).rstrip("\n") + "\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding=encoding)
    return p
