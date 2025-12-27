from __future__ import annotations

from typing import List
from subgen.core_types import Transcript


def _format_ts(sec: float) -> str:
    # SRT timestamp: HH:MM:SS,mmm
    ms = int(round(sec * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(transcript: Transcript) -> str:
    lines: List[str] = []
    for idx, seg in enumerate(transcript.segments, start=1):
        lines.append(str(idx))
        lines.append(f"{_format_ts(seg.start)} --> {_format_ts(seg.end)}")
        text = (seg.text or "").strip()
        lines.append(text if text else " ")
        lines.append("")
    return "\n".join(lines)


def to_srt_bilingual(src: Transcript, zh: Transcript) -> str:
    """
    双语字幕（最小实现）：
    每条字幕两行：
      原文
      中文

    以 index 对齐；适用于你现在“逐段翻译、不改时间戳”的结构。
    """
    src_segs = src.segments
    zh_segs = zh.segments

    n = min(len(src_segs), len(zh_segs))
    lines: List[str] = []

    for idx in range(n):
        s = src_segs[idx]
        z = zh_segs[idx]

        lines.append(str(idx + 1))
        lines.append(f"{_format_ts(s.start)} --> {_format_ts(s.end)}")

        src_text = (s.text or "").strip()
        zh_text = (z.text or "").strip()

        bilingual = f"{src_text}\n{zh_text}".strip()
        lines.append(bilingual if bilingual else " ")
        lines.append("")

    return "\n".join(lines)
