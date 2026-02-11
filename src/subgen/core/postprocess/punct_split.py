from __future__ import annotations

from typing import List, Sequence

from subgen.core_types import Segment, Word

# 句末/强断句标点（含全角/半角）
END_PUNCT = set("。！？!?；;…")


def _join_words(ws: Sequence[Word]) -> str:
    return "".join((w.word or "") for w in ws).strip()


def _slice_words_by_time(words: List[Word], start: float, end: float) -> List[Word]:
    out: List[Word] = []
    for w in words:
        ws = float(w.start)
        we = float(w.end)
        if we <= start:
            continue
        if ws >= end:
            break
        if ws < end and we > start:
            out.append(w)
    return out


def _has_end_punct(token: str) -> bool:
    if not token:
        return False
    return any(ch in END_PUNCT for ch in token)


def split_segments_on_sentence_end_punct(
    *,
    words: List[Word],
    segments: List[Segment],
    min_seg: float = 2.5,
    hard_max: float = 15.0,
    min_tail_chars: int = 4,
) -> List[Segment]:
    """
    在句末标点之后切段（依赖 word timestamps）。
    """
    if not segments:
        return []
    if not words:
        return segments

    words = sorted(words, key=lambda w: (float(w.start), float(w.end)))
    segs = sorted(segments, key=lambda s: (float(s.start), float(s.end)))

    out: List[Segment] = []

    for seg in segs:
        s0 = float(seg.start)
        s1 = float(seg.end)
        if s1 <= s0:
            continue

        win_words = _slice_words_by_time(words, s0, s1)
        if len(win_words) < 2:
            if (seg.text or "").strip():
                out.append(seg)
            continue

        # 尾部字符数：避免句末切完后只剩很少内容
        tail_chars: List[int] = [0] * (len(win_words) + 1)
        acc = 0
        for i in range(len(win_words) - 1, -1, -1):
            tok = (win_words[i].word or "")
            acc += sum(1 for ch in tok if not ch.isspace())
            tail_chars[i] = acc

        cuts: List[int] = []
        seg_start = float(win_words[0].start)

        for i, w in enumerate(win_words[:-1]):
            tok = (w.word or "")
            if not _has_end_punct(tok):
                continue

            cut_time = float(w.end)
            left_dur = cut_time - seg_start
            right_dur = float(win_words[-1].end) - cut_time

            if left_dur < float(min_seg):
                continue
            if right_dur < float(min_seg):
                continue
            if tail_chars[i + 1] < int(min_tail_chars):
                continue

            cuts.append(i)

        if not cuts:
            if (seg.text or "").strip():
                out.append(seg)
            continue

        start_i = 0
        for cut_i in cuts:
            part = win_words[start_i : cut_i + 1]
            if part:
                st = float(part[0].start)
                ed = float(part[-1].end)
                if ed > st and (ed - st) <= float(hard_max):
                    out.append(Segment(start=st, end=ed, text=_join_words(part)))
            start_i = cut_i + 1

        rest = win_words[start_i:]
        if rest:
            st = float(rest[0].start)
            ed = float(rest[-1].end)
            if ed > st and (ed - st) <= float(hard_max):
                out.append(Segment(start=st, end=ed, text=_join_words(rest)))

    out = [s for s in out if (s.text or "").strip()]
    out.sort(key=lambda s: (float(s.start), float(s.end)))
    return out
