from __future__ import annotations

from typing import List, Sequence, Optional

from subgen.core_types import Segment, Word


END_PUNCT = set("。！？!?…")  # 你提到的几个符号（含全角/半角）


def _join_words(ws: Sequence[Word]) -> str:
    # whisper/faster-whisper 的 word 往往自带前导空格，直接拼接
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
    # token 里只要出现句末符号就认为可切（例如 "word。" 或 "？")
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
    强制把 segment 在句末标点（。？！…!?）之后切开，使句末尽量作为段尾。
    依赖 words 的时间戳来定位切点，保证拆分后的时间轴合理。

    安全约束：
    - 切分后两边都尽量 >= min_seg，否则不切
    - 生成段长度仍不超过 hard_max（一般不会变长，这里只是保护）
    """
    if not segments:
        return []
    if not words:
        # 没有 words 就无法安全按时间切分
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

        # 预计算每个位置之后的“尾部字符数”，用于避免在句号后只剩很少东西也切
        tail_chars: List[int] = [0] * (len(win_words) + 1)
        acc = 0
        for i in range(len(win_words) - 1, -1, -1):
            tok = (win_words[i].word or "")
            # 只统计非空白字符
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

            # 两侧太短就不切
            if left_dur < float(min_seg):
                continue
            if right_dur < float(min_seg):
                continue

            # 句号后面如果几乎没内容，不切
            if tail_chars[i + 1] < int(min_tail_chars):
                continue

            cuts.append(i)

        if not cuts:
            # 没有切点，原样保留
            if (seg.text or "").strip():
                out.append(seg)
            continue

        # 根据 cuts 切分成多个子段
        start_i = 0
        for cut_i in cuts:
            part = win_words[start_i : cut_i + 1]
            if part:
                st = float(part[0].start)
                ed = float(part[-1].end)
                if ed > st and (ed - st) <= float(hard_max):
                    out.append(Segment(start=st, end=ed, text=_join_words(part), confidence=getattr(seg, "confidence", None)))
            start_i = cut_i + 1

        rest = win_words[start_i:]
        if rest:
            st = float(rest[0].start)
            ed = float(rest[-1].end)
            if ed > st and (ed - st) <= float(hard_max):
                out.append(Segment(start=st, end=ed, text=_join_words(rest), confidence=getattr(seg, "confidence", None)))

    # 最终按时间排序并过滤空文本
    out = [s for s in out if (s.text or "").strip()]
    out.sort(key=lambda s: (float(s.start), float(s.end)))
    return out
