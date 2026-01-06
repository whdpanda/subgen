from __future__ import annotations

import os
import json
from typing import List, Optional, Tuple

from openai import OpenAI

from subgen.core.segment.base import SegmenterProvider
from subgen.core.segment.rule import RuleSegmenter
from subgen.core_types import Word, Segment
from subgen.utils.logger import get_logger

logger = get_logger()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = """
你是“字幕语义切分器”，只做分段，不做翻译。

输入：连续的 word 列表（每个 word 有 i, w, s, e）。
输出：严格 JSON：{"cuts":[...]}，cuts 是“在该 index 的 word 之后切”。

硬性约束：
1) 每段时长 <= HARD_MAX 秒（绝不超过）。
2) 目标每段 6~10 秒；尽量 <= SOFT_MAX 秒（美观）。允许略超 SOFT_MAX，但不可超过 HARD_MAX。
3) 切分是按照语义进行。
4) 每段时长尽量 >= MIN_SEG 秒（过短会很难看），除非剩余不足。
5) 优先在句末标点/长停顿处切，其次在逗号类停顿处切。
6) cuts 的数量应远小于 words 数量；禁止对每个 word 都切。
只输出 JSON，不要输出解释文字。
""".strip()


def _join_words(ws: List[Word]) -> str:
    # whisper/faster-whisper 的 word 往往自带前导空格，直接拼接
    return "".join((w.word or "") for w in ws).strip()


def _build_segments_from_cuts(win_words: List[Word], cuts: List[int]) -> List[Segment]:
    segs: List[Segment] = []
    start_i = 0
    for cut in cuts:
        if cut < start_i:
            continue
        part = win_words[start_i: cut + 1]
        if part:
            segs.append(Segment(start=part[0].start, end=part[-1].end, text=_join_words(part)))
        start_i = cut + 1
    rest = win_words[start_i:]
    if rest:
        segs.append(Segment(start=rest[0].start, end=rest[-1].end, text=_join_words(rest)))
    return segs


def _seg_stats(segs: List[Segment]) -> Tuple[float, float, float]:
    if not segs:
        return (0.0, 1.0, 1.0)
    durs = [(float(s.end) - float(s.start)) for s in segs]
    avg_dur = sum(durs) / max(1, len(durs))
    short_ratio = sum(1 for d in durs if d < 1.2) / len(durs)
    one_word_like_ratio = sum(
        1 for s, d in zip(segs, durs)
        if d < 2.0 and len((s.text or "").strip()) <= 6
    ) / len(durs)
    return (avg_dur, short_ratio, one_word_like_ratio)


def _slice_words_by_time(words: List[Word], start: float, end: float) -> List[Word]:
    """
    取与 [start,end) 有交集的 words（避免边界词被丢弃）。
    words 需按时间排序。
    """
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


def _overlap_ratio(a: Segment, b: Segment) -> float:
    a0, a1 = float(a.start), float(a.end)
    b0, b1 = float(b.start), float(b.end)
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1e-9, min(a1 - a0, b1 - b0))
    return inter / denom


def _text_density(seg: Segment) -> float:
    t = (seg.text or "").strip()
    dur = max(1e-6, float(seg.end) - float(seg.start))
    eff = sum(1 for ch in t if not ch.isspace())
    return eff / dur


def _dedupe_by_time_overlap(
    segs: List[Segment],
    *,
    start_tol: float = 0.05,
    overlap_th: float = 0.85,
) -> List[Segment]:
    """
    去掉“同一时间窗产生两条高度重叠段”的情况，只保留更好的那条。
    """
    if not segs:
        return []
    segs = sorted(segs, key=lambda x: (float(x.start), float(x.end)))
    out: List[Segment] = []
    for s in segs:
        if not out:
            out.append(s)
            continue
        prev = out[-1]

        if abs(float(s.start) - float(prev.start)) <= start_tol and _overlap_ratio(prev, s) >= overlap_th:
            dp = _text_density(prev)
            ds = _text_density(s)
            if ds > dp * 1.05:
                out[-1] = s
            else:
                # 密度差不多时，保留覆盖更长的
                if (float(s.end) - float(s.start)) > (float(prev.end) - float(prev.start)) + 0.05 and ds >= dp * 0.95:
                    out[-1] = s
            continue

        # 极强重叠（即使 start 不完全相同）
        if _overlap_ratio(prev, s) >= 0.95:
            dp = _text_density(prev)
            ds = _text_density(s)
            if ds > dp * 1.05:
                out[-1] = s
            else:
                if (float(s.end) - float(s.start)) > (float(prev.end) - float(prev.start)) + 0.05 and ds >= dp * 0.95:
                    out[-1] = s
            continue

        out.append(s)
    return out


def _normalize_no_overlap(segs: List[Segment]) -> List[Segment]:
    """
    保证输出 segments 时间不重叠：
    - 强重叠：保留密度更高的
    - 小重叠：把 cur.start clamp 到 prev.end
    """
    if not segs:
        return []
    segs = sorted(segs, key=lambda x: (float(x.start), float(x.end)))
    out: List[Segment] = [segs[0]]

    for cur in segs[1:]:
        prev = out[-1]
        p1 = float(prev.end)
        c0, c1 = float(cur.start), float(cur.end)

        if c1 <= p1:
            if _text_density(cur) > _text_density(prev) * 1.10:
                out[-1] = cur
            continue

        if c0 < p1:
            inter = p1 - c0
            c_len = max(1e-6, c1 - c0)
            overlap_frac = inter / c_len
            if overlap_frac >= 0.60:
                if _text_density(cur) > _text_density(prev) * 1.10:
                    out[-1] = cur
                continue
            cur.start = p1
            if float(cur.end) - float(cur.start) < 0.20:
                continue

        if (cur.text or "").strip():
            out.append(cur)

    return out


class OpenAISegmenter(SegmenterProvider):
    def __init__(
        self,
        model: str = "gpt-5-mini",
        soft_max: float = 7.0,
        hard_max: float = 15.0,
        min_seg: float = 2.5,
        window_s: float = 60.0,
        overlap_s: float = 2.5,
    ):
        self.model = model
        self.soft_max = float(soft_max)
        self.hard_max = float(hard_max)
        self.min_seg = float(min_seg)
        self.window_s = float(window_s)
        self.overlap_s = float(overlap_s)
        self.fallback = RuleSegmenter(soft_max=soft_max, hard_max=hard_max, min_seg=min_seg)

    def _call_llm_for_window(self, win_words: List[Word]) -> Optional[List[int]]:
        payload = [
            {"i": idx, "w": w.word, "s": round(float(w.start), 3), "e": round(float(w.end), 3)}
            for idx, w in enumerate(win_words)
        ]
        user = {
            "SOFT_MAX": self.soft_max,
            "HARD_MAX": self.hard_max,
            "MIN_SEG": self.min_seg,
            "words": payload,
        }

        try:
            resp = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
            )
            txt = (getattr(resp, "output_text", None) or "").strip()
            data = json.loads(txt)
            cuts = data.get("cuts")
            if not isinstance(cuts, list):
                return None

            out: List[int] = []
            for c in cuts:
                if isinstance(c, int):
                    out.append(c)

            out = sorted(set(c for c in out if 0 <= c < len(win_words)))
            return out
        except Exception as e:
            logger.warning(f"OpenAI segmenter failed -> fallback rule. err={e}")
            return None

    def segment(self, words: List[Word]) -> List[Segment]:
        if not words:
            return []

        words = sorted(words, key=lambda w: (float(w.start), float(w.end)))

        segments: List[Segment] = []
        cursor = float(words[0].start)
        end_all = float(words[-1].end)

        left = 0
        n = len(words)
        min_step = max(0.2, self.overlap_s * 0.2)

        while cursor < end_all and left < n:
            win_start = cursor
            win_end = min(cursor + self.window_s, end_all)

            while left < n and float(words[left].end) < win_start:
                left += 1
            right = left
            while right < n and float(words[right].start) <= win_end:
                right += 1

            win_words = words[left:right]
            if len(win_words) < 10:
                # 窗口内词太少：可能是长静默或 timestamps 洞。不要 break，直接推进 cursor 继续扫描。
                next_cursor = win_end - self.overlap_s
                if next_cursor <= cursor + 1e-6:
                    next_cursor = cursor + min_step
                cursor = next_cursor
                continue

            cuts = self._call_llm_for_window(win_words)

            # guardrail：切得过碎就回退 rule
            use_rule = False
            if cuts is None:
                use_rule = True
            else:
                if len(cuts) > max(10, int(len(win_words) * 0.25)):
                    use_rule = True
                else:
                    tmp = _build_segments_from_cuts(win_words, cuts)
                    avg_dur, short_ratio, one_word_like_ratio = _seg_stats(tmp)
                    if avg_dur < 2.0 or short_ratio > 0.35 or one_word_like_ratio > 0.25:
                        use_rule = True

            if use_rule:
                win_segs = self.fallback.segment(win_words)
            else:
                win_segs = _build_segments_from_cuts(win_words, cuts)
                # hard_max 兜底
                fixed: List[Segment] = []
                for s in win_segs:
                    if (float(s.end) - float(s.start)) > self.hard_max:
                        sub = _slice_words_by_time(win_words, float(s.start), float(s.end))
                        fixed.extend(self.fallback.segment(sub) if sub else [s])
                    else:
                        fixed.append(s)
                win_segs = fixed

            cutoff = win_end - self.overlap_s

            for s in win_segs:
                s0 = float(s.start)
                s1 = float(s.end)

                if s1 <= win_start:
                    continue
                if s0 >= cutoff:
                    continue
                if not (s.text or "").strip():
                    continue

                # 跨 cutoff 的段：截断到 cutoff，避免下一窗重复
                if s0 < cutoff < s1:
                    part = _slice_words_by_time(win_words, s0, cutoff)
                    if not part:
                        continue
                    segments.append(Segment(start=part[0].start, end=part[-1].end, text=_join_words(part)))
                    continue

                segments.append(s)

            # 前进 cursor，避免 stuck
            next_cursor = cutoff
            if next_cursor <= cursor + 1e-6:
                next_cursor = cursor + min_step
            cursor = next_cursor

        segments.sort(key=lambda x: (float(x.start), float(x.end)))
        segments = _dedupe_by_time_overlap(segments)
        segments = _normalize_no_overlap(segments)
        return segments
