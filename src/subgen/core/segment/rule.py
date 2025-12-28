from __future__ import annotations

from typing import List
import re

from subgen.core.segment.base import SegmenterProvider
from subgen.core_types import Word, Segment

SENT_END = set("。.!?？！")
SOFT_PUNCT = set("，,、;；:：…")


def _is_end_punct(w: str) -> bool:
    t = (w or "").strip()
    return bool(t) and t[-1] in SENT_END


def _is_soft_punct(w: str) -> bool:
    t = (w or "").strip()
    return bool(t) and t[-1] in SOFT_PUNCT


def _join_words(ws: List[Word]) -> str:
    txt = "".join([w.word for w in ws]).strip()
    txt = re.sub(r"\s+", " ", txt)
    return txt


class RuleSegmenter(SegmenterProvider):
    def __init__(
        self,
        soft_max: float = 7.0,
        hard_max: float = 20.0,
        pause_cut: float = 0.65,
        min_seg: float = 2.5,
        near_margin: float = 1.5,
    ):
        self.soft_max = float(soft_max)
        self.hard_max = float(hard_max)
        self.pause_cut = float(pause_cut)
        self.min_seg = float(min_seg)
        self.near_margin = float(near_margin)

    def segment(self, words: List[Word]) -> List[Segment]:
        if not words:
            return []

        # 候选断点 index：在该 word 后切
        strong = set()
        soft = set()
        pause = set()

        for i in range(len(words) - 1):
            cur = words[i]
            nxt = words[i + 1]
            if _is_end_punct(cur.word):
                strong.add(i)
            elif _is_soft_punct(cur.word):
                soft.add(i)

            gap = nxt.start - cur.end
            if gap >= self.pause_cut:
                pause.add(i)

        out: List[Segment] = []
        i = 0
        n = len(words)

        while i < n:
            start_t = words[i].start

            # 先确定 hard_max 内最远可达
            far = i
            for j in range(i, n):
                if (words[j].end - start_t) <= self.hard_max:
                    far = j
                else:
                    break

            # 在 soft_max 内尽量结束（美观），但不能太短
            soft_far = i
            for j in range(i, n):
                if (words[j].end - start_t) <= self.soft_max:
                    soft_far = j
                else:
                    break

            def ok(k: int) -> bool:
                return (words[k].end - start_t) >= self.min_seg

            # 目标区间优先 soft_far（美观），若太短/没有候选则退到 far（硬上限）
            target = soft_far if soft_far > i else far

            candidates = [k for k in range(i, target + 1) if ok(k)]
            best = target

            near_soft = (words[target].end - start_t) >= (self.soft_max - self.near_margin)

            if candidates:
                # 靠近软上限：优先自然断点
                if near_soft:
                    for k in reversed(candidates):
                        if k in strong:
                            best = k
                            break
                    else:
                        for k in reversed(candidates):
                            if k in pause:
                                best = k
                                break
                        else:
                            for k in reversed(candidates):
                                if k in soft:
                                    best = k
                                    break
                else:
                    # 不靠近软上限：遇到句末/长停顿就切
                    for k in reversed(candidates):
                        if k in strong:
                            best = k
                            break
                    else:
                        for k in reversed(candidates):
                            if k in pause:
                                best = k
                                break

            seg_words = words[i : best + 1]
            out.append(
                Segment(
                    start=seg_words[0].start,
                    end=seg_words[-1].end,
                    text=_join_words(seg_words),
                )
            )

            i = best + 1

        return out
