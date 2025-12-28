from __future__ import annotations

import os
from typing import List

from openai import OpenAI

from subgen.core.translate.base import TranslatorProvider
from subgen.core_types import Transcript, Segment

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = """
你是“受约束字幕翻译器（任意语言→中文）”。

硬性要求：
1) 忠实翻译，不增删事实、不脑补不存在的信息。
2) 不改数字、人名、地名、组织名、产品名（保持原样）。
3) 语言自然、简洁、适合字幕；尽量口语化但不改意义。
4) 只输出【当前段】的中文译文；不要输出解释、不要输出原文、不要加引号。
""".strip()


def _safe(s: str) -> str:
    return (s or "").strip()


class OpenAITranslator(TranslatorProvider):
    def __init__(
        self,
        model: str = "gpt-5-mini",
        prev_k: int = 1,            # 使用前文段数
        next_hint_chars: int = 80,  # 使用后文提示长度（字符）
    ):
        self.model = model
        self.prev_k = max(0, int(prev_k))
        self.next_hint_chars = max(0, int(next_hint_chars))

    def _translate_one(self, src_lang: str, prev: str, cur: str, next_hint: str) -> str:
        user = f"""
源语言：{src_lang}

【前文（用于消歧，可忽略）】
{prev}

【当前段（请只翻译这一段）】
{cur}

【后文提示（用于消歧，可忽略）】
{next_hint}
""".strip()

        resp = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        out = getattr(resp, "output_text", None)
        return (_safe(out) or _safe(cur))

    def translate(self, transcript: Transcript, target_lang: str = "zh") -> Transcript:
        src_lang = transcript.language or "unknown"
        segs = transcript.segments

        new_segments: List[Segment] = []
        for i, s in enumerate(segs):
            cur = _safe(s.text)
            if not cur:
                continue

            # prev context
            if self.prev_k > 0:
                prev_segs = segs[max(0, i - self.prev_k): i]
                prev = "\n".join([_safe(x.text) for x in prev_segs if _safe(x.text)])
            else:
                prev = ""

            # next hint
            if i + 1 < len(segs):
                nh = _safe(segs[i + 1].text)
                next_hint = nh[: self.next_hint_chars]
            else:
                next_hint = ""

            zh = self._translate_one(src_lang=src_lang, prev=prev, cur=cur, next_hint=next_hint)

            new_segments.append(
                Segment(
                    start=s.start,
                    end=s.end,
                    text=zh,
                    confidence=s.confidence,
                )
            )

        return Transcript(language=target_lang, segments=new_segments)
