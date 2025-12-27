from __future__ import annotations

import os
from typing import List

from openai import OpenAI

from subgen.core.translate.base import TranslatorProvider
from subgen.core_types import Transcript, Segment

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = """
你是“多语种→中文的受约束字幕翻译器”。

输入可能是任何非英语语言（也可能包含少量英语夹杂）。
任务：将输入翻译为中文字幕。

要求：
1) 忠实翻译，不要增删事实。
2) 不改数字、日期、金额、单位。
3) 不改人名、地名、组织名、产品名；保持前后一致。
4) 输出自然、简洁，适合字幕。
5) 如原文有省略，按最合理语境补全，但不要虚构信息。
6) 结合上下文进行适当调整，使译文更通顺流畅。
仅输出中文。
""".strip()


class OpenAITranslator(TranslatorProvider):
    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model

    def _translate_text(self, text: str) -> str:
        resp = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text},
            ],
        )
        out = getattr(resp, "output_text", None)
        return (out or text).strip()

    def translate(self, transcript: Transcript, target_lang: str = "zh") -> Transcript:
        new_segments: List[Segment] = []
        for s in transcript.segments:
            jp = (s.text or "").strip()
            if not jp:
                continue
            zh = self._translate_text(jp)
            new_segments.append(
                Segment(
                    start=s.start,
                    end=s.end,
                    text=zh,
                    confidence=s.confidence
                )
            )
        return Transcript(language=target_lang, segments=new_segments)
