from __future__ import annotations

import os
from typing import Any, List, Optional

from subgen.core.translate.base import TranslatorProvider
from subgen.core_types import Transcript, Segment

SYSTEM = """
你是“受约束字幕翻译器（任意语言→中文）”。

硬性要求：
1) 忠实翻译，不增删事实、不脑补不存在的信息。
2) 不改数字、人名、地名、组织名、产品名。
3) 语言自然、简洁、适合字幕；尽量口语化但不改意义。
4) 只输出【当前段】的中文译文；不要输出解释、不要输出原文、不要加引号。
5）不要出现未翻译扔保留原语言的内容。
""".strip()


def _safe(s: str) -> str:
    return (s or "").strip()


# ---- lazy client (重要：避免 pytest collection 时就初始化) ----
_client: Optional[Any] = None


def _get_openai_client(*, api_key: Optional[str] = None) -> Any:
    """
    Lazy init OpenAI client.

    - 不在 import 时初始化，避免测试收集阶段因为缺 OPENAI_API_KEY 直接炸
    - 延迟导入 openai，避免没装 openai 时也在 import 阶段炸
    """
    global _client
    if _client is not None and api_key is None:
        return _client

    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set env OPENAI_API_KEY or pass api_key to OpenAITranslator(api_key=...)."
        )

    try:
        from openai import OpenAI  # lazy import
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'openai'. Install it via: python -m pip install openai"
        ) from e

    client = OpenAI(api_key=key)

    # 只有在使用 env key 的情况下才缓存全局 client
    if api_key is None:
        _client = client

    return client


class OpenAITranslator(TranslatorProvider):
    def __init__(
        self,
        model: str = "gpt-5.2",
        prev_k: int = 1,            # 使用前文段数
        next_hint_chars: int = 80,  # 使用后文提示长度（字符）
        api_key: str | None = None, # 可选：允许显式传 key（测试/多账户更方便）
    ):
        self.model = model
        self.prev_k = max(0, int(prev_k))
        self.next_hint_chars = max(0, int(next_hint_chars))
        self.api_key = api_key

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

        client = _get_openai_client(api_key=self.api_key)

        resp = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        out = str(getattr(resp, "output_text", "") or "")
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
