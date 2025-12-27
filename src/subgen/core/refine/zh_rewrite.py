from __future__ import annotations

import re
from subgen.core_types import Transcript, Segment


def _normalize_punctuation(text: str) -> str:
    """
    Very lightweight Chinese readability cleanup.
    Rules are conservative to avoid changing facts.
    """
    t = text.strip()

    # Normalize common spaces
    t = re.sub(r"\s+", " ", t)

    # Convert common fullwidth/halfwidth punctuation inconsistencies
    # Keep it minimal to avoid over-editing
    t = t.replace(" ,", ",").replace(" .", ".")
    t = t.replace(" !", "!").replace(" ?", "?")
    t = t.replace("， ", "，").replace("。 ", "。")
    t = t.replace("！ ", "！").replace("？ ", "？")

    # Remove duplicate punctuation like "。。", "！！"
    t = re.sub(r"([。！？])\1+", r"\1", t)
    t = re.sub(r"([,，])\1+", r"\1", t)

    # Fix awkward Chinese punctuation spacing
    t = re.sub(r"\s*([，。！？])\s*", r"\1", t)

    return t.strip()


def _light_reflow(text: str) -> str:
    """
    A tiny reflow to reduce obvious machine-translation stiffness.
    We DO NOT add new info or change numbers.
    """
    t = text

    # Very conservative patterns
    # Example: "我认为 这 是" -> "我认为这是"
    t = t.replace(" 这 是", "这是")
    t = t.replace(" 这 个", "这个")
    t = t.replace(" 那 个", "那个")
    t = t.replace(" 我 们", "我们")

    # Remove spaces between Chinese characters (common MT artifact)
    t = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", t)

    return t


def zh_naturalize(transcript: Transcript) -> Transcript:
    """
    Produce zh_final from zh_literal using safe, rule-based rewriting.
    """
    new_segments = []
    for s in transcript.segments:
        text = s.text or ""

        text = _light_reflow(text)
        text = _normalize_punctuation(text)

        new_segments.append(
            Segment(
                start=s.start,
                end=s.end,
                text=text,
                confidence=s.confidence,
            )
        )

    # language stays zh
    return Transcript(language=transcript.language, segments=new_segments)
