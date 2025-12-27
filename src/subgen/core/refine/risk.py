from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict

from subgen.core_types import Segment


_num_re = re.compile(r"""
    (?:
        \d{1,3}(?:,\d{3})+|\d+      # 123,456 or 123
    )
    (?:\.\d+)?                     # optional decimals
    (?:%|％)?                      # optional percent
""", re.X)

# 机器翻译常见“僵硬信号”
_mt_weird_space_cjk = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
_mt_weird_punc_space = re.compile(r"\s*([，。！？])\s*")
_multi_space = re.compile(r"\s{2,}")


@dataclass
class RiskConfig:
    # ASR 低置信阈值
    low_conf_threshold: float = 0.35

    # 数字密集阈值（一条字幕中数字数量）
    number_count_threshold: int = 2

    # 术语密集阈值（命中 glossary 的数量）
    glossary_hit_threshold: int = 1

    # 明显机翻腔判定
    enable_mt_stiff_detector: bool = True


def count_numbers(text: str) -> int:
    return len(_num_re.findall(text or ""))


def count_glossary_hits(text: str, glossary: Dict[str, str]) -> int:
    if not glossary:
        return 0
    hits = 0
    for tgt in glossary.values():
        if tgt and tgt in text:
            hits += 1
    return hits


def looks_mt_stiff(text: str) -> bool:
    """
    Heuristic: detect obvious MT artifacts.
    Keep conservative to avoid false positives.
    """
    t = text or ""

    # 中文字间出现空格
    if _mt_weird_space_cjk.search(t):
        return True

    # 连续多空格
    if _multi_space.search(t):
        return True

    # 标点前后空格异常（只能算弱信号，配合其他条件也行）
    # 这里单独触发也算“可疑”
    if _mt_weird_punc_space.search(t) and ("， " in t or "。 " in t or "！ " in t or "？ " in t):
        return True

    # 超长且缺少常见中文标点
    if len(t) >= 40 and all(p not in t for p in ["，", "。", "！", "？", "；", "："]):
        return True

    return False


def is_high_risk_segment(
    seg: Segment,
    glossary: Dict[str, str] | None = None,
    cfg: RiskConfig | None = None,
) -> bool:
    glossary = glossary or {}
    cfg = cfg or RiskConfig()

    text = (seg.text or "").strip()

    # 1) ASR 低置信
    if seg.confidence is not None and seg.confidence < cfg.low_conf_threshold:
        return True

    # 2) 数字密集
    if count_numbers(text) >= cfg.number_count_threshold:
        return True

    # 3) 术语密集
    if count_glossary_hits(text, glossary) >= cfg.glossary_hit_threshold:
        return True

    # 4) 明显机翻腔
    if cfg.enable_mt_stiff_detector and looks_mt_stiff(text):
        return True

    return False
