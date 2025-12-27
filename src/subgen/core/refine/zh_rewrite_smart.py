from __future__ import annotations

from typing import Dict, List, Callable, Optional

from subgen.core_types import Transcript, Segment
from subgen.utils.logger import get_logger
from subgen.core.refine.zh_rewrite import zh_naturalize
from subgen.core.refine.risk import is_high_risk_segment, RiskConfig

logger = get_logger()


def default_llm_rewrite_text(text: str) -> str:
    """
    LLM rewrite placeholder.
    你之后可以替换为：
      - OpenAI / 其他云模型
      - 本地LLM
    现在先原样返回，保证工程可跑。
    """
    return text


class SmartSelectiveZhRewriter:
    """
    只对高风险片段启用智能润色，其它片段走规则版。
    """

    def __init__(
        self,
        glossary: Dict[str, str] | None = None,
        risk_cfg: RiskConfig | None = None,
        llm_rewrite_fn: Optional[Callable[[str], str]] = None,
    ):
        self.glossary = glossary or {}
        self.risk_cfg = risk_cfg or RiskConfig()
        self.llm_rewrite_fn = llm_rewrite_fn or default_llm_rewrite_text

    def rewrite(self, zh_literal: Transcript) -> Transcript:
        # 先准备规则版全量结果（作为默认）
        rule_final = zh_naturalize(zh_literal)

        new_segments: List[Segment] = []

        for lit_seg, rule_seg in zip(zh_literal.segments, rule_final.segments):
            text = (lit_seg.text or "").strip()
            if not text:
                new_segments.append(rule_seg)
                continue

            high_risk = is_high_risk_segment(lit_seg, self.glossary, self.risk_cfg)

            if not high_risk:
                # 低风险：直接用规则版
                new_segments.append(rule_seg)
                continue

            # 高风险：智能润色
            try:
                candidate = self.llm_rewrite_fn(text).strip() or text
            except Exception as e:
                logger.info(f"[LLM rewrite failed] fallback to rule. err={e}")
                candidate = rule_seg.text

            # 为安全起见：如果 LLM 输出异常空/过长，回退
            if not candidate:
                candidate = rule_seg.text
            if len(candidate) > max(90, len(text) * 2):
                candidate = rule_seg.text

            new_segments.append(
                Segment(
                    start=lit_seg.start,
                    end=lit_seg.end,
                    text=candidate,
                    confidence=lit_seg.confidence,
                )
            )

        return Transcript(language=zh_literal.language, segments=new_segments)
