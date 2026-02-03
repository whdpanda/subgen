from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.tools import StructuredTool

from subgen.agent.knowledge.index import KnowledgeBase
from subgen.agent.knowledge.loader import kb_config_from_env
from subgen.agent.tools.tool_names import KB_SEARCH  # ✅ 统一从这里取真名


class KBResultItem(TypedDict):
    source: str
    score: float
    text: str


class KBSearchResult(TypedDict):
    query: str
    k: int
    kb: Dict[str, Any]
    results: List[KBResultItem]


def _ok(data: Any) -> dict[str, Any]:
    """Success envelope."""
    return {"ok": True, "data": data, "error": None}


def _fail(err_type: str, message: str, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Failure envelope."""
    return {
        "ok": False,
        "data": None,
        "error": {
            "type": err_type,
            "message": message,
            "details": details or {},
        },
    }


def make_kb_search_tool() -> StructuredTool:
    kb = KnowledgeBase(kb_config_from_env())

    def _kb_search(query: str, k: int = 6) -> dict[str, Any]:
        """
        Tool function (wrapped to always return envelope).
        """
        # 1) 轻量参数校验：别让乱参直接炸
        if not isinstance(query, str) or not query.strip():
            return _fail(
                err_type="kb_search.validation_error",
                message="query must be a non-empty string",
                details={"query": query},
            )
        if not isinstance(k, int) or k <= 0:
            return _fail(
                err_type="kb_search.validation_error",
                message="k must be a positive integer",
                details={"k": k},
            )

        # 2) 实际搜索：兜底异常
        try:
            hits = kb.search(query, k=k)
            payload: KBSearchResult = {
                "query": query,
                "k": k,
                "kb": kb.info(),
                "results": [{"source": h.source, "score": h.score, "text": h.text} for h in hits],
            }
            return _ok(payload)
        except Exception as e:
            return _fail(
                err_type="kb_search.runtime_error",
                message=str(e) or e.__class__.__name__,
                details={
                    "exception_class": e.__class__.__name__,
                    "traceback": traceback.format_exc(),
                    "query": query,
                    "k": k,
                },
            )

    tool = StructuredTool.from_function(
        name=KB_SEARCH,  # ✅ 关键：不再手写字符串
        description=(
            "Search SubGen internal knowledge base (docs/knowledge) for "
            "schemas, defaults, CLI/agent usage, and implementation notes. "
            "Use this BEFORE calling other tools if uncertain."
        ),
        func=_kb_search,
    )

    # ✅ 防漂移硬约束
    assert tool.name == KB_SEARCH, f"KB tool name must be '{KB_SEARCH}', got {tool.name!r}"

    return tool
