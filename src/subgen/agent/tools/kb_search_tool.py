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


class KBError(TypedDict, total=False):
    type: str
    message: str
    details: Dict[str, Any]


class KBMeta(TypedDict, total=False):
    error: KBError
    # you can add more runtime hints later (e.g., "index_version", "timing_ms", ...)


class KBSearchResult(TypedDict):
    ok: bool
    query: str
    k: int
    kb: Dict[str, Any]
    results: List[KBResultItem]
    meta: KBMeta


def _ok(payload: Dict[str, Any], *, meta: Optional[Dict[str, Any]] = None) -> KBSearchResult:
    """
    Success: stable FLAT schema.

    Keys are fixed for agent/KB/docs stability:
      ok/query/k/kb/results/meta
    """
    return {
        "ok": True,
        "query": payload["query"],
        "k": payload["k"],
        "kb": payload["kb"],
        "results": payload["results"],
        "meta": meta or {},
    }


def _fail(
    err_type: str,
    message: str,
    *,
    query: Any = "",
    k: Any = 0,
    kb_info: Optional[dict[str, Any]] = None,
    details: Optional[dict[str, Any]] = None,
) -> KBSearchResult:
    """
    Failure: still return the same FLAT schema keys.
    Error goes to meta["error"].
    """
    q = query if isinstance(query, str) else ""
    kk = k if isinstance(k, int) else 0

    return {
        "ok": False,
        "query": q,
        "k": kk,
        "kb": kb_info or {},
        "results": [],
        "meta": {
            "error": {
                "type": err_type,
                "message": message,
                "details": details or {},
            }
        },
    }


def make_kb_search_tool() -> StructuredTool:
    kb = KnowledgeBase(kb_config_from_env())

    def _kb_search(query: str, k: int = 6) -> KBSearchResult:
        """
        Tool function:
        - Validates args (lightweight)
        - Calls KB search
        - Returns stable flat schema
        """
        # 1) Lightweight validation: don't crash on bad args
        if not isinstance(query, str) or not query.strip():
            return _fail(
                err_type="kb_search.validation_error",
                message="query must be a non-empty string",
                query=query,
                k=k,
                kb_info=kb.info(),
                details={"query": query},
            )

        if not isinstance(k, int) or k <= 0:
            return _fail(
                err_type="kb_search.validation_error",
                message="k must be a positive integer",
                query=query,
                k=k,
                kb_info=kb.info(),
                details={"k": k},
            )

        # 2) Actual search with exception safety
        try:
            hits = kb.search(query, k=k)
            payload = {
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
                query=query,
                k=k,
                kb_info=kb.info(),
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
