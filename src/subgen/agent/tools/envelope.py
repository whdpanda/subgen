# subgen/agent/tools/envelope.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Optional, TypeVar
import traceback

T = TypeVar("T")


def _jsonable(x: Any) -> Any:
    """尽量把对象变成可 JSON 序列化的结构，避免 details 里塞进怪对象。"""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool, list, dict)):
        return x
    if is_dataclass(x):
        return asdict(x)
    # pydantic v2 / v1 兼容：有 model_dump / dict 就用
    if hasattr(x, "model_dump") and callable(getattr(x, "model_dump")):
        return x.model_dump()
    if hasattr(x, "dict") and callable(getattr(x, "dict")):
        return x.dict()
    return str(x)


def ok(data: Any) -> Dict[str, Any]:
    return {"ok": True, "data": _jsonable(data), "error": None}


def fail(err_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": {
            "type": err_type,
            "message": message,
            "details": _jsonable(details or {}),
        },
    }


def tool_enveloped(tool_name: str) -> Callable[[Callable[..., T]], Callable[..., Dict[str, Any]]]:
    """
    decorator：保证 tool 永远返回 envelope，而不是抛异常把 runtime 炸掉
    """
    def _decorator(fn: Callable[..., T]) -> Callable[..., Dict[str, Any]]:
        def _wrapped(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            try:
                out = fn(*args, **kwargs)
                # 如果你想允许工具内部已经返回 envelope，就透传：
                if isinstance(out, dict) and set(out.keys()) >= {"ok", "data", "error"}:
                    return out
                return ok(out)
            except Exception as e:
                return fail(
                    err_type=f"{tool_name}.exception",
                    message=str(e) or e.__class__.__name__,
                    details={
                        "exception_class": e.__class__.__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
        return _wrapped
    return _decorator
