# src/subgen/agent/tools/tool_utils.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def now_iso() -> str:
    """UTC ISO8601 timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def abs_path(p: Path) -> Path:
    """Resolve path to an absolute path (relative paths are resolved from cwd)."""
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def resolve_out_dir(
    *,
    requested_out_dir: Optional[Path],
    fallback_parent: Optional[Path] = None,
    default_dirname: str = "out",
) -> Path:
    """
    Best-effort output directory resolver.

    Priority:
      1) requested_out_dir if provided
      2) fallback_parent if provided (e.g. srt_path.parent)
      3) cwd/<default_dirname>

    Always creates the directory (mkdir parents=True, exist_ok=True).
    """
    if requested_out_dir is not None:
        out_dir = abs_path(requested_out_dir)
    elif fallback_parent is not None:
        out_dir = fallback_parent.resolve()
    else:
        out_dir = (Path.cwd() / default_dirname).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def fail_flat(
    *,
    base: dict[str, Any],
    err_type: str,
    message: str,
    details: Optional[dict[str, Any]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build a flat-schema failure payload with a normalized meta.error object.

    Caller provides `base` with tool-specific keys, e.g.
      - {"report_path":..., "report":..., "summary":...}
      - {"fixed_srt_path":..., "quality_before_path":..., ...}

    Returns:
      {"ok": False, **base, "meta": {..., "error": {...}}}
    """
    m = dict(meta or {})
    m["error"] = {
        "type": err_type,
        "message": message,
        "details": details or {},
    }
    return {"ok": False, **base, "meta": m}


def path_to_str(value: Any) -> Any:
    """Convert Path to str while keeping non-Path values unchanged."""
    return str(value) if isinstance(value, Path) else value


def path_values_to_str(payload: dict[str, Any]) -> dict[str, Any]:
    """Shallow-convert Path values in a dict into strings for JSON-safe tool output."""
    return {k: path_to_str(v) for k, v in payload.items()}
