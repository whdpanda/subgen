# tests/tools/test_quality_tool_contract.py
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _fixtures_dir() -> Path:
    # tests/tools/ -> tests/ -> fixtures/
    return Path(__file__).resolve().parents[1] / "fixtures"


def test_quality_check_subtitles_tool_returns_report_path_and_file(tmp_path: Path) -> None:
    from subgen.agent.tools.quality_check_subtitles_tool import quality_check_subtitles_tool  # noqa: WPS433

    srt = _fixtures_dir() / "bad.srt"
    assert srt.exists(), f"fixture missing: {srt}"

    out = quality_check_subtitles_tool(srt_path=str(srt), out_dir=str(tmp_path))

    assert isinstance(out, dict)
    expected_keys = {"ok", "report_path", "report", "summary", "meta"}
    assert set(out.keys()) == expected_keys, f"quality_check_subtitles keys changed: {set(out.keys())}"

    rp = out["report_path"]
    assert isinstance(rp, str) and rp, "report_path must be a non-empty string"
    rp_path = Path(rp)
    assert rp_path.exists(), "report_path must point to an existing file"
    assert rp_path.is_file()

    # Report must be valid JSON
    data = json.loads(rp_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "summary" in data


def test_quality_check_subtitles_tool_best_effort_report_on_missing_srt(tmp_path: Path) -> None:
    """
    PR#4c hard requirement:
    - even if srt_path is invalid / missing, tool MUST return best-effort report_path.
    - report_path MUST exist (tool writes a failure report json).
    """
    from subgen.agent.tools.quality_check_subtitles_tool import quality_check_subtitles_tool  # noqa: WPS433

    missing = tmp_path / "missing.srt"
    assert not missing.exists()

    out = quality_check_subtitles_tool(srt_path=str(missing), out_dir=str(tmp_path))

    assert isinstance(out, dict)
    assert out.get("ok") is False

    rp = out.get("report_path")
    assert isinstance(rp, str) and rp, "best-effort: report_path must still be returned on failure"
    rp_path = Path(rp)
    assert rp_path.exists() and rp_path.is_file(), "best-effort: report_path must exist on disk"

    # Should contain error info in meta and/or report file
    meta = out.get("meta") or {}
    # meta["error"] is preferred (flat schema convention)
    assert "error" in meta or "input" in json.loads(rp_path.read_text(encoding="utf-8"))
