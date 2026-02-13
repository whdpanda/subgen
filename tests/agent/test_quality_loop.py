# tests/agent/test_quality_loop.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


from subgen.agent.loop import run_pr4c_closed_loop
from subgen.agent.tools.tool_names import (
    RUN_SUBGEN_PIPELINE,
    QUALITY_CHECK_SUBTITLES,
    FIX_SUBTITLES,
    BURN_SUBTITLES,
)


@dataclass
class _Call:
    name: str
    payload: Dict[str, Any]


class _StubTool:
    """
    Minimal tool stub that matches what loop.py needs:
    - has .name
    - has .invoke(payload) -> dict
    """

    def __init__(self, name: str, outputs: List[dict], calls: List[_Call]) -> None:
        self.name = name
        self._outputs = outputs
        self._i = 0
        self._calls = calls

    def invoke(self, payload: Dict[str, Any]) -> dict:
        self._calls.append(_Call(self.name, dict(payload)))
        if self._i >= len(self._outputs):
            # default: repeat last output
            return self._outputs[-1]
        out = self._outputs[self._i]
        self._i += 1
        return out


def test_pr4c_loop_success_then_burn(tmp_path: Path) -> None:
    calls: List[_Call] = []

    video_path = str(tmp_path / "v.mp4")
    srt0 = str(tmp_path / "a.zh.srt")
    srt_fixed = str(tmp_path / "a.zh.fixed.srt")
    rep1 = str(tmp_path / "r1.json")
    rep2 = str(tmp_path / "r2.json")
    out_video = str(tmp_path / "burned.mp4")

    pipeline_tool = _StubTool(
        RUN_SUBGEN_PIPELINE,
        outputs=[
            {
                "ok": True,
                "primary_path": srt0,
                "srt_paths": [srt0],
                "outputs": {"mono_srt_path": srt0},
                "artifacts": {},
                "meta": {},
            }
        ],
        calls=calls,
    )

    quality_tool = _StubTool(
        QUALITY_CHECK_SUBTITLES,
        outputs=[
            {
                "ok": False,
                "report_path": rep1,
                "report": {"dummy": 1},
                "summary": {"major_count": 1},
                "meta": {"srt_path": srt0, "profile": {"name": "default"}},
            },
            {
                "ok": True,
                "report_path": rep2,
                "report": {"dummy": 2},
                "summary": {"major_count": 0},
                "meta": {"srt_path": srt_fixed, "profile": {"name": "default"}},
            },
        ],
        calls=calls,
    )

    fix_tool = _StubTool(
        FIX_SUBTITLES,
        outputs=[
            {
                "ok": True,
                "fixed_srt_path": srt_fixed,
                "changed": True,
                "actions": ["fix_wrap"],
                "meta": {},
            }
        ],
        calls=calls,
    )

    burn_tool = _StubTool(
        BURN_SUBTITLES,
        outputs=[
            {
                "ok": True,
                "out_video_path": out_video,
                "artifacts": {},
                "meta": {},
            }
        ],
        calls=calls,
    )

    tool_map = {
        RUN_SUBGEN_PIPELINE: pipeline_tool,
        QUALITY_CHECK_SUBTITLES: quality_tool,
        FIX_SUBTITLES: fix_tool,
        BURN_SUBTITLES: burn_tool,
    }

    res = run_pr4c_closed_loop(
        tool_map,
        pipeline_args={
            "video_path": video_path,
            "out_dir": str(tmp_path / "out"),
            # do not need to specify emit/zh_layout here; loop injects defaults
        },
        burn_args={
            "video_path": video_path,
        },
        max_passes=3,
    )

    assert res.ok is True
    assert res.srt_path == srt_fixed
    assert res.report_path == rep2
    assert res.out_video_path == out_video
    assert res.passes_used == 1

    # Ensure call order: pipeline -> quality -> fix -> quality -> burn
    assert [c.name for c in calls] == [
        RUN_SUBGEN_PIPELINE,
        QUALITY_CHECK_SUBTITLES,
        FIX_SUBTITLES,
        QUALITY_CHECK_SUBTITLES,
        BURN_SUBTITLES,
    ]

    # Ensure burn uses final_srt, not the initial one
    burn_payload = calls[-1].payload
    assert burn_payload["video_path"] == video_path
    assert burn_payload["srt_path"] == srt_fixed


def test_pr4c_loop_hits_max_passes_best_effort_report(tmp_path: Path) -> None:
    calls: List[_Call] = []

    video_path = str(tmp_path / "v.mp4")
    srt0 = str(tmp_path / "a.zh.srt")
    srt_fixed = str(tmp_path / "a.zh.fixed.srt")
    rep1 = str(tmp_path / "r1.json")
    rep2 = str(tmp_path / "r2.json")

    pipeline_tool = _StubTool(
        RUN_SUBGEN_PIPELINE,
        outputs=[
            {
                "ok": True,
                "primary_path": srt0,
                "srt_paths": [srt0],
                "outputs": {"mono_srt_path": srt0},
                "artifacts": {},
                "meta": {},
            }
        ],
        calls=calls,
    )

    # Always fails quality check
    quality_tool = _StubTool(
        QUALITY_CHECK_SUBTITLES,
        outputs=[
            {"ok": False, "report_path": rep1, "report": {}, "summary": {"major_count": 1}, "meta": {}},
            {"ok": False, "report_path": rep2, "report": {}, "summary": {"major_count": 1}, "meta": {}},
        ],
        calls=calls,
    )

    fix_tool = _StubTool(
        FIX_SUBTITLES,
        outputs=[
            {"ok": True, "fixed_srt_path": srt_fixed, "changed": True, "actions": ["fix_wrap"], "meta": {}}
        ],
        calls=calls,
    )

    tool_map = {
        RUN_SUBGEN_PIPELINE: pipeline_tool,
        QUALITY_CHECK_SUBTITLES: quality_tool,
        FIX_SUBTITLES: fix_tool,
    }

    res = run_pr4c_closed_loop(
        tool_map,
        pipeline_args={"video_path": video_path, "out_dir": str(tmp_path / "out")},
        max_passes=1,
        burn_args=None,
    )

    assert res.ok is False
    # best-effort: final_srt is whatever last fix produced
    assert res.srt_path in (srt0, srt_fixed)
    # MUST have best-effort report_path from the last quality check
    assert res.report_path == rep2
    assert res.out_video_path is None
    assert res.passes_used == 1

    assert [c.name for c in calls] == [
        RUN_SUBGEN_PIPELINE,
        QUALITY_CHECK_SUBTITLES,
        FIX_SUBTITLES,
        QUALITY_CHECK_SUBTITLES,
    ]
