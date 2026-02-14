# tests/test_agent_schemas.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


# ---------- A) kb_search schema ----------
def test_kb_search_tool_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    目的：防止未来把 kb_search 输出字段从 text/source/score 改名导致 RAG 注入失效。
    这里不测召回质量，只测 schema + results[0] 的 schema（只要能命中一条）。
    """
    pytest.importorskip("chromadb")

    # 1) 准备临时 KB（Chroma 持久化目录）
    persist_dir = tmp_path / "kb"
    persist_dir.mkdir(parents=True, exist_ok=True)

    from chromadb import PersistentClient  # type: ignore

    client = PersistentClient(path=str(persist_dir))
    collection_name = "subgen-kb-test"
    col = client.get_or_create_collection(collection_name)

    # 2) 插入一条最小文档（metadata 带 source）
    col.add(
        ids=["1"],
        documents=["hello subgen knowledge base"],
        metadatas=[{"source": "docs/knowledge/README.md"}],
    )

    # 3) 让 kb_config_from_env() 指向我们刚建的临时 KB
    monkeypatch.setenv("SUBGEN_KB_DIR", str(persist_dir))
    monkeypatch.setenv("SUBGEN_KB_COLLECTION", collection_name)

    # 4) 真调用 tool（你的 make_kb_search_tool() 不接收参数）
    from subgen.agent.tools.kb_search_tool import make_kb_search_tool

    tool = make_kb_search_tool()
    out = tool.invoke({"query": "hello", "k": 1})

    # 5) 顶层 schema（你真实实现返回 dict）
    assert isinstance(out, dict)
    for key in ("query", "k", "kb", "results"):
        assert key in out, f"kb_search missing top-level key: {key}"

    assert out["query"] == "hello"
    assert out["k"] == 1
    assert isinstance(out["results"], list)

    # 6) 命中至少一条时，校验 results[0] schema（你关心的点）
    assert len(out["results"]) >= 1, (
        "kb_search returned 0 hits; check SUBGEN_KB_DIR/SUBGEN_KB_COLLECTION wiring."
    )
    r0 = out["results"][0]
    assert isinstance(r0, dict)
    for key in ("source", "score", "text"):
        assert key in r0, f"kb_search result missing key: {key}"

    assert isinstance(r0["source"], str) and r0["source"]
    assert isinstance(r0["text"], str) and r0["text"]
    assert isinstance(r0["score"], (int, float))


# ---------- B) run_subgen_pipeline_tool schema ----------
def test_run_subgen_pipeline_tool_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    目的：防止未来改 run_subgen_pipeline_tool 返回字段名，导致 KB/agent 文档与实现不一致。
    不跑真实 pipeline：patch pipeline_tool 里实际调用的 pt.run_pipeline(cfg)。
    """
    import subgen.agent.tools.run_subgen_pipeline_tool as pt

    # 1) fake PipelineResult（只要满足你 tool 里用到的字段）
    class _FakeRes:
        def __init__(self, out_dir: Path) -> None:
            self.primary_path = out_dir / "video_with_subs.mp4"
            self.srt_paths = [out_dir / "a.zh.srt", out_dir / "a.bilingual.srt"]
            self.outputs = {
                "zh_srt": self.srt_paths[0],
                "bilingual_srt": self.srt_paths[1],
            }
            self.artifacts = {"log_path": out_dir / "run.log"}
            self.meta = {"lang": "zh", "translator": "nllb", "segmenter": "rule"}

    def _fake_run_pipeline(cfg: Any) -> Any:
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        return _FakeRes(out_dir)

    # 2) 关键：patch 的点是 pipeline_tool 模块里的 pt.run_pipeline
    #    因为你的文件是：from subgen.core.pipeline import run_pipeline
    monkeypatch.setattr(pt, "run_pipeline", _fake_run_pipeline, raising=True)

    # 3) 直接调用 tool entrypoint（你现在是 run_subgen_pipeline_tool(**kwargs)）
    out = pt.run_subgen_pipeline_tool(video_path=str(tmp_path / "dummy.mp4"))

    # 4) schema 断言：你的真实返回包含 ok 字段
    assert isinstance(out, dict)
    expected_keys = {"ok", "primary_path", "srt_paths", "outputs", "artifacts", "meta"}
    assert set(out.keys()) == expected_keys, f"run_subgen_pipeline_tool keys changed: {set(out.keys())}"

    assert out["ok"] is True
    assert isinstance(out["primary_path"], str)
    assert isinstance(out["srt_paths"], list) and all(isinstance(x, str) for x in out["srt_paths"])
    assert isinstance(out["outputs"], dict)
    assert isinstance(out["artifacts"], dict)
    assert isinstance(out["meta"], dict)
