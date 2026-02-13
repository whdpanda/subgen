from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from subgen.service.app import app


def test_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_run_pipeline_endpoint(monkeypatch, tmp_path: Path):
    class DummyResult:
        primary_path = tmp_path / "out.zh.srt"
        srt_paths = [primary_path]
        outputs = {"primary": primary_path}
        artifacts = {"audio_path": tmp_path / "audio.wav"}
        meta = {"emit": "zh-only", "asr_cache_hit": False}

    def fake_run_pipeline(cfg):
        assert cfg.video_path == tmp_path / "a.mp4"
        assert cfg.out_dir == tmp_path / "out"
        return DummyResult()

    monkeypatch.setattr("subgen.service.app.run_pipeline", fake_run_pipeline)

    client = TestClient(app)
    payload = {
        "video_path": str(tmp_path / "a.mp4"),
        "out_dir": str(tmp_path / "out"),
    }
    resp = client.post("/v1/pipeline/run", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["primary_path"].endswith("out.zh.srt")
    assert data["artifacts"]["audio_path"].endswith("audio.wav")
