from __future__ import annotations

import sys
import types


def _install_fake_faster_whisper() -> None:
    module = types.ModuleType("faster_whisper")

    class _FakeWhisperModel:
        def __init__(self, model_name: str, device: str, compute_type: str):
            self.model_name = model_name
            self.device = device
            self.compute_type = compute_type

    module.WhisperModel = _FakeWhisperModel
    sys.modules["faster_whisper"] = module


def test_float16_on_cpu_falls_back_to_int8(monkeypatch):
    _install_fake_faster_whisper()

    from subgen.core.asr.local_whisper import LocalWhisperASR

    asr = LocalWhisperASR(model_name="tiny", device="cpu", compute_type="float16")

    assert asr.device == "cpu"
    assert asr.compute_type == "int8"


def test_cuda_requested_but_unavailable_falls_back_to_int8(monkeypatch):
    _install_fake_faster_whisper()

    import subgen.core.asr.local_whisper as lw

    monkeypatch.setattr(lw, "_cuda_available", lambda: False)
    monkeypatch.setattr(lw, "_cuda_unavailable_reason", lambda: "mock no cuda")

    asr = lw.LocalWhisperASR(model_name="tiny", device="cuda", compute_type="float16")

    assert asr.device == "cpu"
    assert asr.compute_type == "int8"
