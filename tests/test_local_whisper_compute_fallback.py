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


def test_compute_type_with_spaces_and_uppercase_is_normalized_then_falls_back(monkeypatch):
    _install_fake_faster_whisper()

    from subgen.core.asr.local_whisper import LocalWhisperASR

    asr = LocalWhisperASR(model_name="tiny", device=" CPU ", compute_type=" FP16 ")

    assert asr.device == "cpu"
    assert asr.compute_type == "int8"


def test_cuda_init_failure_retries_with_cpu_int8(monkeypatch):
    module = types.ModuleType("faster_whisper")

    class _FlakyWhisperModel:
        calls = []

        def __init__(self, model_name: str, device: str, compute_type: str):
            self.__class__.calls.append((device, compute_type))
            if device == "cuda" and compute_type == "float16":
                raise RuntimeError("mock cuda init failure")

    module.WhisperModel = _FlakyWhisperModel
    sys.modules["faster_whisper"] = module

    import subgen.core.asr.local_whisper as lw

    monkeypatch.setattr(lw, "_cuda_available", lambda: True)

    asr = lw.LocalWhisperASR(model_name="tiny", device="cuda", compute_type="float16")

    assert asr.device == "cpu"
    assert asr.compute_type == "int8"
    assert _FlakyWhisperModel.calls == [("cuda", "float16"), ("cpu", "int8")]
