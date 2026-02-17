from pathlib import Path
from subgen.core.pipeline import PipelineConfig

def test_pipeline_api_importable(tmp_path: Path):
    cfg = PipelineConfig(
        video_path=tmp_path / "dummy.mp4",
        out_dir=tmp_path / "out",
        language="en",
        target_lang="zh",
    )
    assert cfg.language == "en"


def test_pipeline_config_normalizes_asr_knobs(tmp_path: Path):
    cfg = PipelineConfig(
        video_path=tmp_path / "dummy.mp4",
        out_dir=tmp_path / "out",
        asr_device=" CuDa:0 ",
        asr_compute_type=" FP16 ",
    )

    assert cfg.asr_device == "cuda:0"
    assert cfg.asr_compute_type == "float16"


def test_pipeline_config_forces_int8_on_cpu_fp16(tmp_path: Path):
    cfg = PipelineConfig(
        video_path=tmp_path / "dummy.mp4",
        out_dir=tmp_path / "out",
        asr_device=" cpu ",
        asr_compute_type=" float16 ",
    )

    assert cfg.asr_device == "cpu"
    assert cfg.asr_compute_type == "int8"
