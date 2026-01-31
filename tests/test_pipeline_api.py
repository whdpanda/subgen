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
