from __future__ import annotations

from pathlib import Path

from subgen.core.preprocess.demucs import DemucsPreprocessor


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"wav")


def test_pick_outputs_prefers_matching_input_stem(tmp_path: Path) -> None:
    root = tmp_path / "preprocess" / "demucs"

    # Existing old run for another stem
    _touch(root / "htdemucs" / "old_track" / "vocals.wav")
    _touch(root / "htdemucs" / "old_track" / "no_vocals.wav")

    # Current run for target stem
    _touch(root / "htdemucs" / "target_track" / "vocals.wav")
    _touch(root / "htdemucs" / "target_track" / "accompaniment.wav")

    proc = DemucsPreprocessor()
    vocals, other, meta = proc._pick_outputs(root, input_stem="target_track")

    assert vocals is not None
    assert vocals.parent.name == "target_track"
    assert vocals.name == "vocals.wav"

    assert other is not None
    assert other.name == "accompaniment.wav"

    assert meta["found"] is True
    assert meta["prefer_input_stem"] == "target_track"


def test_pick_outputs_uses_newest_candidate_when_no_stem_match(tmp_path: Path) -> None:
    root = tmp_path / "preprocess" / "demucs"

    older = root / "htdemucs" / "track_a" / "vocals.wav"
    newer = root / "htdemucs" / "track_b" / "vocals.wav"
    _touch(older)
    _touch(newer)

    proc = DemucsPreprocessor()
    vocals, _other, meta = proc._pick_outputs(root, input_stem="missing_stem")

    assert vocals is not None
    assert vocals == newer
    assert meta["found"] is True
