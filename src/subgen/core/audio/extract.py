from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Literal, Optional

from subgen.utils.logger import get_logger
from subgen.utils.io import ensure_dir

logger = get_logger()

AudioPreprocess = Literal["none", "speech_filter", "demucs"]


def _run(cmd: list[str], *, env: Optional[dict[str, str]] = None) -> None:
    """
    Run a command robustly on Windows:
    - force UTF-8 env to avoid GBK unicode crashes in child python processes
    - capture output, decode safely
    """
    base_env = os.environ.copy()
    base_env["PYTHONUTF8"] = "1"
    base_env["PYTHONIOENCODING"] = "utf-8"
    if env:
        base_env.update(env)

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=base_env,
        )
    except subprocess.CalledProcessError as e:
        if e.stdout:
            logger.error(e.stdout)
        if e.stderr:
            logger.error(e.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _extract_wav(
    *,
    video_path: Path,
    out_wav: Path,
    sample_rate: int,
    ffmpeg_af: Optional[str] = None,
) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
    ]
    if ffmpeg_af:
        cmd += ["-af", ffmpeg_af]
    cmd += ["-f", "wav", str(out_wav)]

    logger.info(f"Extracting audio -> {out_wav.name}")
    _run(cmd)
    return out_wav


def _demucs_vocals(
    *,
    in_wav: Path,
    out_dir: Path,
    model: str = "htdemucs",
) -> Path:
    """
    Run demucs and return a vocals wav path.
    Windows-safe: avoid GBK UnicodeEncodeError by using ASCII temp filename + UTF-8 env.
    """
    import os
    import sys
    import shutil
    import tempfile

    demucs_root = out_dir / "demucs"
    ensure_dir(demucs_root)

    # 1) Make an ASCII-only temp copy to avoid demucs printing non-ASCII paths
    tmp_dir = out_dir / "_tmp"
    ensure_dir(tmp_dir)
    ascii_in = tmp_dir / "__demucs_input.wav"
    shutil.copyfile(in_wav, ascii_in)

    # 2) Force UTF-8 for the subprocess (helps even when other prints happen)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["LANG"] = "C.UTF-8"
    env["LC_ALL"] = "C.UTF-8"

    cmd = [
        sys.executable,   # IMPORTANT: use venv python
        "-m", "demucs",
        "-n", model,
        "--two-stems=vocals",
        "-o", str(demucs_root),
        str(ascii_in),
    ]
    logger.info(f"Demucs vocal isolation (model={model}) -> {in_wav.name}")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    except subprocess.CalledProcessError as e:
        # Print output safely (avoid re-triggering encoding problems)
        if e.stdout:
            logger.error(e.stdout.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
        if e.stderr:
            logger.error(e.stderr.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
        raise RuntimeError(f"Command failed: {' '.join(cmd)}") from e

    # 3) Locate vocals.wav robustly
    candidates = list(demucs_root.rglob("vocals.wav"))
    if not candidates:
        candidates = [p for p in demucs_root.rglob("*.wav") if p.name.lower().startswith("vocals")]
    if not candidates:
        raise RuntimeError("Demucs finished but vocals wav not found under demucs output directory.")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    vocals_src = candidates[0]

    vocals_out = out_dir / f"{in_wav.stem}.vocals.wav"
    vocals_out.write_bytes(vocals_src.read_bytes())

    logger.info(f"Using vocals audio -> {vocals_out.name}")
    return vocals_out



def extract_audio(
    video_path: Path,
    out_dir: Path,
    sample_rate: int = 16000,
    *,
    preprocess: AudioPreprocess = "none",
    demucs_model: str = "htdemucs",
) -> Path:
    """
    Extract audio for ASR.

    preprocess:
      - none: raw mono wav
      - speech_filter: lightweight speech-friendly ffmpeg filters
      - demucs: isolate vocals (best for background music)
    """
    ensure_dir(out_dir)

    base_wav = out_dir / f"{video_path.stem}.audio.wav"

    if preprocess == "none":
        return _extract_wav(video_path=video_path, out_wav=base_wav, sample_rate=sample_rate)

    if preprocess == "speech_filter":
        af = "highpass=f=80,lowpass=f=8000,dynaudnorm=f=150:g=15,afftdn=nf=-25"
        filtered_wav = out_dir / f"{video_path.stem}.audio.speech.wav"
        return _extract_wav(
            video_path=video_path,
            out_wav=filtered_wav,
            sample_rate=sample_rate,
            ffmpeg_af=af,
        )

    if preprocess == "demucs":
        wav = _extract_wav(video_path=video_path, out_wav=base_wav, sample_rate=sample_rate)
        return _demucs_vocals(in_wav=wav, out_dir=out_dir, model=demucs_model)

    raise ValueError(f"Unknown preprocess: {preprocess}")
