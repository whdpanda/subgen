from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from subgen.utils.io import ensure_dir
from subgen.utils.logger import get_logger

logger = get_logger()


def _ffmpeg_filter_escape_value(s: str) -> str:
    """
    Escape a string that will be embedded into an ffmpeg filtergraph argument value
    and wrapped by single quotes.

    Handles Windows paths and filtergraph-sensitive characters.
    """
    # Normalize Windows path separators early
    s = s.replace("\\", "/")

    # Filtergraph escape (order matters a bit; keep it consistent)
    s = s.replace("\\", r"\\")   # robustness (even after normalization)
    s = s.replace(":", r"\:")    # drive letter C:
    s = s.replace("'", r"\'")    # because we wrap with single quotes
    s = s.replace(",", r"\,")    # filterchain separator
    s = s.replace("[", r"\[")
    s = s.replace("]", r"\]")
    return s


def _decode_ffmpeg_bytes(b: bytes) -> str:
    """Decode ffmpeg output robustly on Windows/macOS/Linux."""
    if not b:
        return ""
    # Try utf-8 first; fallback to Windows ANSI codepage
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("mbcs", errors="replace")


def burn_subtitles(
    video_path: Path,
    srt_path: Path,
    out_path: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    force_style: Optional[str] = None,
    crf: int = 18,
    preset: str = "veryfast",
    copy_audio: bool = False,
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not srt_path.exists():
        raise FileNotFoundError(srt_path)

    ensure_dir(out_path.parent)

    # 1) Build vf with the most stable syntax: subtitles=filename='...'
    sub = _ffmpeg_filter_escape_value(str(srt_path))
    vf = f"subtitles=filename='{sub}'"

    if force_style:
        safe_style = _ffmpeg_filter_escape_value(force_style)
        vf += f":force_style='{safe_style}'"

    # 2) Force low-noise, agent-friendly ffmpeg flags
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
    ]

    if copy_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", audio_codec]
        if audio_bitrate:
            cmd += ["-b:a", audio_bitrate]

    cmd += [str(out_path)]

    logger.info("ffmpeg burn cmd: %s", " ".join(cmd))

    # 3) Env: keep, but don't rely on it for ffmpeg's encoding
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")

    try:
        # 4) Capture bytes and decode ourselves (prevents UnicodeDecodeError)
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
    except FileNotFoundError as e:
        # ffmpeg not found / wrong path
        raise RuntimeError(f"ffmpeg not found: {ffmpeg_bin}") from e

    if proc.returncode != 0:
        stdout = _decode_ffmpeg_bytes(proc.stdout).strip()
        stderr = _decode_ffmpeg_bytes(proc.stderr).strip()
        raise RuntimeError(
            "ffmpeg burn failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout: {stdout[:2000]}\n"
            f"stderr: {stderr[:4000]}"
        )

    return out_path
