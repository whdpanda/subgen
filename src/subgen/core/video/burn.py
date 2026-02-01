from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from subgen.utils.io import ensure_dir
from subgen.utils.logger import get_logger

logger = get_logger()


def _ffmpeg_filter_escape_path(p: Path) -> str:
    """
    Escape file path for ffmpeg subtitles filter argument.

    Notes:
    - Use forward slashes on Windows.
    - Escape ':' in drive letter as '\\:'.
    - Escape single quote as "\\'".
    - Escape filtergraph-sensitive chars: ',', '[', ']'.
    """
    s = str(p).replace("\\", "/")

    # Windows drive: C:/... -> C\:/...
    s = s.replace(":", r"\:")

    # Quotes
    s = s.replace("'", r"\'")

    # Filtergraph-sensitive chars
    s = s.replace(",", r"\,")
    s = s.replace("[", r"\[")
    s = s.replace("]", r"\]")

    return s


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
    """
    Burn subtitles into video (hard-sub).
    - Video: libx264, CRF/preset configurable.
    - Audio: default to AAC re-encode for maximum mp4 compatibility.

    force_style example (ASS style string):
      "FontName=Arial,FontSize=16,BorderStyle=1,Outline=1,Shadow=0"
    """
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not srt_path.exists():
        raise FileNotFoundError(srt_path)

    ensure_dir(out_path.parent)

    sub = _ffmpeg_filter_escape_path(srt_path)

    # Use single quotes around the path; we escaped inner quotes already.
    vf = f"subtitles='{sub}'"

    if force_style:
        # Escape characters that can break filtergraph parsing.
        safe_style = (
            force_style
            .replace("\\", "/")
            .replace("'", r"\'")
            .replace(",", r"\,")
            .replace("[", r"\[")
            .replace("]", r"\]")
        )
        vf += f":force_style='{safe_style}'"

    cmd = [
        ffmpeg_bin,
        "-y",
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
        # For AAC, bitrate is the common knob. If empty, omit.
        if audio_bitrate:
            cmd += ["-b:a", audio_bitrate]

    cmd += [str(out_path)]

    logger.info("ffmpeg burn cmd: %s", " ".join(cmd))

    # Windows robustness: force UTF-8 env to avoid encoding issues
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        raise RuntimeError(
            "ffmpeg burn failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout: {stdout[:2000]}\n"
            f"stderr: {stderr[:4000]}"
        ) from e

    return out_path
