import subprocess
from pathlib import Path
from subgen.utils.logger import get_logger
from subgen.utils.io import ensure_dir

logger = get_logger()

def extract_audio(
    video_path: Path,
    out_dir: Path,
    sample_rate: int = 16000,
) -> Path:
    """
    Extract mono wav audio from video using ffmpeg.
    Output: {stem}.wav
    """
    ensure_dir(out_dir)
    audio_path = out_dir / f"{video_path.stem}.audio.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",                 # no video
        "-ac", "1",            # mono
        "-ar", str(sample_rate),
        "-f", "wav",
        str(audio_path),
    ]

    logger.info(f"Extracting audio -> {audio_path.name}")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr.decode(errors="ignore"))
        raise RuntimeError("ffmpeg failed. Make sure ffmpeg is installed and in PATH.")

    return audio_path
