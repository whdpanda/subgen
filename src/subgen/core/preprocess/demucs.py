from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from subgen.core.preprocess.base import PreprocessResult, PreprocessSpec


class DemucsPreprocessor:
    """
    Demucs preprocessor.

    Output layout (stable, PR6-friendly):
      <out_dir>/preprocess/demucs/
        vocals.wav
        no_vocals.wav   (or accompaniment.wav depending on demucs)
        meta.json       (optional if you later persist)
    """

    name = "demucs"

    def _pick_outputs(self, demucs_out_root: Path, input_stem: str) -> Tuple[Optional[Path], Optional[Path], Dict[str, Any]]:
        """
        Demucs output folder usually looks like:
          <out_root>/<model>/<track_stem>/{vocals.wav, no_vocals.wav or accompaniment.wav}
        We best-effort locate vocals and the other stem.
        """
        meta: Dict[str, Any] = {"search_root": str(demucs_out_root)}

        if not demucs_out_root.exists():
            meta["found"] = False
            return None, None, meta

        # Find best matching folder containing vocals.wav
        vocals = None
        other = None

        # Fast path: most demucs versions output <root>/<model>/<input_stem>/vocals.wav
        stem_candidates = list(demucs_out_root.glob(f"*/{input_stem}/vocals.wav"))
        candidates = stem_candidates or list(demucs_out_root.rglob("vocals.wav"))
        # Prefer newest artifact in case preprocess dir already contains older runs.
        # Use nanosecond mtimes and ctimes to avoid coarse timestamp ties on some filesystems.
        candidates.sort(
            key=lambda p: (
                p.stat().st_mtime_ns,
                p.stat().st_ctime_ns,
                str(p),
            ),
            reverse=True,
        )

        meta["vocals_candidates"] = [str(p) for p in candidates[:20]]
        meta["prefer_input_stem"] = input_stem
        if candidates:
            vocals = candidates[0]

            # Prefer no_vocals.wav, else accompaniment.wav
            no_vocals = vocals.with_name("no_vocals.wav")
            accomp = vocals.with_name("accompaniment.wav")
            if no_vocals.exists():
                other = no_vocals
            elif accomp.exists():
                other = accomp
            else:
                # best-effort: pick any other wav in same folder that isn't vocals
                wavs = [p for p in vocals.parent.glob("*.wav") if p.name != "vocals.wav"]
                other = wavs[0] if wavs else None

        meta["found"] = bool(vocals)
        meta["vocals"] = str(vocals) if vocals else None
        meta["other"] = str(other) if other else None
        return vocals, other, meta

    def run(self, *, audio_in_path: str, out_dir: str, spec: PreprocessSpec) -> PreprocessResult:
        t0 = time.time()

        audio_in = Path(audio_in_path).expanduser().resolve()
        out_root = Path(out_dir).expanduser().resolve()
        stage_dir = out_root / "preprocess" / "demucs"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # If demucs not installed, raise a clean error
        # (we don't import demucs directly; we check CLI availability by running python -m)
        python_exe = shutil.which("python") or "python"

        # Cache dir: prefer spec.cache_dir; else env; else None
        cache_dir = spec.cache_dir or os.getenv("SUBGEN_CACHE_DIR") or os.getenv("TORCH_HOME") or None

        # Demucs CLI:
        # python -m demucs.separate -o <stage_dir> -n <model> --two-stems <stems> <audio_in>
        cmd = [
            python_exe,
            "-m",
            "demucs.separate",
            "-o",
            str(stage_dir),
            "-n",
            str(spec.model or "htdemucs"),
            "--two-stems",
            str(spec.stems or "vocals"),
        ]

        # device flag best-effort (supported in most demucs releases)
        # If unsupported, demucs will exit non-zero; caller will see stderr in meta.
        if spec.device:
            cmd += ["--device", str(spec.device)]

        # extra params: pass-through only for simple scalar flags, e.g. {"shifts": 1, "overlap": 0.25}
        for k, v in (spec.params or {}).items():
            if v is None:
                continue
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            else:
                cmd += [flag, str(v)]

        cmd.append(str(audio_in))

        env = os.environ.copy()
        if cache_dir:
            # torch cache location (best-effort)
            env.setdefault("TORCH_HOME", cache_dir)

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except Exception as e:
            return PreprocessResult(
                ok=False,
                audio_path_for_asr=str(audio_in),
                artifacts={"stage_dir": str(stage_dir)},
                meta={
                    "preprocess": "demucs",
                    "error": {"type": "demucs.spawn_failed", "message": str(e) or e.__class__.__name__},
                    "cmd": cmd,
                },
            )

        elapsed = time.time() - t0

        meta: Dict[str, Any] = {
            "preprocess": "demucs",
            "cmd": cmd,
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 3),
            "stdout_tail": (proc.stdout[-4000:] if proc.stdout else ""),
            "stderr_tail": (proc.stderr[-4000:] if proc.stderr else ""),
            "cache_dir": cache_dir,
        }

        if proc.returncode != 0:
            # demucs failed, keep pipeline alive (best-effort) by returning original audio path
            meta["error"] = {"type": "demucs.failed", "message": "demucs command returned non-zero"}
            return PreprocessResult(
                ok=False,
                audio_path_for_asr=str(audio_in),
                artifacts={"stage_dir": str(stage_dir)},
                meta=meta,
            )

        # Locate outputs
        vocals, other, find_meta = self._pick_outputs(stage_dir, audio_in.stem)
        meta["find"] = find_meta

        if not vocals or not vocals.exists():
            meta["error"] = {"type": "demucs.no_vocals", "message": "cannot find vocals.wav in demucs output"}
            return PreprocessResult(
                ok=False,
                audio_path_for_asr=str(audio_in),
                artifacts={"stage_dir": str(stage_dir)},
                meta=meta,
            )

        artifacts: Dict[str, Any] = {
            "stage_dir": str(stage_dir),
            "vocals_path": str(vocals),
        }
        if other and other.exists():
            artifacts["other_path"] = str(other)

        return PreprocessResult(
            ok=True,
            audio_path_for_asr=str(vocals),
            artifacts=artifacts,
            meta=meta,
        )
