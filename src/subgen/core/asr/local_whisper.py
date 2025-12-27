from __future__ import annotations

from pathlib import Path
from typing import Optional
import re
import math
import subprocess
import tempfile 

from subgen.core_types import Transcript, Segment
from subgen.core.asr.base import ASRProvider
from subgen.utils.logger import get_logger

logger = get_logger()

# ---- 可疑长段的判定阈值（你可以以后根据经验微调）----
SUSPECT_DURATION = 20.0    # 持续时间 >= 20 秒
SUSPECT_MAX_CHARS = 150     # 但只说了很短的一两句（字数 <= 150）


def _auto_device() -> str:
    try:
        import torch  # optional
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _avg_logprob_to_conf(avg_logprob: Optional[float]) -> Optional[float]:
    """
    faster-whisper 的 avg_logprob → 粗略 [0,1] 置信度
    """
    if avg_logprob is None:
        return None
    try:
        conf = math.exp(avg_logprob)
        conf = max(0.0, min(1.0, conf))
        return conf
    except Exception:
        return None


def _norm_for_repeat(text: str) -> str:
    """
    用于重复检测 / 相似度判断的归一化：
    - 去首尾空格
    - 合并中间连续空格
    - 去掉句末标点
    - 转小写
    """
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[，。\.！!？?、…：:;]+$", "", t)
    return t.lower()


def _is_suspect_segment(seg: Segment) -> bool:
    """
    判定“可疑长段”：
    - 时间很长
    - 但文本很短（可能是 ASR 听崩，只留了一句废话）
    """
    duration = seg.end - seg.start
    if duration < SUSPECT_DURATION:
        return False
    text = (seg.text or "").strip()
    if len(text) > SUSPECT_MAX_CHARS:
        return False
    return True


def _retry_segment_with_different_params(
    model,
    audio_path: Path,
    seg: Segment,
    language: str | None,
) -> list[Segment] | None:
    """
    对“可疑长段”重新做一次 ASR：
    - 只截取 seg 时间范围的音频
    - 用另一套参数（不同于全局 transcribe）
    - 返回新的 Segment 列表（时间按原始 seg 对齐）
    """
    start = float(seg.start)
    end = float(seg.end)
    if end <= start:
        return None

    # 1) 用 ffmpeg 截取这一小段音频到临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{start}",
        "-to", f"{end}",
        "-i", str(audio_path),
        str(tmp_path),
    ]

    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        logger.warning(f"ffmpeg cut failed for suspect segment: {e}")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    try:
        # 2) 用一套“更宽松”的参数重跑这一小段
        segments_iter, _info = model.transcribe(
            str(tmp_path),
            language=language,
            beam_size=1,          # 降低 beam，避免长时间卡在一个假设
            best_of=1,
            vad_filter=False,     # 不要再做 VAD，尽量把能听到的都吐出来
            temperature=0.4,      # 稍微放一点随机性
        )

        new_segs: list[Segment] = []
        for s in segments_iter:
            text = (s.text or "").strip()
            if not text:
                continue
            conf = _avg_logprob_to_conf(getattr(s, "avg_logprob", None))

            # 注意：这里 s.start/s.end 是“子片段内部时间”，要平移回原始时间轴
            new_segs.append(
                Segment(
                    start=start + float(s.start),
                    end=start + float(s.end),
                    text=text,
                    confidence=conf,
                )
            )

        if not new_segs:
            return None

        return new_segs

    except Exception as e:
        logger.warning(f"retry ASR on suspect segment failed: {e}")
        return None
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

class LocalWhisperASR(ASRProvider):
    """
    Real ASR implementation using faster-whisper.

    High-accuracy-first defaults:
    - beam search
    - VAD filter enabled
    - conservative temperature
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "auto",
        compute_type: Optional[str] = None,
        beam_size: int = 5,
        best_of: int = 5,
        vad_filter: bool = True,
    ):
        self.model_name = model_name
        self.device = _auto_device() if device == "auto" else device

        if compute_type is None:
            compute_type = "float16" if self.device == "cuda" else "int8"
        self.compute_type = compute_type

        self.beam_size = beam_size
        self.best_of = best_of
        self.vad_filter = vad_filter

        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            ) from e

        logger.info(
            f"Loading faster-whisper model={self.model_name}, "
            f"device={self.device}, compute_type={self.compute_type}"
        )
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio_path: Path, language: str = "auto") -> Transcript:
        """
        Transcribe audio into segments with timestamps.
        language:
          - 'auto' lets model detect language
          - or pass e.g. 'ja', 'ko', 'fr', etc.
        """
        logger.info(f"Transcribing -> {audio_path.name}")

        # 1) 第一次全局转录（较保守参数）
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=None if language == "auto" else language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            vad_filter=self.vad_filter,
            temperature=0.0,              # 减少随机性
            condition_on_previous_text=True,
            chunk_length=30,              # 内部以 30 秒为粒度切块
        )

        # 2) 收集并做“相邻重复合并”
        merged: list[Segment] = []

        for s in segments_iter:
            text = (s.text or "").strip()
            if not text:
                continue

            conf = _avg_logprob_to_conf(getattr(s, "avg_logprob", None))
            norm = _norm_for_repeat(text)

            if merged:
                last = merged[-1]
                last_norm = _norm_for_repeat(last.text)

                close_in_time = float(s.start) - last.end <= 2.0
                similar_text = (
                    norm == last_norm
                    or norm in last_norm
                    or last_norm in norm
                )

                if close_in_time and similar_text:
                    # 合并成一条更长的 segment
                    last.end = float(s.end)
                    if conf is not None:
                        if last.confidence is None:
                            last.confidence = conf
                        else:
                            last.confidence = max(last.confidence, conf)
                    continue

            merged.append(
                Segment(
                    start=float(s.start),
                    end=float(s.end),
                    text=text,
                    confidence=conf,
                )
            )

        # 3) 对“可疑长段”做二次 ASR 重试
        final_segs: list[Segment] = []
        # 确定重试时用的语言：如果 language="auto"，就用 info.language
        retry_lang = None if language == "auto" else language
        if retry_lang is None:
            retry_lang = getattr(info, "language", None)

        for seg in merged:
            if _is_suspect_segment(seg) and retry_lang is not None:
                logger.info(
                    f"Suspect long segment detected "
                    f"(dur={seg.end - seg.start:.1f}s, text='{seg.text[:40]}...'), retrying ASR..."
                )
                improved = _retry_segment_with_different_params(
                    self.model,
                    audio_path,
                    seg,
                    retry_lang,
                )
                if improved:
                    logger.info(
                        f"  -> Retry succeeded, replaced 1 segment with {len(improved)} segments."
                    )
                    final_segs.extend(improved)
                    continue  # 不保留原 seg，直接用改进后的
                else:
                    logger.info("  -> Retry failed, keep original segment.")

            # 非可疑段 / 重试失败 → 原样加入
            final_segs.append(seg)

        detected_lang = getattr(info, "language", None) or "unknown"
        return Transcript(language=detected_lang, segments=final_segs)

