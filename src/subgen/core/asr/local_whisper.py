from __future__ import annotations

import math
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List

from subgen.core.asr.base import ASRProvider
from subgen.core_types import Transcript, Segment, Word
from subgen.utils.logger import get_logger

logger = get_logger()

SPACE_RE = re.compile(r"\s+")


def _auto_device() -> str:
    try:
        import torch  # optional
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _avg_logprob_to_conf(avg_logprob: Optional[float]) -> Optional[float]:
    if avg_logprob is None:
        return None
    try:
        conf = math.exp(avg_logprob)
        return max(0.0, min(1.0, conf))
    except Exception:
        return None


def _cut_audio_range(audio_path: Path, start: float, end: float) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start}",
        "-to",
        f"{end}",
        "-i",
        str(audio_path),
        str(tmp_path),
    ]
    subprocess.run(cmd, check=True)
    return tmp_path


def _seg_has_text(s) -> bool:
    return bool((getattr(s, "text", "") or "").strip())


def _collect_words_from_fw_segments(fw_segments: List, shift: float = 0.0) -> List[Word]:
    words: List[Word] = []
    for s in fw_segments:
        ws = getattr(s, "words", None) or []
        for w in ws:
            txt = (getattr(w, "word", "") or "").strip()
            if not txt:
                continue
            words.append(
                Word(
                    start=shift + float(getattr(w, "start", 0.0)),
                    end=shift + float(getattr(w, "end", 0.0)),
                    word=getattr(w, "word", ""),
                )
            )
    words.sort(key=lambda x: (float(x.start), float(x.end)))
    return words


def _collect_pseudo_words_from_fw_segments(fw_segments: List, shift: float = 0.0) -> List[Word]:
    """
    回退方案：用 segment.text 拆 token，均分时间，生成 word-like units。
    注意：这是保底方案，稳定优先；精度取决于 fw_segments 的 start/end。
    """
    out: List[Word] = []
    first_global = True

    for s in fw_segments:
        text = (getattr(s, "text", "") or "").strip()
        if not text:
            continue

        st = shift + float(getattr(s, "start", 0.0))
        ed = shift + float(getattr(s, "end", 0.0))
        if ed <= st:
            continue

        tokens = [t for t in SPACE_RE.split(text) if t]
        if not tokens:
            tokens = [text]

        dur = ed - st
        n = len(tokens)
        step = dur / max(1, n)

        for i, tok in enumerate(tokens):
            ts = st + i * step
            te = st + (i + 1) * step
            if first_global:
                wtxt = tok
                first_global = False
            else:
                wtxt = " " + tok
            out.append(Word(start=float(ts), end=float(te), word=wtxt))

    out.sort(key=lambda x: (float(x.start), float(x.end)))
    return out


def _basic_word_stats_ok(words: List[Word]) -> bool:
    if not words or len(words) < 30:
        return False

    durs = []
    bad = 0
    for w in words:
        dur = float(w.end) - float(w.start)
        if dur <= 0 or dur > 8.0:
            bad += 1
            continue
        durs.append(dur)

    if not durs:
        return False

    bad_ratio = bad / max(1, len(words))
    p50 = statistics.median(durs)

    if bad_ratio > 0.2:
        return False
    if p50 > 1.2:
        return False
    return True


def _covered_by_words(words: List[Word], start: float, end: float) -> float:
    if end <= start or not words:
        return 0.0
    covered = 0.0
    for w in words:
        ws = float(w.start)
        we = float(w.end)
        if we <= start:
            continue
        if ws >= end:
            break
        covered += max(0.0, min(we, end) - max(ws, start))
    return covered


def _word_coverage_has_holes(
    words: List[Word],
    fw_segments: List,
    audio_end: float,
    *,
    hole_check_min_seg_dur: float = 1.5,
    hole_ratio_th: float = 0.15,
    hole_total_th_sec: float = 3.0,
    hole_total_th_ratio: float = 0.05,
) -> bool:
    """
    检查 word 覆盖是否“有洞（holes）”：
    - 仅对有文本的 fw_segments 检查覆盖率
    - 覆盖率过低的 segment 计入 hole_total
    - hole_total 超过阈值即判不可靠

    注意：这里的 fw_segments 与 words 必须在同一时间坐标系下。
    在 range 场景：它们都是 cut 音频的相对时间 (0..cut_dur)。
    """
    if audio_end <= 0 or not fw_segments:
        return False

    if not words:
        return any(_seg_has_text(s) for s in fw_segments)

    words = sorted(words, key=lambda x: (float(x.start), float(x.end)))

    hole_total = 0.0
    voiced_total = 0.0

    for s in fw_segments:
        if not _seg_has_text(s):
            continue
        st = float(getattr(s, "start", 0.0))
        ed = float(getattr(s, "end", 0.0))
        if ed <= st:
            continue

        st = max(0.0, st)
        ed = min(audio_end, ed)
        dur = ed - st
        if dur < hole_check_min_seg_dur:
            continue

        voiced_total += dur
        cov = _covered_by_words(words, st, ed)
        if (cov / max(1e-6, dur)) < hole_ratio_th:
            hole_total += dur

    if voiced_total <= 0:
        return False

    if hole_total >= hole_total_th_sec:
        return True
    if hole_total >= audio_end * hole_total_th_ratio:
        return True
    return False


def _words_are_reliable(words: List[Word], fw_segments: List, audio_end: float) -> bool:
    if not _basic_word_stats_ok(words):
        return False
    if fw_segments and audio_end > 0 and _word_coverage_has_holes(words, fw_segments, audio_end):
        return False
    return True


def _clamp_words_to_range(words: List[Word], start: float, end: float) -> List[Word]:
    """
    把 words 裁回 [start,end]。
    - 保留有交集的 word
    - start/end clamp 到边界，避免 padding 溢出
    """
    if not words or end <= start:
        return []
    out: List[Word] = []
    for w in words:
        ws = float(w.start)
        we = float(w.end)
        if we <= start:
            continue
        if ws >= end:
            break
        ns = max(ws, start)
        ne = min(we, end)
        if ne <= ns:
            continue
        w.start = float(ns)
        w.end = float(ne)
        if (w.word or "").strip():
            out.append(w)
    out.sort(key=lambda x: (float(x.start), float(x.end)))
    return out


class LocalWhisperASR(ASRProvider):
    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "auto",
        compute_type: Optional[str] = None,
        beam_size: int = 1,
        best_of: int = 1,
        vad_filter: bool = False,
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
            raise ImportError("faster-whisper is not installed. Run: pip install faster-whisper") from e

        logger.info(f"Loading faster-whisper model={self.model_name}, device={self.device}, compute_type={self.compute_type}")
        self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

    def _fw_transcribe(self, audio_path: Path, language: Optional[str]):
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            vad_filter=self.vad_filter,
            temperature=0.0,
            condition_on_previous_text=True,
            chunk_length=30,
            word_timestamps=True,
        )
        return list(segments_iter), info

    def _fw_transcribe_range(
        self,
        audio_path: Path,
        start: float,
        end: float,
        language: Optional[str],
        *,
        beam_size: int,
        best_of: int,
        vad_filter: bool,
        temperature: float,
        condition_on_previous_text: bool,
        pad_left: float = 0.5,
        pad_right: float = 0.8,
    ):
        """
        返回：
          (fw_segments, info, cut_start, cut_end)
        其中 fw_segments 的时间戳是相对 cut 音频 (0..cut_dur) 的。
        """
        if end <= start:
            return [], None, float(start), float(end)

        cut_start = max(0.0, float(start) - float(pad_left))
        cut_end = float(end) + float(pad_right)

        tmp_path = None
        try:
            tmp_path = _cut_audio_range(audio_path, cut_start, cut_end)
            segments_iter, info = self.model.transcribe(
                str(tmp_path),
                language=language,
                beam_size=beam_size,
                best_of=best_of,
                vad_filter=vad_filter,
                temperature=temperature,
                condition_on_previous_text=condition_on_previous_text,
                word_timestamps=True,
            )
            return list(segments_iter), info, float(cut_start), float(cut_end)
        finally:
            if tmp_path:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _shift_words(self, words: List[Word], shift: float) -> List[Word]:
        if not words or shift == 0.0:
            return words
        for w in words:
            w.start = float(w.start) + float(shift)
            w.end = float(w.end) + float(shift)
        words.sort(key=lambda x: (float(x.start), float(x.end)))
        return words

    def _transcribe_units_range(
        self,
        audio_path: Path,
        start: float,
        end: float,
        language: Optional[str],
        *,
        beam_size: int,
        best_of: int,
        vad_filter: bool,
        temperature: float,
        condition_on_previous_text: bool,
    ) -> List[Word]:
        if end <= start:
            return []

        # 这里 _fw_transcribe_range 返回 4 个值（含 padding 后的 cut_start/cut_end）
        fw_segments, _info, cut_start, cut_end = self._fw_transcribe_range(
            audio_path=audio_path,
            start=start,
            end=end,
            language=language,
            beam_size=beam_size,
            best_of=best_of,
            vad_filter=vad_filter,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
        )

        # IMPORTANT FIX:
        # - fw_segments times are RELATIVE to the cut audio (0..cut_dur)
        # - do reliability checks in RELATIVE time, then shift to ABSOLUTE time at the end
        cut_dur = float(cut_end) - float(cut_start)

        words_rel = _collect_words_from_fw_segments(fw_segments, shift=0.0)
        if _words_are_reliable(words_rel, fw_segments, cut_dur):
            words_abs = self._shift_words(words_rel, float(cut_start))
            return _clamp_words_to_range(words_abs, float(start), float(end))

        logger.warning("Range word timestamps unreliable/holes -> fallback pseudo-words.")
        pseudo_rel = _collect_pseudo_words_from_fw_segments(fw_segments, shift=0.0)
        pseudo_abs = self._shift_words(pseudo_rel, float(cut_start))
        return _clamp_words_to_range(pseudo_abs, float(start), float(end))

    def transcribe_words_range(
        self,
        audio_path: Path,
        start: float,
        end: float,
        language: Optional[str] = None,
        *,
        beam_size: int = 1,
        best_of: int = 1,
        vad_filter: bool = False,
        temperature: float = 0.0,
        condition_on_previous_text: bool = False,
    ) -> List[Word]:
        return self._transcribe_units_range(
            audio_path=audio_path,
            start=start,
            end=end,
            language=language,
            beam_size=beam_size,
            best_of=best_of,
            vad_filter=vad_filter,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
        )

    def transcribe(self, audio_path: Path, language: str = "auto") -> Transcript:
        logger.info(f"Transcribing -> {audio_path.name}")

        lang = None if language == "auto" else language
        fw_segments, info = self._fw_transcribe(audio_path, lang)

        detected_lang = getattr(info, "language", None) or "unknown"
        audio_end = float(getattr(fw_segments[-1], "end", 0.0)) if fw_segments else 0.0

        # 1) primary output for V1.2: words
        words = _collect_words_from_fw_segments(fw_segments, shift=0.0)
        if not _words_are_reliable(words, fw_segments, audio_end):
            logger.warning("Word timestamps unreliable/holes -> fallback pseudo-words.")
            words = _collect_pseudo_words_from_fw_segments(fw_segments, shift=0.0)

        # 2) V1.2 contract: keep segments as a minimal placeholder (pipeline will re-segment)
        segments: List[Segment] = []
        if audio_end > 0:
            segments = [Segment(start=0.0, end=audio_end, text="", confidence=None)]

        # 3) return
        return Transcript(language=detected_lang, segments=segments, words=words)
