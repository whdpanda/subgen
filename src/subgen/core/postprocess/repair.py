from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import re
import wave

from subgen.core_types import Transcript, Segment, Word
from subgen.utils.logger import get_logger

logger = get_logger()

SPACE_RE = re.compile(r"\s+")
TAIL_PUNCT_RE = re.compile(r"[。\.！!？\?…։؛、，,;:]+$")
HYPHEN_SPACE_RE = re.compile(r"\s*-\s*")


@dataclass
class RepairConfig:
    soft_max: float = 7.0
    hard_max: float = 20.0

    suspect_dur: float = 16.0
    suspect_cps: float = 1.5

    short_min_dur: float = 6.0
    short_max_words: int = 1
    short_max_letters: int = 2
    short_max_wps: float = 0.2

    repeat_gap_tol: float = 2.0
    repeat_min_chars: int = 12
    repeat_similarity: float = 0.92
    repeat_long_chars: int = 50

    stuck_min_words: int = 12
    stuck_top1_ratio: float = 0.45
    stuck_min_run: int = 8

    backtrack: float = 0.0
    lookahead: float = 1.5

    max_actions: int = 30
    max_same_range_retry: int = 1
    range_round: float = 0.1


def _wav_duration_seconds(path: Path) -> Optional[float]:
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return float(frames) / float(rate)
    except Exception:
        return None


def _norm_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = HYPHEN_SPACE_RE.sub("-", t)
    t = SPACE_RE.sub(" ", t)
    t = TAIL_PUNCT_RE.sub("", t).strip()
    return t.lower()


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(1, max(len(a), len(b)))
    common = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            common += 1
        else:
            break
    return common / max(1, min(len(a), len(b)))


def _count_words(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    return len([x for x in SPACE_RE.split(t) if x])


def _count_letters(text: str) -> int:
    return sum(1 for ch in (text or "") if ch.isalpha())


def _has_replacement_char(text: str) -> bool:
    return "�" in (text or "")


def _is_stuck_repetition(text: str, cfg: RepairConfig) -> bool:
    t = _norm_text(text)
    if not t:
        return False
    words = [w for w in t.split(" ") if w]
    if len(words) < cfg.stuck_min_words:
        return False

    freq: Dict[str, int] = {}
    for w in words[:80]:
        freq[w] = freq.get(w, 0) + 1
    top1 = max(freq.values()) if freq else 0
    denom = max(1, min(len(words), 80))
    if (top1 / denom) >= cfg.stuck_top1_ratio:
        return True

    run = 1
    best = 1
    for i in range(1, min(len(words), 120)):
        if words[i] == words[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best >= cfg.stuck_min_run


def _is_short_low_info(seg: Segment, cfg: RepairConfig) -> bool:
    dur = float(seg.end) - float(seg.start)
    if dur < cfg.short_min_dur:
        return False
    txt = (seg.text or "").strip()
    wc = _count_words(txt)
    lc = _count_letters(txt)
    wps = (wc / dur) if dur > 0 else 0.0
    return (wc <= cfg.short_max_words) or (lc <= cfg.short_max_letters) or (wps <= cfg.short_max_wps)


def _is_suspect(seg: Segment, cfg: RepairConfig) -> bool:
    dur = float(seg.end) - float(seg.start)
    if dur < cfg.suspect_dur:
        return False
    txt = (seg.text or "").strip()
    if not txt:
        return True
    cps = (len(txt) / dur) if dur > 0 else 0.0
    return cps <= cfg.suspect_cps


def _is_repeat(prev: Segment, cur: Segment, cfg: RepairConfig) -> bool:
    if len((prev.text or "").strip()) < cfg.repeat_min_chars:
        return False
    if len((cur.text or "").strip()) < cfg.repeat_min_chars:
        return False

    a = _norm_text(prev.text)
    b = _norm_text(cur.text)
    if not a or not b:
        return False

    if (a == b or a in b or b in a) and min(len(a), len(b)) >= cfg.repeat_long_chars:
        return True

    if (float(cur.start) - float(prev.end)) > cfg.repeat_gap_tol:
        return False

    return _similarity(a, b) >= cfg.repeat_similarity


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return (a0 < b1) and (b0 < a1)


def _join_units_text(units: List[Word]) -> str:
    return "".join((u.word or "") for u in units).strip()


def _build_segments_from_units(units: List[Word], cfg: RepairConfig) -> List[Segment]:
    if not units:
        return []
    units = sorted(units, key=lambda w: (float(w.start), float(w.end)))

    segs: List[Segment] = []
    buf: List[Word] = []
    seg_start = float(units[0].start)

    def flush(end_time: float):
        nonlocal buf, seg_start
        if not buf:
            return
        text = _join_units_text(buf)
        if text:
            segs.append(Segment(start=float(seg_start), end=float(end_time), text=text, confidence=None))
        buf = []

    for u in units:
        if not buf:
            seg_start = float(u.start)
            buf.append(u)
            continue

        if (float(u.end) - float(seg_start)) > cfg.hard_max:
            flush(float(buf[-1].end))
            seg_start = float(u.start)
            buf = [u]
            continue

        buf.append(u)
        if (float(buf[-1].end) - float(seg_start)) >= cfg.soft_max:
            flush(float(buf[-1].end))

    if buf:
        flush(float(buf[-1].end))
    return segs


def _replace_by_relisten(
    *,
    asr,
    audio_path: Path,
    language: Optional[str],
    segs: List[Segment],
    idx: int,
    start: float,
    end: float,
    cfg: RepairConfig,
    segmenter_rule=None,
) -> Tuple[List[Segment], int]:
    if end <= start:
        return segs, idx + 1

    units: List[Word] = asr.transcribe_words_range(
        audio_path=audio_path,
        start=start,
        end=end,
        language=language,
        beam_size=1,
        best_of=1,
        vad_filter=False,
        temperature=0.0,
        condition_on_previous_text=False,
    )

    if not units:
        return segs, idx + 1

    if segmenter_rule is not None:
        try:
            new_segs = segmenter_rule.segment(units)
        except Exception:
            new_segs = _build_segments_from_units(units, cfg)
    else:
        new_segs = _build_segments_from_units(units, cfg)

    new_segs = [s for s in new_segs if (s.text or "").strip()]
    if not new_segs:
        return segs, idx + 1

    del_l = idx
    while del_l > 0 and _overlaps(float(segs[del_l - 1].start), float(segs[del_l - 1].end), start, end):
        del_l -= 1
    del_r = idx
    while del_r < len(segs) and _overlaps(float(segs[del_r].start), float(segs[del_r].end), start, end):
        del_r += 1

    segs = segs[:del_l] + new_segs + segs[del_r:]
    next_idx = del_l + len(new_segs)
    return segs, next_idx


def repair_transcript(
    transcript: Transcript,
    *,
    asr,
    audio_path: Path,
    language: Optional[str],
    cfg: Optional[RepairConfig] = None,
    segmenter_rule=None,
    audio_end: Optional[float] = None,
) -> Transcript:
    if cfg is None:
        cfg = RepairConfig()

    segs = list(transcript.segments or [])
    if len(segs) < 2:
        return transcript

    if audio_end is None:
        audio_end = _wav_duration_seconds(audio_path)
    if audio_end is None:
        audio_end = float(segs[-1].end)

    actions = 0
    i = 0
    seen_count: Dict[Tuple[float, float], int] = {}

    while i < len(segs) and actions < cfg.max_actions:
        cur = segs[i]
        prev = segs[i - 1] if i > 0 else None
        if prev is None:
            i += 1
            continue

        relisten_start = max(0.0, float(prev.end) - cfg.backtrack)
        next_seg = segs[i + 1] if (i + 1) < len(segs) else None
        next_start = float(next_seg.start) if next_seg is not None else float(cur.end)
        relisten_end = min(max(float(cur.end) + cfg.lookahead, next_start + 0.2), float(audio_end))


        key = (
            round(relisten_start / cfg.range_round) * cfg.range_round,
            round(relisten_end / cfg.range_round) * cfg.range_round,
        )

        def relisten(reason: str) -> bool:
            nonlocal segs, i, actions
            c = seen_count.get(key, 0)
            if c >= cfg.max_same_range_retry:
                return False
            seen_count[key] = c + 1

            logger.warning(f"[REPAIR:{reason}] idx={i} relisten {relisten_start:.2f}-{relisten_end:.2f}")
            segs, i = _replace_by_relisten(
                asr=asr,
                audio_path=audio_path,
                language=language,
                segs=segs,
                idx=i,
                start=relisten_start,
                end=relisten_end,
                cfg=cfg,
                segmenter_rule=segmenter_rule,
            )
            actions += 1
            return True

        if _has_replacement_char(cur.text) or _is_stuck_repetition(cur.text, cfg):
            if relisten("STUCK_OR_INVALID"):
                continue

        if _is_repeat(prev, cur, cfg):
            if relisten("REPEAT"):
                continue

        if _is_short_low_info(cur, cfg):
            if relisten("SHORT_LOW_INFO"):
                continue

        if _is_suspect(cur, cfg):
            if relisten("SUSPECT"):
                continue

        i += 1

    transcript.segments = segs
    return transcript


def repair_segments_with_tail_listen(
    *,
    audio_path: Path,
    segments: List[Segment],
    asr,
    language: Optional[str],
    segmenter_rule,
    soft_max: float,
    hard_max: float,
    suspect_dur: float,
    suspect_cps: float,
) -> List[Segment]:
    cfg = RepairConfig(
        soft_max=float(soft_max),
        hard_max=float(hard_max),
        suspect_dur=float(suspect_dur),
        suspect_cps=float(suspect_cps),
        backtrack=0.0,
    )

    t = Transcript(language=language or "unknown", segments=list(segments))
    t2 = repair_transcript(
        t,
        asr=asr,
        audio_path=audio_path,
        language=language,
        cfg=cfg,
        segmenter_rule=segmenter_rule,
        audio_end=_wav_duration_seconds(audio_path),
    )
    return list(t2.segments or [])
