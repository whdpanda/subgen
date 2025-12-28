# subgen/core/postprocess/dedupe.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Callable
import re
import unicodedata
from difflib import SequenceMatcher


# If your project has its own logger, pass it in via `logger=` when calling.
# Expected interface: logger.info(str) / logger.debug(str) / logger.warning(str)
LoggerLike = object


@dataclass(frozen=True)
class DedupeConfig:
    """
    Final-pass segment de-duplication / conflict resolution.

    Philosophy:
    - This runs AFTER all candidate-generation steps (repair/relisten, segmenter, etc.)
    - It cleans the output by removing near-duplicate segments produced by overlapping windows
      or insert-vs-replace behaviors upstream.

    Tunables:
    - time_window_s: only consider segments "near" each other in time for duplicate checks
    - similarity_threshold: SequenceMatcher ratio on normalized text to treat as duplicates
    - short_text_chars: segments shorter than this are considered "snippets" and are easier to drop
    - gap_merge_s: if a short snippet sits between two near-duplicates, drop snippet first
    - keep_strategy: prefer higher confidence, longer duration, longer text (in that order)
    """

    # Compare duplicates only within this time proximity (start time delta)
    time_window_s: float = 30.0

    # Normalized-text similarity threshold for duplicates
    similarity_threshold: float = 0.90

    # If a segment is very short, treat it as a snippet and drop more aggressively
    short_text_chars: int = 10

    # Consider segments adjacent if gap is small (helps eliminate snippet-only segments)
    adjacency_gap_s: float = 2.0

    # If True, drop segments whose text becomes empty after normalization
    drop_empty: bool = True

    # If True, aggressively drop short snippets that are contained in a nearby longer segment
    drop_contained_snippets: bool = True

    # If True, when duplicates are found, keep the "best" one rather than first/last
    choose_best_on_duplicate: bool = True

    # Logging
    log_level: str = "info"  # "debug" | "info" | "warning" | "none"


# ----------------------------
# Text normalization utilities
# ----------------------------

_PUNCT_RE = re.compile(r"[\s\u00A0]+", re.UNICODE)
_STRIP_RE = re.compile(r"[^\w\u4e00-\u9fff\u3040-\u30ff\u0400-\u04ff\u0530-\u058F]+", re.UNICODE)

# Common “slash pronoun” patterns you showed: 他/她, 他／她, 他\她 etc.
_PRONOUN_SLASH_RE = re.compile(r"(他|她|他們|她們|他们|她们)\s*[/／\\]\s*(他|她|他們|她們|他们|她们)")


def normalize_text(text: str) -> str:
    """
    Normalize text for duplicate detection.

    Goals:
    - Reduce differences from punctuation/spacing
    - Canonicalize common slash-pronoun templates (e.g., 他/她) to a single token
    - Remove most punctuation to focus on content
    """
    if not text:
        return ""

    # Unicode canonicalization (NFKC is good for normalizing fullwidth forms)
    t = unicodedata.normalize("NFKC", text)

    # Canonicalize "他/她" style patterns
    t = _PRONOUN_SLASH_RE.sub("TA", t)

    # Collapse whitespace
    t = _PUNCT_RE.sub(" ", t).strip().lower()

    # Strip punctuation-like characters while keeping:
    # - word characters
    # - CJK unified ideographs
    # - Japanese kana
    # - Cyrillic
    # - Armenian (since you are dealing with hy)
    t = _STRIP_RE.sub("", t)

    return t


def text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def is_contained(shorter: str, longer: str) -> bool:
    if not shorter or not longer:
        return False
    return shorter in longer


# ----------------------------
# Segment helpers
# ----------------------------

def _seg_duration(seg) -> float:
    try:
        return max(0.0, float(seg.end) - float(seg.start))
    except Exception:
        return 0.0


def _seg_text(seg) -> str:
    return getattr(seg, "text", "") or ""


def _seg_conf(seg) -> Optional[float]:
    # many of your segments have confidence=None; handle gracefully
    c = getattr(seg, "confidence", None)
    try:
        return None if c is None else float(c)
    except Exception:
        return None


def _seg_key(seg) -> Tuple[float, float, str]:
    # For stable logging / debug
    return (float(getattr(seg, "start", 0.0)), float(getattr(seg, "end", 0.0)), _seg_text(seg)[:50])


def _prefer(seg_a, seg_b) -> object:
    """
    Choose which segment to keep among duplicates.

    Heuristic:
    1) Higher confidence wins (if both non-null and differ)
    2) Longer duration wins
    3) Longer text (normalized length) wins
    4) Earlier start wins (stability)
    """
    ca, cb = _seg_conf(seg_a), _seg_conf(seg_b)
    if ca is not None and cb is not None and ca != cb:
        return seg_a if ca > cb else seg_b
    if ca is not None and cb is None:
        return seg_a
    if cb is not None and ca is None:
        return seg_b

    da, db = _seg_duration(seg_a), _seg_duration(seg_b)
    if da != db:
        return seg_a if da > db else seg_b

    na = len(normalize_text(_seg_text(seg_a)))
    nb = len(normalize_text(_seg_text(seg_b)))
    if na != nb:
        return seg_a if na > nb else seg_b

    sa = float(getattr(seg_a, "start", 0.0))
    sb = float(getattr(seg_b, "start", 0.0))
    return seg_a if sa <= sb else seg_b


# ----------------------------
# Core dedupe pass
# ----------------------------

def dedupe_segments(
    segments: Iterable[object],
    *,
    cfg: Optional[DedupeConfig] = None,
    logger: Optional[LoggerLike] = None,
) -> List[object]:
    """
    Final-pass deduplication for Segment-like objects.
    Works with your Segment dataclass / pydantic model as long as it has:
      - start: float
      - end: float
      - text: str
      - confidence: Optional[float] (optional)

    Returns a NEW list; does not mutate input segments.
    """
    cfg = cfg or DedupeConfig()
    segs = list(segments)

    # Sort by start time; stable sort keeps original order for ties
    segs.sort(key=lambda s: (float(getattr(s, "start", 0.0)), float(getattr(s, "end", 0.0))))

    def _log(msg: str, level: str = "info") -> None:
        if cfg.log_level == "none" or logger is None:
            return
        if cfg.log_level == "warning":
            if level in ("warning",):
                getattr(logger, "warning", print)(msg)
            return
        if cfg.log_level == "info":
            if level in ("info", "warning"):
                getattr(logger, level, print)(msg)
            return
        # debug
        getattr(logger, level, print)(msg)

    # Optional: drop empty-ish segments
    if cfg.drop_empty:
        filtered = []
        for s in segs:
            nt = normalize_text(_seg_text(s))
            if nt:
                filtered.append(s)
            else:
                _log(f"[dedupe] drop empty seg { _seg_key(s) }", "debug")
        segs = filtered

    kept: List[object] = []

    # Sliding window index into kept segments for comparisons
    # Because kept is time-sorted, we can pop old items from a separate list of indices.
    for seg in segs:
        seg_start = float(getattr(seg, "start", 0.0))
        seg_end = float(getattr(seg, "end", 0.0))
        seg_text = _seg_text(seg)
        seg_norm = normalize_text(seg_text)
        seg_norm_len = len(seg_norm)

        if not seg_norm and cfg.drop_empty:
            continue

        # Step 1: remove trivial snippets that are contained in the most recent kept segment
        if cfg.drop_contained_snippets and kept:
            prev = kept[-1]
            prev_start = float(getattr(prev, "start", 0.0))
            if seg_start - prev_start <= cfg.time_window_s:
                prev_norm = normalize_text(_seg_text(prev))
                if seg_norm_len <= cfg.short_text_chars and is_contained(seg_norm, prev_norm):
                    _log(
                        f"[dedupe] drop contained snippet seg={_seg_key(seg)} "
                        f"contained_in prev={_seg_key(prev)}",
                        "info",
                    )
                    continue

        # Step 2: check duplicates against recent kept segments within time_window
        # We scan from the end backwards until out of time_window for efficiency.
        duplicate_idx: Optional[int] = None
        best_keep = None

        for i in range(len(kept) - 1, -1, -1):
            cand = kept[i]
            cand_start = float(getattr(cand, "start", 0.0))
            if seg_start - cand_start > cfg.time_window_s:
                break  # older than window; stop scanning

            cand_norm = normalize_text(_seg_text(cand))
            if not cand_norm:
                continue

            # Hard containment rule: short snippet vs longer segment
            if cfg.drop_contained_snippets:
                if seg_norm_len <= cfg.short_text_chars and is_contained(seg_norm, cand_norm):
                    _log(
                        f"[dedupe] drop contained snippet seg={_seg_key(seg)} "
                        f"contained_in cand={_seg_key(cand)}",
                        "info",
                    )
                    duplicate_idx = -1  # special marker: dropped
                    break
                if len(cand_norm) <= cfg.short_text_chars and is_contained(cand_norm, seg_norm):
                    # Replace short snippet with longer current seg
                    duplicate_idx = i
                    best_keep = seg
                    _log(
                        f"[dedupe] replace short snippet cand={_seg_key(cand)} "
                        f"with longer seg={_seg_key(seg)} (containment)",
                        "info",
                    )
                    break

            sim = text_similarity(seg_norm, cand_norm)
            if sim >= cfg.similarity_threshold:
                duplicate_idx = i
                if cfg.choose_best_on_duplicate:
                    best_keep = _prefer(cand, seg)
                else:
                    best_keep = cand  # keep existing by default
                _log(
                    f"[dedupe] near-dup sim={sim:.3f} seg={_seg_key(seg)} cand={_seg_key(cand)} "
                    f"-> keep={_seg_key(best_keep)}",
                    "info",
                )
                break

            # Prefix/suffix style duplicates (common in overlapping windows)
            # Example: "他/她承诺要为...如果他/她" (truncated) vs full sentence
            if (seg_norm and cand_norm) and (
                seg_norm.startswith(cand_norm) or seg_norm.endswith(cand_norm) or
                cand_norm.startswith(seg_norm) or cand_norm.endswith(seg_norm)
            ):
                # treat as duplicate if time proximity is reasonable
                # and at least one side is short-ish
                if min(len(seg_norm), len(cand_norm)) <= max(cfg.short_text_chars * 2, 20):
                    duplicate_idx = i
                    best_keep = _prefer(cand, seg) if cfg.choose_best_on_duplicate else cand
                    _log(
                        f"[dedupe] affix-dup seg={_seg_key(seg)} cand={_seg_key(cand)} "
                        f"-> keep={_seg_key(best_keep)}",
                        "info",
                    )
                    break

        if duplicate_idx == -1:
            # dropped due to containment
            continue

        if duplicate_idx is None:
            kept.append(seg)
            continue

        # duplicate_idx is an index into kept: resolve keep/replace
        assert best_keep is not None
        if best_keep is kept[duplicate_idx]:
            # keep existing, drop current
            continue
        else:
            # replace existing with current best
            kept[duplicate_idx] = best_keep

    # Step 3: small adjacency cleanup
    # Drop tiny snippets that sit adjacent to a larger segment and are highly similar/contained.
    cleaned: List[object] = []
    for seg in kept:
        if not cleaned:
            cleaned.append(seg)
            continue

        prev = cleaned[-1]
        gap = float(getattr(seg, "start", 0.0)) - float(getattr(prev, "end", 0.0))
        if gap <= cfg.adjacency_gap_s:
            prev_norm = normalize_text(_seg_text(prev))
            seg_norm = normalize_text(_seg_text(seg))
            if cfg.drop_contained_snippets:
                if len(seg_norm) <= cfg.short_text_chars and is_contained(seg_norm, prev_norm):
                    _log(
                        f"[dedupe] drop adjacent snippet seg={_seg_key(seg)} "
                        f"contained_in prev={_seg_key(prev)} gap={gap:.2f}s",
                        "info",
                    )
                    continue
                if len(prev_norm) <= cfg.short_text_chars and is_contained(prev_norm, seg_norm):
                    _log(
                        f"[dedupe] drop adjacent snippet prev={_seg_key(prev)} "
                        f"contained_in seg={_seg_key(seg)} gap={gap:.2f}s",
                        "info",
                    )
                    cleaned[-1] = seg
                    continue

            sim = text_similarity(prev_norm, seg_norm)
            if sim >= cfg.similarity_threshold:
                best = _prefer(prev, seg) if cfg.choose_best_on_duplicate else prev
                _log(
                    f"[dedupe] drop adjacent near-dup sim={sim:.3f} prev={_seg_key(prev)} seg={_seg_key(seg)} "
                    f"-> keep={_seg_key(best)}",
                    "info",
                )
                cleaned[-1] = best
                continue

        cleaned.append(seg)

    # Final sort to ensure monotonicity after replacements
    cleaned.sort(key=lambda s: (float(getattr(s, "start", 0.0)), float(getattr(s, "end", 0.0))))
    return cleaned


# Convenience alias (naming aligned with typical coalesce pipeline usage)
def dedupe_and_resolve_conflicts(
    segments: Iterable[object],
    *,
    cfg: Optional[DedupeConfig] = None,
    logger: Optional[LoggerLike] = None,
) -> List[object]:
    return dedupe_segments(segments, cfg=cfg, logger=logger)
