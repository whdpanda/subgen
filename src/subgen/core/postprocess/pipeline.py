from __future__ import annotations

from subgen.core.postprocess.coalesce import coalesce_segments
from subgen.core.postprocess.punct_split import split_segments_on_sentence_end_punct
from subgen.core_types import Segment, Word


def apply_postprocess_pipeline(
    *,
    words: list[Word],
    segments: list[Segment],
    min_seg: float,
    soft_max: float,
    hard_max: float,
    min_chars: int = 8,
) -> list[Segment]:
    """Apply postprocess operators with a stable execution order.

    This function intentionally centralizes postprocessing orchestration so
    callers don't have to manually chain operators and parameters.
    """

    split_segments = split_segments_on_sentence_end_punct(
        words=words,
        segments=segments,
        min_seg=min_seg,
        hard_max=hard_max,
    )

    return coalesce_segments(
        split_segments,
        min_dur=min_seg,
        min_chars=min_chars,
        target_dur=soft_max,
        hard_max=hard_max,
    )
