from subgen.core.postprocess.coalesce import coalesce_segments
from subgen.core.postprocess.pipeline import apply_postprocess_pipeline
from subgen.core.postprocess.punct_split import split_segments_on_sentence_end_punct
from subgen.core_types import Segment, Word


def test_apply_postprocess_pipeline_matches_atomic_chain() -> None:
    words = [
        Word(start=0.0, end=0.6, word="Hello"),
        Word(start=0.6, end=1.2, word="world."),
        Word(start=1.2, end=1.8, word="How"),
        Word(start=1.8, end=2.4, word="are"),
        Word(start=2.4, end=3.0, word="you?"),
    ]
    segments = [Segment(start=0.0, end=3.0, text="Hello world. How are you?")]

    direct = coalesce_segments(
        split_segments_on_sentence_end_punct(
            words=words,
            segments=segments,
            min_seg=1.0,
            hard_max=10.0,
        ),
        min_dur=1.0,
        min_chars=1,
        target_dur=4.0,
        hard_max=10.0,
    )

    piped = apply_postprocess_pipeline(
        words=words,
        segments=segments,
        min_seg=1.0,
        soft_max=4.0,
        hard_max=10.0,
        min_chars=1,
    )

    assert [s.model_dump() for s in piped] == [s.model_dump() for s in direct]
