from pathlib import Path

from subgen.core.subtitle.srt_io import read_srt, write_srt


def test_srt_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "in.srt"
    src.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nWorld\n\n",
        encoding="utf-8",
    )

    doc = read_srt(src)
    assert doc.total_cues == 2
    assert doc.cues[0].text == "Hello"
    assert doc.cues[1].text == "World"

    out = tmp_path / "out.srt"
    write_srt(doc, out)

    doc2 = read_srt(out)
    assert doc2.total_cues == 2
    assert doc2.cues[0].start_ms == 0
    assert doc2.cues[0].end_ms == 1000
    assert doc2.cues[1].start_ms == 1000
    assert doc2.cues[1].end_ms == 2000
    assert doc2.cues[0].text == "Hello"
    assert doc2.cues[1].text == "World"
