from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Optional

from subgen.core_types import Transcript

# 主优先：句末/强断句标点
_SENT_PUNCT = set("。！？!?；;…")
# 次优先/辅助：弱断句标点（长度兜底时更优先切）
_WEAK_PUNCT = set("，、,:：")
# 避免 cue/行 以这些标点开头
_LEADING_PUNCT = set("，。、！？!?；;：:）)]】》」』”’…")
# 避免在这些符号后面硬切（行尾是开引号/括号类会很丑）
_TRAILING_BAD = set("（([【《「『“‘")

_SPACE_RE = re.compile(r"\s+")


def _normalize_zh_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = _SPACE_RE.sub(" ", s)
    s = re.sub(r"\s*([，。！？；：、])\s*", r"\1", s)
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    return s.strip()


def _display_len(s: str) -> int:
    return len(s.replace("\n", ""))


def _count_reading_chars(text: str) -> int:
    # 与项目里 CPS/字符计数口径一致：非空白字符
    if not text:
        return 0
    return sum(1 for ch in text if not ch.isspace())


def _can_break_here(prev: str, nxt: str) -> bool:
    if not prev:
        return False
    if prev[-1] in _TRAILING_BAD:
        return False
    if nxt and nxt[0] in _LEADING_PUNCT:
        return False
    return True


def _split_by_sentence_punct_keep(text: str) -> List[str]:
    """主优先：按句末标点断句，标点保留在左侧。"""
    text = (text or "").strip()
    if not text:
        return []
    out: List[str] = []
    buf: List[str] = []
    for ch in text:
        buf.append(ch)
        if ch in _SENT_PUNCT:
            piece = "".join(buf).strip()
            if piece:
                out.append(piece)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def _regex_tokenize(text: str) -> List[str]:
    """无依赖 token：CJK块 / ASCII块 / 单标点。"""
    text = (text or "").strip()
    if not text:
        return []
    pattern = re.compile(
        r"[\u4e00-\u9fff]+"
        r"|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?"
        r"|[^\s]"
    )
    return [m.group(0) for m in pattern.finditer(text)]


def _tokenize(text: str) -> List[str]:
    """
    次优先：词/语义边界
    - 若环境有 jieba，则优先用
    - 否则 fallback regex tokenize
    """
    text = (text or "").strip()
    if not text:
        return []
    try:
        import jieba  # type: ignore

        toks = [t for t in jieba.cut(text, cut_all=False) if t and not t.isspace()]
        return toks if toks else _regex_tokenize(text)
    except Exception:
        return _regex_tokenize(text)


def _append_piece(out: List[str], piece: str) -> None:
    piece = (piece or "").strip()
    if not piece:
        return

    # 避免以标点开头：并回上一段
    while out and piece and piece[0] in _LEADING_PUNCT:
        out[-1] = (out[-1] + piece[0]).strip()
        piece = piece[1:].strip()

    if piece:
        out.append(piece)


def _split_tokens_by_max_chars(tokens: List[str], max_chars: int) -> List[str]:
    """辅助：按 max_chars 切（优先 token 边界），再兜底逐字。"""
    if max_chars <= 0:
        return ["".join(tokens).strip()] if tokens else []

    out: List[str] = []
    buf: List[str] = []

    def flush():
        if buf:
            _append_piece(out, "".join(buf))
            buf.clear()

    for tok in tokens:
        if not tok:
            continue

        cand = "".join(buf) + tok
        if _display_len(cand) <= max_chars:
            buf.append(tok)
            continue

        # tok 自身超长 -> 逐字兜底
        if _display_len(tok) > max_chars:
            flush()
            chunk = ""
            for ch in tok:
                if _display_len(chunk + ch) <= max_chars:
                    chunk += ch
                else:
                    _append_piece(out, chunk)
                    chunk = ch
            _append_piece(out, chunk)
            continue

        # buf+tok 超长 -> 先 flush 再放 tok
        flush()
        buf.append(tok)

    flush()
    return [x for x in out if x.strip()]


def _wrap_greedy_cjk(text: str, max_line_len: int) -> List[str]:
    """
    cue 内换行（不做 cue 切分）：
    - 优先强标点，其次弱标点，否则硬切
    """
    text = (text or "").strip()
    if not text:
        return []

    lines: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        remaining = text[i:]
        if _display_len(remaining) <= max_line_len:
            lines.append(remaining.strip())
            break

        cut = min(n, i + max_line_len)

        best_cut: Optional[int] = None
        for j in range(cut - 1, i, -1):
            if text[j] in _SENT_PUNCT:
                left = text[i : j + 1]
                right = text[j + 1 :]
                if _can_break_here(left, right):
                    best_cut = j + 1
                    break

        if best_cut is None:
            for j in range(cut - 1, i, -1):
                if text[j] in _WEAK_PUNCT:
                    left = text[i : j + 1]
                    right = text[j + 1 :]
                    if _can_break_here(left, right):
                        best_cut = j + 1
                        break

        if best_cut is None:
            best_cut = cut
            if best_cut < n and text[best_cut] in _LEADING_PUNCT and best_cut > i + 1:
                best_cut -= 1

        piece = text[i:best_cut].strip()
        if piece:
            lines.append(piece)
        i = best_cut

    if len(lines) >= 2 and _display_len(lines[-1]) == 1:
        lines[-2] = (lines[-2] + lines[-1]).strip()
        lines.pop()

    return [ln for ln in lines if ln.strip()]


def wrap_zh(
    text: str,
    *,
    max_line_len: int = 18,
    max_lines: int = 2,
    line_len_cap: int = 42,  # ✅ 保留旧参数，防止上游调用炸
) -> str:
    """
    cue 内排版：插入 '\n'（不拆 cue）
    - line_len_cap 为旧接口参数：这里不改变行为，只保持兼容；
      真正的 cue 字符上限由 apply_zh_layout_split_to_cues 决定。
    """
    _ = line_len_cap  # compatibility noop
    text = _normalize_zh_text(text)
    if not text:
        return text

    lines = _wrap_greedy_cjk(text, max_line_len=max_line_len)

    if max_lines > 0 and len(lines) > max_lines:
        kept = lines[: max_lines - 1]
        tail = "".join(lines[max_lines - 1 :]).strip()
        kept.append(tail)
        lines = kept

    return "\n".join(lines)


def _fits_line_constraints(text: str, max_line_len: int, max_lines: int) -> bool:
    if max_lines <= 0:
        return True
    lines = _wrap_greedy_cjk(_normalize_zh_text(text), max_line_len=max_line_len)
    return len(lines) <= max_lines


def _split_sentence_to_cues(
    sentence: str,
    *,
    max_chars_per_cue: int,
    max_line_len: int,
    max_lines: int,
) -> List[str]:
    sentence = _normalize_zh_text(sentence)
    if not sentence:
        return []

    if (max_chars_per_cue <= 0 or _display_len(sentence) <= max_chars_per_cue) and _fits_line_constraints(
        sentence, max_line_len=max_line_len, max_lines=max_lines
    ):
        return [sentence]

    tokens = _tokenize(sentence)
    pieces = _split_tokens_by_max_chars(tokens, max_chars=max_chars_per_cue)

    out: List[str] = []
    for p in pieces:
        if _fits_line_constraints(p, max_line_len=max_line_len, max_lines=max_lines):
            _append_piece(out, p)
            continue

        # 行数超了：进一步缩短兜底
        smaller = max(1, max_chars_per_cue // 2) if max_chars_per_cue > 0 else max(1, max_line_len * max(1, max_lines))
        sub = _split_tokens_by_max_chars(_tokenize(p), max_chars=smaller)
        for sp in sub:
            _append_piece(out, sp)

    return [x for x in out if x.strip()]


def _alloc_times_by_chars(start: float, end: float, parts: List[str]) -> List[Tuple[float, float]]:
    """按非空白字符数比例分配时长（单字所占时间一致）。"""
    n = len(parts)
    if n <= 0:
        return []
    dur = float(end) - float(start)
    if dur <= 0:
        eps = 0.001
        tmp_spans = []
        cur = float(start)
        for _ in range(n):
            tmp_spans.append((cur, cur + eps))
            cur += eps
        return tmp_spans

    weights = [_count_reading_chars(p) for p in parts]
    total = sum(weights)

    if total <= 0:
        step = dur / n
        even_spans = []
        for i in range(n):
            s = float(start) + step * i
            e = float(start) + step * (i + 1)
            even_spans.append((s, e))
        even_spans[-1] = (even_spans[-1][0], float(end))
        return even_spans

    out: List[Tuple[float, float]] = []
    acc = 0.0
    for w in weights:
        s = float(start) + dur * (acc / total)
        acc += float(w)
        e = float(start) + dur * (acc / total)
        out.append((s, e))

    fixed: List[Tuple[float, float]] = []
    last_s = float(start)
    for i, (s, e) in enumerate(out):
        s = max(s, last_s)
        if e <= s:
            e = s + 0.001
        if i == len(out) - 1:
            e = float(end)
            if e <= s:
                e = s + 0.001
        fixed.append((s, e))
        last_s = e
    return fixed


def _get_text_field(seg_dict: Dict[str, Any]) -> Tuple[str, str]:
    if "text" in seg_dict and isinstance(seg_dict["text"], str):
        return "text", seg_dict["text"]
    if "t" in seg_dict and isinstance(seg_dict["t"], str):
        return "t", seg_dict["t"]
    return "", ""


def apply_zh_layout(
    transcript: Transcript,
    *,
    max_line_len: int = 18,
    max_lines: int = 2,
    line_len_cap: int = 42,  # ✅ 兼容旧参数
) -> Transcript:
    d = transcript.model_dump()
    segs = d.get("segments")
    if not isinstance(segs, list):
        return transcript

    out_segs: List[Dict[str, Any]] = []
    for seg in segs:
        if not isinstance(seg, dict):
            continue

        field, txt = _get_text_field(seg)
        if not field or not txt:
            out_segs.append(seg)
            continue

        seg2 = dict(seg)
        seg2[field] = wrap_zh(
            txt,
            max_line_len=max_line_len,
            max_lines=max_lines,
            line_len_cap=line_len_cap,
        )
        out_segs.append(seg2)

    d["segments"] = out_segs
    return Transcript.model_validate(d)


def apply_zh_layout_split_to_cues(
    transcript: Transcript,
    *,
    max_line_len: int = 18,
    max_lines: int = 2,
    line_len_cap: int = 42,  # ✅ 兼容旧参数：旧 pipeline 还在传这个
    max_chars_per_cue: Optional[int] = None,  # ✅ 新参数：你要的“cue 字符上限”
) -> Transcript:
    """
    你要求的切分优先级：
    1) 标点断句（强句末标点）
    2) 语义/词边界（分词/token）
    3) 长度约束兜底（max_chars_per_cue / max_line_len / max_lines）
       - 若未传 max_chars_per_cue，则沿用 line_len_cap（兼容旧行为）
    时长：按字符数比例分配（单字时间一致）
    """
    if max_chars_per_cue is None:
        max_chars_per_cue = int(line_len_cap)

    d = transcript.model_dump()
    segs = d.get("segments")
    if not isinstance(segs, list):
        return transcript

    out_segs: List[Dict[str, Any]] = []

    for seg in segs:
        if not isinstance(seg, dict):
            continue

        if "start" not in seg or "end" not in seg:
            out_segs.append(seg)
            continue

        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except Exception:
            out_segs.append(seg)
            continue

        field, txt = _get_text_field(seg)
        if not field or not txt:
            out_segs.append(seg)
            continue

        text = _normalize_zh_text(txt)
        if not text:
            out_segs.append(seg)
            continue

        # 1) 主优先：标点断句
        sentences = _split_by_sentence_punct_keep(text)

        # 2) 次优先：token 边界 + 3) 长度/行约束兜底
        cue_texts: List[str] = []
        for sentence in sentences:
            parts = _split_sentence_to_cues(
                sentence,
                max_chars_per_cue=max_chars_per_cue,
                max_line_len=max_line_len,
                max_lines=max_lines,
            )
            for p in parts:
                _append_piece(cue_texts, p)

        if not cue_texts:
            out_segs.append(seg)
            continue

        if len(cue_texts) == 1:
            seg2 = dict(seg)
            seg2[field] = wrap_zh(
                cue_texts[0],
                max_line_len=max_line_len,
                max_lines=max_lines,
                line_len_cap=line_len_cap,
            )
            out_segs.append(seg2)
            continue

        spans = _alloc_times_by_chars(start, end, cue_texts)

        for (span_start, span_end), cue_text in zip(spans, cue_texts):
            seg2 = dict(seg)
            seg2["start"] = str(float(span_start)) if isinstance(seg.get("start"), str) else float(span_start)
            seg2["end"] = str(float(span_end)) if isinstance(seg.get("end"), str) else float(span_end)
            seg2[field] = wrap_zh(
                cue_text,
                max_line_len=max_line_len,
                max_lines=max_lines,
                line_len_cap=line_len_cap,
            )
            out_segs.append(seg2)

    d["segments"] = out_segs
    return Transcript.model_validate(d)
