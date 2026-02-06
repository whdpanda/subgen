from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from subgen.core.quality.report import QualityProfile, quality_check_srt
from subgen.core.quality.pipeline import FixBudget, apply_fixes
from subgen.core.subtitle.srt_io import read_srt, write_srt


def _json_dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _action_label(action: Any) -> str:
    """
    Best-effort: 兼容 FixAction 字段变动。
    优先从对象属性拿；没有则从 to_dict 里拿；再不行用类名兜底。
    """
    # 1) 先拿 dict（如果有）
    d: Dict[str, Any] = {}
    if hasattr(action, "to_dict") and callable(getattr(action, "to_dict")):
        try:
            d = action.to_dict() or {}
        except Exception:
            d = {}

    # 2) 优先级：type/kind/action_type/name（你项目里最常见的命名）
    label = (
        getattr(action, "type", None)
        or getattr(action, "kind", None)
        or getattr(action, "action_type", None)
        or getattr(action, "name", None)
        or d.get("type")
        or d.get("kind")
        or d.get("action_type")
        or d.get("name")
    )

    # 3) 转成稳定字符串
    if label is None:
        return action.__class__.__name__

    # Enum / 自定义类型：尽量取 .name
    if hasattr(label, "name"):
        try:
            return str(label.name)
        except Exception:
            pass

    return str(label)


def _print_violations(rep, limit: int = 10) -> None:
    print("ok:", rep.ok())
    print("violations:", len(rep.violations))
    for v in rep.violations[:limit]:
        print("-", v.type, "cue=", v.cue_index, "msg=", v.message)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    srt_in = root / "tests" / "fixtures" / "bad.srt"
    if not srt_in.exists():
        raise FileNotFoundError(f"missing fixture: {srt_in}")

    out_dir = root / "out" / "quality_verify"
    out_dir.mkdir(parents=True, exist_ok=True)

    srt_fixed = out_dir / "bad.fixed.srt"
    rep_before_path = out_dir / "quality_before.json"
    rep_after_path = out_dir / "quality_after.json"

    # 1) Quality profile（你的默认阈值）
    profile = QualityProfile(
        name="verify",
        max_cps=17.0,
        max_line_len=42,
        max_lines=2,
        min_dur_ms=700,
        max_dur_ms=7000,
        max_overlap_ms=0,
    )

    # 2) BEFORE
    rep_before = quality_check_srt(str(srt_in), profile)
    _json_dump(rep_before_path, rep_before.to_dict())

    print("=== BEFORE ===")
    _print_violations(rep_before, limit=10)

    # 3) APPLY FIXES
    doc = read_srt(srt_in).sorted()
    budget = FixBudget(max_passes=2)
    fix_res = apply_fixes(doc, profile, budget=budget)
    write_srt(fix_res.fixed_doc, srt_fixed)

    # 3.1) Print FIX ACTIONS (robust)
    print("\n=== FIX ACTIONS ===")
    print("passes:", fix_res.passes)
    print("actions:", len(fix_res.actions))

    # (A) 类型统计
    cnt = Counter(_action_label(a) for a in fix_res.actions)
    if cnt:
        print("\n-- action type counts --")
        for k, v in cnt.most_common():
            print(f"  {k}: {v}")

    # (B) 逐条打印（含 to_dict）
    print("\n-- actions detail --")
    for i, a in enumerate(fix_res.actions):
        label = _action_label(a)
        a_dict = a.to_dict() if hasattr(a, "to_dict") else {"repr": repr(a)}
        print(f"- [{i}] {label}: {a_dict}")

    # 4) AFTER
    rep_after = quality_check_srt(str(srt_fixed), profile)
    _json_dump(rep_after_path, rep_after.to_dict())

    print("\n=== AFTER ===")
    print("fixed_srt:", srt_fixed)
    _print_violations(rep_after, limit=10)

    # 5) Gate: violations 不应变多
    if len(rep_after.violations) > len(rep_before.violations):
        raise AssertionError("violations increased after fixes (regression)")

    print("\n✅ Verification OK: violations did not increase.")
    if rep_after.ok():
        print("✅ Passed quality gate (no violations).")
    else:
        print("⚠️ Still has violations; check quality_after.json for details.")


if __name__ == "__main__":
    main()
