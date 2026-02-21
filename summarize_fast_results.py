#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def read_macro(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    value = payload.get("macro_auc")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.6f}"


def main() -> None:
    experiments = [
        ("H0_base", "artifacts_fast_base"),
        ("H1_quantize", "artifacts_fast_h1_quantize"),
        ("H1_quantize_lite", "artifacts_fast_h1_quantize_lite_v2"),
        ("H2_te", "artifacts_fast_h2_te"),
        ("H3_quantize_te", "artifacts_fast_h3_quantize_te"),
        ("H6_targetaware", "artifacts_fast_h6_targetaware"),
        ("H6_targetaware_pcat200", "artifacts_fast_h6_targetaware_pcat200"),
        ("H6_targetaware_ultrasafe", "artifacts_fast_h6_targetaware_ultrasafe"),
        ("H4_xgb", "artifacts_fast_h4_xgb"),
        ("H5_crossstack", "artifacts_fast_h5_crossstack"),
    ]

    rows = []
    for name, out_dir in experiments:
        p = Path(out_dir) / "scores"
        cat = read_macro(p / "catboost_scores.json")
        lgb = read_macro(p / "lightgbm_scores.json")
        blend = read_macro(p / "blend_scores.json")
        stack = read_macro(p / "stack_scores.json")
        rows.append((name, cat, lgb, blend, stack))

    baseline_blend = rows[0][3]
    baseline_stack = rows[0][4]

    print("experiment\tcatboost\tlightgbm\tblend\tstack\td_blend\td_stack")
    for name, cat, lgb, blend, stack in rows:
        d_blend = (blend - baseline_blend) if blend is not None and baseline_blend is not None else None
        d_stack = (stack - baseline_stack) if stack is not None and baseline_stack is not None else None
        d_blend_s = "-" if d_blend is None else f"{d_blend:+.6f}"
        d_stack_s = "-" if d_stack is None else f"{d_stack:+.6f}"
        print(f"{name}\t{fmt(cat)}\t{fmt(lgb)}\t{fmt(blend)}\t{fmt(stack)}\t{d_blend_s}\t{d_stack_s}")


if __name__ == "__main__":
    main()
