#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Build target correlation report")
    p.add_argument("--train-target", required=True, help="Path to train_target.parquet")
    p.add_argument("--top-k", type=int, default=6, help="Top neighbors per target")
    p.add_argument("--output", required=True, help="Output JSON path")
    args = p.parse_args()

    train_target = pd.read_parquet(Path(args.train_target))
    target_cols = [c for c in train_target.columns if c.startswith("target_")]
    if not target_cols:
        raise ValueError("No target_* columns found")

    corr_abs = train_target[target_cols].corr().abs()
    top_k = int(args.top_k)

    per_target: Dict[str, List[Dict[str, Any]]] = {}
    pair_rows: List[Dict[str, Any]] = []

    for target in target_cols:
        neighbors = (
            corr_abs[target]
            .drop(labels=[target], errors="ignore")
            .sort_values(ascending=False)
            .head(top_k)
        )
        per_target[target] = [
            {"target": str(idx), "corr_abs": float(val)}
            for idx, val in neighbors.items()
        ]

    for i, left in enumerate(target_cols):
        for right in target_cols[i + 1 :]:
            pair_rows.append(
                {
                    "left": left,
                    "right": right,
                    "corr_abs": float(corr_abs.loc[left, right]),
                }
            )
    pair_rows.sort(key=lambda x: x["corr_abs"], reverse=True)

    payload = {
        "n_targets": len(target_cols),
        "top_k": top_k,
        "top_pairs": pair_rows[: min(80, len(pair_rows))],
        "neighbors": per_target,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[target_corr] saved {out_path}")


if __name__ == "__main__":
    main()

