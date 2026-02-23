#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_target_scores(path: Path | None) -> Dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = payload.get("target_scores", payload)
    if not isinstance(scores, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in scores.items():
        if str(k).startswith("target_"):
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build target groups (rare/hard/common) for specialist experiments"
    )
    p.add_argument("--train-target", required=True, help="Path to train_target.parquet")
    p.add_argument(
        "--scores-json",
        default=None,
        help="Optional scores JSON with target_scores (e.g. blend_scores.json) for hard targets",
    )
    p.add_argument(
        "--rare-top-k",
        type=int,
        default=10,
        help="Number of rarest targets to include in group 'rare'",
    )
    p.add_argument(
        "--hard-top-k",
        type=int,
        default=10,
        help="Number of lowest-AUC targets to include in group 'hard' (if scores provided)",
    )
    p.add_argument(
        "--rare-max-rate",
        type=float,
        default=0.01,
        help="Optional cap on positive rate for rare group (targets above cap are skipped before top-k). 0=off",
    )
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    train_target = pd.read_parquet(Path(args.train_target))
    target_cols = [c for c in train_target.columns if c.startswith("target_")]
    if not target_cols:
        raise ValueError("No target_* columns found")

    pos_rate = {
        t: float(pd.to_numeric(train_target[t], errors="coerce").fillna(0).mean()) for t in target_cols
    }
    rare_candidates = sorted(target_cols, key=lambda t: (pos_rate[t], t))
    if float(args.rare_max_rate) > 0:
        capped = [t for t in rare_candidates if pos_rate[t] <= float(args.rare_max_rate)]
        if capped:
            rare_candidates = capped
    rare = rare_candidates[: max(0, int(args.rare_top_k))]

    scores = load_target_scores(Path(args.scores_json) if args.scores_json else None)
    hard: List[str] = []
    if scores:
        hard_candidates = [t for t in target_cols if t in scores]
        hard_candidates.sort(key=lambda t: (scores[t], t))
        hard = hard_candidates[: max(0, int(args.hard_top_k))]

    special = set(rare) | set(hard)
    common = [t for t in target_cols if t not in special]

    out_payload: Dict[str, Any] = {
        "groups": {
            "rare": rare,
            "hard": hard,
            "common": common,
        },
        "stats": {
            "n_targets": len(target_cols),
            "rare_top_k": int(args.rare_top_k),
            "hard_top_k": int(args.hard_top_k),
            "rare_max_rate": float(args.rare_max_rate),
            "positive_rate": pos_rate,
            "hard_scores": {t: scores[t] for t in hard} if scores else {},
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_path}")
    print(f"rare={len(rare)} hard={len(hard)} common={len(common)}")


if __name__ == "__main__":
    main()
