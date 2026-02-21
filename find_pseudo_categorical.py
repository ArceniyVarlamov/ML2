#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import polars as pl


def discover_float_cols(path: Path) -> List[str]:
    schema = pl.read_parquet(path, n_rows=1).schema
    float_types = {pl.Float32, pl.Float64}
    cols: List[str] = []
    for col, dtype in schema.items():
        if col == "customer_id":
            continue
        if dtype in float_types:
            cols.append(col)
    return cols


def main() -> None:
    p = argparse.ArgumentParser(description="Find pseudo-categorical float columns by n_unique")
    p.add_argument("--train-extra", required=True, help="Path to train_extra_features.parquet")
    p.add_argument("--test-extra", required=True, help="Path to test_extra_features.parquet")
    p.add_argument("--max-unique", type=int, default=50, help="Max unique values to treat as pseudo-categorical")
    p.add_argument("--min-unique", type=int, default=2, help="Min unique values to keep")
    p.add_argument("--sample-rows", type=int, default=250000, help="Rows to read from each split (0=all)")
    p.add_argument("--output", required=True, help="Output JSON path")
    args = p.parse_args()

    train_path = Path(args.train_extra)
    test_path = Path(args.test_extra)
    float_cols = discover_float_cols(train_path)
    if not float_cols:
        raise ValueError("No float columns found in train extra features")

    n_rows = None if int(args.sample_rows) <= 0 else int(args.sample_rows)
    train_df = pl.read_parquet(train_path, columns=float_cols, n_rows=n_rows)
    test_df = pl.read_parquet(test_path, columns=float_cols, n_rows=n_rows)
    combined = pl.concat([train_df, test_df], how="vertical_relaxed")

    exprs = [pl.col(c).drop_nulls().n_unique().alias(c) for c in float_cols]
    stats = combined.select(exprs).to_dicts()[0]

    min_unique = int(args.min_unique)
    max_unique = int(args.max_unique)
    selected = sorted(
        [
            c
            for c in float_cols
            if min_unique <= int(stats.get(c, 0)) <= max_unique
        ]
    )

    payload = {
        "max_unique": max_unique,
        "min_unique": min_unique,
        "sample_rows": int(args.sample_rows),
        "n_float_cols": len(float_cols),
        "n_selected": len(selected),
        "pseudo_categorical_cols": selected,
        "n_unique_stats": {c: int(stats.get(c, 0)) for c in selected},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pseudo_cat_scan] saved {out_path} selected={len(selected)} / {len(float_cols)}")


if __name__ == "__main__":
    main()

