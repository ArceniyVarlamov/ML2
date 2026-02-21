#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def target_to_predict_col(target_col: str) -> str:
    return target_col.replace("target_", "predict_", 1)


def predict_to_target_col(predict_col: str) -> str:
    return predict_col.replace("predict_", "target_", 1)


def rank_normalize(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    return ranks


def parse_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",") if x.strip()]
        return [int(v) for v in parts]
    return []


def load_pseudo_cat_cols(cfg: Dict[str, Any], feature_cols: Sequence[str]) -> List[str]:
    features_cfg = cfg.get("features", {})
    feature_set = set(feature_cols)
    out: List[str] = []

    inline = features_cfg.get("pseudo_cat_cols", [])
    if isinstance(inline, str):
        inline = [x.strip() for x in inline.split(",") if x.strip()]
    if isinstance(inline, list):
        out.extend([str(c) for c in inline])

    file_path = features_cfg.get("pseudo_cat_cols_file")
    if file_path:
        p = Path(str(file_path))
        if p.exists():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    out.extend([str(c) for c in payload])
                elif isinstance(payload, dict):
                    for key in ["pseudo_categorical_cols", "columns", "pseudo_cat_cols"]:
                        value = payload.get(key)
                        if isinstance(value, list):
                            out.extend([str(c) for c in value])
                            break
            except Exception as exc:
                print(f"[pseudo_cat] failed to parse {p}: {exc}")
        else:
            print(f"[pseudo_cat] file does not exist: {p}")

    filtered = [c for c in dict.fromkeys(out) if c in feature_set]
    if filtered:
        print(f"[pseudo_cat] loaded columns={len(filtered)}")
    return filtered


def discover_columns(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    train_main_path = Path(cfg["data"]["train_main"])
    train_extra_path = Path(cfg["data"]["train_extra"])
    train_target_path = Path(cfg["data"]["train_target"])

    train_main_cols = pl.read_parquet(train_main_path, n_rows=1).columns
    train_extra_cols = pl.read_parquet(train_extra_path, n_rows=1).columns
    train_target_cols = pl.read_parquet(train_target_path, n_rows=1).columns

    feature_cols_main = [
        c for c in train_main_cols if c.startswith("num_feature") or c.startswith("cat_feature")
    ]
    target_cols = [c for c in train_target_cols if c.startswith("target_")]
    extra_cols = [c for c in train_extra_cols if c != "customer_id"]
    cat_cols = [c for c in feature_cols_main if c.startswith("cat_feature")]

    return {
        "feature_cols_main": feature_cols_main,
        "extra_cols": extra_cols,
        "target_cols": target_cols,
        "cat_cols": cat_cols,
    }


def select_extra_columns(cfg: Dict[str, Any], available_extra_cols: Sequence[str]) -> List[str]:
    top_k = int(cfg["features"].get("extra_top_k", 0))
    method = str(cfg["features"].get("extra_selection", "first_k"))
    sample_rows = int(cfg["features"].get("variance_sample_rows", 120_000))
    train_extra_path = Path(cfg["data"]["train_extra"])

    if top_k <= 0 or top_k >= len(available_extra_cols):
        return list(available_extra_cols)

    if method != "variance":
        return list(available_extra_cols[:top_k])

    print(f"[select_extra_columns] variance selection: top_k={top_k}, sample_rows={sample_rows}")
    sample_df = pl.read_parquet(train_extra_path, columns=list(available_extra_cols), n_rows=sample_rows)
    exprs = [pl.col(c).cast(pl.Float32).fill_null(0.0).var().alias(c) for c in available_extra_cols]
    variances = sample_df.select(exprs).to_pandas().iloc[0]
    ranked = variances.sort_values(ascending=False).index.tolist()
    return ranked[:top_k]


def add_row_stats(df_extra: pl.DataFrame, extra_cols: Sequence[str]) -> pl.DataFrame:
    if not extra_cols:
        return df_extra

    n = float(len(extra_cols))
    null_exprs = [pl.col(c).is_null().cast(pl.Int16) for c in extra_cols]
    zero_exprs = [(pl.col(c).fill_null(0.0) == 0.0).cast(pl.Int16) for c in extra_cols]

    df_extra = df_extra.with_columns(
        [
            pl.mean_horizontal(extra_cols).cast(pl.Float32).alias("row_extra_mean"),
            pl.min_horizontal(extra_cols).cast(pl.Float32).alias("row_extra_min"),
            pl.max_horizontal(extra_cols).cast(pl.Float32).alias("row_extra_max"),
            pl.sum_horizontal(extra_cols).cast(pl.Float32).alias("row_extra_sum"),
            pl.sum_horizontal(null_exprs).cast(pl.Float32).alias("row_extra_null_count"),
            pl.sum_horizontal(zero_exprs).cast(pl.Float32).alias("row_extra_zero_count"),
        ]
    ).with_columns(
        [
            (pl.col("row_extra_null_count") / n).cast(pl.Float32).alias("row_extra_null_share"),
            (pl.col("row_extra_zero_count") / n).cast(pl.Float32).alias("row_extra_zero_share"),
        ]
    )
    return df_extra


def load_raw_joined_data(cfg: Dict[str, Any], selected_extra_cols: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_main = pl.read_parquet(Path(cfg["data"]["train_main"]))
    test_main = pl.read_parquet(Path(cfg["data"]["test_main"]))
    train_target = pl.read_parquet(Path(cfg["data"]["train_target"]))

    train_extra = pl.read_parquet(
        Path(cfg["data"]["train_extra"]), columns=["customer_id"] + list(selected_extra_cols)
    )
    test_extra = pl.read_parquet(
        Path(cfg["data"]["test_extra"]), columns=["customer_id"] + list(selected_extra_cols)
    )

    if bool(cfg["features"].get("add_row_stats", True)):
        print("[load_raw_joined_data] adding row stats")
        train_extra = add_row_stats(train_extra, selected_extra_cols)
        test_extra = add_row_stats(test_extra, selected_extra_cols)

    train_df = (
        train_main.join(train_extra, on="customer_id", how="inner")
        .join(train_target, on="customer_id", how="inner")
        .to_pandas()
    )
    test_df = test_main.join(test_extra, on="customer_id", how="inner").to_pandas()

    return train_df, test_df


def encode_categories(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: Sequence[str]) -> None:
    for col in cat_cols:
        combined = pd.concat([train_df[col], test_df[col]], axis=0, ignore_index=True)
        combined = combined.astype("string").fillna("__MISSING__")
        codes, _ = pd.factorize(combined, sort=False)
        train_df[col] = codes[: len(train_df)].astype(np.int32)
        test_df[col] = codes[len(train_df) :].astype(np.int32)


def cast_numeric_float32(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: Sequence[str]) -> None:
    for col in feature_cols:
        if str(train_df[col].dtype).startswith("int") and str(test_df[col].dtype).startswith("int"):
            continue
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").astype(np.float32)
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").astype(np.float32)


def add_quantized_num_features(
    cfg: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[List[str], pd.DataFrame, pd.DataFrame]:
    decimals = parse_int_list(cfg["features"].get("quantize_num_decimals", []))
    if not decimals:
        return [], train_df, test_df

    prefixes = cfg["features"].get("quantize_num_prefixes", ["num_feature"])
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    prefixes = [str(x) for x in prefixes]

    source_cols = [c for c in feature_cols if any(c.startswith(p) for p in prefixes)]
    if not source_cols:
        return [], train_df, test_df

    max_cols = int(cfg["features"].get("quantize_max_cols", 0))
    if max_cols > 0 and len(source_cols) > max_cols:
        source_cols = source_cols[:max_cols]

    new_cols: List[str] = []
    train_blocks: List[np.ndarray] = []
    test_blocks: List[np.ndarray] = []
    print(
        f"[quantize] source_cols={len(source_cols)} decimals={decimals} max_cols={max_cols if max_cols > 0 else 'all'}"
    )
    for col in source_cols:
        tr = pd.to_numeric(train_df[col], errors="coerce")
        te = pd.to_numeric(test_df[col], errors="coerce")
        for d in decimals:
            qcol = f"{col}_q{d}"
            new_cols.append(qcol)
            train_blocks.append(tr.round(d).to_numpy(dtype=np.float32, copy=False))
            test_blocks.append(te.round(d).to_numpy(dtype=np.float32, copy=False))

    if new_cols:
        train_q = np.column_stack(train_blocks).astype(np.float32, copy=False)
        test_q = np.column_stack(test_blocks).astype(np.float32, copy=False)

        train_q_df = pd.DataFrame(train_q, columns=new_cols, index=train_df.index)
        test_q_df = pd.DataFrame(test_q, columns=new_cols, index=test_df.index)
        # Concat avoids repeated internal inserts and frame fragmentation warnings.
        train_df = pd.concat([train_df, train_q_df], axis=1)
        test_df = pd.concat([test_df, test_q_df], axis=1)
    return new_cols, train_df, test_df


def make_smoothed_mapping(
    col_values: np.ndarray,
    y: np.ndarray,
    alpha: float,
    global_mean: float,
) -> pd.Series:
    tmp = pd.DataFrame({"key": col_values, "y": y})
    grp = tmp.groupby("key", observed=True)["y"].agg(["mean", "count"])
    enc = (grp["mean"] * grp["count"] + global_mean * alpha) / (grp["count"] + alpha)
    return enc.astype(np.float32)


def add_oof_global_target_encoding(
    cfg: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_cols: Sequence[str],
    cat_cols: Sequence[str],
    fold_ids: np.ndarray,
) -> tuple[List[str], pd.DataFrame, pd.DataFrame]:
    if not bool(cfg["features"].get("oof_global_target_encoding", False)):
        return [], train_df, test_df
    if not cat_cols:
        return [], train_df, test_df

    alpha = float(cfg["features"].get("oof_te_alpha", 20.0))
    max_cols = int(cfg["features"].get("oof_te_max_cat_cols", 0))
    use_cols = list(cat_cols)
    if max_cols > 0 and len(use_cols) > max_cols:
        use_cols = use_cols[:max_cols]

    y_global = (train_df[list(target_cols)].sum(axis=1).to_numpy() > 0).astype(np.int8)
    global_mean = float(y_global.mean())
    n_folds = int(np.max(fold_ids) + 1)

    print(
        f"[oof_te] cat_cols={len(use_cols)} alpha={alpha} global_mean={global_mean:.6f} folds={n_folds}"
    )
    new_cols: List[str] = []
    train_blocks: List[np.ndarray] = []
    test_blocks: List[np.ndarray] = []
    for col in use_cols:
        tr_col = train_df[col].to_numpy()
        te_col = test_df[col].to_numpy()

        oof_encoded = np.zeros(len(train_df), dtype=np.float32)
        for fold in range(n_folds):
            idx_tr = np.where(fold_ids != fold)[0]
            idx_val = np.where(fold_ids == fold)[0]
            mapping = make_smoothed_mapping(
                col_values=tr_col[idx_tr],
                y=y_global[idx_tr],
                alpha=alpha,
                global_mean=global_mean,
            )
            val_encoded = pd.Series(tr_col[idx_val]).map(mapping).fillna(global_mean).to_numpy(np.float32)
            oof_encoded[idx_val] = val_encoded

        full_mapping = make_smoothed_mapping(
            col_values=tr_col,
            y=y_global,
            alpha=alpha,
            global_mean=global_mean,
        )
        test_encoded = pd.Series(te_col).map(full_mapping).fillna(global_mean).to_numpy(np.float32)

        new_col = f"{col}_te_global"
        new_cols.append(new_col)
        train_blocks.append(oof_encoded.astype(np.float32, copy=False))
        test_blocks.append(test_encoded.astype(np.float32, copy=False))

    if new_cols:
        train_te_df = pd.DataFrame(
            np.column_stack(train_blocks).astype(np.float32, copy=False),
            columns=new_cols,
            index=train_df.index,
        )
        test_te_df = pd.DataFrame(
            np.column_stack(test_blocks).astype(np.float32, copy=False),
            columns=new_cols,
            index=test_df.index,
        )
        train_df = pd.concat([train_df, train_te_df], axis=1)
        test_df = pd.concat([test_df, test_te_df], axis=1)

    return new_cols, train_df, test_df


def add_svd_features(
    cfg: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    svd_source_cols: Sequence[str],
) -> List[str]:
    if not bool(cfg["features"].get("add_svd", False)):
        return []
    if not svd_source_cols:
        return []

    n_components = int(cfg["features"].get("svd_components", 64))
    fit_rows = int(cfg["features"].get("svd_fit_rows", 200_000))
    batch_size = int(cfg["features"].get("svd_batch_size", 100_000))
    seed = int(cfg.get("seed", 42))

    n_components = min(n_components, len(svd_source_cols) - 1)
    if n_components < 2:
        return []

    rng = np.random.default_rng(seed)
    if fit_rows >= len(train_df):
        fit_idx = np.arange(len(train_df))
    else:
        fit_idx = rng.choice(len(train_df), size=fit_rows, replace=False)

    print(
        f"[add_svd_features] fit_rows={len(fit_idx)}, n_components={n_components}, "
        f"source_cols={len(svd_source_cols)}"
    )
    fit_matrix = (
        train_df.iloc[fit_idx][list(svd_source_cols)].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    )
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(fit_matrix)

    train_svd = np.empty((len(train_df), n_components), dtype=np.float32)
    for start in range(0, len(train_df), batch_size):
        end = min(start + batch_size, len(train_df))
        block = train_df.iloc[start:end][list(svd_source_cols)].fillna(0.0).to_numpy(np.float32, copy=False)
        train_svd[start:end] = svd.transform(block).astype(np.float32)

    test_svd = np.empty((len(test_df), n_components), dtype=np.float32)
    for start in range(0, len(test_df), batch_size):
        end = min(start + batch_size, len(test_df))
        block = test_df.iloc[start:end][list(svd_source_cols)].fillna(0.0).to_numpy(np.float32, copy=False)
        test_svd[start:end] = svd.transform(block).astype(np.float32)

    svd_cols: List[str] = []
    for i in range(n_components):
        col = f"svd_extra_{i:03d}"
        train_df[col] = train_svd[:, i]
        test_df[col] = test_svd[:, i]
        svd_cols.append(col)

    return svd_cols


def build_fold_ids(cfg: Dict[str, Any], y: np.ndarray) -> np.ndarray:
    n_splits = int(cfg["folds"].get("n_splits", 5))
    seed = int(cfg.get("seed", 42))
    method = str(cfg["folds"].get("method", "multilabel"))

    fold_ids = np.full(len(y), fill_value=-1, dtype=np.int16)

    if method == "multilabel":
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

            mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold, (_, val_idx) in enumerate(mskf.split(np.zeros((len(y), 1)), y)):
                fold_ids[val_idx] = fold
            return fold_ids
        except Exception:
            print("[build_fold_ids] iterstrat not found, fallback to surrogate multilabel strata")

    positives = y.sum(axis=1).astype(np.int16)
    prevalence = y.mean(axis=0)
    top_targets = np.argsort(prevalence)[::-1][:2]

    strata = positives.astype(str)
    for idx in top_targets:
        strata = np.char.add(np.char.add(strata, "_"), y[:, idx].astype(str))

    # Some strata can be too rare for n_splits. Fallback to positives-only buckets.
    vc = pd.Series(strata).value_counts()
    if (vc < n_splits).any():
        strata = positives.astype(str)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(strata)), strata)):
        fold_ids[val_idx] = fold
    return fold_ids


def prepare_dataset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(cfg["output_dir"])
    ensure_dir(out_dir)
    ensure_dir(out_dir / "meta")

    cols = discover_columns(cfg)
    target_cols = cols["target_cols"]
    base_cat_cols = cols["cat_cols"]
    selected_extra_cols = select_extra_columns(cfg, cols["extra_cols"])

    print(
        f"[prepare_dataset] main_features={len(cols['feature_cols_main'])}, "
        f"extra_selected={len(selected_extra_cols)}, targets={len(target_cols)}"
    )
    train_df, test_df = load_raw_joined_data(cfg, selected_extra_cols)

    row_stat_cols = [c for c in train_df.columns if c.startswith("row_extra_")]
    feature_cols = list(cols["feature_cols_main"]) + list(selected_extra_cols) + row_stat_cols

    pseudo_cat_cols = load_pseudo_cat_cols(cfg, feature_cols)
    cat_cols = list(dict.fromkeys(base_cat_cols + pseudo_cat_cols))

    encode_categories(train_df, test_df, cat_cols)
    cast_numeric_float32(train_df, test_df, feature_cols)

    y = train_df[target_cols].to_numpy(dtype=np.int8, copy=False)
    fold_ids = build_fold_ids(cfg, y)

    quantized_cols, train_df, test_df = add_quantized_num_features(cfg, train_df, test_df, feature_cols)
    feature_cols.extend(quantized_cols)

    te_cols, train_df, test_df = add_oof_global_target_encoding(
        cfg=cfg,
        train_df=train_df,
        test_df=test_df,
        target_cols=target_cols,
        cat_cols=cat_cols,
        fold_ids=fold_ids,
    )
    feature_cols.extend(te_cols)

    svd_cols = add_svd_features(cfg, train_df, test_df, selected_extra_cols)
    feature_cols.extend(svd_cols)

    fold_path = out_dir / "meta" / "fold_ids.npy"
    np.save(fold_path, fold_ids)

    meta = {
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "cat_cols": cat_cols,
        "pseudo_cat_cols": pseudo_cat_cols,
        "selected_extra_cols": list(selected_extra_cols),
        "row_stat_cols": row_stat_cols,
        "quantized_cols": quantized_cols,
        "te_cols": te_cols,
        "svd_cols": svd_cols,
    }
    with (out_dir / "meta" / "columns.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "cat_cols": cat_cols,
        "fold_ids": fold_ids,
    }


def sample_rows_by_class(y: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or len(y) <= max_rows:
        return np.arange(len(y))

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return rng.choice(len(y), size=max_rows, replace=False)

    pos_share = len(pos_idx) / len(y)
    pos_take = int(round(max_rows * pos_share))
    pos_take = max(1, min(pos_take, len(pos_idx) - 1))
    neg_take = max_rows - pos_take
    neg_take = max(1, min(neg_take, len(neg_idx) - 1))

    sampled_pos = rng.choice(pos_idx, size=pos_take, replace=False)
    sampled_neg = rng.choice(neg_idx, size=neg_take, replace=False)
    idx = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(idx)
    return idx


def sample_rows_random(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return rng.choice(n_rows, size=max_rows, replace=False)


def build_per_target_feature_map(
    cfg: Dict[str, Any],
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    cat_cols: Sequence[str],
) -> Dict[str, List[str]]:
    per_target_top_k = int(cfg["features"].get("per_target_top_k", 0))
    if per_target_top_k <= 0:
        return {}

    seed = int(cfg.get("seed", 42))
    sample_rows = int(cfg["features"].get("per_target_sample_rows", 200_000))
    per_target_union_cap = int(cfg["features"].get("per_target_union_cap", 0))
    per_target_mi_source_top_k = int(cfg["features"].get("per_target_mi_source_top_k", 0))

    out_dir = Path(cfg["output_dir"])
    ensure_dir(out_dir / "meta")
    cache_path = out_dir / "meta" / "per_target_features.json"
    reuse_cache = bool(cfg["features"].get("reuse_per_target_feature_map", True))

    if reuse_cache and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        cached = payload.get("target_feature_cols", {})
        cached_top_k = int(payload.get("per_target_top_k", -1))
        cached_sample_rows = int(payload.get("sample_rows", -1))
        cached_union_cap = int(payload.get("per_target_union_cap", 0))
        cached_mi_source_top_k = int(payload.get("per_target_mi_source_top_k", 0))
        if (
            cached_top_k == per_target_top_k
            and cached_sample_rows == sample_rows
            and cached_union_cap == per_target_union_cap
            and cached_mi_source_top_k == per_target_mi_source_top_k
            and all(t in cached for t in target_cols)
        ):
            print(f"[feature_map] loaded cache: {cache_path}")
            return {t: list(cached[t]) for t in target_cols}

    always_keep_prefixes = list(
        cfg["features"].get("per_target_always_keep_prefixes", ["row_extra_", "svd_extra_"])
    )
    always_keep = [
        c for c in feature_cols if any(c.startswith(prefix) for prefix in always_keep_prefixes)
    ]
    base_keep = [
        c for c in feature_cols if c.startswith("cat_feature") or c.startswith("num_feature")
    ][: min(40, len(feature_cols))]

    mi_feature_pool = list(feature_cols)
    if 0 < per_target_mi_source_top_k < len(mi_feature_pool):
        mi_feature_pool = mi_feature_pool[:per_target_mi_source_top_k]
    mi_feature_pool = list(dict.fromkeys(mi_feature_pool + always_keep + base_keep))

    idx_sample = sample_rows_random(n_rows=len(train_df), max_rows=sample_rows, seed=seed)
    x_sample = (
        train_df.iloc[idx_sample][mi_feature_pool]
        .fillna(0.0)
        .to_numpy(dtype=np.float32, copy=False)
    )
    cat_set = set(cat_cols)
    discrete_mask = np.array([c in cat_set for c in mi_feature_pool], dtype=bool)

    feature_map: Dict[str, List[str]] = {}
    top_selected_by_target: Dict[str, List[str]] = {}
    print(
        f"[feature_map] computing MI map: targets={len(target_cols)} sample_rows={len(idx_sample)} "
        f"top_k={per_target_top_k} union_cap={per_target_union_cap if per_target_union_cap > 0 else 'off'} "
        f"mi_pool={len(mi_feature_pool)}"
    )
    for i, target in enumerate(target_cols):
        y = train_df.iloc[idx_sample][target].to_numpy(dtype=np.int8, copy=False)
        positives = int(y.sum())
        if positives == 0 or positives == len(y):
            selected = list(mi_feature_pool[:per_target_top_k])
        else:
            mi = mutual_info_classif(
                x_sample,
                y,
                discrete_features=discrete_mask,
                random_state=seed + i,
                n_neighbors=3,
            )
            order = np.argsort(mi)[::-1]
            selected = [mi_feature_pool[idx] for idx in order[:per_target_top_k]]

        top_selected_by_target[target] = list(dict.fromkeys(selected))
        if (i + 1) % 5 == 0:
            print(f"[feature_map] progress {i + 1}/{len(target_cols)} targets")

    if per_target_union_cap > 0:
        freq: Dict[str, int] = {}
        rank_sum: Dict[str, int] = {}
        for selected in top_selected_by_target.values():
            for rank, feature in enumerate(selected):
                freq[feature] = freq.get(feature, 0) + 1
                rank_sum[feature] = rank_sum.get(feature, 0) + rank

        ranked_union = sorted(
            freq.keys(),
            key=lambda c: (-freq[c], rank_sum[c] / max(freq[c], 1), c),
        )
        keep_union = set(ranked_union[:per_target_union_cap])
        print(
            f"[feature_map] union cap applied: raw_union={len(ranked_union)} "
            f"kept={len(keep_union)}"
        )

        global_fallback = ranked_union[: min(64, len(ranked_union))]
        for target in target_cols:
            selected = top_selected_by_target[target]
            filtered = [c for c in selected if c in keep_union]
            if not filtered:
                filtered = list(global_fallback)
            merged = list(dict.fromkeys(filtered + always_keep + base_keep))
            feature_map[target] = merged
    else:
        for target in target_cols:
            selected = top_selected_by_target[target]
            merged = list(dict.fromkeys(selected + always_keep + base_keep))
            feature_map[target] = merged

    payload = {
        "per_target_top_k": per_target_top_k,
        "sample_rows": sample_rows,
        "per_target_union_cap": per_target_union_cap,
        "per_target_mi_source_top_k": per_target_mi_source_top_k,
        "target_feature_cols": feature_map,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[feature_map] saved cache: {cache_path}")
    return feature_map


def fit_predict_one_fold(
    model_name: str,
    model_params: Dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    x_test: pd.DataFrame,
    cat_cols: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "catboost":
        from catboost import CatBoostClassifier, CatBoostError

        params = dict(model_params)
        auto_weight = bool(params.pop("auto_scale_pos_weight", True))
        if auto_weight:
            pos = int(y_train.sum())
            neg = int(len(y_train) - pos)
            ratio = (neg / max(pos, 1)) if pos > 0 else 1.0
            params["scale_pos_weight"] = float(ratio)

        cat_idx = [x_train.columns.get_loc(c) for c in cat_cols if c in x_train.columns]
        model = CatBoostClassifier(**params)
        try:
            model.fit(x_train, y_train, cat_features=cat_idx, eval_set=(x_val, y_val), verbose=False)
        except CatBoostError as e:
            err = str(e).lower()
            if params.get("task_type") == "GPU" and (
                "environment for task type [gpu] not found" in err
                or "cuda" in err
            ):
                cpu_params = dict(params)
                cpu_params.pop("devices", None)
                cpu_params["task_type"] = "CPU"
                print("[catboost] GPU unavailable, fallback to CPU")
                model = CatBoostClassifier(**cpu_params)
                model.fit(
                    x_train,
                    y_train,
                    cat_features=cat_idx,
                    eval_set=(x_val, y_val),
                    verbose=False,
                )
            else:
                raise
        val_pred = model.predict_proba(x_val)[:, 1].astype(np.float32)
        test_pred = model.predict_proba(x_test)[:, 1].astype(np.float32)
        return val_pred, test_pred

    if model_name == "lightgbm":
        import lightgbm as lgb

        params = dict(model_params)
        early_stopping_rounds = int(params.pop("early_stopping_rounds", 100))
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="auc",
            categorical_feature=[c for c in cat_cols if c in x_train.columns],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )
        val_pred = model.predict_proba(x_val)[:, 1].astype(np.float32)
        test_pred = model.predict_proba(x_test)[:, 1].astype(np.float32)
        return val_pred, test_pred

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        params = dict(model_params)
        early_stopping_rounds = int(params.pop("early_stopping_rounds", 100))
        model = XGBClassifier(**params)
        fit_kwargs: Dict[str, Any] = {"eval_set": [(x_val, y_val)], "verbose": False}
        if early_stopping_rounds > 0:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        model.fit(x_train, y_train, **fit_kwargs)
        val_pred = model.predict_proba(x_val)[:, 1].astype(np.float32)
        test_pred = model.predict_proba(x_test)[:, 1].astype(np.float32)
        return val_pred, test_pred

    if model_name == "mlp":
        params = dict(model_params)
        max_train_rows = int(params.pop("max_train_rows", 0))
        random_state = int(params.get("random_state", 42))

        x_train_np = x_train.fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        y_train_np = y_train.astype(np.int8, copy=False)
        x_val_np = x_val.fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        x_test_np = x_test.fillna(0.0).to_numpy(dtype=np.float32, copy=False)

        if max_train_rows > 0 and len(x_train_np) > max_train_rows:
            sample_idx = sample_rows_by_class(
                y=y_train_np,
                max_rows=max_train_rows,
                seed=random_state,
            )
            x_train_np = x_train_np[sample_idx]
            y_train_np = y_train_np[sample_idx]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_np)
        x_val_scaled = scaler.transform(x_val_np)
        x_test_scaled = scaler.transform(x_test_np)

        model = MLPClassifier(**params)
        model.fit(x_train_scaled, y_train_np)
        val_pred = model.predict_proba(x_val_scaled)[:, 1].astype(np.float32)
        test_pred = model.predict_proba(x_test_scaled)[:, 1].astype(np.float32)
        return val_pred, test_pred

    raise ValueError(f"Unsupported model: {model_name}")


def train_base_model(
    cfg: Dict[str, Any],
    model_name: str,
    model_cfg: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    cat_cols: Sequence[str],
    fold_ids: np.ndarray,
    target_feature_map: Dict[str, List[str]] | None = None,
) -> None:
    out_dir = Path(cfg["output_dir"])
    ensure_dir(out_dir / "base")
    ensure_dir(out_dir / "scores")

    n_folds = int(cfg["folds"].get("n_splits", 5))
    x_all = train_df[list(feature_cols)]
    x_test_all = test_df[list(feature_cols)]

    oof_df = pd.DataFrame({"customer_id": train_df["customer_id"].values})
    test_pred_df = pd.DataFrame({"customer_id": test_df["customer_id"].values})
    target_scores: Dict[str, float] = {}

    print(f"[train_base_model] model={model_name} targets={len(target_cols)} folds={n_folds}")
    for target in target_cols:
        target_features = (
            target_feature_map.get(target, list(feature_cols))
            if target_feature_map is not None
            else list(feature_cols)
        )
        target_cat_cols = [c for c in cat_cols if c in target_features]
        x_target = x_all[target_features]
        x_test = x_test_all[target_features]

        y = train_df[target].to_numpy(dtype=np.int8, copy=False)
        positives = int(y.sum())
        if positives == 0 or positives == len(y):
            constant = float(y.mean())
            oof = np.full(len(y), constant, dtype=np.float32)
            test_pred = np.full(len(test_df), constant, dtype=np.float32)
            oof_df[target] = oof
            test_pred_df[target_to_predict_col(target)] = test_pred
            target_scores[target] = float("nan")
            continue

        oof = np.zeros(len(y), dtype=np.float32)
        test_pred = np.zeros(len(test_df), dtype=np.float32)
        for fold in range(n_folds):
            idx_val = np.where(fold_ids == fold)[0]
            idx_tr = np.where(fold_ids != fold)[0]

            x_train = x_target.iloc[idx_tr]
            y_train = y[idx_tr]
            x_val = x_target.iloc[idx_val]
            y_val = y[idx_val]

            val_pred, fold_test_pred = fit_predict_one_fold(
                model_name=model_name,
                model_params=model_cfg["params"],
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                cat_cols=target_cat_cols,
            )
            oof[idx_val] = val_pred
            test_pred += fold_test_pred / n_folds

        score = roc_auc_score(y, oof)
        target_scores[target] = float(score)
        oof_df[target] = oof
        test_pred_df[target_to_predict_col(target)] = test_pred
        print(f"  - {target}: AUC={score:.6f} pos={positives}")

    valid_scores = [v for v in target_scores.values() if not math.isnan(v)]
    macro_auc = float(np.mean(valid_scores)) if valid_scores else float("nan")
    print(f"[train_base_model] model={model_name} macro_auc={macro_auc:.6f}")

    oof_path = out_dir / "base" / f"{model_name}_oof.parquet"
    test_path = out_dir / "base" / f"{model_name}_test.parquet"
    oof_df.to_parquet(oof_path, index=False)
    test_pred_df.to_parquet(test_path, index=False)

    score_payload = {
        "model": model_name,
        "macro_auc": macro_auc,
        "target_scores": target_scores,
    }
    with (out_dir / "scores" / f"{model_name}_scores.json").open("w", encoding="utf-8") as f:
        json.dump(score_payload, f, ensure_ascii=False, indent=2)


def simplex_grid(n_models: int, step: float) -> np.ndarray:
    units = int(round(1.0 / step))
    combos: List[List[int]] = []

    def rec(depth: int, remaining: int, current: List[int]) -> None:
        if depth == n_models - 1:
            combos.append(current + [remaining])
            return
        for value in range(remaining + 1):
            rec(depth + 1, remaining - value, current + [value])

    rec(0, units, [])
    return np.array(combos, dtype=np.float32) / units


def load_base_predictions(
    output_dir: Path, model_names: Sequence[str]
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    oof_map: Dict[str, pd.DataFrame] = {}
    test_map: Dict[str, pd.DataFrame] = {}
    for model_name in model_names:
        oof_path = output_dir / "base" / f"{model_name}_oof.parquet"
        test_path = output_dir / "base" / f"{model_name}_test.parquet"
        if not oof_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Missing base predictions for model={model_name}. "
                f"Expected {oof_path} and {test_path}"
            )
        oof_map[model_name] = pd.read_parquet(oof_path)
        test_map[model_name] = pd.read_parquet(test_path)
    return oof_map, test_map


def run_blend(cfg: Dict[str, Any], model_names: Sequence[str]) -> None:
    out_dir = Path(cfg["output_dir"])
    ensure_dir(out_dir / "ensemble")
    ensure_dir(out_dir / "scores")

    train_target = pd.read_parquet(Path(cfg["data"]["train_target"]))
    target_cols = [c for c in train_target.columns if c.startswith("target_")]

    oof_map, test_map = load_base_predictions(out_dir, model_names)
    base_customer = oof_map[model_names[0]]["customer_id"]
    test_customer = test_map[model_names[0]]["customer_id"]

    step = float(cfg["blend"].get("weight_step", 0.05))
    use_rank = bool(cfg["blend"].get("use_rank", True))
    weights_grid = simplex_grid(len(model_names), step)

    blend_oof = pd.DataFrame({"customer_id": base_customer.values})
    blend_test = pd.DataFrame({"customer_id": test_customer.values})
    target_scores: Dict[str, float] = {}
    target_weights: Dict[str, List[float]] = {}

    for target in target_cols:
        y = train_target[target].to_numpy(dtype=np.int8, copy=False)
        oof_parts = [oof_map[m][target].to_numpy(dtype=np.float32, copy=False) for m in model_names]
        test_parts = [
            test_map[m][target_to_predict_col(target)].to_numpy(dtype=np.float32, copy=False)
            for m in model_names
        ]

        positives = int(y.sum())
        if positives == 0 or positives == len(y):
            uniform_w = [1.0 / len(model_names)] * len(model_names)
            blended_oof = np.mean(np.column_stack(oof_parts), axis=1).astype(np.float32)
            blended_test = np.mean(np.column_stack(test_parts), axis=1).astype(np.float32)
            target_scores[target] = float("nan")
            target_weights[target] = uniform_w
            blend_oof[target] = blended_oof
            blend_test[target_to_predict_col(target)] = blended_test
            print(f"[blend] {target} constant target, use uniform weights={uniform_w}")
            continue

        if use_rank:
            oof_parts = [rank_normalize(v) for v in oof_parts]
            test_parts = [rank_normalize(v) for v in test_parts]

        best_auc = -1.0
        best_w = None
        best_oof = None
        best_test = None

        for w in weights_grid:
            candidate_oof = np.zeros(len(y), dtype=np.float32)
            candidate_test = np.zeros(len(test_customer), dtype=np.float32)
            for i, wi in enumerate(w):
                candidate_oof += wi * oof_parts[i]
                candidate_test += wi * test_parts[i]
            auc = roc_auc_score(y, candidate_oof)
            if auc > best_auc:
                best_auc = float(auc)
                best_w = w
                best_oof = candidate_oof
                best_test = candidate_test

        assert best_oof is not None and best_test is not None and best_w is not None
        target_scores[target] = best_auc
        target_weights[target] = [float(x) for x in best_w.tolist()]
        blend_oof[target] = best_oof
        blend_test[target_to_predict_col(target)] = best_test
        print(f"[blend] {target} auc={best_auc:.6f} weights={target_weights[target]}")

    valid_scores = [v for v in target_scores.values() if not math.isnan(v)]
    macro_auc = float(np.mean(valid_scores)) if valid_scores else float("nan")
    print(f"[blend] macro_auc={macro_auc:.6f}")

    blend_oof.to_parquet(out_dir / "ensemble" / "blend_oof.parquet", index=False)
    blend_test.to_parquet(out_dir / "ensemble" / "blend_test.parquet", index=False)

    payload = {
        "models": list(model_names),
        "macro_auc": macro_auc,
        "target_scores": target_scores,
        "target_weights": target_weights,
        "use_rank": use_rank,
        "step": step,
    }
    with (out_dir / "scores" / "blend_scores.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_stack(cfg: Dict[str, Any], model_names: Sequence[str]) -> None:
    out_dir = Path(cfg["output_dir"])
    ensure_dir(out_dir / "ensemble")
    ensure_dir(out_dir / "scores")

    train_target = pd.read_parquet(Path(cfg["data"]["train_target"]))
    target_cols = [c for c in train_target.columns if c.startswith("target_")]
    fold_ids = np.load(out_dir / "meta" / "fold_ids.npy")
    n_folds = int(cfg["folds"].get("n_splits", 5))

    oof_map, test_map = load_base_predictions(out_dir, model_names)
    base_customer = oof_map[model_names[0]]["customer_id"]
    test_customer = test_map[model_names[0]]["customer_id"]

    max_iter = int(cfg["stack"].get("max_iter", 1200))
    c_value = float(cfg["stack"].get("C", 0.5))
    class_weight = cfg["stack"].get("class_weight", "balanced")
    cross_target_top_k = int(cfg["stack"].get("cross_target_top_k", 0))
    cross_target_models_cfg = cfg["stack"].get("cross_target_models", list(model_names))
    if isinstance(cross_target_models_cfg, str):
        cross_target_models = [x.strip() for x in cross_target_models_cfg.split(",") if x.strip()]
    else:
        cross_target_models = [str(x) for x in cross_target_models_cfg]
    cross_target_models = [m for m in cross_target_models if m in model_names]
    corr_abs = train_target[target_cols].corr().abs() if cross_target_top_k > 0 else None

    stack_oof = pd.DataFrame({"customer_id": base_customer.values})
    stack_test = pd.DataFrame({"customer_id": test_customer.values})
    target_scores: Dict[str, float] = {}

    for target in target_cols:
        y = train_target[target].to_numpy(dtype=np.int8, copy=False)
        oof_parts = [oof_map[m][target].to_numpy(dtype=np.float32, copy=False) for m in model_names]
        test_parts = [
            test_map[m][target_to_predict_col(target)].to_numpy(dtype=np.float32, copy=False)
            for m in model_names
        ]

        if corr_abs is not None and cross_target_models and cross_target_top_k > 0:
            neighbors = (
                corr_abs[target]
                .drop(labels=[target], errors="ignore")
                .sort_values(ascending=False)
                .head(cross_target_top_k)
                .index.tolist()
            )
            for neighbor in neighbors:
                neighbor_pred_col = target_to_predict_col(neighbor)
                for model_name in cross_target_models:
                    if neighbor not in oof_map[model_name].columns:
                        continue
                    if neighbor_pred_col not in test_map[model_name].columns:
                        continue
                    oof_parts.append(
                        oof_map[model_name][neighbor].to_numpy(dtype=np.float32, copy=False)
                    )
                    test_parts.append(
                        test_map[model_name][neighbor_pred_col].to_numpy(dtype=np.float32, copy=False)
                    )

        x_meta = np.column_stack(oof_parts)
        x_meta_test = np.column_stack(test_parts)

        if int(y.sum()) == 0 or int(y.sum()) == len(y):
            constant = float(y.mean())
            oof_pred = np.full(len(y), constant, dtype=np.float32)
            test_pred = np.full(len(test_customer), constant, dtype=np.float32)
            stack_oof[target] = oof_pred
            stack_test[target_to_predict_col(target)] = test_pred
            target_scores[target] = float("nan")
            continue

        oof_pred = np.zeros(len(y), dtype=np.float32)
        test_pred = np.zeros(len(test_customer), dtype=np.float32)

        for fold in range(n_folds):
            idx_val = np.where(fold_ids == fold)[0]
            idx_tr = np.where(fold_ids != fold)[0]

            model = LogisticRegression(
                C=c_value,
                max_iter=max_iter,
                class_weight=class_weight,
            )
            model.fit(x_meta[idx_tr], y[idx_tr])
            oof_pred[idx_val] = model.predict_proba(x_meta[idx_val])[:, 1].astype(np.float32)
            test_pred += model.predict_proba(x_meta_test)[:, 1].astype(np.float32) / n_folds

        auc = roc_auc_score(y, oof_pred)
        target_scores[target] = float(auc)
        stack_oof[target] = oof_pred
        stack_test[target_to_predict_col(target)] = test_pred
        print(f"[stack] {target} auc={auc:.6f}")

    valid_scores = [v for v in target_scores.values() if not math.isnan(v)]
    macro_auc = float(np.mean(valid_scores)) if valid_scores else float("nan")
    print(f"[stack] macro_auc={macro_auc:.6f}")

    stack_oof.to_parquet(out_dir / "ensemble" / "stack_oof.parquet", index=False)
    stack_test.to_parquet(out_dir / "ensemble" / "stack_test.parquet", index=False)

    payload = {
        "models": list(model_names),
        "macro_auc": macro_auc,
        "target_scores": target_scores,
        "meta_model": "logistic_regression",
        "cross_target_top_k": cross_target_top_k,
        "cross_target_models": cross_target_models,
    }
    with (out_dir / "scores" / "stack_scores.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_feature_only_dataset(
    cfg: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    cols = discover_columns(cfg)
    selected_extra_cols = select_extra_columns(cfg, cols["extra_cols"])
    train_main = pl.read_parquet(Path(cfg["data"]["train_main"]))
    test_main = pl.read_parquet(Path(cfg["data"]["test_main"]))

    train_extra = pl.read_parquet(
        Path(cfg["data"]["train_extra"]), columns=["customer_id"] + list(selected_extra_cols)
    )
    test_extra = pl.read_parquet(
        Path(cfg["data"]["test_extra"]), columns=["customer_id"] + list(selected_extra_cols)
    )

    if bool(cfg["features"].get("add_row_stats", True)):
        train_extra = add_row_stats(train_extra, selected_extra_cols)
        test_extra = add_row_stats(test_extra, selected_extra_cols)

    train_df = train_main.join(train_extra, on="customer_id", how="inner").to_pandas()
    test_df = test_main.join(test_extra, on="customer_id", how="inner").to_pandas()

    row_stat_cols = [c for c in train_df.columns if c.startswith("row_extra_")]
    feature_cols = list(cols["feature_cols_main"]) + list(selected_extra_cols) + row_stat_cols
    base_cat_cols = [c for c in cols["cat_cols"] if c in feature_cols]
    pseudo_cat_cols = load_pseudo_cat_cols(cfg, feature_cols)
    cat_cols = list(dict.fromkeys(base_cat_cols + pseudo_cat_cols))

    encode_categories(train_df, test_df, cat_cols)
    cast_numeric_float32(train_df, test_df, feature_cols)
    quantized_cols, train_df, test_df = add_quantized_num_features(cfg, train_df, test_df, feature_cols)
    feature_cols.extend(quantized_cols)
    return train_df, test_df, feature_cols, cat_cols, row_stat_cols


def run_advval(cfg: Dict[str, Any]) -> None:
    out_dir = Path(cfg["output_dir"])
    ensure_dir(out_dir / "scores")

    adv_cfg = cfg.get("advval", {})
    sample_rows_per_class = int(adv_cfg.get("sample_rows_per_class", 150_000))
    n_splits = int(adv_cfg.get("n_splits", 3))
    seed = int(cfg.get("seed", 42))
    model_name = str(adv_cfg.get("model", "lightgbm"))
    shift_threshold = float(adv_cfg.get("shift_threshold", 0.55))

    train_df, test_df, feature_cols, cat_cols, _ = load_feature_only_dataset(cfg)

    idx_train = sample_rows_random(len(train_df), sample_rows_per_class, seed)
    idx_test = sample_rows_random(len(test_df), sample_rows_per_class, seed + 17)
    x_train = train_df.iloc[idx_train][feature_cols]
    x_test = test_df.iloc[idx_test][feature_cols]

    x_all = pd.concat([x_train, x_test], axis=0, ignore_index=True)
    y_all = np.concatenate(
        [
            np.zeros(len(x_train), dtype=np.int8),
            np.ones(len(x_test), dtype=np.int8),
        ]
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs: List[float] = []
    fold_importances: List[np.ndarray] = []
    for fold, (idx_tr, idx_val) in enumerate(skf.split(x_all, y_all)):
        x_tr = x_all.iloc[idx_tr]
        y_tr = y_all[idx_tr]
        x_val = x_all.iloc[idx_val]
        y_val = y_all[idx_val]

        if model_name == "lightgbm":
            import lightgbm as lgb

            params = dict(
                adv_cfg.get(
                    "lightgbm_params",
                    {
                        "n_estimators": 1200,
                        "learning_rate": 0.03,
                        "num_leaves": 127,
                        "subsample": 0.85,
                        "colsample_bytree": 0.85,
                        "objective": "binary",
                        "n_jobs": -1,
                        "random_state": seed + fold,
                    },
                )
            )
            early_stopping_rounds = int(adv_cfg.get("early_stopping_rounds", 120))
            model = lgb.LGBMClassifier(**params)
            model.fit(
                x_tr,
                y_tr,
                eval_set=[(x_val, y_val)],
                eval_metric="auc",
                categorical_feature=[c for c in cat_cols if c in x_tr.columns],
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
            )
            val_pred = model.predict_proba(x_val)[:, 1]
            imp = model.feature_importances_.astype(np.float64, copy=False)
            imp_sum = float(np.sum(imp))
            if imp_sum > 0:
                imp = imp / imp_sum
            fold_importances.append(imp)
        else:
            scaler = StandardScaler()
            x_tr_np = scaler.fit_transform(x_tr.fillna(0.0).to_numpy(dtype=np.float32, copy=False))
            x_val_np = scaler.transform(x_val.fillna(0.0).to_numpy(dtype=np.float32, copy=False))
            model = LogisticRegression(max_iter=600)
            model.fit(x_tr_np, y_tr)
            val_pred = model.predict_proba(x_val_np)[:, 1]
            imp = np.abs(model.coef_[0]).astype(np.float64, copy=False)
            imp_sum = float(np.sum(imp))
            if imp_sum > 0:
                imp = imp / imp_sum
            fold_importances.append(imp)

        auc = roc_auc_score(y_val, val_pred)
        fold_aucs.append(float(auc))
        print(f"[advval] fold={fold} auc={auc:.6f}")

    mean_auc = float(np.mean(fold_aucs))
    is_shift = bool(mean_auc > shift_threshold)
    print(f"[advval] mean_auc={mean_auc:.6f} threshold={shift_threshold:.3f} shift_detected={is_shift}")

    top_k = int(adv_cfg.get("top_drift_features", 30))
    top_drift_features: List[Dict[str, Any]] = []
    if fold_importances:
        imp_mean = np.mean(np.vstack(fold_importances), axis=0)
        order = np.argsort(imp_mean)[::-1]
        top_idx = order[: min(top_k, len(feature_cols))]
        for idx in top_idx:
            top_drift_features.append(
                {
                    "feature": feature_cols[int(idx)],
                    "importance": float(imp_mean[int(idx)]),
                }
            )
        if top_drift_features:
            print(
                "[advval] top_drift_features="
                + ", ".join([f"{x['feature']}:{x['importance']:.4f}" for x in top_drift_features[:10]])
            )

    payload = {
        "model": model_name,
        "n_splits": n_splits,
        "sample_rows_per_class": sample_rows_per_class,
        "fold_aucs": fold_aucs,
        "mean_auc": mean_auc,
        "shift_threshold": shift_threshold,
        "shift_detected": is_shift,
        "top_drift_features": top_drift_features,
    }
    with (out_dir / "scores" / "advval.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def make_submission(cfg: Dict[str, Any], source_path: Path, output_path: Path) -> None:
    sample_path = Path(cfg["data"]["sample_submit"])
    sample = pd.read_parquet(sample_path)
    pred = pd.read_parquet(source_path)

    submission = pd.DataFrame()
    for col in sample.columns:
        expected_dtype = sample[col].dtype
        if col == "customer_id":
            if "customer_id" not in pred.columns:
                raise KeyError(f"Missing required submission column 'customer_id' in {source_path}")
            submission[col] = pred["customer_id"].astype(expected_dtype)
            continue

        if col in pred.columns:
            submission[col] = pd.to_numeric(pred[col], errors="coerce").astype(expected_dtype)
            continue

        target_col = predict_to_target_col(col)
        if target_col in pred.columns:
            submission[col] = pd.to_numeric(pred[target_col], errors="coerce").astype(expected_dtype)
            continue

        raise KeyError(f"Missing required submission column '{col}' in {source_path}")

    submission = submission[sample.columns.tolist()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(output_path, index=False)
    print(f"[submit] saved {output_path}")


def run_base_models(cfg: Dict[str, Any], selected_models: Sequence[str] | None) -> None:
    prepared = prepare_dataset(cfg)
    train_df = prepared["train_df"]
    test_df = prepared["test_df"]
    feature_cols = prepared["feature_cols"]
    target_cols = prepared["target_cols"]
    cat_cols = prepared["cat_cols"]
    fold_ids = prepared["fold_ids"]

    all_models = [name for name, mc in cfg["models"].items() if bool(mc.get("enabled", False))]
    model_names = list(selected_models) if selected_models else all_models
    if not model_names:
        raise ValueError("No enabled models found. Check config['models'][*]['enabled'].")

    target_feature_map = build_per_target_feature_map(
        cfg=cfg,
        train_df=train_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        cat_cols=cat_cols,
    )

    for model_name in model_names:
        if model_name not in cfg["models"]:
            raise KeyError(f"Model '{model_name}' is not defined in config")
        train_base_model(
            cfg=cfg,
            model_name=model_name,
            model_cfg=cfg["models"][model_name],
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_cols=target_cols,
            cat_cols=cat_cols,
            fold_ids=fold_ids,
            target_feature_map=target_feature_map,
        )


def parse_models_arg(text: str | None) -> List[str] | None:
    if text is None:
        return None
    items = [x.strip() for x in text.split(",") if x.strip()]
    return items or None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Top-1 oriented ensemble pipeline for Cyber Shelf task")
    p.add_argument("--config", required=True, help="Path to JSON config")

    sub = p.add_subparsers(dest="command", required=True)

    p_bases = sub.add_parser("run-bases", help="Train all base models and save OOF/test predictions")
    p_bases.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset of models (e.g. catboost,lightgbm,xgboost)",
    )

    p_blend = sub.add_parser("blend", help="Per-target OOF blend with simplex grid search")
    p_blend.add_argument("--models", required=True, help="Comma-separated model names for blending")

    p_stack = sub.add_parser("stack", help="L2 stacking on OOF base predictions")
    p_stack.add_argument("--models", required=True, help="Comma-separated model names for stacking")

    sub.add_parser("advval", help="Run adversarial validation train-vs-test shift check")

    p_submit = sub.add_parser("submit", help="Create competition parquet from prediction parquet")
    p_submit.add_argument(
        "--source",
        required=True,
        help="Prediction parquet path (blend_test.parquet, stack_test.parquet, etc.)",
    )
    p_submit.add_argument("--output", required=True, help="Final submission parquet path")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    if args.command == "run-bases":
        run_base_models(cfg, parse_models_arg(args.models))
        return

    if args.command == "blend":
        model_names = parse_models_arg(args.models)
        if not model_names:
            raise ValueError("--models is required for blend")
        run_blend(cfg, model_names)
        return

    if args.command == "stack":
        model_names = parse_models_arg(args.models)
        if not model_names:
            raise ValueError("--models is required for stack")
        run_stack(cfg, model_names)
        return

    if args.command == "advval":
        run_advval(cfg)
        return

    if args.command == "submit":
        make_submission(cfg, Path(args.source), Path(args.output))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
