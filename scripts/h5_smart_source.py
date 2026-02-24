#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import top1_pipeline as p1


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("sources", [])
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("Unsupported manifest JSON format")
    out: List[Dict[str, str]] = []
    for item in items:
        name = item.get("name", item.get("label"))
        if not name:
            raise KeyError("Manifest source item must contain 'name' or 'label'")
        out.append({
            "name": str(name),
            "oof_path": str(item["oof_path"]),
            "test_path": str(item["test_path"]),
        })
    return out


def _read_source_pair(spec: Dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = pd.read_parquet(spec["oof_path"])
    tst = pd.read_parquet(spec["test_path"])
    return oof, tst


def _resolve_sources(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    c = cfg.get("h5_smart", {})
    sources: List[Dict[str, str]] = []
    if c.get("source_manifest"):
        sources.extend(_load_manifest(Path(c["source_manifest"])))
    for s in c.get("sources", []):
        sources.append({"name": str(s["name"]), "oof_path": str(s["oof_path"]), "test_path": str(s["test_path"])})
    if not sources:
        raise ValueError("Provide h5_smart.source_manifest or h5_smart.sources")
    return sources


def _target_to_predcol(target: str) -> str:
    return p1.target_to_predict_col(target)


def _load_state(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "scores" / "h5smart_state.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"schema_version": 1, "created_at": now_ts(), "updated_at": now_ts(), "targets": {}, "global": {"targets_done": 0, "targets_failed": 0}}


def _save_state(out_dir: Path, state: Dict[str, Any]) -> None:
    p1.ensure_dir(out_dir / "scores")
    state["updated_at"] = now_ts()
    (out_dir / "scores" / "h5smart_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _auc_safe(y: np.ndarray, p: np.ndarray) -> float:
    pos = int(y.sum())
    if pos == 0 or pos == len(y):
        return float("nan")
    return float(roc_auc_score(y, p))


def _get_source_col(df: pd.DataFrame, target: str, *, is_test: bool) -> str | None:
    cand = _target_to_predcol(target) if is_test else target
    if cand in df.columns:
        return cand
    if (not is_test) and _target_to_predcol(target) in df.columns:
        return _target_to_predcol(target)
    return None


def _make_meta_features(
    target: str,
    neighbors: Sequence[str],
    target_source_names: Sequence[str],
    neighbor_source_names: Sequence[str],
    oof_map: Dict[str, pd.DataFrame],
    test_map: Dict[str, pd.DataFrame],
    add_pairwise_products: bool,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    oof_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    names: List[str] = []

    for s in target_source_names:
        c_oof = _get_source_col(oof_map[s], target, is_test=False)
        c_tst = _get_source_col(test_map[s], target, is_test=True)
        if c_oof and c_tst:
            oof_parts.append(oof_map[s][c_oof].to_numpy(dtype=np.float32, copy=False))
            test_parts.append(test_map[s][c_tst].to_numpy(dtype=np.float32, copy=False))
            names.append(f"self::{s}")

    for n in neighbors:
        for s in neighbor_source_names:
            c_oof = _get_source_col(oof_map[s], n, is_test=False)
            c_tst = _get_source_col(test_map[s], n, is_test=True)
            if c_oof and c_tst:
                oof_parts.append(oof_map[s][c_oof].to_numpy(dtype=np.float32, copy=False))
                test_parts.append(test_map[s][c_tst].to_numpy(dtype=np.float32, copy=False))
                names.append(f"nbr::{n}::{s}")

    if not oof_parts:
        raise ValueError(f"No available meta features for {target}")

    x_oof = np.column_stack(oof_parts).astype(np.float32)
    x_tst = np.column_stack(test_parts).astype(np.float32)

    if add_pairwise_products and len(target_source_names) >= 2:
        # pairwise products only among current-target source preds (first |target_source_names| columns, subset if missing)
        self_cols = [i for i, n in enumerate(names) if n.startswith("self::")]
        prod_oof_parts = []
        prod_tst_parts = []
        prod_names = []
        for i, j in itertools.combinations(self_cols, 2):
            prod_oof_parts.append((x_oof[:, i] * x_oof[:, j]).astype(np.float32))
            prod_tst_parts.append((x_tst[:, i] * x_tst[:, j]).astype(np.float32))
            prod_names.append(f"prod::{names[i]}*{names[j]}")
        if prod_oof_parts:
            x_oof = np.column_stack([x_oof] + prod_oof_parts).astype(np.float32)
            x_tst = np.column_stack([x_tst] + prod_tst_parts).astype(np.float32)
            names.extend(prod_names)

    return x_oof, x_tst, names


def run_h5_smart(cfg: Dict[str, Any], resume: bool = True) -> None:
    hcfg = cfg.get("h5_smart", {})
    out_dir = Path(cfg["output_dir"])
    p1.ensure_dir(out_dir / "base")
    p1.ensure_dir(out_dir / "ensemble")
    p1.ensure_dir(out_dir / "scores")

    train_target = pd.read_parquet(Path(cfg["data"]["train_target"]))
    target_cols = [c for c in train_target.columns if c.startswith("target_")]
    target_cols = p1.filter_target_columns(cfg, target_cols)

    sources = _resolve_sources(cfg)
    oof_map: Dict[str, pd.DataFrame] = {}
    test_map: Dict[str, pd.DataFrame] = {}
    for s in sources:
        oof_df, test_df = _read_source_pair(s)
        oof_map[s["name"]] = oof_df
        test_map[s["name"]] = test_df

    base_customer = next(iter(oof_map.values()))["customer_id"].to_numpy(copy=False)
    test_customer = next(iter(test_map.values()))["customer_id"].to_numpy(copy=False)
    fold_ids = np.load(Path(hcfg.get("fold_ids_path", "artifacts_fast_base/meta/fold_ids.npy")))
    n_folds = int(np.max(fold_ids)) + 1

    corr = train_target[target_cols].corr().abs()
    top_k_neighbors = int(hcfg.get("top_k_neighbors", 6))
    add_pairwise_products = bool(hcfg.get("add_pairwise_products", True))
    target_sources = [str(x) for x in hcfg.get("target_sources", [s["name"] for s in sources])]
    neighbor_sources = [str(x) for x in hcfg.get("neighbor_sources", target_sources)]
    target_sources = [s for s in target_sources if s in oof_map]
    neighbor_sources = [s for s in neighbor_sources if s in oof_map]
    C = float(hcfg.get("C", 0.5))
    class_weight = hcfg.get("class_weight", "balanced")
    max_iter = int(hcfg.get("max_iter", 1200))
    min_self_sources = int(hcfg.get("min_self_sources", 2))

    oof_path = out_dir / "base" / "h5smart_oof.parquet"
    test_path = out_dir / "base" / "h5smart_test.parquet"
    if resume and oof_path.exists() and test_path.exists():
        oof_out = pd.read_parquet(oof_path)
        test_out = pd.read_parquet(test_path)
    else:
        oof_out = pd.DataFrame({"customer_id": base_customer})
        test_out = pd.DataFrame({"customer_id": test_customer})

    state = _load_state(out_dir)
    target_scores: Dict[str, float] = {}
    target_details: Dict[str, Any] = {}

    for target in target_cols:
        pred_col = _target_to_predcol(target)
        if resume and target in oof_out.columns and pred_col in test_out.columns:
            continue
        state["targets"][target] = {"status": "running"}
        _save_state(out_dir, state)
        y = train_target[target].to_numpy(dtype=np.int8, copy=False)
        pos = int(y.sum())
        try:
            if pos == 0 or pos == len(y):
                const = float(y.mean())
                oof_pred = np.full(len(y), const, dtype=np.float32)
                test_pred = np.full(len(test_customer), const, dtype=np.float32)
                auc = float("nan")
                used_neighbors: List[str] = []
                used_features = 1
            else:
                neighbors = (
                    corr[target]
                    .drop(labels=[target], errors="ignore")
                    .sort_values(ascending=False)
                    .head(top_k_neighbors)
                    .index.tolist()
                )
                # require sufficient self-source coverage or skip to constant fallback
                self_available = 0
                for s in target_sources:
                    if _get_source_col(oof_map[s], target, is_test=False) and _get_source_col(test_map[s], target, is_test=True):
                        self_available += 1
                if self_available < min_self_sources:
                    raise ValueError(f"insufficient self sources for {target}: {self_available} < {min_self_sources}")

                x_meta, x_meta_test, feat_names = _make_meta_features(
                    target=target,
                    neighbors=neighbors,
                    target_source_names=target_sources,
                    neighbor_source_names=neighbor_sources,
                    oof_map=oof_map,
                    test_map=test_map,
                    add_pairwise_products=add_pairwise_products,
                )

                oof_pred = np.zeros(len(y), dtype=np.float32)
                test_fold_preds = []
                for fold in range(n_folds):
                    idx_val = np.where(fold_ids == fold)[0]
                    idx_tr = np.where(fold_ids != fold)[0]
                    if int(y[idx_tr].sum()) == 0 or int(y[idx_tr].sum()) == len(idx_tr):
                        oof_pred[idx_val] = float(y[idx_tr].mean())
                        test_fold_preds.append(np.full(len(test_customer), float(y[idx_tr].mean()), dtype=np.float32))
                        continue
                    scaler = StandardScaler()
                    x_tr = scaler.fit_transform(x_meta[idx_tr])
                    x_val = scaler.transform(x_meta[idx_val])
                    x_tst = scaler.transform(x_meta_test)
                    lr = LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        class_weight=class_weight,
                        solver="lbfgs",
                    )
                    lr.fit(x_tr, y[idx_tr])
                    oof_pred[idx_val] = lr.predict_proba(x_val)[:, 1].astype(np.float32)
                    test_fold_preds.append(lr.predict_proba(x_tst)[:, 1].astype(np.float32))
                test_pred = np.mean(np.column_stack(test_fold_preds), axis=1).astype(np.float32)
                auc = _auc_safe(y, oof_pred)
                used_neighbors = neighbors
                used_features = int(x_meta.shape[1])

            oof_out[target] = oof_pred
            test_out[pred_col] = test_pred
            target_scores[target] = auc
            target_details[target] = {"auc": auc, "pos": pos, "neighbors": used_neighbors, "n_features": used_features}
            state["targets"][target] = {"status": "done", "auc": auc, "pos": pos}
            state["global"]["targets_done"] = int(sum(1 for v in state["targets"].values() if v.get("status") == "done"))
            _save_state(out_dir, state)
            print(f"[h5_smart] {target} auc={auc:.6f} pos={pos} features={target_details[target]['n_features']}")
            oof_out.to_parquet(oof_path, index=False)
            test_out.to_parquet(test_path, index=False)
        except Exception as exc:
            state["targets"][target] = {"status": "failed", "error": str(exc), "pos": pos}
            state["global"]["targets_failed"] = int(sum(1 for v in state["targets"].values() if v.get("status") == "failed"))
            _save_state(out_dir, state)
            raise

    valid_scores = [v for v in target_scores.values() if not (isinstance(v, float) and math.isnan(v))]
    macro_auc = float(np.mean(valid_scores)) if valid_scores else float("nan")
    # emit base + ensemble(blend-like) so it can be used directly as source and by submit helper if needed.
    oof_out.to_parquet(oof_path, index=False)
    test_out.to_parquet(test_path, index=False)
    blend_oof = pd.DataFrame({"customer_id": oof_out["customer_id"].values})
    blend_test = pd.DataFrame({"customer_id": test_out["customer_id"].values})
    for t in target_cols:
        if t in oof_out.columns:
            blend_oof[t] = oof_out[t].astype(np.float32)
        pc = _target_to_predcol(t)
        if pc in test_out.columns:
            blend_test[pc] = test_out[pc].astype(np.float32)
    blend_oof.to_parquet(out_dir / "ensemble" / "blend_oof.parquet", index=False)
    blend_test.to_parquet(out_dir / "ensemble" / "blend_test.parquet", index=False)

    (out_dir / "scores" / "h5smart_scores.json").write_text(
        json.dumps({"model": "h5smart", "macro_auc": macro_auc, "target_scores": target_scores, "target_details": target_details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "scores" / "blend_scores.json").write_text(
        json.dumps({"models": ["h5smart"], "macro_auc": macro_auc, "target_scores": target_scores, "target_weights": {t: [1.0] for t in target_scores}, "use_rank": False, "step": 1.0}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[h5_smart] macro_auc={macro_auc:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="H5 Smart source builder (safe OOF cross-target meta source)")
    ap.add_argument("--config", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("status")
    args = ap.parse_args()
    cfg = p1.load_config(Path(args.config))
    if args.cmd == "run":
        run_h5_smart(cfg, resume=True)
    elif args.cmd == "status":
        print((_load_state(Path(cfg["output_dir"])) ))


if __name__ == "__main__":
    main()
