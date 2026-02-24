#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import top1_pipeline as p1


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def stable_hash_obj(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def stable_int_seed(parts: Sequence[Any], nbytes: int = 8) -> int:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:nbytes], byteorder="little", signed=False)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_audit_cfg(path: Path) -> Dict[str, Any]:
    return load_json(path)


def build_pipeline_cfg(audit_cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_cfg = p1.load_config(Path(audit_cfg["base_pipeline_config"]))
    cfg = copy.deepcopy(base_cfg)
    cfg = deep_merge(cfg, audit_cfg.get("pipeline_overrides", {}))
    scope = audit_cfg.get("target_scope", {})
    if scope:
        cfg["targets"] = deep_merge(cfg.get("targets", {}), scope)
    work_dir = Path(audit_cfg["output_dir"])
    cfg["output_dir"] = str(work_dir / "prep")
    return cfg


def load_state(path: Path, audit_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if path.exists():
        return load_json(path)
    return {
        "schema_version": 1,
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "audit_config_sha256": stable_hash_obj(audit_cfg),
        "targets": {},
        "global": {"targets_done": 0, "targets_failed": 0},
    }


def save_state(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = now_ts()
    ensure_dir(path.parent)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_rows_stratified(idx: np.ndarray, y_full: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or len(idx) <= max_rows:
        return idx
    y = y_full[idx]
    pos = idx[y == 1]
    neg = idx[y == 0]
    rng = np.random.default_rng(seed)
    if len(pos) == 0 or len(neg) == 0:
        return np.array(rng.choice(idx, size=max_rows, replace=False), dtype=np.int64)
    pos_share = len(pos) / len(idx)
    pos_take = max(1, min(len(pos), int(round(max_rows * pos_share))))
    neg_take = max(1, min(len(neg), max_rows - pos_take))
    if pos_take + neg_take < max_rows:
        neg_take = min(len(neg), max_rows - pos_take)
    pos_sample = rng.choice(pos, size=pos_take, replace=False)
    neg_sample = rng.choice(neg, size=neg_take, replace=False)
    out = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(out)
    return np.array(out, dtype=np.int64)


@dataclass
class FoldArtifacts:
    model: Any
    x_val: pd.DataFrame
    y_val: np.ndarray
    baseline_auc: float
    feature_names: List[str]
    importances: np.ndarray


def fit_xgb_model(x_train: pd.DataFrame, y_train: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray, params: Dict[str, Any]):
    from xgboost import XGBClassifier

    p = dict(params)
    early_stopping_rounds = int(p.pop("early_stopping_rounds", 0))
    m = XGBClassifier(**p)
    fit_kwargs: Dict[str, Any] = {"eval_set": [(x_val, y_val)], "verbose": False}
    if early_stopping_rounds > 0:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    try:
        m.fit(x_train, y_train, **fit_kwargs)
    except TypeError as e:
        if "early_stopping_rounds" in str(e):
            fit_kwargs.pop("early_stopping_rounds", None)
            m.fit(x_train, y_train, **fit_kwargs)
        else:
            raise
    return m


def auc_safe(y: np.ndarray, pred: np.ndarray) -> float:
    if int(y.sum()) == 0 or int(y.sum()) == len(y):
        return float("nan")
    return float(roc_auc_score(y, pred))


def rank_features_by_real_importance(folds: Sequence[FoldArtifacts], candidate_feats: Sequence[str]) -> List[str]:
    if not candidate_feats:
        return []
    score = {f: 0.0 for f in candidate_feats}
    cnt = {f: 0 for f in candidate_feats}
    cset = set(candidate_feats)
    for fa in folds:
        for name, imp in zip(fa.feature_names, fa.importances):
            if name in cset:
                score[name] += float(imp)
                cnt[name] += 1
    ranked = sorted(candidate_feats, key=lambda f: (-(score[f] / max(cnt[f], 1)), f))
    return ranked


def permutation_delta_for_feature(model: Any, x_val: pd.DataFrame, y_val: np.ndarray, feature: str, baseline_auc: float, seed: int) -> float:
    if feature not in x_val.columns or math.isnan(baseline_auc):
        return float("nan")
    x_perm = x_val.copy()
    rng = np.random.default_rng(seed)
    vals = x_perm[feature].to_numpy(copy=True)
    rng.shuffle(vals)
    x_perm[feature] = vals
    pred = p1.xgb_predict_proba_binary(model, x_perm)
    auc_perm = auc_safe(y_val, pred)
    if math.isnan(auc_perm):
        return float("nan")
    return float(baseline_auc - auc_perm)


def get_candidate_columns(audit_cfg: Dict[str, Any], prepared: Dict[str, Any]) -> List[str]:
    mode = str(audit_cfg.get("audit", {}).get("candidate_mode", "selected_extra"))
    feature_cols = list(prepared["feature_cols"])
    train_df = prepared["train_df"]
    cols_meta_path = Path(build_pipeline_cfg(audit_cfg)["output_dir"]) / "meta" / "columns.json"
    selected_extra: List[str] = []
    row_stats: List[str] = []
    if cols_meta_path.exists():
        meta = load_json(cols_meta_path)
        selected_extra = [str(x) for x in meta.get("selected_extra_cols", [])]
        row_stats = [str(x) for x in meta.get("row_stat_cols", [])]
    if mode == "selected_extra":
        out = [c for c in selected_extra if c in train_df.columns]
    elif mode == "selected_extra_plus_row_stats":
        out = [c for c in selected_extra + row_stats if c in train_df.columns]
    elif mode == "feature_cols":
        out = [c for c in feature_cols if c in train_df.columns]
    elif mode == "extra_and_num":
        out = [c for c in feature_cols if c.startswith("num_feature") or c in set(selected_extra)]
    else:
        raise ValueError(f"Unknown audit.candidate_mode: {mode}")

    include_prefixes = [str(x) for x in audit_cfg.get("audit", {}).get("force_include_prefixes", [])]
    if include_prefixes:
        extra = [c for c in feature_cols if any(c.startswith(p) for p in include_prefixes)]
        out = list(dict.fromkeys(out + extra))

    exclude_prefixes = [str(x) for x in audit_cfg.get("audit", {}).get("exclude_prefixes", [])]
    if exclude_prefixes:
        out = [c for c in out if not any(c.startswith(p) for p in exclude_prefixes)]

    hard_cap = int(audit_cfg.get("audit", {}).get("candidate_hard_cap", 0))
    if hard_cap > 0:
        out = out[:hard_cap]
    return out


def collect_real_folds(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
    fold_ids: np.ndarray,
    xgb_params: Dict[str, Any],
    search_cfg: Dict[str, Any],
) -> tuple[List[FoldArtifacts], List[float]]:
    y = train_df[target].to_numpy(dtype=np.int8, copy=False)
    n_folds = int(np.max(fold_ids)) + 1
    folds: List[FoldArtifacts] = []
    fold_aucs: List[float] = []
    x_all = train_df[list(feature_cols)]
    sample_tr = int(search_cfg.get("sample_train_rows_per_fold", 0))
    sample_val = int(search_cfg.get("sample_val_rows_per_fold", 0))
    seed = int(search_cfg.get("seed", 42))

    for fold in range(n_folds):
        idx_val = np.where(fold_ids == fold)[0]
        idx_tr = np.where(fold_ids != fold)[0]
        idx_tr = sample_rows_stratified(idx_tr, y, sample_tr, seed + 1000 + fold)
        idx_val = sample_rows_stratified(idx_val, y, sample_val, seed + 2000 + fold)

        x_train = x_all.iloc[idx_tr]
        y_train = y[idx_tr]
        x_val = x_all.iloc[idx_val]
        y_val = y[idx_val]

        model = fit_xgb_model(x_train, y_train, x_val, y_val, xgb_params)
        pred = p1.xgb_predict_proba_binary(model, x_val)
        auc = auc_safe(y_val, pred)
        fold_aucs.append(float(auc))
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            importances = np.zeros(len(x_train.columns), dtype=np.float32)
        folds.append(
            FoldArtifacts(
                model=model,
                x_val=x_val,
                y_val=y_val,
                baseline_auc=float(auc),
                feature_names=list(x_train.columns),
                importances=np.asarray(importances, dtype=np.float32),
            )
        )
    return folds, fold_aucs


def collect_null_deltas(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
    fold_ids: np.ndarray,
    xgb_params: Dict[str, Any],
    search_cfg: Dict[str, Any],
    features_to_perm: Sequence[str],
    null_trials: int,
    seed: int,
) -> Dict[str, List[float]]:
    y_true = train_df[target].to_numpy(dtype=np.int8, copy=False)
    n_folds = int(np.max(fold_ids)) + 1
    x_all = train_df[list(feature_cols)]
    sample_tr = int(search_cfg.get("sample_train_rows_per_fold", 0))
    sample_val = int(search_cfg.get("sample_val_rows_per_fold", 0))

    deltas: Dict[str, List[float]] = defaultdict(list)
    for nt in range(null_trials):
        y_perm_global = y_true.copy()
        rng_global = np.random.default_rng(seed + 10_000 + nt)
        rng_global.shuffle(y_perm_global)
        for fold in range(n_folds):
            idx_val = np.where(fold_ids == fold)[0]
            idx_tr = np.where(fold_ids != fold)[0]
            idx_tr = sample_rows_stratified(idx_tr, y_true, sample_tr, seed + nt * 100 + 1000 + fold)
            idx_val = sample_rows_stratified(idx_val, y_true, sample_val, seed + nt * 100 + 2000 + fold)

            x_train = x_all.iloc[idx_tr]
            y_train = y_perm_global[idx_tr]
            x_val = x_all.iloc[idx_val]
            y_val = y_perm_global[idx_val]

            if int(y_train.sum()) == 0 or int(y_train.sum()) == len(y_train) or int(y_val.sum()) in (0, len(y_val)):
                continue
            model = fit_xgb_model(x_train, y_train, x_val, y_val, xgb_params)
            pred = p1.xgb_predict_proba_binary(model, x_val)
            baseline_auc = auc_safe(y_val, pred)
            for fi, feat in enumerate(features_to_perm):
                d = permutation_delta_for_feature(
                    model=model,
                    x_val=x_val,
                    y_val=y_val,
                    feature=feat,
                    baseline_auc=baseline_auc,
                    seed=seed + nt * 100_000 + fold * 1_000 + fi,
                )
                if not math.isnan(d):
                    deltas[feat].append(float(d))
    return deltas


def mean_std(vals: Sequence[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def summarize_feature_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    delta_vals = [r["mean_delta_auc"] for r in records if not math.isnan(r.get("mean_delta_auc", float("nan")))]
    null_vals = [r["null_mean_delta_auc"] for r in records if not math.isnan(r.get("null_mean_delta_auc", float("nan")))]
    z_vals = [r["z_vs_null"] for r in records if not math.isnan(r.get("z_vs_null", float("nan")))]
    return {
        "targets_seen": len(records),
        "mean_delta_auc": float(np.mean(delta_vals)) if delta_vals else float("nan"),
        "median_delta_auc": float(np.median(delta_vals)) if delta_vals else float("nan"),
        "mean_null_delta_auc": float(np.mean(null_vals)) if null_vals else float("nan"),
        "median_z_vs_null": float(np.median(z_vals)) if z_vals else float("nan"),
        "mean_z_vs_null": float(np.mean(z_vals)) if z_vals else float("nan"),
        "targets_positive_delta": int(sum(1 for r in records if r.get("mean_delta_auc", -1) > 0)),
        "targets_strong_signal": int(sum(1 for r in records if r.get("z_vs_null", -1e9) >= 2.0)),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_audit(cfg_path: Path) -> None:
    audit_cfg = load_audit_cfg(cfg_path)
    work_dir = Path(audit_cfg["output_dir"])
    ensure_dir(work_dir)
    ensure_dir(work_dir / "targets")
    ensure_dir(work_dir / "reports")
    state_path = work_dir / "state.json"
    state = load_state(state_path, audit_cfg)
    (work_dir / "audit_config_snapshot.json").write_text(json.dumps(audit_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    pipeline_cfg = build_pipeline_cfg(audit_cfg)
    prepared = p1.prepare_dataset(pipeline_cfg)
    train_df = prepared["train_df"]
    feature_cols = list(prepared["feature_cols"])
    target_cols = list(prepared["target_cols"])
    fold_ids = prepared["fold_ids"]
    cat_cols = list(prepared["cat_cols"])
    target_feature_map = p1.build_per_target_feature_map(
        cfg=pipeline_cfg,
        train_df=train_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        cat_cols=cat_cols,
    )

    candidate_cols = get_candidate_columns(audit_cfg, prepared)
    print(f"[audit] candidate features: {len(candidate_cols)}")

    audit = audit_cfg.get("audit", {})
    seed = int(audit.get("seed", audit_cfg.get("seed", 42)))
    max_targets = int(audit.get("max_targets", 0))
    feature_limit = int(audit.get("per_target_feature_limit", 64))
    null_trials = int(audit.get("null_trials", 3))
    use_target_feature_map = bool(audit.get("use_target_feature_map", True))
    xgb_params = dict(audit_cfg.get("xgboost_params") or pipeline_cfg["models"]["xgboost"]["params"])
    if bool(audit.get("force_cpu", False)):
        xgb_params.pop("device", None)
        xgb_params["tree_method"] = "hist"

    search_cfg = audit.get("fold_eval", {})
    targets = list(target_cols[:max_targets] if max_targets > 0 else target_cols)
    if max_targets > 0:
        print(f"[audit] max_targets={max_targets}; evaluating first {len(targets)} targets")

    for ti, target in enumerate(targets, start=1):
        target_state = state["targets"].get(target, {})
        if target_state.get("status") == "done" and (work_dir / "targets" / f"{target}.json").exists():
            print(f"[audit] skip {target} (done)")
            continue
        print(f"[audit] target {ti}/{len(targets)}: {target}")
        state["targets"][target] = {"status": "running"}
        save_state(state_path, state)
        try:
            y = train_df[target].to_numpy(dtype=np.int8, copy=False)
            positives = int(y.sum())
            if positives == 0 or positives == len(y):
                payload = {
                    "target": target,
                    "status": "constant_target",
                    "positives": positives,
                    "feature_rows": [],
                }
                (work_dir / "targets" / f"{target}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                state["targets"][target] = {"status": "done", "positives": positives, "constant": True}
                state["global"]["targets_done"] = int(state["global"].get("targets_done", 0)) + 1
                save_state(state_path, state)
                continue

            target_feats_full = target_feature_map.get(target, feature_cols) if use_target_feature_map else feature_cols
            target_feats_full = [c for c in target_feats_full if c in train_df.columns]
            candidate_for_target = [c for c in candidate_cols if c in target_feats_full]
            if not candidate_for_target:
                payload = {
                    "target": target,
                    "status": "no_candidate_features",
                    "positives": positives,
                    "feature_rows": [],
                }
                (work_dir / "targets" / f"{target}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                state["targets"][target] = {"status": "done", "positives": positives, "empty": True}
                state["global"]["targets_done"] = int(state["global"].get("targets_done", 0)) + 1
                save_state(state_path, state)
                continue

            folds_real, fold_aucs = collect_real_folds(
                train_df=train_df,
                feature_cols=target_feats_full,
                target=target,
                fold_ids=fold_ids,
                xgb_params=xgb_params,
                search_cfg=search_cfg,
            )
            ranked = rank_features_by_real_importance(folds_real, candidate_for_target)
            shortlist = ranked[:feature_limit] if feature_limit > 0 else ranked
            print(f"[audit] {target}: candidates={len(candidate_for_target)} shortlist={len(shortlist)}")

            per_feature_real: Dict[str, List[float]] = defaultdict(list)
            for fold_idx, fa in enumerate(folds_real):
                for fi, feat in enumerate(shortlist):
                    d = permutation_delta_for_feature(
                        model=fa.model,
                        x_val=fa.x_val,
                        y_val=fa.y_val,
                        feature=feat,
                        baseline_auc=fa.baseline_auc,
                        seed=seed + stable_int_seed([target, "real", fold_idx, feat]) % 1_000_000,
                    )
                    if not math.isnan(d):
                        per_feature_real[feat].append(float(d))

            per_feature_null = collect_null_deltas(
                train_df=train_df,
                feature_cols=target_feats_full,
                target=target,
                fold_ids=fold_ids,
                xgb_params=xgb_params,
                search_cfg=search_cfg,
                features_to_perm=shortlist,
                null_trials=null_trials,
                seed=seed + stable_int_seed([target, "null"]),
            )

            feature_rows: List[Dict[str, Any]] = []
            for feat in shortlist:
                real_vals = [float(x) for x in per_feature_real.get(feat, []) if not math.isnan(float(x))]
                null_vals = [float(x) for x in per_feature_null.get(feat, []) if not math.isnan(float(x))]
                mean_real, std_real = mean_std(real_vals)
                mean_null, std_null = mean_std(null_vals)
                z = float("nan")
                if not math.isnan(mean_real) and not math.isnan(mean_null):
                    denom = (0.0 if math.isnan(std_null) else std_null) + 1e-9
                    z = float((mean_real - mean_null) / denom) if denom > 0 else float("nan")
                row = {
                    "target": target,
                    "feature": feat,
                    "n_real": len(real_vals),
                    "n_null": len(null_vals),
                    "mean_delta_auc": mean_real,
                    "std_delta_auc": std_real,
                    "null_mean_delta_auc": mean_null,
                    "null_std_delta_auc": std_null,
                    "z_vs_null": z,
                    "pos_rate_real": float(sum(1 for v in real_vals if v > 0) / len(real_vals)) if real_vals else float("nan"),
                }
                feature_rows.append(row)

            payload = {
                "target": target,
                "status": "done",
                "positives": positives,
                "fold_aucs": fold_aucs,
                "candidate_count": len(candidate_for_target),
                "shortlist_count": len(shortlist),
                "feature_rows": feature_rows,
            }
            (work_dir / "targets" / f"{target}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            append_jsonl(work_dir / "reports" / "audit_events.jsonl", {
                "ts_utc": now_ts(),
                "event": "target_done",
                "target": target,
                "shortlist_count": len(shortlist),
                "positives": positives,
                "mean_fold_auc": float(np.nanmean(np.asarray(fold_aucs, dtype=np.float64))) if fold_aucs else float("nan"),
            })
            state["targets"][target] = {
                "status": "done",
                "positives": positives,
                "candidate_count": len(candidate_for_target),
                "shortlist_count": len(shortlist),
            }
            state["global"]["targets_done"] = int(state["global"].get("targets_done", 0)) + 1
            save_state(state_path, state)
        except Exception as exc:
            state["targets"][target] = {"status": "failed", "error": str(exc)}
            state["global"]["targets_failed"] = int(state["global"].get("targets_failed", 0)) + 1
            save_state(state_path, state)
            append_jsonl(work_dir / "reports" / "audit_events.jsonl", {"ts_utc": now_ts(), "event": "target_failed", "target": target, "error": str(exc)})
            print(f"[audit] target FAILED: {target}: {exc}")
            raise

    aggregate_reports(cfg_path)


def aggregate_reports(cfg_path: Path) -> None:
    audit_cfg = load_audit_cfg(cfg_path)
    work_dir = Path(audit_cfg["output_dir"])
    target_files = sorted((work_dir / "targets").glob("target_*.json"))
    per_feature_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    target_rows_all: List[Dict[str, Any]] = []
    target_summaries: List[Dict[str, Any]] = []
    for p in target_files:
        rec = load_json(p)
        target_summaries.append({
            "target": rec.get("target"),
            "status": rec.get("status"),
            "positives": rec.get("positives"),
            "candidate_count": rec.get("candidate_count"),
            "shortlist_count": rec.get("shortlist_count"),
            "mean_fold_auc": float(np.nanmean(np.asarray(rec.get("fold_aucs", []), dtype=np.float64))) if rec.get("fold_aucs") else float("nan"),
        })
        for row in rec.get("feature_rows", []):
            target_rows_all.append(row)
            per_feature_records[str(row["feature"])].append(row)

    feature_summary_rows: List[Dict[str, Any]] = []
    for feat, rows in per_feature_records.items():
        s = summarize_feature_records(rows)
        feature_summary_rows.append({"feature": feat, **s})

    feature_summary_rows.sort(
        key=lambda r: (
            -(r.get("targets_strong_signal") or 0),
            -((r.get("mean_z_vs_null") if isinstance(r.get("mean_z_vs_null"), (int, float)) and not math.isnan(r.get("mean_z_vs_null")) else -1e18)),
            -((r.get("mean_delta_auc") if isinstance(r.get("mean_delta_auc"), (int, float)) and not math.isnan(r.get("mean_delta_auc")) else -1e18)),
            r["feature"],
        )
    )

    audit = audit_cfg.get("audit", {})
    decision = audit.get("decision", {})
    keep_min_targets = int(decision.get("keep_min_targets", 2))
    keep_min_pos_rate = float(decision.get("keep_min_pos_rate", 0.6))
    keep_min_z = float(decision.get("keep_min_z", 1.5))
    keep_min_mean_delta = float(decision.get("keep_min_mean_delta", 0.0002))
    drop_min_targets = int(decision.get("drop_min_targets", 2))
    drop_max_z = float(decision.get("drop_max_z", 0.2))
    drop_max_mean_delta = float(decision.get("drop_max_mean_delta", 0.0))

    keep_extra: List[str] = []
    drop_cols: List[str] = []
    for r in feature_summary_rows:
        t_seen = int(r.get("targets_seen") or 0)
        mean_delta = float(r.get("mean_delta_auc")) if r.get("mean_delta_auc") is not None else float("nan")
        mean_z = float(r.get("mean_z_vs_null")) if r.get("mean_z_vs_null") is not None else float("nan")
        pos_rate = (int(r.get("targets_positive_delta") or 0) / t_seen) if t_seen else 0.0

        is_keep = (
            t_seen >= keep_min_targets
            and not math.isnan(mean_delta)
            and mean_delta >= keep_min_mean_delta
            and not math.isnan(mean_z)
            and mean_z >= keep_min_z
            and pos_rate >= keep_min_pos_rate
        )
        is_drop = (
            t_seen >= drop_min_targets
            and (math.isnan(mean_delta) or mean_delta <= drop_max_mean_delta)
            and (math.isnan(mean_z) or mean_z <= drop_max_z)
        )
        if is_keep:
            keep_extra.append(str(r["feature"]))
        elif is_drop:
            drop_cols.append(str(r["feature"]))

    reports_dir = work_dir / "reports"
    ensure_dir(reports_dir)
    write_csv(reports_dir / "feature_audit_target_rows.csv", target_rows_all)
    write_csv(reports_dir / "feature_audit_feature_summary.csv", feature_summary_rows)
    write_csv(reports_dir / "feature_audit_target_summary.csv", target_summaries)

    (work_dir / "keep_extra.json").write_text(json.dumps({"columns": keep_extra}, ensure_ascii=False, indent=2), encoding="utf-8")
    (work_dir / "drop_columns.json").write_text(json.dumps({"columns": drop_cols}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "ts_utc": now_ts(),
        "audit_config_path": str(cfg_path),
        "targets_total_files": len(target_files),
        "targets_with_feature_rows": int(sum(1 for x in target_summaries if x.get("shortlist_count"))),
        "features_scored": len(feature_summary_rows),
        "keep_extra_count": len(keep_extra),
        "drop_columns_count": len(drop_cols),
        "reports": {
            "feature_summary_csv": str(reports_dir / "feature_audit_feature_summary.csv"),
            "target_rows_csv": str(reports_dir / "feature_audit_target_rows.csv"),
            "target_summary_csv": str(reports_dir / "feature_audit_target_summary.csv"),
        },
        "outputs": {
            "keep_extra": str(work_dir / "keep_extra.json"),
            "drop_columns": str(work_dir / "drop_columns.json"),
        },
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[audit] aggregate complete: features_scored={len(feature_summary_rows)} keep={len(keep_extra)} drop={len(drop_cols)}")


def print_status(cfg_path: Path) -> None:
    audit_cfg = load_audit_cfg(cfg_path)
    work_dir = Path(audit_cfg["output_dir"])
    state_path = work_dir / "state.json"
    if not state_path.exists():
        print(f"[audit] no state file yet: {state_path}")
        return
    state = load_json(state_path)
    print("[audit] state:", state_path)
    print("[audit] global:", state.get("global", {}))
    for target, rec in sorted(state.get("targets", {}).items()):
        print(f"  - {target}: {rec.get('status')} positives={rec.get('positives')} shortlist={rec.get('shortlist_count')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quality-first OOF feature audit (permutation + null importance + stability)")
    p.add_argument("--config", required=True, help="Path to audit JSON config")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Run/resume audit and aggregate reports")
    sub.add_parser("aggregate", help="Aggregate existing per-target audit outputs")
    sub.add_parser("status", help="Print audit state")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg_path = Path(args.config)
    if args.command == "run":
        run_audit(cfg_path)
        return
    if args.command == "aggregate":
        aggregate_reports(cfg_path)
        return
    if args.command == "status":
        print_status(cfg_path)
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
