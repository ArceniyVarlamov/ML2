#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in [REPO_ROOT, SCRIPTS_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import top1_pipeline as p1
import hpo_xgb_targets as hx


SUPPORTED_MODELS = {"xgboost", "lightgbm", "catboost"}


def load_hpo_cfg(path: Path) -> Dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    model_name = str(cfg.get("model_name", "")).lower().strip()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"config.model_name must be one of {sorted(SUPPORTED_MODELS)}")
    cfg["model_name"] = model_name
    return cfg


def build_pipeline_cfg(hpo_cfg: Dict[str, Any], mode: str) -> Dict[str, Any]:
    base_cfg = p1.load_config(Path(hpo_cfg["base_pipeline_config"]))
    cfg = copy.deepcopy(base_cfg)
    cfg = hx.deep_merge(cfg, hpo_cfg.get("pipeline_overrides", {}))
    scope = hpo_cfg.get("target_scope", {})
    if scope:
        cfg["targets"] = hx.deep_merge(cfg.get("targets", {}), scope)

    model_name = hpo_cfg["model_name"]
    models = cfg.setdefault("models", {})
    for m in ["catboost", "lightgbm", "xgboost"]:
        models.setdefault(m, {"enabled": False, "params": {}})
        models[m]["enabled"] = (m == model_name)

    if mode == "search":
        cfg["output_dir"] = str(Path(hpo_cfg["output_dir"]) / "search_prep")
    elif mode == "train_source":
        out_dir = hpo_cfg.get("train_source", {}).get("output_dir")
        if not out_dir:
            raise ValueError("train_source.output_dir is required")
        cfg["output_dir"] = str(out_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return cfg


def apply_tuned_params_to_cfg(cfg: Dict[str, Any], model_name: str, params_by_target: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("models", {}).setdefault(model_name, {}).setdefault("params", {})
    out["models"][model_name]["enabled"] = True
    out["models"][model_name]["params_by_target"] = copy.deepcopy(params_by_target)
    return out


def fit_fold_val_only(
    model_name: str,
    params: Dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: Sequence[str],
) -> np.ndarray:
    val_pred, _ = p1.fit_predict_one_fold(
        model_name=model_name,
        model_params=params,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_val,
        cat_cols=[c for c in cat_cols if c in x_train.columns],
    )
    return val_pred.astype(np.float32, copy=False)


def evaluate_params_oof(
    *,
    model_name: str,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    cat_cols: Sequence[str],
    target: str,
    fold_ids: np.ndarray,
    params: Dict[str, Any],
    target_features: Sequence[str] | None,
    sample_train_rows_per_fold: int,
    sample_val_rows_per_fold: int,
    seed: int,
) -> Dict[str, Any]:
    y = train_df[target].to_numpy(dtype=np.int8, copy=False)
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        return {"auc": float("nan"), "positives": positives, "fold_aucs": [], "status": "constant_target"}

    feats = list(target_features) if target_features is not None else list(feature_cols)
    x_target = train_df[feats]
    t_cat_cols = [c for c in cat_cols if c in feats]
    n_folds = int(np.max(fold_ids)) + 1
    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_aucs: List[float] = []

    for fold in range(n_folds):
        idx_val = np.where(fold_ids == fold)[0]
        idx_tr = np.where(fold_ids != fold)[0]
        idx_tr = hx.sample_fold_rows(idx_tr, y, int(sample_train_rows_per_fold), seed + 1000 + fold, stratified=True)
        idx_val = hx.sample_fold_rows(idx_val, y, int(sample_val_rows_per_fold), seed + 2000 + fold, stratified=True)
        x_train = x_target.iloc[idx_tr]
        y_train = y[idx_tr]
        x_val = x_target.iloc[idx_val]
        y_val = y[idx_val]
        val_pred = fit_fold_val_only(model_name, params, x_train, y_train, x_val, y_val, t_cat_cols)
        oof[idx_val] = val_pred
        fold_aucs.append(float(roc_auc_score(y_val, val_pred)))

    valid_mask = ~np.isnan(oof)
    auc = float(roc_auc_score(y[valid_mask], oof[valid_mask]))
    return {
        "auc": auc,
        "positives": positives,
        "fold_aucs": fold_aucs,
        "status": "ok",
        "filled_rows": int(valid_mask.sum()),
        "total_rows": int(len(y)),
    }


def run_search(hpo_cfg_path: Path) -> None:
    hpo_cfg = load_hpo_cfg(hpo_cfg_path)
    model_name = hpo_cfg["model_name"]
    work_dir = Path(hpo_cfg["output_dir"])
    hx.ensure_dir(work_dir)
    hx.ensure_dir(work_dir / "logs")
    state_path = work_dir / "state.json"
    trials_log = work_dir / "trials.jsonl"
    best_params_path = work_dir / "best_params_by_target.json"
    summary_path = work_dir / "summary.json"
    (work_dir / "hpo_config_snapshot.json").write_text(json.dumps(hpo_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    search_cfg = build_pipeline_cfg(hpo_cfg, mode="search")
    prepared = p1.prepare_dataset(search_cfg)
    train_df = prepared["train_df"]
    feature_cols = prepared["feature_cols"]
    target_cols = prepared["target_cols"]
    fold_ids = prepared["fold_ids"]
    cat_cols = prepared["cat_cols"]
    target_feature_map = p1.build_per_target_feature_map(
        cfg=search_cfg,
        train_df=train_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        cat_cols=cat_cols,
    )

    m_cfg = search_cfg["models"][model_name]
    base_params = dict(m_cfg["params"])
    # Optional force CPU for search to reduce GPU queue pressure even on A100.
    if hpo_cfg.get("search", {}).get("force_cpu_for_search", False):
        if model_name == "xgboost":
            base_params.pop("device", None)
            base_params["tree_method"] = "hist"
        elif model_name == "catboost":
            base_params.pop("devices", None)
            base_params["task_type"] = "CPU"

    search = hpo_cfg.get("search", {})
    search_space = hpo_cfg["space"]
    trials_per_target = int(search.get("trials_per_target", 20))
    max_trials_total = int(search.get("max_trials_total", 0))
    max_seconds_total = float(search.get("max_seconds_total", 0))
    max_seconds_per_target = float(search.get("max_seconds_per_target", 0))
    sample_train_rows = int(search.get("sample_train_rows_per_fold", 0))
    sample_val_rows = int(search.get("sample_val_rows_per_fold", 0))
    global_seed = int(search.get("seed", hpo_cfg.get("seed", 42)))
    max_targets = int(search.get("max_targets", 0))

    state = hx.load_state(state_path, hpo_cfg)
    start_ts = time.time()
    completed_global = int(state.get("global", {}).get("trials_completed", 0))
    failed_global = int(state.get("global", {}).get("trials_failed", 0))

    targets = list(target_cols)
    if max_targets > 0:
        targets = targets[:max_targets]

    for ti, target in enumerate(targets, start=1):
        t0 = time.time()
        tstate = state["targets"].setdefault(target, {"status": "pending", "trials": {}, "best_trial_idx": None, "best_auc": None, "best_params": None})
        print(f"[hpo:{model_name}] target {ti}/{len(targets)}: {target}")
        target_seed = global_seed + (hx.stable_int_seed([hpo_cfg.get("name", "hpo"), model_name, target]) % 1000003)
        best_auc = float(tstate["best_auc"]) if tstate["best_auc"] is not None else -1.0
        tstate["status"] = "running"

        for trial_idx in range(trials_per_target):
            if max_trials_total > 0 and completed_global >= max_trials_total:
                print("[hpo] global max_trials_total reached")
                break
            if max_seconds_total > 0 and (time.time() - start_ts) >= max_seconds_total:
                print("[hpo] global max_seconds_total reached")
                break
            if max_seconds_per_target > 0 and (time.time() - t0) >= max_seconds_per_target:
                print(f"[hpo] target {target} max_seconds_per_target reached")
                break

            trial_key = str(trial_idx)
            if trial_key in tstate["trials"] and tstate["trials"][trial_key].get("status") == "ok":
                continue

            params = hx.make_trial_params(base_params=base_params, search_space=search_space, seed=target_seed, target=target, trial_idx=trial_idx)
            t_trial = time.time()
            row = {"ts_utc": hx.now_ts(), "event": "hpo_trial", "model": model_name, "target": target, "trial_idx": trial_idx, "params": params}
            try:
                result = evaluate_params_oof(
                    model_name=model_name,
                    train_df=train_df,
                    feature_cols=feature_cols,
                    cat_cols=cat_cols,
                    target=target,
                    fold_ids=fold_ids,
                    params=params,
                    target_features=target_feature_map.get(target),
                    sample_train_rows_per_fold=sample_train_rows,
                    sample_val_rows_per_fold=sample_val_rows,
                    seed=target_seed + trial_idx * 17,
                )
                auc = result["auc"]
                dur = time.time() - t_trial
                row.update({"status": "ok", "auc": auc, "duration_sec": dur, "fold_aucs": result.get("fold_aucs", []), "filled_rows": result.get("filled_rows"), "total_rows": result.get("total_rows"), "positives": result.get("positives")})
                tstate["trials"][trial_key] = {"status": "ok", "auc": auc, "duration_sec": dur, "params": params}
                completed_global += 1
                if not math.isnan(auc) and auc > best_auc:
                    best_auc = float(auc)
                    tstate["best_auc"] = best_auc
                    tstate["best_params"] = params
                    tstate["best_trial_idx"] = trial_idx
                    print(f"[hpo:{model_name}] {target} trial={trial_idx} NEW BEST auc={best_auc:.6f}")
                else:
                    print(f"[hpo:{model_name}] {target} trial={trial_idx} auc={auc:.6f}")
            except Exception as exc:
                dur = time.time() - t_trial
                row.update({"status": "failed", "duration_sec": dur, "error": str(exc)})
                tstate["trials"][trial_key] = {"status": "failed", "duration_sec": dur, "error": str(exc), "params": params}
                failed_global += 1
                print(f"[hpo:{model_name}] {target} trial={trial_idx} FAILED: {exc}")

            hx.append_jsonl(trials_log, row)
            state["global"]["trials_completed"] = completed_global
            state["global"]["trials_failed"] = failed_global
            hx.save_state(state_path, state)

        tstate["status"] = "done"
        state["global"]["targets_completed"] = int(sum(1 for t in state["targets"].values() if t.get("status") == "done"))
        hx.save_state(state_path, state)
        if (max_trials_total > 0 and completed_global >= max_trials_total) or (max_seconds_total > 0 and (time.time() - start_ts) >= max_seconds_total):
            break

    best_params_by_target = {}
    summary_targets: Dict[str, Any] = {}
    for target in targets:
        tstate = state["targets"].get(target, {})
        if isinstance(tstate.get("best_params"), dict):
            best_params_by_target[target] = tstate["best_params"]
        summary_targets[target] = {
            "status": tstate.get("status"),
            "best_auc": tstate.get("best_auc"),
            "best_trial_idx": tstate.get("best_trial_idx"),
            "trials_done": int(sum(1 for v in tstate.get("trials", {}).values() if v.get("status") == "ok")),
            "trials_failed": int(sum(1 for v in tstate.get("trials", {}).values() if v.get("status") == "failed")),
        }
    best_params_path.write_text(json.dumps(best_params_by_target, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "ts_utc": hx.now_ts(),
        "model_name": model_name,
        "hpo_config_path": str(hpo_cfg_path),
        "hpo_config_sha256": hx.stable_hash_obj(hpo_cfg),
        "targets_total": len(targets),
        "targets_with_best_params": len(best_params_by_target),
        "global": state["global"],
        "targets": summary_targets,
        "best_params_by_target_path": str(best_params_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hpo:{model_name}] saved best params: {best_params_path}")


def export_tuned_pipeline_config(hpo_cfg_path: Path) -> Path:
    hpo_cfg = load_hpo_cfg(hpo_cfg_path)
    model_name = hpo_cfg["model_name"]
    work_dir = Path(hpo_cfg["output_dir"])
    best_params_path = work_dir / "best_params_by_target.json"
    if not best_params_path.exists():
        raise FileNotFoundError(f"Missing best params file: {best_params_path}")
    best_params = json.loads(best_params_path.read_text(encoding="utf-8"))
    train_cfg = build_pipeline_cfg(hpo_cfg, mode="train_source")
    train_cfg = apply_tuned_params_to_cfg(train_cfg, model_name, best_params)
    train_source = hpo_cfg.get("train_source", {})
    tuned_cfg_path = Path(train_source.get("tuned_config_path", work_dir / f"tuned_{model_name}_pipeline_config.json"))
    hx.ensure_dir(tuned_cfg_path.parent)
    tuned_cfg_path.write_text(json.dumps(train_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hpo:{model_name}] exported tuned pipeline config: {tuned_cfg_path}")
    return tuned_cfg_path


def run_train_source(hpo_cfg_path: Path, skip_base: bool = False, skip_blend: bool = False, skip_stack: bool = True) -> None:
    hpo_cfg = load_hpo_cfg(hpo_cfg_path)
    model_name = hpo_cfg["model_name"]
    tuned_cfg_path = export_tuned_pipeline_config(hpo_cfg_path)
    cfg = p1.load_config(tuned_cfg_path)
    train_source = hpo_cfg.get("train_source", {})
    model_names = [str(x) for x in train_source.get("models", [model_name])]
    if not skip_base:
        p1.run_base_models(cfg, model_names)
    if bool(train_source.get("run_blend", True)) and not skip_blend:
        p1.run_blend(cfg, model_names)
    if bool(train_source.get("run_stack", False)) and not skip_stack:
        p1.run_stack(cfg, model_names)
    if bool(train_source.get("emit_submission", False)):
        src = Path(cfg["output_dir"]) / "ensemble" / "blend_test.parquet"
        out = Path(cfg["output_dir"]) / "submissions" / f"sub_hpo_{model_name}_blend_float64.parquet"
        fallback = train_source.get("fallback_source")
        p1.make_submission(cfg, src, out, Path(fallback) if fallback else None)


def print_status(hpo_cfg_path: Path) -> None:
    hpo_cfg = load_hpo_cfg(hpo_cfg_path)
    work_dir = Path(hpo_cfg["output_dir"])
    state_path = work_dir / "state.json"
    summary_path = work_dir / "summary.json"
    if not state_path.exists():
        print(f"[hpo] no state file yet: {state_path}")
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    print("[hpo] state:", state_path)
    print("[hpo] global:", state.get("global"))
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print("[hpo] summary targets_with_best_params:", summary.get("targets_with_best_params"))
    for target, rec in sorted(state.get("targets", {}).items()):
        print(f"  - {target}: status={rec.get('status')} best_auc={rec.get('best_auc')} best_trial_idx={rec.get('best_trial_idx')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resumable target HPO for boosting models (catboost/lightgbm/xgboost)")
    p.add_argument("--config", required=True, help="Path to HPO JSON config")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("search")
    sub.add_parser("export-config")
    p_train = sub.add_parser("train-source")
    p_train.add_argument("--skip-base", action="store_true")
    p_train.add_argument("--skip-blend", action="store_true")
    p_train.add_argument("--run-stack", action="store_true")
    sub.add_parser("search-and-train")
    sub.add_parser("status")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg_path = Path(args.config)
    if args.command == "search":
        run_search(cfg_path)
    elif args.command == "export-config":
        export_tuned_pipeline_config(cfg_path)
    elif args.command == "train-source":
        run_train_source(cfg_path, skip_base=bool(args.skip_base), skip_blend=bool(args.skip_blend), skip_stack=not bool(args.run_stack))
    elif args.command == "search-and-train":
        run_search(cfg_path)
        run_train_source(cfg_path, skip_base=False, skip_blend=False, skip_stack=True)
    elif args.command == "status":
        print_status(cfg_path)


if __name__ == "__main__":
    main()
