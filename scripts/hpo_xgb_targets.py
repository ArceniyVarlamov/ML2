#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Allow importing top-level project modules when script is launched as scripts/...
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


def load_hpo_cfg(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pipeline_cfg(hpo_cfg: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """
    Build pipeline config from base config + overrides.
    mode:
      - search: output_dir points to hpo work dir / search_prep
      - train_source: output_dir points to train_source.output_dir
    """
    base_cfg_path = Path(hpo_cfg["base_pipeline_config"])
    base_cfg = p1.load_config(base_cfg_path)
    cfg = copy.deepcopy(base_cfg)
    cfg = deep_merge(cfg, hpo_cfg.get("pipeline_overrides", {}))

    scope = hpo_cfg.get("target_scope", {})
    if scope:
        cfg["targets"] = deep_merge(cfg.get("targets", {}), scope)

    if mode == "search":
        work_dir = Path(hpo_cfg["output_dir"])
        cfg["output_dir"] = str(work_dir / "search_prep")
    elif mode == "train_source":
        train_source = hpo_cfg.get("train_source", {})
        out_dir = train_source.get("output_dir")
        if not out_dir:
            raise ValueError("train_source.output_dir is required for train_source mode")
        cfg["output_dir"] = str(out_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return cfg


def apply_tuned_params_to_cfg(cfg: Dict[str, Any], params_by_target: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    models = out.setdefault("models", {})
    if "xgboost" not in models:
        raise KeyError("Pipeline config has no models.xgboost")
    models["xgboost"]["enabled"] = True
    models["xgboost"]["params_by_target"] = copy.deepcopy(params_by_target)
    return out


def target_list_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    cols = p1.discover_columns(cfg)
    return list(cols["target_cols"])


def load_state(path: Path, hpo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "hpo_config_sha256": stable_hash_obj(hpo_cfg),
        "targets": {},
        "global": {
            "trials_completed": 0,
            "trials_failed": 0,
            "targets_completed": 0,
        },
    }


def save_state(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = now_ts()
    ensure_dir(path.parent)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_space_value(spec: Dict[str, Any], rng: np.random.Generator) -> Any:
    typ = str(spec.get("type", "float"))
    if typ == "categorical":
        choices = list(spec["choices"])
        idx = int(rng.integers(0, len(choices)))
        return choices[idx]
    if typ == "bool":
        return bool(rng.integers(0, 2))
    if typ == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        if high < low:
            raise ValueError(f"Invalid int range {low}>{high}")
        step = int(spec.get("step", 1))
        n = ((high - low) // step) + 1
        k = int(rng.integers(0, n))
        return int(low + k * step)
    if typ == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        if high < low:
            raise ValueError(f"Invalid float range {low}>{high}")
        if bool(spec.get("log", False)):
            if low <= 0 or high <= 0:
                raise ValueError("Log-uniform float range must be positive")
            x = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        else:
            x = float(rng.uniform(low, high))
        step = spec.get("step")
        if step is not None:
            step_f = float(step)
            x = round(x / step_f) * step_f
        if "round" in spec:
            x = round(x, int(spec["round"]))
        return float(x)
    raise ValueError(f"Unsupported search space type: {typ}")


def make_trial_params(
    base_params: Dict[str, Any],
    search_space: Dict[str, Dict[str, Any]],
    seed: int,
    target: str,
    trial_idx: int,
) -> Dict[str, Any]:
    digest = hashlib.sha256(f"{seed}|{target}|{trial_idx}".encode("utf-8")).digest()
    trial_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(trial_seed)
    params = dict(base_params)
    for key, spec in search_space.items():
        params[key] = sample_space_value(spec, rng)
    params["random_state"] = int(params.get("random_state", seed))
    return params


def sample_fold_rows(
    idx: np.ndarray,
    y_full: np.ndarray,
    max_rows: int,
    seed: int,
    stratified: bool = True,
) -> np.ndarray:
    if max_rows <= 0 or len(idx) <= max_rows:
        return idx
    if not stratified:
        rng = np.random.default_rng(seed)
        return np.array(rng.choice(idx, size=max_rows, replace=False), dtype=np.int64)
    y = y_full[idx]
    pos = idx[y == 1]
    neg = idx[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        rng = np.random.default_rng(seed)
        return np.array(rng.choice(idx, size=max_rows, replace=False), dtype=np.int64)
    pos_share = len(pos) / len(idx)
    pos_take = int(round(max_rows * pos_share))
    pos_take = max(1, min(pos_take, len(pos)))
    neg_take = max_rows - pos_take
    neg_take = max(1, min(neg_take, len(neg)))
    rng = np.random.default_rng(seed)
    pos_sample = rng.choice(pos, size=pos_take, replace=False)
    neg_sample = rng.choice(neg, size=neg_take, replace=False)
    out = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(out)
    return np.array(out, dtype=np.int64)


def fit_xgb_fold_val_only(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    from xgboost import XGBClassifier

    params = dict(params)
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 0))
    model = XGBClassifier(**params)
    fit_kwargs: Dict[str, Any] = {"eval_set": [(x_val, y_val)], "verbose": False}
    if early_stopping_rounds > 0:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    model.fit(x_train, y_train, **fit_kwargs)
    return model.predict_proba(x_val)[:, 1].astype(np.float32)


def evaluate_xgb_params_oof(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
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
        return {
            "auc": float("nan"),
            "positives": positives,
            "fold_aucs": [],
            "status": "constant_target",
        }

    feats = list(target_features) if target_features is not None else list(feature_cols)
    x_target = train_df[feats]
    n_folds = int(np.max(fold_ids)) + 1
    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_aucs: List[float] = []

    for fold in range(n_folds):
        idx_val = np.where(fold_ids == fold)[0]
        idx_tr = np.where(fold_ids != fold)[0]

        idx_tr = sample_fold_rows(
            idx_tr, y, max_rows=int(sample_train_rows_per_fold), seed=seed + 1000 + fold, stratified=True
        )
        idx_val = sample_fold_rows(
            idx_val, y, max_rows=int(sample_val_rows_per_fold), seed=seed + 2000 + fold, stratified=True
        )

        x_train = x_target.iloc[idx_tr]
        y_train = y[idx_tr]
        x_val = x_target.iloc[idx_val]
        y_val = y[idx_val]

        val_pred = fit_xgb_fold_val_only(x_train, y_train, x_val, y_val, params=params)
        oof[idx_val] = val_pred
        fold_aucs.append(float(roc_auc_score(y_val, val_pred)))

    valid_mask = ~np.isnan(oof)  # only sampled val rows may be filled
    if valid_mask.all():
        auc = float(roc_auc_score(y, oof))
    else:
        auc = float(roc_auc_score(y[valid_mask], oof[valid_mask]))
    return {
        "auc": auc,
        "positives": positives,
        "fold_aucs": fold_aucs,
        "status": "ok",
        "filled_rows": int(valid_mask.sum()),
        "total_rows": int(len(y)),
    }


def select_targets_for_search(cfg: Dict[str, Any]) -> List[str]:
    return target_list_from_cfg(cfg)


def run_search(hpo_cfg_path: Path) -> None:
    hpo_cfg = load_hpo_cfg(hpo_cfg_path)
    work_dir = Path(hpo_cfg["output_dir"])
    ensure_dir(work_dir)
    ensure_dir(work_dir / "logs")
    state_path = work_dir / "state.json"
    trials_log = work_dir / "trials.jsonl"
    best_params_path = work_dir / "best_params_by_target.json"
    summary_path = work_dir / "summary.json"
    run_cfg_snapshot = work_dir / "hpo_config_snapshot.json"
    run_cfg_snapshot.write_text(json.dumps(hpo_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

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

    xgb_cfg = search_cfg["models"]["xgboost"]
    base_params = dict(xgb_cfg["params"])
    base_params.pop("device", None) if hpo_cfg.get("search", {}).get("force_cpu_for_search", False) else None
    if hpo_cfg.get("search", {}).get("force_cpu_for_search", False):
        base_params["tree_method"] = "hist"

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

    state = load_state(state_path, hpo_cfg)
    start_ts = time.time()
    completed_global = int(state.get("global", {}).get("trials_completed", 0))
    failed_global = int(state.get("global", {}).get("trials_failed", 0))

    targets = list(target_cols)
    if max_targets > 0:
        targets = targets[:max_targets]

    for ti, target in enumerate(targets, start=1):
        t0 = time.time()
        tstate = state["targets"].setdefault(
            target,
            {
                "status": "pending",
                "trials": {},
                "best_trial_idx": None,
                "best_auc": None,
                "best_params": None,
            },
        )
        print(f"[hpo] target {ti}/{len(targets)}: {target}")
        target_seed = global_seed + (stable_int_seed([hpo_cfg.get("name", "hpo"), target]) % 1000003)
        best_auc = float(tstate["best_auc"]) if tstate["best_auc"] is not None else -1.0
        best_params = tstate["best_params"]
        best_trial_idx = tstate["best_trial_idx"]
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

            params = make_trial_params(
                base_params=base_params,
                search_space=search_space,
                seed=target_seed,
                target=target,
                trial_idx=trial_idx,
            )
            t_trial = time.time()
            row = {
                "ts_utc": now_ts(),
                "event": "hpo_trial",
                "target": target,
                "trial_idx": trial_idx,
                "params": params,
            }
            try:
                result = evaluate_xgb_params_oof(
                    train_df=train_df,
                    feature_cols=feature_cols,
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
                row.update(
                    {
                        "status": "ok",
                        "auc": auc,
                        "duration_sec": dur,
                        "fold_aucs": result.get("fold_aucs", []),
                        "filled_rows": result.get("filled_rows"),
                        "total_rows": result.get("total_rows"),
                        "positives": result.get("positives"),
                    }
                )
                tstate["trials"][trial_key] = {
                    "status": "ok",
                    "auc": auc,
                    "duration_sec": dur,
                    "params": params,
                }
                completed_global += 1
                if not math.isnan(auc) and auc > best_auc:
                    best_auc = float(auc)
                    best_params = params
                    best_trial_idx = trial_idx
                    tstate["best_auc"] = best_auc
                    tstate["best_params"] = best_params
                    tstate["best_trial_idx"] = best_trial_idx
                    print(f"[hpo] {target} trial={trial_idx} NEW BEST auc={best_auc:.6f}")
                else:
                    print(f"[hpo] {target} trial={trial_idx} auc={auc:.6f}")
            except Exception as exc:
                dur = time.time() - t_trial
                row.update({"status": "failed", "duration_sec": dur, "error": str(exc)})
                tstate["trials"][trial_key] = {
                    "status": "failed",
                    "duration_sec": dur,
                    "error": str(exc),
                    "params": params,
                }
                failed_global += 1
                print(f"[hpo] {target} trial={trial_idx} FAILED: {exc}")

            append_jsonl(trials_log, row)
            state["global"]["trials_completed"] = completed_global
            state["global"]["trials_failed"] = failed_global
            save_state(state_path, state)

        tstate["status"] = "done"
        state["global"]["targets_completed"] = int(
            sum(1 for t in state["targets"].values() if t.get("status") == "done")
        )
        save_state(state_path, state)

        if (max_trials_total > 0 and completed_global >= max_trials_total) or (
            max_seconds_total > 0 and (time.time() - start_ts) >= max_seconds_total
        ):
            break

    best_params_by_target: Dict[str, Dict[str, Any]] = {}
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
            "trials_failed": int(
                sum(1 for v in tstate.get("trials", {}).values() if v.get("status") == "failed")
            ),
        }

    best_params_path.write_text(json.dumps(best_params_by_target, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "ts_utc": now_ts(),
        "hpo_config_path": str(hpo_cfg_path),
        "hpo_config_sha256": stable_hash_obj(hpo_cfg),
        "targets_total": len(targets),
        "targets_with_best_params": len(best_params_by_target),
        "global": state["global"],
        "targets": summary_targets,
        "best_params_by_target_path": str(best_params_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hpo] saved state: {state_path}")
    print(f"[hpo] saved best params: {best_params_path}")
    print(f"[hpo] saved summary: {summary_path}")


def export_tuned_pipeline_config(hpo_cfg_path: Path) -> Path:
    hpo_cfg = load_hpo_cfg(hpo_cfg_path)
    work_dir = Path(hpo_cfg["output_dir"])
    best_params_path = work_dir / "best_params_by_target.json"
    if not best_params_path.exists():
        raise FileNotFoundError(f"Missing best params file: {best_params_path}")
    best_params = json.loads(best_params_path.read_text(encoding="utf-8"))

    train_cfg = build_pipeline_cfg(hpo_cfg, mode="train_source")
    train_cfg = apply_tuned_params_to_cfg(train_cfg, best_params)

    train_source = hpo_cfg.get("train_source", {})
    tuned_cfg_path = Path(train_source.get("tuned_config_path", work_dir / "tuned_pipeline_config.json"))
    ensure_dir(tuned_cfg_path.parent)
    tuned_cfg_path.write_text(json.dumps(train_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hpo] exported tuned pipeline config: {tuned_cfg_path}")
    return tuned_cfg_path


def run_train_source(hpo_cfg_path: Path, skip_base: bool = False, skip_blend: bool = False, skip_stack: bool = True) -> None:
    hpo_cfg = load_hpo_cfg(hpo_cfg_path)
    tuned_cfg_path = export_tuned_pipeline_config(hpo_cfg_path)
    cfg = p1.load_config(tuned_cfg_path)

    train_source = hpo_cfg.get("train_source", {})
    model_names = [str(x) for x in train_source.get("models", ["xgboost"])]
    if not model_names:
        raise ValueError("train_source.models must contain at least one model")

    if not skip_base:
        p1.run_base_models(cfg, model_names)
    if bool(train_source.get("run_blend", True)) and not skip_blend:
        p1.run_blend(cfg, model_names)
    if bool(train_source.get("run_stack", False)) and not skip_stack:
        p1.run_stack(cfg, model_names)

    if bool(train_source.get("emit_submission", False)):
        src = Path(cfg["output_dir"]) / "ensemble" / "blend_test.parquet"
        out = Path(cfg["output_dir"]) / "submissions" / "sub_hpo_xgb_blend_float64.parquet"
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
        print(
            f"  - {target}: status={rec.get('status')} best_auc={rec.get('best_auc')} "
            f"best_trial_idx={rec.get('best_trial_idx')}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quality-first XGBoost HPO for target subsets/full set with resumable state")
    p.add_argument("--config", required=True, help="Path to HPO JSON config")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("search", help="Run/resume HPO search and export best_params_by_target.json")
    sub.add_parser("export-config", help="Export tuned pipeline config with params_by_target")
    p_train = sub.add_parser("train-source", help="Train tuned xgboost source using exported params_by_target")
    p_train.add_argument("--skip-base", action="store_true")
    p_train.add_argument("--skip-blend", action="store_true")
    p_train.add_argument("--run-stack", action="store_true", help="Also run stack if enabled in HPO config")
    sub.add_parser("search-and-train", help="Run search, then export config and train source")
    sub.add_parser("status", help="Print status from state.json/summary.json")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg_path = Path(args.config)
    if args.command == "search":
        run_search(cfg_path)
        return
    if args.command == "export-config":
        export_tuned_pipeline_config(cfg_path)
        return
    if args.command == "train-source":
        run_train_source(
            cfg_path,
            skip_base=bool(args.skip_base),
            skip_blend=bool(args.skip_blend),
            skip_stack=not bool(args.run_stack),
        )
        return
    if args.command == "search-and-train":
        run_search(cfg_path)
        run_train_source(cfg_path, skip_base=False, skip_blend=False, skip_stack=True)
        return
    if args.command == "status":
        print_status(cfg_path)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
