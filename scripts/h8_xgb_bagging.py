#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import top1_pipeline as p1


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def stable_int_seed(parts: Sequence[Any], nbytes: int = 8) -> int:
    import hashlib

    raw = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:nbytes], byteorder="little", signed=False)


def _load_or_init_base_frames(out_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p1.ensure_dir(out_dir / "base")
    oof_path = out_dir / "base" / "xgboost_oof.parquet"
    test_path = out_dir / "base" / "xgboost_test.parquet"
    if oof_path.exists() and test_path.exists():
        return pd.read_parquet(oof_path), pd.read_parquet(test_path)
    return (
        pd.DataFrame({"customer_id": train_df["customer_id"].values}),
        pd.DataFrame({"customer_id": test_df["customer_id"].values}),
    )


def _load_or_init_scores(out_dir: Path) -> Dict[str, Any]:
    p1.ensure_dir(out_dir / "scores")
    sp = out_dir / "scores" / "xgboost_scores.json"
    if sp.exists():
        payload = json.loads(sp.read_text(encoding="utf-8"))
        payload.setdefault("target_scores", {})
        payload.setdefault("target_param_overrides", {})
        payload.setdefault("bagging", {})
        return payload
    return {
        "model": "xgboost",
        "macro_auc": float("nan"),
        "target_scores": {},
        "target_param_overrides": {},
        "bagging": {},
    }


def _save_base_outputs(out_dir: Path, oof_df: pd.DataFrame, test_df: pd.DataFrame, score_payload: Dict[str, Any]) -> None:
    p1.ensure_dir(out_dir / "base")
    p1.ensure_dir(out_dir / "scores")
    oof_df.to_parquet(out_dir / "base" / "xgboost_oof.parquet", index=False)
    test_df.to_parquet(out_dir / "base" / "xgboost_test.parquet", index=False)
    with (out_dir / "scores" / "xgboost_scores.json").open("w", encoding="utf-8") as f:
        json.dump(score_payload, f, ensure_ascii=False, indent=2)


def _save_state(out_dir: Path, state: Dict[str, Any]) -> None:
    p1.ensure_dir(out_dir / "scores")
    state["updated_at"] = now_ts()
    (out_dir / "scores" / "xgb_bagging_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_state(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "scores" / "xgb_bagging_state.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "targets": {},
        "global": {"targets_done": 0, "targets_failed": 0},
    }


def _sample_bag_indices(
    y_train: np.ndarray,
    *,
    n_bags: int,
    bag_idx: int,
    seed: int,
    neg_to_pos_ratio: float,
    max_neg_per_bag: int,
    pos_bootstrap: bool,
    neg_replace: bool,
    train_row_cap_per_bag: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + bag_idx)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.arange(len(y_train), dtype=np.int64)

    if pos_bootstrap:
        pos_sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    else:
        pos_sample = pos_idx.copy()

    neg_take = int(round(len(pos_sample) * float(neg_to_pos_ratio)))
    neg_take = max(1, neg_take)
    if max_neg_per_bag > 0:
        neg_take = min(neg_take, int(max_neg_per_bag))
    neg_take = min(neg_take, len(neg_idx)) if not neg_replace else neg_take
    neg_sample = rng.choice(neg_idx, size=neg_take, replace=bool(neg_replace))

    idx = np.concatenate([pos_sample, neg_sample]).astype(np.int64, copy=False)
    rng.shuffle(idx)

    if train_row_cap_per_bag > 0 and len(idx) > train_row_cap_per_bag:
        # Preserve all positives, downsample negatives to fit cap.
        y_sub = y_train[idx]
        pos_mask = y_sub == 1
        pos_local = idx[pos_mask]
        neg_local = idx[~pos_mask]
        max_neg_keep = max(1, train_row_cap_per_bag - len(pos_local))
        if len(neg_local) > max_neg_keep:
            neg_local = rng.choice(neg_local, size=max_neg_keep, replace=False)
        idx = np.concatenate([pos_local, neg_local]).astype(np.int64, copy=False)
        rng.shuffle(idx)
    return idx


def _fit_xgb_with_fallback(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    x_test: pd.DataFrame,
    params: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    from xgboost import XGBClassifier

    p = dict(params)
    early_stopping_rounds = int(p.pop("early_stopping_rounds", 0))
    model = XGBClassifier(**p)
    fit_kwargs: Dict[str, Any] = {"eval_set": [(x_val, y_val)], "verbose": False}
    if early_stopping_rounds > 0:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    try:
        model.fit(x_train, y_train, **fit_kwargs)
    except Exception as exc:
        err = str(exc).lower()
        if ("cuda" in err or "gpu" in err) and (p.get("device") == "cuda" or p.get("tree_method") in {"gpu_hist"}):
            cpu_p = dict(p)
            cpu_p.pop("device", None)
            cpu_p["tree_method"] = "hist"
            model = XGBClassifier(**cpu_p)
            model.fit(x_train, y_train, **fit_kwargs)
        else:
            raise
    val_pred = model.predict_proba(x_val)[:, 1].astype(np.float32)
    test_pred = model.predict_proba(x_test)[:, 1].astype(np.float32)
    return val_pred, test_pred


def train_xgb_bagging(cfg: Dict[str, Any], resume: bool = True) -> None:
    prepared = p1.prepare_dataset(cfg)
    train_df = prepared["train_df"]
    test_df = prepared["test_df"]
    feature_cols = prepared["feature_cols"]
    target_cols = prepared["target_cols"]
    cat_cols = prepared["cat_cols"]
    fold_ids = prepared["fold_ids"]
    out_dir = Path(cfg["output_dir"])

    target_feature_map = p1.build_per_target_feature_map(
        cfg=cfg,
        train_df=train_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        cat_cols=cat_cols,
    )

    bag = cfg.get("bagging", {})
    n_bags = int(bag.get("n_bags", 5))
    neg_to_pos_ratio = float(bag.get("neg_to_pos_ratio", 20.0))
    max_neg_per_bag = int(bag.get("max_neg_per_bag", 0))
    pos_bootstrap = bool(bag.get("pos_bootstrap", False))
    neg_replace = bool(bag.get("neg_replace", False))
    train_row_cap_per_bag = int(bag.get("train_row_cap_per_bag", 0))
    global_seed = int(bag.get("seed", cfg.get("seed", 42)))

    model_cfg = cfg["models"]["xgboost"]
    if not bool(model_cfg.get("enabled", True)):
        raise ValueError("xgboost must be enabled in config for h8_xgb_bagging")

    oof_df, test_pred_df = _load_or_init_base_frames(out_dir, train_df, test_df) if resume else (
        pd.DataFrame({"customer_id": train_df["customer_id"].values}),
        pd.DataFrame({"customer_id": test_df["customer_id"].values}),
    )
    score_payload = _load_or_init_scores(out_dir) if resume else {
        "model": "xgboost",
        "macro_auc": float("nan"),
        "target_scores": {},
        "target_param_overrides": {},
        "bagging": {},
    }
    score_payload["bagging"] = {
        "n_bags": n_bags,
        "neg_to_pos_ratio": neg_to_pos_ratio,
        "max_neg_per_bag": max_neg_per_bag,
        "pos_bootstrap": pos_bootstrap,
        "neg_replace": neg_replace,
        "train_row_cap_per_bag": train_row_cap_per_bag,
    }
    state = _load_state(out_dir) if resume else {
        "schema_version": 1,
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "targets": {},
        "global": {"targets_done": 0, "targets_failed": 0},
    }

    n_folds = int(cfg["folds"].get("n_splits", 3))
    x_all = train_df[list(feature_cols)]
    x_test_all = test_df[list(feature_cols)]

    print(f"[h8_bagging] model=xgboost targets={len(target_cols)} folds={n_folds} bags={n_bags}")
    for target in target_cols:
        predict_col = p1.target_to_predict_col(target)
        if resume and target in oof_df.columns and predict_col in test_pred_df.columns and target in score_payload.get("target_scores", {}):
            print(f"  - {target}: skip (already present)")
            continue

        state["targets"][target] = {"status": "running"}
        _save_state(out_dir, state)
        try:
            target_features = target_feature_map.get(target, list(feature_cols)) if target_feature_map else list(feature_cols)
            target_cat_cols = [c for c in cat_cols if c in target_features]
            if target_cat_cols:
                # XGBoost in this runner treats all features as numeric (already encoded).
                target_cat_cols = []
            x_target = x_all[target_features]
            x_test = x_test_all[target_features]
            y = train_df[target].to_numpy(dtype=np.int8, copy=False)
            positives = int(y.sum())

            target_model_params = p1.resolve_model_params_for_target(cfg, "xgboost", model_cfg, target)
            base_params = dict(model_cfg["params"])
            diff = {k: v for k, v in target_model_params.items() if k not in base_params or base_params[k] != v}
            if diff:
                score_payload.setdefault("target_param_overrides", {})[target] = diff

            if positives == 0 or positives == len(y):
                const = float(y.mean())
                oof_df[target] = np.full(len(y), const, dtype=np.float32)
                test_pred_df[predict_col] = np.full(len(test_df), const, dtype=np.float32)
                score_payload.setdefault("target_scores", {})[target] = float("nan")
                state["targets"][target] = {"status": "done", "constant": True, "positives": positives}
                _save_base_outputs(out_dir, oof_df, test_pred_df, score_payload)
                _save_state(out_dir, state)
                print(f"  - {target}: constant target pos={positives}")
                continue

            oof = np.zeros(len(y), dtype=np.float32)
            test_pred = np.zeros(len(test_df), dtype=np.float32)
            for fold in range(n_folds):
                idx_val = np.where(fold_ids == fold)[0]
                idx_tr = np.where(fold_ids != fold)[0]
                x_val = x_target.iloc[idx_val]
                y_val = y[idx_val]
                x_train_full = x_target.iloc[idx_tr]
                y_train_full = y[idx_tr]

                fold_val_acc = np.zeros(len(idx_val), dtype=np.float32)
                fold_test_acc = np.zeros(len(test_df), dtype=np.float32)
                for bag_idx in range(n_bags):
                    bag_seed = global_seed + (stable_int_seed([target, fold, bag_idx]) % 1_000_000)
                    local_idx = _sample_bag_indices(
                        y_train_full,
                        n_bags=n_bags,
                        bag_idx=bag_idx,
                        seed=bag_seed,
                        neg_to_pos_ratio=neg_to_pos_ratio,
                        max_neg_per_bag=max_neg_per_bag,
                        pos_bootstrap=pos_bootstrap,
                        neg_replace=neg_replace,
                        train_row_cap_per_bag=train_row_cap_per_bag,
                    )
                    x_train = x_train_full.iloc[local_idx]
                    y_train = y_train_full[local_idx]
                    val_pred, bag_test_pred = _fit_xgb_with_fallback(
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        x_test=x_test,
                        params=target_model_params,
                    )
                    fold_val_acc += val_pred / n_bags
                    fold_test_acc += bag_test_pred / n_bags

                oof[idx_val] = fold_val_acc
                test_pred += fold_test_acc / n_folds

            score = float(roc_auc_score(y, oof))
            oof_df[target] = oof
            test_pred_df[predict_col] = test_pred.astype(np.float32)
            score_payload.setdefault("target_scores", {})[target] = score
            state["targets"][target] = {
                "status": "done",
                "positives": positives,
                "auc": score,
                "n_bags": n_bags,
            }
            _save_base_outputs(out_dir, oof_df, test_pred_df, score_payload)
            _save_state(out_dir, state)
            print(f"  - {target}: AUC={score:.6f} pos={positives} bags={n_bags}")
        except Exception as exc:
            state["targets"][target] = {"status": "failed", "error": str(exc)}
            state.setdefault("global", {}).setdefault("targets_failed", 0)
            state["global"]["targets_failed"] += 1
            _save_state(out_dir, state)
            raise

    valid_scores = [v for v in score_payload.get("target_scores", {}).values() if not (isinstance(v, float) and math.isnan(v))]
    score_payload["macro_auc"] = float(np.mean(valid_scores)) if valid_scores else float("nan")
    _save_base_outputs(out_dir, oof_df, test_pred_df, score_payload)
    state["global"]["targets_done"] = int(sum(1 for r in state.get("targets", {}).values() if r.get("status") == "done"))
    _save_state(out_dir, state)
    p1.append_experiment_log(
        cfg,
        event="h8_xgb_bagging_base",
        payload={
            "macro_auc": score_payload["macro_auc"],
            "targets": len(target_cols),
            "oof_path": str(out_dir / "base" / "xgboost_oof.parquet"),
            "test_path": str(out_dir / "base" / "xgboost_test.parquet"),
            "n_bags": n_bags,
            "neg_to_pos_ratio": neg_to_pos_ratio,
        },
    )
    print(f"[h8_bagging] macro_auc={score_payload['macro_auc']:.6f}")


def print_status(cfg: Dict[str, Any]) -> None:
    out_dir = Path(cfg["output_dir"])
    state_path = out_dir / "scores" / "xgb_bagging_state.json"
    if not state_path.exists():
        print(f"[h8_bagging] no state file yet: {state_path}")
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    print("[h8_bagging] state:", state_path)
    print("[h8_bagging] global:", state.get("global", {}))
    for target, rec in sorted(state.get("targets", {}).items()):
        print(f"  - {target}: {rec.get('status')} auc={rec.get('auc')} pos={rec.get('positives')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="H8 XGBoost bagging runner (partial-source specialists, resumable)")
    p.add_argument("--config", required=True, help="Path to pipeline-style JSON config")
    sub = p.add_subparsers(dest="command", required=True)
    p_rb = sub.add_parser("run-bases", help="Train XGBoost bagging base predictions")
    p_rb.add_argument("--no-resume", action="store_true")
    sub.add_parser("blend", help="Run pipeline blend (usually xgboost-only)")
    sub.add_parser("stack", help="Run pipeline stack (optional)")
    sub.add_parser("write-meta", help="Write standardized meta only")
    sub.add_parser("status", help="Show target-level bagging status")
    p_all = sub.add_parser("run-all", help="run-bases + blend (+ stack optional)")
    p_all.add_argument("--run-stack", action="store_true")
    p_all.add_argument("--no-resume", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = p1.load_config(Path(args.config))
    if args.command == "run-bases":
        train_xgb_bagging(cfg, resume=not bool(args.no_resume))
        return
    if args.command == "blend":
        p1.run_blend(cfg, ["xgboost"])
        return
    if args.command == "stack":
        p1.run_stack(cfg, ["xgboost"])
        return
    if args.command == "write-meta":
        p1.run_write_meta(cfg)
        return
    if args.command == "status":
        print_status(cfg)
        return
    if args.command == "run-all":
        train_xgb_bagging(cfg, resume=not bool(args.no_resume))
        p1.run_blend(cfg, ["xgboost"])
        if bool(args.run_stack):
            p1.run_stack(cfg, ["xgboost"])
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
