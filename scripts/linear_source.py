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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import top1_pipeline as p1


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _save_state(out_dir: Path, state: Dict[str, Any]) -> None:
    p1.ensure_dir(out_dir / "scores")
    state["updated_at"] = now_ts()
    (out_dir / "scores" / "linear_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_state(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "scores" / "linear_state.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "targets": {},
        "global": {"targets_done": 0, "targets_failed": 0},
    }


def _load_or_init_base_frames(out_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame):
    p1.ensure_dir(out_dir / "base")
    oof_path = out_dir / "base" / "linear_oof.parquet"
    test_path = out_dir / "base" / "linear_test.parquet"
    if oof_path.exists() and test_path.exists():
        return pd.read_parquet(oof_path), pd.read_parquet(test_path)
    return (
        pd.DataFrame({"customer_id": train_df["customer_id"].values}),
        pd.DataFrame({"customer_id": test_df["customer_id"].values}),
    )


def _load_or_init_scores(out_dir: Path) -> Dict[str, Any]:
    p1.ensure_dir(out_dir / "scores")
    sp = out_dir / "scores" / "linear_scores.json"
    if sp.exists():
        payload = json.loads(sp.read_text(encoding="utf-8"))
        payload.setdefault("target_scores", {})
        payload.setdefault("target_param_overrides", {})
        return payload
    return {
        "model": "linear",
        "family": None,
        "macro_auc": float("nan"),
        "target_scores": {},
        "target_param_overrides": {},
    }


def _save_base_outputs(out_dir: Path, oof_df: pd.DataFrame, test_df: pd.DataFrame, score_payload: Dict[str, Any]) -> None:
    p1.ensure_dir(out_dir / "base")
    p1.ensure_dir(out_dir / "scores")
    oof_df.to_parquet(out_dir / "base" / "linear_oof.parquet", index=False)
    test_df.to_parquet(out_dir / "base" / "linear_test.parquet", index=False)
    with (out_dir / "scores" / "linear_scores.json").open("w", encoding="utf-8") as f:
        json.dump(score_payload, f, ensure_ascii=False, indent=2)


def _auc_safe(y: np.ndarray, pred: np.ndarray) -> float:
    pos = int(y.sum())
    if pos == 0 or pos == len(y):
        return float("nan")
    return float(roc_auc_score(y, pred))


def _build_estimator(cfg: Dict[str, Any], *, seed: int) -> Any:
    lin = cfg.get("linear", {})
    family = str(lin.get("family", "sgd_log")).lower()
    imputer_strategy = str(lin.get("imputer", "median"))
    scaler_with_mean = bool(lin.get("scaler_with_mean", False))

    if family == "logreg":
        params = {
            "C": float(lin.get("C", 1.0)),
            "max_iter": int(lin.get("max_iter", 200)),
            "class_weight": lin.get("class_weight", "balanced"),
            "solver": str(lin.get("solver", "saga")),
            "penalty": str(lin.get("penalty", "l2")),
            "n_jobs": int(lin.get("n_jobs", -1)),
            "random_state": seed,
        }
        est = LogisticRegression(**params)
    elif family in {"sgd", "sgd_log", "sgd_logloss"}:
        params = {
            "loss": "log_loss",
            "penalty": str(lin.get("penalty", "elasticnet")),
            "alpha": float(lin.get("alpha", 1e-5)),
            "l1_ratio": float(lin.get("l1_ratio", 0.15)),
            "max_iter": int(lin.get("max_iter", 1000)),
            "tol": float(lin.get("tol", 1e-3)),
            "class_weight": lin.get("class_weight", "balanced"),
            "random_state": seed,
        }
        est = SGDClassifier(**params)
    else:
        raise ValueError(f"Unsupported linear.family: {family}")

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("scaler", StandardScaler(with_mean=scaler_with_mean)),
            ("model", est),
        ]
    )
    return pipe


def _predict_proba_1d(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1].astype(np.float32)
        if p.ndim == 1:
            return p.astype(np.float32)
    if hasattr(model, "decision_function"):
        z = model.decision_function(x)
        z = np.asarray(z, dtype=np.float32)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    pred = np.asarray(model.predict(x), dtype=np.float32)
    return pred


def train_linear_source(cfg: Dict[str, Any], resume: bool = True) -> None:
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

    lin = cfg.get("linear", {})
    family = str(lin.get("family", "sgd_log")).lower()
    sample_train_rows = int(lin.get("sample_train_rows_per_fold", 0))
    sample_val_rows = int(lin.get("sample_val_rows_per_fold", 0))
    seed = int(lin.get("seed", cfg.get("seed", 42)))

    oof_df, test_pred_df = _load_or_init_base_frames(out_dir, train_df, test_df) if resume else (
        pd.DataFrame({"customer_id": train_df["customer_id"].values}),
        pd.DataFrame({"customer_id": test_df["customer_id"].values}),
    )
    score_payload = _load_or_init_scores(out_dir) if resume else {
        "model": "linear", "family": family, "macro_auc": float("nan"), "target_scores": {}, "target_param_overrides": {}
    }
    score_payload["family"] = family
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
    print(f"[linear_source] family={family} targets={len(target_cols)} folds={n_folds}")

    for target in target_cols:
        predict_col = p1.target_to_predict_col(target)
        if resume and target in oof_df.columns and predict_col in test_pred_df.columns and target in score_payload.get("target_scores", {}):
            print(f"  - {target}: skip (already present)")
            continue

        state["targets"][target] = {"status": "running"}
        _save_state(out_dir, state)
        try:
            target_features = target_feature_map.get(target, list(feature_cols)) if target_feature_map else list(feature_cols)
            x_target = x_all[target_features]
            x_test = x_test_all[target_features]
            y = train_df[target].to_numpy(dtype=np.int8, copy=False)
            pos = int(y.sum())
            if pos == 0 or pos == len(y):
                const = float(y.mean())
                oof_pred = np.full(len(train_df), const, dtype=np.float32)
                test_pred = np.full(len(test_df), const, dtype=np.float32)
                auc = float("nan")
            else:
                oof_pred = np.zeros(len(train_df), dtype=np.float32)
                test_fold_preds = []
                for fold in range(n_folds):
                    idx_val = np.where(fold_ids == fold)[0]
                    idx_tr = np.where(fold_ids != fold)[0]
                    if sample_train_rows > 0 and len(idx_tr) > sample_train_rows:
                        idx_tr = p1.sample_rows_by_class(y=y[idx_tr], max_rows=sample_train_rows, seed=seed + 13 * (fold + 1))
                        # sample_rows_by_class returns local indices if passed y slice, fix to global mapping
                        if len(idx_tr) and idx_tr.max() < len(np.where(fold_ids != fold)[0]):
                            idx_tr = np.where(fold_ids != fold)[0][idx_tr]
                    if sample_val_rows > 0 and len(idx_val) > sample_val_rows:
                        idx_val_local = p1.sample_rows_by_class(y=y[idx_val], max_rows=sample_val_rows, seed=seed + 29 * (fold + 1))
                        if len(idx_val_local) and idx_val_local.max() < len(np.where(fold_ids == fold)[0]):
                            idx_val = np.where(fold_ids == fold)[0][idx_val_local]
                    x_tr = x_target.iloc[idx_tr]
                    y_tr = y[idx_tr]
                    x_val = x_target.iloc[idx_val]
                    y_val = y[idx_val]
                    model = _build_estimator(cfg, seed=seed + fold)
                    model.fit(x_tr, y_tr)
                    val_pred = _predict_proba_1d(model, x_val)
                    oof_pred[idx_val] = val_pred
                    test_fold_preds.append(_predict_proba_1d(model, x_test))
                test_pred = np.mean(np.column_stack(test_fold_preds), axis=1).astype(np.float32)
                auc = _auc_safe(y, oof_pred)

            oof_df[target] = oof_pred
            test_pred_df[predict_col] = test_pred
            score_payload.setdefault("target_scores", {})[target] = auc
            score_payload.setdefault("target_param_overrides", {})[target] = {}
            valid = [v for v in score_payload["target_scores"].values() if not (isinstance(v, float) and math.isnan(v))]
            score_payload["macro_auc"] = float(np.mean(valid)) if valid else float("nan")
            _save_base_outputs(out_dir, oof_df, test_pred_df, score_payload)
            state["targets"][target] = {"status": "done", "auc": auc, "pos": pos}
            state["global"]["targets_done"] = int(sum(1 for v in state["targets"].values() if v.get("status") == "done"))
            _save_state(out_dir, state)
            print(f"  - {target}: AUC={auc:.6f} pos={pos}")
        except Exception as exc:
            state["targets"][target] = {"status": "failed", "error": str(exc)}
            state["global"]["targets_failed"] = int(sum(1 for v in state["targets"].values() if v.get("status") == "failed"))
            _save_state(out_dir, state)
            raise

    p1.append_experiment_log(
        cfg,
        event="linear_source_run_bases",
        payload={
            "family": family,
            "targets": len(target_cols),
            "oof_path": str(out_dir / "base" / "linear_oof.parquet"),
            "test_path": str(out_dir / "base" / "linear_test.parquet"),
            "macro_auc": score_payload.get("macro_auc"),
        },
    )
    print(f"[linear_source] macro_auc={float(score_payload.get('macro_auc', float('nan'))):.6f}")


def run_write_meta(cfg: Dict[str, Any]) -> None:
    prepared = p1.prepare_dataset(cfg)
    out_dir = Path(cfg["output_dir"])
    p1.ensure_dir(out_dir / "meta")
    np.save(out_dir / "meta" / "fold_ids.npy", prepared["fold_ids"])
    cols_payload = {
        "feature_cols": list(prepared["feature_cols"]),
        "cat_cols": list(prepared["cat_cols"]),
        "target_cols": list(prepared["target_cols"]),
    }
    (out_dir / "meta" / "columns.json").write_text(json.dumps(cols_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write-meta] targets={len(prepared['target_cols'])} features={len(prepared['feature_cols'])}")


def run_status(cfg: Dict[str, Any]) -> None:
    out_dir = Path(cfg["output_dir"])
    state = _load_state(out_dir)
    scores = _load_or_init_scores(out_dir)
    done = int(sum(1 for v in state.get("targets", {}).values() if v.get("status") == "done"))
    failed = int(sum(1 for v in state.get("targets", {}).values() if v.get("status") == "failed"))
    print(json.dumps({
        "output_dir": str(out_dir),
        "targets_done": done,
        "targets_failed": failed,
        "macro_auc": scores.get("macro_auc"),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Linear source runner (OOF/test) compatible with top1 pipeline")
    ap.add_argument("--config", required=True, help="Path to pipeline-like config JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("write-meta")
    sub.add_parser("run-bases")
    p_blend = sub.add_parser("blend")
    p_blend.add_argument("--models", default="linear")
    p_stack = sub.add_parser("stack")
    p_stack.add_argument("--models", default="linear")
    sub.add_parser("status")
    p_all = sub.add_parser("run-all")
    p_all.add_argument("--skip-stack", action="store_true")

    args = ap.parse_args()
    cfg = p1.load_config(Path(args.config))

    if args.cmd == "write-meta":
        run_write_meta(cfg)
        return
    if args.cmd == "run-bases":
        train_linear_source(cfg, resume=True)
        return
    if args.cmd == "blend":
        p1.run_blend(cfg, [m.strip() for m in args.models.split(",") if m.strip()])
        return
    if args.cmd == "stack":
        p1.run_stack(cfg, [m.strip() for m in args.models.split(",") if m.strip()])
        return
    if args.cmd == "status":
        run_status(cfg)
        return
    if args.cmd == "run-all":
        run_write_meta(cfg)
        train_linear_source(cfg, resume=True)
        p1.run_blend(cfg, ["linear"])
        if not args.skip_stack:
            p1.run_stack(cfg, ["linear"])
        return


if __name__ == "__main__":
    main()
