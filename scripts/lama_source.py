#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import top1_pipeline as p1


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _save_state(out_dir: Path, state: Dict[str, Any]) -> None:
    p1.ensure_dir(out_dir / "scores")
    state["updated_at"] = now_ts()
    (out_dir / "scores" / "lama_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_state(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "scores" / "lama_state.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"schema_version": 1, "created_at": now_ts(), "updated_at": now_ts(), "targets": {}, "global": {"targets_done": 0, "targets_failed": 0}}


def _load_or_init_base_frames(out_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame):
    p1.ensure_dir(out_dir / "base")
    oof_path = out_dir / "base" / "lama_oof.parquet"
    test_path = out_dir / "base" / "lama_test.parquet"
    if oof_path.exists() and test_path.exists():
        return pd.read_parquet(oof_path), pd.read_parquet(test_path)
    return pd.DataFrame({"customer_id": train_df["customer_id"].values}), pd.DataFrame({"customer_id": test_df["customer_id"].values})


def _load_or_init_scores(out_dir: Path) -> Dict[str, Any]:
    p1.ensure_dir(out_dir / "scores")
    sp = out_dir / "scores" / "lama_scores.json"
    if sp.exists():
        payload = json.loads(sp.read_text(encoding="utf-8"))
        payload.setdefault("target_scores", {})
        return payload
    return {"model": "lama", "macro_auc": float("nan"), "target_scores": {}, "runtime": {}}


def _save_base_outputs(out_dir: Path, oof_df: pd.DataFrame, test_df: pd.DataFrame, score_payload: Dict[str, Any]) -> None:
    p1.ensure_dir(out_dir / "base")
    p1.ensure_dir(out_dir / "scores")
    oof_df.to_parquet(out_dir / "base" / "lama_oof.parquet", index=False)
    test_df.to_parquet(out_dir / "base" / "lama_test.parquet", index=False)
    with (out_dir / "scores" / "lama_scores.json").open("w", encoding="utf-8") as f:
        json.dump(score_payload, f, ensure_ascii=False, indent=2)


def _auc_safe(y: np.ndarray, pred: np.ndarray) -> float:
    pos = int(y.sum())
    if pos == 0 or pos == len(y):
        return float("nan")
    return float(roc_auc_score(y, pred))


def _import_lama():
    try:
        from lightautoml.automl.presets.tabular_presets import TabularAutoML
        from lightautoml.tasks import Task
        return TabularAutoML, Task
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "LightAutoML is not available. Install lightautoml on the A100 runtime before running lama_source.py"
        ) from exc


def _to_pred_array(obj: Any) -> np.ndarray:
    data = getattr(obj, "data", obj)
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return arr.astype(np.float32, copy=False)


def _fit_fold_lama(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    lama_cfg: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    TabularAutoML, Task = _import_lama()
    timeout = int(lama_cfg.get("timeout_sec_per_fold", 1800))
    cpu_limit = int(lama_cfg.get("cpu_limit", 8))
    random_state = int(lama_cfg.get("random_state", 42))
    use_algos = lama_cfg.get("use_algos")
    general_params = dict(lama_cfg.get("general_params", {}))
    if use_algos and "use_algos" not in general_params:
        general_params["use_algos"] = use_algos

    train = x_train.copy()
    train["__target__"] = y_train.astype(np.int8)
    valid = x_val.copy()
    valid["__target__"] = 0  # placeholder

    automl = TabularAutoML(
        task=Task("binary"),
        timeout=timeout,
        cpu_limit=cpu_limit,
        reader_params={"random_state": random_state},
        general_params=general_params if general_params else None,
    )

    roles = {"target": "__target__"}
    try:
        oof_valid = automl.fit_predict(train, roles=roles, valid_data=valid)
    except TypeError:
        # Some versions do not accept valid_data; fallback to internal fit then predict val.
        automl.fit_predict(train, roles=roles)
        oof_valid = automl.predict(valid.drop(columns=["__target__"]))
    val_pred = _to_pred_array(oof_valid)
    test_pred = _to_pred_array(automl.predict(x_test))
    return val_pred, test_pred


def train_lama_source(cfg: Dict[str, Any], resume: bool = True) -> None:
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

    lama_cfg = cfg.get("lama", {})
    seed = int(lama_cfg.get("seed", cfg.get("seed", 42)))
    sample_train_rows = int(lama_cfg.get("sample_train_rows_per_fold", 0))
    sample_val_rows = int(lama_cfg.get("sample_val_rows_per_fold", 0))

    oof_df, test_pred_df = _load_or_init_base_frames(out_dir, train_df, test_df) if resume else (
        pd.DataFrame({"customer_id": train_df["customer_id"].values}),
        pd.DataFrame({"customer_id": test_df["customer_id"].values}),
    )
    score_payload = _load_or_init_scores(out_dir) if resume else {"model": "lama", "macro_auc": float("nan"), "target_scores": {}, "runtime": {}}
    state = _load_state(out_dir) if resume else {"schema_version": 1, "created_at": now_ts(), "updated_at": now_ts(), "targets": {}, "global": {"targets_done": 0, "targets_failed": 0}}

    n_folds = int(cfg["folds"].get("n_splits", 3))
    x_all = train_df[list(feature_cols)].copy()
    x_test_all = test_df[list(feature_cols)].copy()
    # LAMA handles numeric tables better if types are compact.
    for c in x_all.columns:
        if x_all[c].dtype.kind in "iu":
            x_all[c] = x_all[c].astype(np.int32, copy=False)
            x_test_all[c] = x_test_all[c].astype(np.int32, copy=False)
        else:
            x_all[c] = x_all[c].astype(np.float32, copy=False)
            x_test_all[c] = x_test_all[c].astype(np.float32, copy=False)

    print(f"[lama_source] targets={len(target_cols)} folds={n_folds}")
    for target in target_cols:
        predict_col = p1.target_to_predict_col(target)
        if resume and target in oof_df.columns and predict_col in test_pred_df.columns and target in score_payload.get("target_scores", {}):
            print(f"  - {target}: skip (already present)")
            continue
        state["targets"][target] = {"status": "running"}
        _save_state(out_dir, state)
        t0 = time.time()
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
                test_fold_preds: List[np.ndarray] = []
                for fold in range(n_folds):
                    idx_val = np.where(fold_ids == fold)[0]
                    idx_tr = np.where(fold_ids != fold)[0]
                    if sample_train_rows > 0 and len(idx_tr) > sample_train_rows:
                        idx_local = p1.sample_rows_by_class(y=y[idx_tr], max_rows=sample_train_rows, seed=seed + 101 * (fold + 1))
                        idx_tr = np.where(fold_ids != fold)[0][idx_local]
                    if sample_val_rows > 0 and len(idx_val) > sample_val_rows:
                        idx_local = p1.sample_rows_by_class(y=y[idx_val], max_rows=sample_val_rows, seed=seed + 211 * (fold + 1))
                        idx_val = np.where(fold_ids == fold)[0][idx_local]
                    val_pred, test_pred_fold = _fit_fold_lama(
                        x_train=x_target.iloc[idx_tr],
                        y_train=y[idx_tr],
                        x_val=x_target.iloc[idx_val],
                        x_test=x_test,
                        lama_cfg=dict(lama_cfg, random_state=seed + fold),
                    )
                    oof_pred[idx_val] = val_pred
                    test_fold_preds.append(test_pred_fold)
                test_pred = np.mean(np.column_stack(test_fold_preds), axis=1).astype(np.float32)
                auc = _auc_safe(y, oof_pred)

            oof_df[target] = oof_pred
            test_pred_df[predict_col] = test_pred
            score_payload.setdefault("target_scores", {})[target] = auc
            score_payload.setdefault("runtime", {})[target] = float(time.time() - t0)
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
        event="lama_source_run_bases",
        payload={
            "targets": len(target_cols),
            "oof_path": str(out_dir / "base" / "lama_oof.parquet"),
            "test_path": str(out_dir / "base" / "lama_test.parquet"),
            "macro_auc": score_payload.get("macro_auc"),
        },
    )
    print(f"[lama_source] macro_auc={float(score_payload.get('macro_auc', float('nan'))):.6f}")


def run_write_meta(cfg: Dict[str, Any]) -> None:
    prepared = p1.prepare_dataset(cfg)
    out_dir = Path(cfg["output_dir"])
    p1.ensure_dir(out_dir / "meta")
    np.save(out_dir / "meta" / "fold_ids.npy", prepared["fold_ids"])
    payload = {"feature_cols": list(prepared["feature_cols"]), "cat_cols": list(prepared["cat_cols"]), "target_cols": list(prepared["target_cols"])}
    (out_dir / "meta" / "columns.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write-meta] targets={len(prepared['target_cols'])} features={len(prepared['feature_cols'])}")


def run_status(cfg: Dict[str, Any]) -> None:
    out_dir = Path(cfg["output_dir"])
    state = _load_state(out_dir)
    scores = _load_or_init_scores(out_dir)
    print(json.dumps({
        "output_dir": str(out_dir),
        "targets_done": int(sum(1 for v in state.get('targets', {}).values() if v.get('status') == 'done')),
        "targets_failed": int(sum(1 for v in state.get('targets', {}).values() if v.get('status') == 'failed')),
        "macro_auc": scores.get("macro_auc")
    }, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="LightAutoML source runner (OOF/test) compatible with top1 pipeline")
    ap.add_argument("--config", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("write-meta")
    sub.add_parser("run-bases")
    p_blend = sub.add_parser("blend")
    p_blend.add_argument("--models", default="lama")
    p_stack = sub.add_parser("stack")
    p_stack.add_argument("--models", default="lama")
    sub.add_parser("status")
    p_all = sub.add_parser("run-all")
    p_all.add_argument("--skip-stack", action="store_true")
    args = ap.parse_args()

    cfg = p1.load_config(Path(args.config))
    if args.cmd == "write-meta":
        run_write_meta(cfg)
    elif args.cmd == "run-bases":
        train_lama_source(cfg, resume=True)
    elif args.cmd == "blend":
        p1.run_blend(cfg, [m.strip() for m in args.models.split(",") if m.strip()])
    elif args.cmd == "stack":
        p1.run_stack(cfg, [m.strip() for m in args.models.split(",") if m.strip()])
    elif args.cmd == "status":
        run_status(cfg)
    elif args.cmd == "run-all":
        run_write_meta(cfg)
        train_lama_source(cfg, resume=True)
        p1.run_blend(cfg, ["lama"])
        if not args.skip_stack:
            p1.run_stack(cfg, ["lama"])


if __name__ == "__main__":
    main()
