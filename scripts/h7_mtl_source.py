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
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import top1_pipeline as p1


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        return torch, nn, DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed on the runtime. Install torch before running h7_mtl_source.py") from exc


def _auc_safe(y: np.ndarray, pred: np.ndarray) -> float:
    pos = int(y.sum())
    if pos == 0 or pos == len(y):
        return float("nan")
    return float(roc_auc_score(y, pred))


def _macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for j in range(y_true.shape[1]):
        s = _auc_safe(y_true[:, j], y_pred[:, j])
        if not math.isnan(s):
            scores.append(s)
    return float(np.mean(scores)) if scores else float("nan")


def _load_state(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "scores" / "h7_mtl_state.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"schema_version": 1, "created_at": now_ts(), "updated_at": now_ts(), "folds": {}, "global": {"folds_done": 0, "folds_failed": 0}}


def _save_state(out_dir: Path, state: Dict[str, Any]) -> None:
    p1.ensure_dir(out_dir / "scores")
    state["updated_at"] = now_ts()
    (out_dir / "scores" / "h7_mtl_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_or_init_base(out_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame):
    p1.ensure_dir(out_dir / "base")
    oof_path = out_dir / "base" / "mtl_oof.parquet"
    test_path = out_dir / "base" / "mtl_test.parquet"
    if oof_path.exists() and test_path.exists():
        return pd.read_parquet(oof_path), pd.read_parquet(test_path)
    return pd.DataFrame({"customer_id": train_df["customer_id"].values}), pd.DataFrame({"customer_id": test_df["customer_id"].values})


def _load_or_init_scores(out_dir: Path) -> Dict[str, Any]:
    p1.ensure_dir(out_dir / "scores")
    sp = out_dir / "scores" / "mtl_scores.json"
    if sp.exists():
        payload = json.loads(sp.read_text(encoding="utf-8"))
        payload.setdefault("target_scores", {})
        payload.setdefault("fold_scores", {})
        return payload
    return {"model": "mtl", "macro_auc": float("nan"), "target_scores": {}, "fold_scores": {}}


def _save_outputs(out_dir: Path, oof_df: pd.DataFrame, test_df: pd.DataFrame, score_payload: Dict[str, Any]) -> None:
    p1.ensure_dir(out_dir / "base")
    p1.ensure_dir(out_dir / "scores")
    oof_df.to_parquet(out_dir / "base" / "mtl_oof.parquet", index=False)
    test_df.to_parquet(out_dir / "base" / "mtl_test.parquet", index=False)
    with (out_dir / "scores" / "mtl_scores.json").open("w", encoding="utf-8") as f:
        json.dump(score_payload, f, ensure_ascii=False, indent=2)


def _build_model(nn, in_dim: int, out_dim: int, hidden: List[int], dropout: float):
    layers = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def _make_loaders(torch, DataLoader, TensorDataset, x_tr, y_tr, x_val, y_val, batch_size: int, num_workers: int):
    tr_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    va_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    va_ld = DataLoader(va_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=False)
    return tr_ld, va_ld


def _train_fold(torch, nn, DataLoader, TensorDataset, cfg: Dict[str, Any], x_tr, y_tr, x_val, y_val, x_test):
    mcfg = cfg.get("h7_mtl", {})
    device_name = str(mcfg.get("device", "cuda_if_available"))
    if device_name == "cuda_if_available":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    hidden = [int(x) for x in mcfg.get("hidden_dims", [512, 256, 128])]
    dropout = float(mcfg.get("dropout", 0.3))
    lr = float(mcfg.get("lr", 1e-3))
    weight_decay = float(mcfg.get("weight_decay", 1e-5))
    epochs = int(mcfg.get("epochs", 20))
    batch_size = int(mcfg.get("batch_size", 512))
    num_workers = int(mcfg.get("num_workers", 0))
    patience = int(mcfg.get("patience", 4))
    grad_clip = float(mcfg.get("grad_clip", 0.0))
    use_pos_weight = bool(mcfg.get("use_pos_weight", True))

    model = _build_model(nn, x_tr.shape[1], y_tr.shape[1], hidden, dropout).to(device)
    pos = y_tr.sum(axis=0)
    neg = y_tr.shape[0] - pos
    pos_weight_np = np.clip(neg / np.maximum(pos, 1.0), 1.0, float(mcfg.get("pos_weight_clip", 50.0))).astype(np.float32)
    pos_weight = torch.from_numpy(pos_weight_np).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight if use_pos_weight else None)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    tr_ld, va_ld = _make_loaders(torch, DataLoader, TensorDataset, x_tr, y_tr, x_val, y_val, batch_size, num_workers)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in tr_ld:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in va_ld:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(float(loss.item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(x_val).to(device)).detach().cpu().numpy()
        test_logits = model(torch.from_numpy(x_test).to(device)).detach().cpu().numpy()
    val_pred = (1.0 / (1.0 + np.exp(-val_logits))).astype(np.float32)
    test_pred = (1.0 / (1.0 + np.exp(-test_logits))).astype(np.float32)
    return val_pred, test_pred, {"best_val_loss": best_val_loss, "best_epoch": best_epoch}


def train_h7_mtl_source(cfg: Dict[str, Any], resume: bool = True) -> None:
    torch, nn, DataLoader, TensorDataset = _import_torch()
    prepared = p1.prepare_dataset(cfg)
    train_df = prepared["train_df"]
    test_df = prepared["test_df"]
    feature_cols = list(prepared["feature_cols"])
    target_cols = list(prepared["target_cols"])
    fold_ids = prepared["fold_ids"]
    out_dir = Path(cfg["output_dir"])

    state = _load_state(out_dir) if resume else {"schema_version": 1, "created_at": now_ts(), "updated_at": now_ts(), "folds": {}, "global": {"folds_done": 0, "folds_failed": 0}}
    oof_df, test_pred_df = _load_or_init_base(out_dir, train_df, test_df) if resume else (
        pd.DataFrame({"customer_id": train_df["customer_id"].values}),
        pd.DataFrame({"customer_id": test_df["customer_id"].values}),
    )
    score_payload = _load_or_init_scores(out_dir) if resume else {"model": "mtl", "macro_auc": float("nan"), "target_scores": {}, "fold_scores": {}}

    mcfg = cfg.get("h7_mtl", {})
    seed = int(mcfg.get("seed", cfg.get("seed", 42)))
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train_raw = train_df[feature_cols]
    x_test_raw = test_df[feature_cols]
    y_all = train_df[target_cols].to_numpy(dtype=np.float32, copy=False)

    # Resume artifacts arrays if possible.
    if all(t in oof_df.columns for t in target_cols):
        oof_mat = oof_df[target_cols].to_numpy(dtype=np.float32, copy=False)
    else:
        oof_mat = np.zeros((len(train_df), len(target_cols)), dtype=np.float32)
    if all(p1.target_to_predict_col(t) in test_pred_df.columns for t in target_cols):
        test_accum = test_pred_df[[p1.target_to_predict_col(t) for t in target_cols]].to_numpy(dtype=np.float32, copy=False)
    else:
        test_accum = np.zeros((len(test_df), len(target_cols)), dtype=np.float32)
    test_folds_done = np.zeros(int(np.max(fold_ids)) + 1, dtype=np.int8)
    for fk, rec in state.get("folds", {}).items():
        if rec.get("status") == "done":
            try:
                test_folds_done[int(fk)] = 1
            except Exception:
                pass

    n_folds = int(np.max(fold_ids)) + 1
    print(f"[h7_mtl] targets={len(target_cols)} folds={n_folds} features={len(feature_cols)}")
    for fold in range(n_folds):
        if state.get("folds", {}).get(str(fold), {}).get("status") == "done":
            print(f"[h7_mtl] fold={fold} skip (already done)")
            continue
        state.setdefault("folds", {})[str(fold)] = {"status": "running"}
        _save_state(out_dir, state)
        try:
            idx_val = np.where(fold_ids == fold)[0]
            idx_tr = np.where(fold_ids != fold)[0]
            x_tr_df = x_train_raw.iloc[idx_tr]
            x_val_df = x_train_raw.iloc[idx_val]
            x_te_df = x_test_raw
            y_tr = y_all[idx_tr]
            y_val = y_all[idx_val]

            imputer = SimpleImputer(strategy=str(mcfg.get("imputer", "median")))
            scaler = StandardScaler(with_mean=bool(mcfg.get("scaler_with_mean", False)))
            x_tr = imputer.fit_transform(x_tr_df)
            x_val = imputer.transform(x_val_df)
            x_te = imputer.transform(x_te_df)
            x_tr = scaler.fit_transform(x_tr).astype(np.float32, copy=False)
            x_val = scaler.transform(x_val).astype(np.float32, copy=False)
            x_te = scaler.transform(x_te).astype(np.float32, copy=False)

            val_pred, test_pred_fold, fold_info = _train_fold(torch, nn, DataLoader, TensorDataset, cfg, x_tr, y_tr.astype(np.float32), x_val, y_val.astype(np.float32), x_te)
            oof_mat[idx_val, :] = val_pred
            test_accum += (test_pred_fold / float(n_folds)).astype(np.float32)
            state["folds"][str(fold)] = {"status": "done", **fold_info}
            state["global"]["folds_done"] = int(sum(1 for v in state["folds"].values() if v.get("status") == "done"))
            _save_state(out_dir, state)

            for j, t in enumerate(target_cols):
                oof_df[t] = oof_mat[:, j]
                test_pred_df[p1.target_to_predict_col(t)] = test_accum[:, j]
            _save_outputs(out_dir, oof_df, test_pred_df, score_payload)
            print(f"[h7_mtl] fold={fold} done val_macro_auc={_macro_auc(y_val, val_pred):.6f}")
        except Exception as exc:
            state["folds"][str(fold)] = {"status": "failed", "error": str(exc)}
            state["global"]["folds_failed"] = int(sum(1 for v in state["folds"].values() if v.get("status") == "failed"))
            _save_state(out_dir, state)
            raise

    for j, t in enumerate(target_cols):
        score_payload["target_scores"][t] = _auc_safe(y_all[:, j].astype(np.int8), oof_mat[:, j])
    score_payload["macro_auc"] = _macro_auc(y_all, oof_mat)
    score_payload["fold_scores"] = {k: v for k, v in state.get("folds", {}).items()}
    _save_outputs(out_dir, oof_df, test_pred_df, score_payload)
    p1.append_experiment_log(
        cfg,
        event="h7_mtl_run_bases",
        payload={
            "targets": len(target_cols),
            "features": len(feature_cols),
            "macro_auc": score_payload.get("macro_auc"),
            "oof_path": str(out_dir / "base" / "mtl_oof.parquet"),
            "test_path": str(out_dir / "base" / "mtl_test.parquet"),
        },
    )
    print(f"[h7_mtl] macro_auc={float(score_payload['macro_auc']):.6f}")


def run_write_meta(cfg: Dict[str, Any]) -> None:
    prepared = p1.prepare_dataset(cfg)
    out_dir = Path(cfg["output_dir"])
    p1.ensure_dir(out_dir / "meta")
    np.save(out_dir / "meta" / "fold_ids.npy", prepared["fold_ids"])
    (out_dir / "meta" / "columns.json").write_text(
        json.dumps({"feature_cols": list(prepared["feature_cols"]), "target_cols": list(prepared["target_cols"]), "cat_cols": list(prepared["cat_cols"])}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[write-meta] targets={len(prepared['target_cols'])} features={len(prepared['feature_cols'])}")


def run_status(cfg: Dict[str, Any]) -> None:
    out_dir = Path(cfg["output_dir"])
    state = _load_state(out_dir)
    scores = _load_or_init_scores(out_dir)
    print(json.dumps({
        "output_dir": str(out_dir),
        "folds_done": int(sum(1 for v in state.get('folds', {}).values() if v.get('status') == 'done')),
        "folds_failed": int(sum(1 for v in state.get('folds', {}).values() if v.get('status') == 'failed')),
        "macro_auc": scores.get("macro_auc"),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="H7 MTL source runner (PyTorch, multilabel OOF/test export)")
    ap.add_argument("--config", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("write-meta")
    sub.add_parser("run-bases")
    p_bl = sub.add_parser("blend")
    p_bl.add_argument("--models", default="mtl")
    p_st = sub.add_parser("stack")
    p_st.add_argument("--models", default="mtl")
    sub.add_parser("status")
    p_all = sub.add_parser("run-all")
    p_all.add_argument("--skip-stack", action="store_true")
    args = ap.parse_args()
    cfg = p1.load_config(Path(args.config))
    if args.cmd == "write-meta":
        run_write_meta(cfg)
    elif args.cmd == "run-bases":
        train_h7_mtl_source(cfg, resume=True)
    elif args.cmd == "blend":
        p1.run_blend(cfg, [m.strip() for m in args.models.split(",") if m.strip()])
    elif args.cmd == "stack":
        p1.run_stack(cfg, [m.strip() for m in args.models.split(",") if m.strip()])
    elif args.cmd == "status":
        run_status(cfg)
    elif args.cmd == "run-all":
        run_write_meta(cfg)
        train_h7_mtl_source(cfg, resume=True)
        p1.run_blend(cfg, ["mtl"])
        if not args.skip_stack:
            p1.run_stack(cfg, ["mtl"])


if __name__ == "__main__":
    main()
