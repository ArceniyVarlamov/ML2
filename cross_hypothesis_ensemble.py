#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def target_to_predict_col(target_col: str) -> str:
    return target_col.replace("target_", "predict_", 1)


def parse_sources(items: List[str]) -> List[Dict[str, str]]:
    """
    Source format:
    label:oof_path:test_path
    """
    out: List[Dict[str, str]] = []
    for item in items:
        parts = item.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid source '{item}'. Expected format label:oof_path:test_path"
            )
        label, oof_path, test_path = parts
        out.append({"label": label, "oof_path": oof_path, "test_path": test_path})
    return out


def ensure_same_ids(dfs: List[pd.DataFrame], id_col: str, name: str) -> None:
    base = dfs[0][id_col].to_numpy()
    for idx, df in enumerate(dfs[1:], start=1):
        if not np.array_equal(base, df[id_col].to_numpy()):
            raise ValueError(f"{name} id mismatch at source index {idx}")


def sample_rows_binary(y: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or len(y) <= max_rows:
        return np.arange(len(y))

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return rng.choice(len(y), size=max_rows, replace=False)

    pos_share = len(pos_idx) / len(y)
    pos_take = int(round(max_rows * pos_share))
    pos_take = max(1, min(pos_take, len(pos_idx)))
    neg_take = max_rows - pos_take
    neg_take = max(1, min(neg_take, len(neg_idx)))

    picked_pos = rng.choice(pos_idx, size=pos_take, replace=False)
    picked_neg = rng.choice(neg_idx, size=neg_take, replace=False)
    out = np.concatenate([picked_pos, picked_neg])
    rng.shuffle(out)
    return out


def project_weights_bounded(
    w: np.ndarray,
    min_w: float,
    max_w: float,
    sum_target: float = 1.0,
    max_iter: int = 100,
) -> np.ndarray:
    out = np.clip(w.astype(np.float64, copy=True), min_w, max_w)
    for _ in range(max_iter):
        diff = float(sum_target - out.sum())
        if abs(diff) < 1e-12:
            return out
        if diff > 0:
            room = max_w - out
            total_room = float(room.sum())
            if total_room <= 1e-12:
                return out
            out += diff * (room / total_room)
        else:
            room = out - min_w
            total_room = float(room.sum())
            if total_room <= 1e-12:
                return out
            out += diff * (room / total_room)
        out = np.clip(out, min_w, max_w)
    return out


def optimize_target_weights(
    oof_matrix: np.ndarray,
    y: np.ndarray,
    alpha: float,
    min_w: float,
    max_w: float,
    max_iter: int,
    finite_diff_eps: float,
) -> tuple[np.ndarray, Dict[str, Any]]:
    try:
        from scipy.optimize import minimize
    except Exception as exc:
        raise ImportError(
            "scipy is required for --mode optimize_weights. Install via: pip install scipy"
        ) from exc

    n_models = int(oof_matrix.shape[1])
    if n_models == 1:
        return np.array([1.0], dtype=np.float64), {
            "success": True,
            "message": "single source",
            "auc": float(roc_auc_score(y, oof_matrix[:, 0])),
            "reg": 0.0,
            "loss": float(-roc_auc_score(y, oof_matrix[:, 0])),
        }

    if n_models * min_w > 1.0 + 1e-12:
        raise ValueError(
            f"Infeasible bounds: n_models * min_weight = {n_models * min_w:.6f} > 1.0"
        )
    if n_models * max_w < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible bounds: n_models * max_weight = {n_models * max_w:.6f} < 1.0"
        )

    uniform = np.full(n_models, 1.0 / n_models, dtype=np.float64)
    x0 = project_weights_bounded(uniform, min_w=min_w, max_w=max_w)

    def objective(w: np.ndarray) -> float:
        pred = oof_matrix @ w
        auc = float(roc_auc_score(y, pred))
        reg = alpha * float(np.sum((w - uniform) ** 2))
        return -auc + reg

    bounds = [(min_w, max_w)] * n_models
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": max_iter,
            "ftol": 1e-7,
            "eps": float(finite_diff_eps),
            "disp": False,
        },
    )

    if result.success:
        raw_w = result.x
    else:
        raw_w = x0
    w = project_weights_bounded(raw_w, min_w=min_w, max_w=max_w)
    w = w / max(float(w.sum()), 1e-12)
    w = project_weights_bounded(w, min_w=min_w, max_w=max_w)

    pred = oof_matrix @ w
    auc = float(roc_auc_score(y, pred))
    reg = alpha * float(np.sum((w - uniform) ** 2))
    loss = -auc + reg
    meta = {
        "success": bool(result.success),
        "message": str(result.message),
        "n_iter": int(getattr(result, "nit", -1)),
        "auc": auc,
        "reg": reg,
        "loss": loss,
    }
    return w, meta


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cross-hypothesis ensemble by per-target OOF performance"
    )
    p.add_argument("--train-target", required=True, help="Path to train_target.parquet")
    p.add_argument("--sample-submit", required=True, help="Path to sample_submit.parquet")
    p.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source in format label:oof_path:test_path. Repeat --source multiple times.",
    )
    p.add_argument(
        "--mode",
        choices=["winner", "top2_weighted", "optimize_weights"],
        default="winner",
        help=(
            "winner: choose best source per target; "
            "top2_weighted: weighted average of top-2 by OOF AUC; "
            "optimize_weights: SLSQP with L2 regularization"
        ),
    )
    p.add_argument(
        "--opt-alpha",
        type=float,
        default=0.1,
        help="L2 regularization strength towards uniform weights for optimize_weights",
    )
    p.add_argument(
        "--opt-min-weight",
        type=float,
        default=0.01,
        help="Lower bound for each source weight in optimize_weights",
    )
    p.add_argument(
        "--opt-max-weight",
        type=float,
        default=0.99,
        help="Upper bound for each source weight in optimize_weights",
    )
    p.add_argument(
        "--opt-maxiter",
        type=int,
        default=300,
        help="Maximum SLSQP iterations for optimize_weights",
    )
    p.add_argument(
        "--opt-eps",
        type=float,
        default=0.02,
        help="Finite-difference epsilon for SLSQP gradient approximation",
    )
    p.add_argument(
        "--opt-max-rows",
        type=int,
        default=150000,
        help="Max train rows for weight optimization objective (0=use all rows)",
    )
    p.add_argument(
        "--opt-seed",
        type=int,
        default=42,
        help="Random seed for optimize_weights row sampling",
    )
    p.add_argument("--out-parquet", required=True, help="Output submission parquet path")
    p.add_argument("--out-report", required=True, help="Output JSON report path")
    args = p.parse_args()

    train_target = pd.read_parquet(args.train_target)
    target_cols = [c for c in train_target.columns if c.startswith("target_")]
    sample = pd.read_parquet(args.sample_submit)

    sources = parse_sources(args.source)
    oof_dfs = [pd.read_parquet(s["oof_path"]) for s in sources]
    test_dfs = [pd.read_parquet(s["test_path"]) for s in sources]

    ensure_same_ids(oof_dfs, "customer_id", "OOF")
    ensure_same_ids(test_dfs, "customer_id", "TEST")

    submission = pd.DataFrame({"customer_id": test_dfs[0]["customer_id"].astype(sample["customer_id"].dtype)})
    report = {
        "mode": args.mode,
        "sources": sources,
        "opt_params": {
            "alpha": float(args.opt_alpha),
            "min_weight": float(args.opt_min_weight),
            "max_weight": float(args.opt_max_weight),
            "maxiter": int(args.opt_maxiter),
            "eps": float(args.opt_eps),
            "max_rows": int(args.opt_max_rows),
            "seed": int(args.opt_seed),
        },
        "targets": {},
        "macro_auc": None,
    }
    aucs: List[float] = []

    for target in target_cols:
        y = train_target[target].to_numpy(dtype=np.int8, copy=False)
        positives = int(y.sum())
        is_constant_target = positives == 0 or positives == len(y)
        per_source = []
        for src, oof_df, test_df in zip(sources, oof_dfs, test_dfs):
            if target not in oof_df.columns:
                raise KeyError(f"Missing target column '{target}' in {src['oof_path']}")
            pcol = target_to_predict_col(target)
            if pcol not in test_df.columns:
                raise KeyError(f"Missing predict column '{pcol}' in {src['test_path']}")

            oof_pred = oof_df[target].to_numpy(dtype=np.float64, copy=False)
            auc = float(roc_auc_score(y, oof_pred)) if not is_constant_target else float("nan")
            per_source.append(
                {
                    "label": src["label"],
                    "auc": auc,
                    "oof_pred": oof_pred,
                    "test_pred": test_df[pcol].to_numpy(dtype=np.float64, copy=False),
                }
            )

        per_source.sort(key=lambda x: x["auc"], reverse=True)

        if args.mode == "winner":
            best = per_source[0]
            oof_final = best["oof_pred"]
            test_final = best["test_pred"]
            chosen = {"strategy": "winner", "sources": [{"label": best["label"], "weight": 1.0}]}
        elif args.mode == "top2_weighted":
            top = per_source[:2]
            w1, w2 = 0.65, 0.35
            oof_final = w1 * top[0]["oof_pred"] + w2 * top[1]["oof_pred"]
            test_final = w1 * top[0]["test_pred"] + w2 * top[1]["test_pred"]
            chosen = {
                "strategy": "top2_weighted",
                "sources": [
                    {"label": top[0]["label"], "weight": w1},
                    {"label": top[1]["label"], "weight": w2},
                ],
            }
        else:
            if is_constant_target:
                n_sources = len(per_source)
                w = np.full(n_sources, 1.0 / max(n_sources, 1), dtype=np.float64)
                oof_final = np.column_stack([x["oof_pred"] for x in per_source]) @ w
                test_final = np.column_stack([x["test_pred"] for x in per_source]) @ w
                chosen = {
                    "strategy": "optimize_weights",
                    "sources": [
                        {"label": src["label"], "weight": float(weight)}
                        for src, weight in zip(per_source, w)
                    ],
                    "optimizer": {"success": True, "message": "constant target"},
                }
            else:
                oof_matrix = np.column_stack([x["oof_pred"] for x in per_source])
                test_matrix = np.column_stack([x["test_pred"] for x in per_source])
                fit_idx = sample_rows_binary(
                    y=y,
                    max_rows=int(args.opt_max_rows),
                    seed=int(args.opt_seed),
                )
                y_fit = y[fit_idx]
                oof_fit = oof_matrix[fit_idx]
                w_opt, meta = optimize_target_weights(
                    oof_matrix=oof_fit,
                    y=y_fit,
                    alpha=float(args.opt_alpha),
                    min_w=float(args.opt_min_weight),
                    max_w=float(args.opt_max_weight),
                    max_iter=int(args.opt_maxiter),
                    finite_diff_eps=float(args.opt_eps),
                )
                oof_final = oof_matrix @ w_opt
                test_final = test_matrix @ w_opt
                chosen = {
                    "strategy": "optimize_weights",
                    "sources": [
                        {"label": src["label"], "weight": float(weight)}
                        for src, weight in zip(per_source, w_opt)
                    ],
                    "optimizer": {
                        **meta,
                        "fit_rows": int(len(fit_idx)),
                    },
                }

        if is_constant_target:
            auc_final = float("nan")
        else:
            auc_final = float(roc_auc_score(y, oof_final))
            aucs.append(auc_final)

        pcol = target_to_predict_col(target)
        submission[pcol] = pd.Series(test_final).astype(sample[pcol].dtype)
        report["targets"][target] = {
            "final_auc": auc_final,
            "per_source_auc": {x["label"]: x["auc"] for x in per_source},
            "chosen": chosen,
        }

    report["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    submission = submission[sample.columns.tolist()]
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(out_path, index=False)

    print(f"Saved ensemble submission: {out_path}")
    print(f"Saved ensemble report: {report_path}")
    print(f"OOF macro AUC: {report['macro_auc']:.6f}")


if __name__ == "__main__":
    main()
