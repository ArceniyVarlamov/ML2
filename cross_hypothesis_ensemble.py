#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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


def load_json_or_yaml(path: Path) -> Any:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ImportError(
                "YAML manifest requires PyYAML. Install via: pip install pyyaml"
            ) from exc
        return yaml.safe_load(text)
    return json.loads(text)


def parse_sources_manifest(path: Path) -> List[Dict[str, str]]:
    payload = load_json_or_yaml(path)
    if isinstance(payload, dict):
        raw_sources = payload.get("sources", payload.get("items"))
        if raw_sources is None:
            raise ValueError(f"Manifest {path} must contain 'sources' list")
    elif isinstance(payload, list):
        raw_sources = payload
    else:
        raise ValueError(f"Manifest {path} must be a list or dict with 'sources'")

    out: List[Dict[str, str]] = []
    for idx, item in enumerate(raw_sources):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest source #{idx} must be an object")
        label = item.get("label")
        oof_path = item.get("oof_path")
        test_path = item.get("test_path")
        if not all([label, oof_path, test_path]):
            raise ValueError(
                f"Manifest source #{idx} must contain label/oof_path/test_path"
            )
        out.append(
            {
                "label": str(label),
                "oof_path": str(oof_path),
                "test_path": str(test_path),
            }
        )
    return out


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def source_signature(src: Dict[str, str]) -> Dict[str, str]:
    oof_p = Path(src["oof_path"])
    test_p = Path(src["test_path"])
    return {
        "oof_sha256": sha256_file(oof_p),
        "test_sha256": sha256_file(test_p),
    }


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
        default=[],
        help="Source in format label:oof_path:test_path. Repeat --source multiple times.",
    )
    p.add_argument(
        "--source-manifest",
        default=None,
        help="Path to JSON/YAML manifest with sources (list or {sources:[...]}).",
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
    p.add_argument(
        "--max-sources-per-target",
        type=int,
        default=0,
        help="Cap candidate sources per target after sorting/filtering (0=off).",
    )
    p.add_argument(
        "--min-delta-auc",
        type=float,
        default=0.0,
        help=(
            "Keep only sources within this OOF AUC gap from the best source per target "
            "(0=off). The best source is always kept."
        ),
    )
    p.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Check duplicate sources by SHA256 of OOF+test parquet and fail if found.",
    )
    p.add_argument(
        "--allow-partial-sources",
        action="store_true",
        help="Allow sources that contain only a subset of targets (useful for specialist models).",
    )
    p.add_argument(
        "--experiment-log",
        default=None,
        help="Optional JSONL experiment log path to append ensemble run summary.",
    )
    p.add_argument("--out-parquet", required=True, help="Output submission parquet path")
    p.add_argument("--out-report", required=True, help="Output JSON report path")
    args = p.parse_args()

    train_target = pd.read_parquet(args.train_target)
    target_cols = [c for c in train_target.columns if c.startswith("target_")]
    sample = pd.read_parquet(args.sample_submit)

    sources: List[Dict[str, str]] = []
    if args.source_manifest:
        sources.extend(parse_sources_manifest(Path(args.source_manifest)))
    if args.source:
        sources.extend(parse_sources(args.source))
    if not sources:
        raise ValueError("Provide at least one source via --source or --source-manifest")
    labels = [s["label"] for s in sources]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate source labels are not allowed")
    if args.check_duplicates:
        seen_signatures: Dict[tuple[str, str], str] = {}
        for src in sources:
            sig = source_signature(src)
            src["signature"] = sig
            key = (sig["oof_sha256"], sig["test_sha256"])
            if key in seen_signatures:
                raise ValueError(
                    f"Duplicate source content detected: {src['label']} matches {seen_signatures[key]}"
                )
            seen_signatures[key] = src["label"]
    oof_dfs = [pd.read_parquet(s["oof_path"]) for s in sources]
    test_dfs = [pd.read_parquet(s["test_path"]) for s in sources]

    ensure_same_ids(oof_dfs, "customer_id", "OOF")
    ensure_same_ids(test_dfs, "customer_id", "TEST")

    submission = pd.DataFrame({"customer_id": test_dfs[0]["customer_id"].astype(sample["customer_id"].dtype)})
    report = {
        "mode": args.mode,
        "sources": sources,
        "selection_params": {
            "max_sources_per_target": int(args.max_sources_per_target),
            "min_delta_auc": float(args.min_delta_auc),
            "check_duplicates": bool(args.check_duplicates),
            "allow_partial_sources": bool(args.allow_partial_sources),
        },
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
        "source_usage": {},
        "macro_auc": None,
    }
    aucs: List[float] = []
    usage_summary: Dict[str, Dict[str, Any]] = {
        s["label"]: {
            "targets_selected_count": 0,
            "weight_sum": 0.0,
            "max_weight_seen": 0.0,
            "candidate_pool_count": 0,
        }
        for s in sources
    }

    for target in target_cols:
        y = train_target[target].to_numpy(dtype=np.int8, copy=False)
        positives = int(y.sum())
        is_constant_target = positives == 0 or positives == len(y)
        per_source = []
        for src, oof_df, test_df in zip(sources, oof_dfs, test_dfs):
            if target not in oof_df.columns:
                if args.allow_partial_sources:
                    continue
                raise KeyError(f"Missing target column '{target}' in {src['oof_path']}")
            pcol = target_to_predict_col(target)
            if pcol in test_df.columns:
                test_arr = test_df[pcol].to_numpy(dtype=np.float64, copy=False)
            elif target in test_df.columns:
                # Allow base predictions saved with target_* names in test files.
                test_arr = test_df[target].to_numpy(dtype=np.float64, copy=False)
            else:
                if args.allow_partial_sources:
                    continue
                raise KeyError(f"Missing predict column '{pcol}' (or '{target}') in {src['test_path']}")

            oof_pred = oof_df[target].to_numpy(dtype=np.float64, copy=False)
            auc = float(roc_auc_score(y, oof_pred)) if not is_constant_target else float("nan")
            per_source.append(
                {
                    "label": src["label"],
                    "auc": auc,
                    "oof_pred": oof_pred,
                    "test_pred": test_arr,
                }
            )
        if not per_source:
            raise ValueError(
                f"No available sources for target={target}. "
                "If using specialists, ensure at least one fallback source covers every target."
            )

        per_source.sort(key=lambda x: x["auc"], reverse=True)
        candidate_pool = list(per_source)
        if (not is_constant_target) and float(args.min_delta_auc) > 0.0 and candidate_pool:
            best_auc = float(candidate_pool[0]["auc"])
            filtered = [candidate_pool[0]]
            for src_item in candidate_pool[1:]:
                if (best_auc - float(src_item["auc"])) <= float(args.min_delta_auc):
                    filtered.append(src_item)
            # Preserve minimum width for top2 mode if available.
            if args.mode == "top2_weighted" and len(filtered) < min(2, len(candidate_pool)):
                filtered = list(candidate_pool[: min(2, len(candidate_pool))])
            candidate_pool = filtered

        if int(args.max_sources_per_target) > 0 and len(candidate_pool) > int(args.max_sources_per_target):
            max_n = int(args.max_sources_per_target)
            if args.mode == "top2_weighted":
                max_n = max(max_n, min(2, len(candidate_pool)))
            candidate_pool = candidate_pool[:max_n]

        for src_item in candidate_pool:
            usage_summary[src_item["label"]]["candidate_pool_count"] += 1

        if args.mode == "winner":
            best = candidate_pool[0]
            oof_final = best["oof_pred"]
            test_final = best["test_pred"]
            chosen = {"strategy": "winner", "sources": [{"label": best["label"], "weight": 1.0}]}
        elif args.mode == "top2_weighted":
            top = candidate_pool[:2]
            if len(top) == 1:
                oof_final = top[0]["oof_pred"]
                test_final = top[0]["test_pred"]
                chosen = {
                    "strategy": "top2_weighted",
                    "sources": [{"label": top[0]["label"], "weight": 1.0}],
                }
            else:
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
                n_sources = len(candidate_pool)
                w = np.full(n_sources, 1.0 / max(n_sources, 1), dtype=np.float64)
                oof_final = np.column_stack([x["oof_pred"] for x in candidate_pool]) @ w
                test_final = np.column_stack([x["test_pred"] for x in candidate_pool]) @ w
                chosen = {
                    "strategy": "optimize_weights",
                    "sources": [
                        {"label": src["label"], "weight": float(weight)}
                        for src, weight in zip(candidate_pool, w)
                    ],
                    "optimizer": {"success": True, "message": "constant target"},
                }
            else:
                oof_matrix = np.column_stack([x["oof_pred"] for x in candidate_pool])
                test_matrix = np.column_stack([x["test_pred"] for x in candidate_pool])
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
                        for src, weight in zip(candidate_pool, w_opt)
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
        for src_meta in chosen.get("sources", []):
            label = str(src_meta["label"])
            w = float(src_meta["weight"])
            usage_summary[label]["targets_selected_count"] += 1
            usage_summary[label]["weight_sum"] += w
            usage_summary[label]["max_weight_seen"] = max(
                float(usage_summary[label]["max_weight_seen"]), w
            )
        report["targets"][target] = {
            "final_auc": auc_final,
            "per_source_auc": {x["label"]: x["auc"] for x in per_source},
            "candidate_pool": [
                {
                    "label": x["label"],
                    "auc": x["auc"],
                }
                for x in candidate_pool
            ],
            "chosen": chosen,
        }

    report["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    report["source_usage"] = {
        label: {
            **stats,
            "avg_weight_when_selected": (
                float(stats["weight_sum"]) / max(int(stats["targets_selected_count"]), 1)
            ),
        }
        for label, stats in usage_summary.items()
    }

    submission = submission[sample.columns.tolist()]
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(out_path, index=False)
    report["out_parquet"] = str(out_path)
    report["out_parquet_sha256"] = sha256_file(out_path)
    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved ensemble submission: {out_path}")
    print(f"Saved ensemble report: {report_path}")
    print(f"OOF macro AUC: {report['macro_auc']:.6f}")
    if args.experiment_log:
        exp_path = Path(args.experiment_log)
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        with exp_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": "cross_hyp_ensemble",
                        "mode": args.mode,
                        "macro_auc": report["macro_auc"],
                        "out_parquet": str(out_path),
                        "out_parquet_sha256": report["out_parquet_sha256"],
                        "report_path": str(report_path),
                        "sources": [s["label"] for s in sources],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
