# Top-1 Pipeline (Cyber Shelf)

This pipeline is isolated from `Outsorce/*` and is focused on leaderboard growth via:
- unified multilabel folds;
- stronger features (`extra_top_k` + row stats + optional SVD);
- per-target feature selection (MI-based, cached);
- diverse base models (`catboost`, `lightgbm`, `xgboost`, optional `mlp`);
- per-target OOF blend and L2 stack with optional cross-target meta-features;
- adversarial validation for train/test shift diagnostics.

## 1. Install dependencies

Recommended: Python `3.11` (or `3.10/3.12`). Some ML wheels can be unavailable for `3.14`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

```bash
python -m pip install -r requirements-top1.txt
```

## 2. Check train/test shift (recommended first)

```bash
python3 top1_pipeline.py --config configs/top1.json advval
```

Result file:
- `artifacts_top1/scores/advval.json`

## 3. Train base models (OOF + test preds)

```bash
python3 top1_pipeline.py --config configs/top1.json run-bases
```

Optional subset:

```bash
python3 top1_pipeline.py --config configs/top1.json run-bases --models catboost,lightgbm
```

Optional with NN diversity:

```bash
python3 top1_pipeline.py --config configs/top1.json run-bases --models catboost,lightgbm,xgboost,mlp
```

Artifacts:
- `artifacts_top1/base/<model>_oof.parquet`
- `artifacts_top1/base/<model>_test.parquet`
- `artifacts_top1/scores/<model>_scores.json`

## 4. Per-target blend (weights from OOF)

```bash
python3 top1_pipeline.py --config configs/top1.json blend --models catboost,lightgbm,xgboost
```

Artifacts:
- `artifacts_top1/ensemble/blend_oof.parquet`
- `artifacts_top1/ensemble/blend_test.parquet`
- `artifacts_top1/scores/blend_scores.json`

## 5. L2 stacking (LogReg on OOF)

```bash
python3 top1_pipeline.py --config configs/top1.json stack --models catboost,lightgbm,xgboost
```

Artifacts:
- `artifacts_top1/ensemble/stack_oof.parquet`
- `artifacts_top1/ensemble/stack_test.parquet`
- `artifacts_top1/scores/stack_scores.json`

## 6. Build final submission parquet

From blend:

```bash
python3 top1_pipeline.py --config configs/top1.json submit \
  --source artifacts_top1/ensemble/blend_test.parquet \
  --output artifacts_top1/submissions/sub_blend.parquet
```

From stack:

```bash
python3 top1_pipeline.py --config configs/top1.json submit \
  --source artifacts_top1/ensemble/stack_test.parquet \
  --output artifacts_top1/submissions/sub_stack.parquet
```

## Suggested run order (high ROI)

1. Run `advval` and check if `mean_auc > 0.55` in `advval.json`.
2. Start with `catboost + lightgbm` and no SVD (`add_svd=false`) for a fast robust baseline.
3. Add `xgboost`, then enable SVD and widen `extra_top_k` gradually (1000 -> 1400 -> 1800).
4. Use `cross_target_top_k` in stack (default enabled) and compare against pure blend.
5. Enable `mlp` only after stable GBDT baseline is confirmed.
6. Keep 2 final submissions for private LB:
- aggressive (`stack_test`);
- conservative (`blend_test`).

## Fast hypothesis mode

Use ready short-run configs:
- `configs/top1_fast_base.json`
- `configs/top1_fast_h1_quantize.json`
- `configs/top1_fast_h2_te.json`
- `configs/top1_fast_h3_quantize_te.json`

Protocol and promotion rules:
- `FAST_HYPOTHESIS_PROTOCOL.md`
