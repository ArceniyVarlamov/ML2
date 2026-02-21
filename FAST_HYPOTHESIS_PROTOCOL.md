# Fast Hypothesis Protocol

Goal: evaluate hypotheses quickly, then promote only strong ones to long runs.

## Fixed sequence

1. `H0` baseline:
   - `configs/top1_fast_base.json`
2. `H1` quantized numeric features:
   - `configs/top1_fast_h1_quantize.json`
3. `H2` OOF global target encoding:
   - `configs/top1_fast_h2_te.json`
4. `H3` quantization + OOF TE:
   - `configs/top1_fast_h3_quantize_te.json`
5. `H6` target-aware per-target feature map:
   - `configs/top1_fast_h6_targetaware.json`

All runs keep the same short regime:
- 3 folds
- reduced iterations
- `catboost + lightgbm`
- no SVD / no per-target FS / no cross-target stack

## Commands per hypothesis

```bash
./.venv/bin/python top1_pipeline.py --config <CONFIG> run-bases --models catboost,lightgbm
./.venv/bin/python top1_pipeline.py --config <CONFIG> blend --models catboost,lightgbm
./.venv/bin/python top1_pipeline.py --config <CONFIG> stack --models catboost,lightgbm
```

## Decision rules

1. Primary score: `blend` macro AUC from `artifacts_*/scores/blend_scores.json`.
2. Promotion threshold to long run:
   - strong promote: `+0.0010` vs `H0`
   - conditional promote: `+0.0005` to `+0.0010` with per-target gains in at least 12 targets
3. Rejection:
   - `< +0.0005` and no stable target-level gains

Quick summary command:

```bash
./.venv/bin/python summarize_fast_results.py
```

## Long-run reveal safeguard

Some hypotheses improve only with longer training. For each rejected/conditional hypothesis:

1. Run a medium check:
   - 5 folds
   - same features as hypothesis
   - moderate iterations (`catboost~1500`, `lightgbm~2500`)
2. If medium check still does not beat baseline, drop permanently.

This avoids wasting 20+ hour full runs on weak ideas while not missing late-blooming ones.
