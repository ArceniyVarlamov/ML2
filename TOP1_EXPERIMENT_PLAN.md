# Top-1 Experiment Plan

## Wave A: fast diversity baseline (high ROI)

1. Run shift check first:
- `python3 top1_pipeline.py --config configs/top1.json advval`
2. In `configs/top1.json`, set:
   - `"add_svd": false`
   - `"extra_top_k": 1000`
   - `"per_target_top_k": 700`
3. Train base models:
   - `python3 top1_pipeline.py --config configs/top1.json run-bases --models catboost,lightgbm`
4. Build ensembles:
   - `python3 top1_pipeline.py --config configs/top1.json blend --models catboost,lightgbm`
   - `python3 top1_pipeline.py --config configs/top1.json stack --models catboost,lightgbm`
5. Submit both outputs (`blend_test`, `stack_test`).

Stop criterion:
- If best ensemble is below previous LB result, do not move to heavier runs until fold setup/features are validated.

## Wave B: add third model + richer features

1. Enable:
   - `"xgboost": { "enabled": true }`
   - `"add_svd": true`
   - `"svd_components": 64`
   - `"extra_top_k": 1400`
   - `"per_target_top_k": 950`
   - `"stack.cross_target_top_k": 6`
2. Run full base training:
   - `python3 top1_pipeline.py --config configs/top1.json run-bases --models catboost,lightgbm,xgboost`
3. Rebuild blend/stack for 3 models.
4. Submit blend and stack again.

Stop criterion:
- Keep only models that improve OOF macro-AUC and LB stability.

## Wave C: push for #1 with controlled risk

1. Increase feature width:
   - `"extra_top_k": 1800`
   - `"svd_components": 96` (if RAM allows)
2. Optional diversity add-on:
   - enable `mlp` and run `--models catboost,lightgbm,xgboost,mlp`
3. Keep two final candidates for private:
   - aggressive: `stack_test`
   - conservative: `blend_test`
4. Track per-target AUC deltas from `artifacts_top1/scores/*.json`.

Decision rule:
- If improvements are concentrated in a few targets and public score is volatile, prefer conservative blend for one of final private slots.
