# A100 Readiness Runbook (Quality-First, Resumable)

This file is for the moment A100 becomes available. The goal is to run a prepared, high-quality campaign without improvisation, while still being able to use short windows efficiently.

## Principles

1. Use A100 for **new heavy sources**, not for small no-train tweaks.
2. Keep Mac as the orchestrator:
   - assemble top-level ensembles
   - produce submissions
   - track results
3. Save `OOF` + `test` artifacts early and often (sessions can die).

## Preflight (run first on A100 machine / Colab terminal)

From repo root:

```bash
bash scripts/a100_preflight.sh
```

If `.venv` is not available in the A100 environment, create one and install deps first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-top1.txt
```

Then rerun:

```bash
bash scripts/a100_preflight.sh
```

## Priority Queue (A100)

### Prepared campaign (resumable queue)

If you have a short/unstable A100 window, use the campaign runner:

```bash
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_max_v1.json
```

Useful flags:

```bash
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_max_v1.json --dry-run
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_max_v1.json --from-task h9_run_bases_xgb
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_max_v1.json --only h9_run_bases_xgb,h9_blend
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_max_v1.json --force
```

Campaign state/logs:

- `artifacts_campaigns/a100_max_v1/state.json`
- `artifacts_campaigns/a100_max_v1/*.out.log`
- `artifacts_campaigns/a100_max_v1/*.err.log`

### HPO campaign (quality-first specialists/full41, resumable)

This is the prepared XGBoost HPO framework. It supports `rare/hard` pilot scope and `all41` using the same runner and artifact format.

Quick status check (safe, no training):

```bash
./.venv/bin/python scripts/hpo_xgb_targets.py --config configs/hpo/xgb_targets_rarehard_v1.json status
./.venv/bin/python scripts/hpo_xgb_targets.py --config configs/hpo/xgb_targets_all41_v1.json status
```

Run the HPO campaign queue:

```bash
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_hpo_xgb_targets_v1.json
```

Useful staged runs:

```bash
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_hpo_xgb_targets_v1.json --only hpo_rarehard_search
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_hpo_xgb_targets_v1.json --only hpo_rarehard_train_source
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_hpo_xgb_targets_v1.json --from-task hpo_all41_search
```

Key outputs (rarehard pilot):

- `artifacts_hpo_xgb_targets_rarehard_v1/state.json`
- `artifacts_hpo_xgb_targets_rarehard_v1/best_params_by_target.json`
- `configs/generated/top1_h8_hpo_xgb_rarehard_v1_tuned.json`
- `artifacts_h8_hpo_xgb_rarehard_v1/base/xgboost_oof.parquet`
- `artifacts_h8_hpo_xgb_rarehard_v1/ensemble/blend_oof.parquet`

### CatBoost/LightGBM target-HPO campaign (quality-first, resumable)

This extends the same target-HPO idea beyond XGBoost using a generic boosting HPO runner.

```bash
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_hpo_catlgb_targets_v1.json
```

Useful staged runs:

```bash
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_hpo_catlgb_targets_v1.json --only hpo_lgb_rarehard_search
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_hpo_catlgb_targets_v1.json --only hpo_cat_rarehard_search
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_hpo_catlgb_targets_v1.json --from-task hpo_lgb_all41_search
```

### Feature-audit campaign (anti-noise layer, resumable)

Run before aggressive feature pruning decisions and before building `FS_signal_only` / `FS_drift_pruned` from heuristics alone:

```bash
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_feature_audit_xgb_v1.json
```

Outputs:

- `artifacts_feature_audit_xgb_rarehard_v1/summary.json`
- `artifacts_feature_audit_xgb_rarehard_v1/keep_extra.json`
- `artifacts_feature_audit_xgb_rarehard_v1/drop_columns.json`
- `artifacts_feature_audit_xgb_all41_v1/summary.json`
- `artifacts_feature_audit_xgb_all41_v1/keep_extra.json`
- `artifacts_feature_audit_xgb_all41_v1/drop_columns.json`

### H8 XGBoost bagging campaign (specialists v2, partial sources)

This is the first bagging-based specialist branch (`rare` + `hard`) using undersampling per bag.

```bash
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_h8_xgb_bagging_v1.json
```

Key outputs:

- `artifacts_h8_rare_xgb_bagging_a100_v1/ensemble/blend_oof.parquet`
- `artifacts_h8_rare_xgb_bagging_a100_v1/ensemble/blend_test.parquet`
- `artifacts_h8_hard_xgb_bagging_a100_v1/ensemble/blend_oof.parquet`
- `artifacts_h8_hard_xgb_bagging_a100_v1/ensemble/blend_test.parquet`

### Linear source campaign (orthogonal GLM-like source)

```bash
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_linear_sources_v1.json
```

Key outputs:

- `artifacts_hlinear_sgd_a100_v1/base/linear_oof.parquet`
- `artifacts_hlinear_sgd_a100_v1/ensemble/blend_oof.parquet`

### LightAutoML source campaign (black-box orthogonal source)

Requires `lightautoml` installed in the A100 runtime.

```bash
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_lama_sources_v1.json
```

Key outputs:

- `artifacts_hlama_a100_v1/base/lama_oof.parquet`
- `artifacts_hlama_a100_v1/ensemble/blend_oof.parquet`

### H5 Smart v2 source campaign (safe OOF cross-target interactions)

```bash
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_h5_smart_v2.json
```

Key outputs:

- `artifacts_h5_smart_v2_source/base/h5smart_oof.parquet`
- `artifacts_h5_smart_v2_source/ensemble/blend_oof.parquet`

### H7 MTL source campaign (PyTorch multilabel source)

Requires `torch` installed in the A100 runtime.

```bash
./.venv/bin/python scripts/a100_campaign.py --campaign configs/campaigns/a100_h7_mtl_source_v1.json
```

Key outputs:

- `artifacts_h7_mtl_source_v1/base/mtl_oof.parquet`
- `artifacts_h7_mtl_source_v1/ensemble/blend_oof.parquet`

### Priority 1 — `H9_full_xgb` (full/near-full extra features)

Config:

- `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/top1_a100_h9_full_xgb_v1.json`

Run:

```bash
./.venv/bin/python top1_pipeline.py --config configs/top1_a100_h9_full_xgb_v1.json run-bases --models xgboost
./.venv/bin/python top1_pipeline.py --config configs/top1_a100_h9_full_xgb_v1.json blend --models xgboost
./.venv/bin/python top1_pipeline.py --config configs/top1_a100_h9_full_xgb_v1.json stack --models xgboost
```

Artifacts to copy back:

- `artifacts_a100_h9_full_xgb_v1/base/xgboost_oof.parquet`
- `artifacts_a100_h9_full_xgb_v1/base/xgboost_test.parquet`
- `artifacts_a100_h9_full_xgb_v1/ensemble/blend_oof.parquet`
- `artifacts_a100_h9_full_xgb_v1/ensemble/blend_test.parquet`
- `artifacts_a100_h9_full_xgb_v1/scores/xgboost_scores.json`
- `artifacts_a100_h9_full_xgb_v1/scores/blend_scores.json`
- `artifacts_a100_h9_full_xgb_v1/meta/*`

### Priority 2 — `H6_targetaware_strong` (GPU-enabled stronger H6)

Config:

- `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/top1_a100_h6_targetaware_strong_gpu_v1.json`

Run:

```bash
./.venv/bin/python top1_pipeline.py --config configs/top1_a100_h6_targetaware_strong_gpu_v1.json run-bases --models catboost,lightgbm,xgboost
./.venv/bin/python top1_pipeline.py --config configs/top1_a100_h6_targetaware_strong_gpu_v1.json blend --models catboost,lightgbm,xgboost
./.venv/bin/python top1_pipeline.py --config configs/top1_a100_h6_targetaware_strong_gpu_v1.json stack --models catboost,lightgbm,xgboost
```

Artifacts to copy back:

- `artifacts_a100_h6_targetaware_strong_gpu_v1/base/*`
- `artifacts_a100_h6_targetaware_strong_gpu_v1/ensemble/*`
- `artifacts_a100_h6_targetaware_strong_gpu_v1/scores/*`
- `artifacts_a100_h6_targetaware_strong_gpu_v1/meta/*`

## Immediate Integration On Mac (after copying artifacts back)

Use the current champion source manifest:

- `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/manifests/champion_current_best_sources.json`

Example: add `H9_xgb` as a new top-level source.

```bash
./.venv/bin/python cross_hypothesis_ensemble.py \
  --train-target Data/Main/train_target.parquet \
  --sample-submit Data/Main/sample_submit.parquet \
  --source-manifest configs/manifests/champion_current_best_sources.json \
  --source H9_xgb:artifacts_a100_h9_full_xgb_v1/base/xgboost_oof.parquet:artifacts_a100_h9_full_xgb_v1/base/xgboost_test.parquet \
  --allow-partial-sources \
  --mode top2_weighted \
  --check-duplicates \
  --max-sources-per-target 8 \
  --min-delta-auc 0.01 \
  --out-parquet artifacts_cross_hyp/submissions/sub_CHAMP_plus_H9xgb_top2w.parquet \
  --out-report artifacts_cross_hyp/reports/CHAMP_plus_H9xgb_top2w.json \
  --experiment-log artifacts_cross_hyp/reports/experiment_log.jsonl
```

Example: add `H6_strong_blend`:

```bash
./.venv/bin/python cross_hypothesis_ensemble.py \
  --train-target Data/Main/train_target.parquet \
  --sample-submit Data/Main/sample_submit.parquet \
  --source-manifest configs/manifests/champion_current_best_sources.json \
  --source H6strong_blend:artifacts_a100_h6_targetaware_strong_gpu_v1/ensemble/blend_oof.parquet:artifacts_a100_h6_targetaware_strong_gpu_v1/ensemble/blend_test.parquet \
  --allow-partial-sources \
  --mode top2_weighted \
  --check-duplicates \
  --max-sources-per-target 8 \
  --min-delta-auc 0.01 \
  --out-parquet artifacts_cross_hyp/submissions/sub_CHAMP_plus_H6strong_blend_top2w.parquet \
  --out-report artifacts_cross_hyp/reports/CHAMP_plus_H6strong_blend_top2w.json \
  --experiment-log artifacts_cross_hyp/reports/experiment_log.jsonl
```

Example: add tuned HPO XGBoost specialists source (partial `rare/hard`):

```bash
./.venv/bin/python cross_hypothesis_ensemble.py \
  --train-target Data/Main/train_target.parquet \
  --sample-submit Data/Main/sample_submit.parquet \
  --source-manifest configs/manifests/champion_current_best_sources.json \
  --source H8hpo_xgb_rarehard_blend:artifacts_h8_hpo_xgb_rarehard_v1/ensemble/blend_oof.parquet:artifacts_h8_hpo_xgb_rarehard_v1/ensemble/blend_test.parquet \
  --allow-partial-sources \
  --mode top2_weighted \
  --check-duplicates \
  --max-sources-per-target 10 \
  --min-delta-auc 0.01 \
  --out-parquet artifacts_cross_hyp/submissions/sub_CHAMP_plus_H8hpoXgbRareHard_top2w.parquet \
  --out-report artifacts_cross_hyp/reports/CHAMP_plus_H8hpoXgbRareHard_top2w.json
```

## If A100 window is short (triage mode)

1. Run only `H9_full_xgb run-bases --models xgboost`
2. Copy `base/xgboost_oof.parquet` and `base/xgboost_test.parquet` first
3. Blend/stack can be done later or skipped (xgboost source is enough for first integration)

## Top-level ensemble modes after A100 sources arrive

Default remains `top2_weighted` (best Public LB so far), but now the ensemble supports:

- `top3_weighted` with fixed weights (default normalized to `0.45/0.35/0.20`)

Example:

```bash
./.venv/bin/python cross_hypothesis_ensemble.py \
  --train-target Data/Main/train_target.parquet \
  --sample-submit Data/Main/sample_submit.parquet \
  --source-manifest configs/manifests/champion_current_best_sources.json \
  --source H9_xgb:artifacts_a100_h9_full_xgb_v1/base/xgboost_oof.parquet:artifacts_a100_h9_full_xgb_v1/base/xgboost_test.parquet \
  --allow-partial-sources \
  --mode top3_weighted \
  --top3-weights 0.45,0.35,0.20 \
  --check-duplicates \
  --max-sources-per-target 8 \
  --min-delta-auc 0.01 \
  --out-parquet artifacts_cross_hyp/submissions/sub_CHAMP_plus_H9xgb_top3w.parquet \
  --out-report artifacts_cross_hyp/reports/CHAMP_plus_H9xgb_top3w.json
```

## What not to do when A100 opens

1. Do not spend the first hour tuning `optimize_weights`.
2. Do not rerun already-closed no-train source expansions (`H4_cat/H4_lgb`, `H3_cat/H3_lgb`) in current `top2w` core.
3. Do not start with pseudo-labeling.
4. Do not run manual ad-hoc HPO loops if `scripts/hpo_xgb_targets.py` + campaign spec already cover the same scope.

## Notes on H2O/AutoML/GLM

H2O GLM and some AutoML paths are valid ideas, but they are often CPU/RAM-heavy more than GPU-heavy.
If the A100 environment has strong CPU/RAM and stable session time, they are good candidates after `H9` / `H6_strong`.
