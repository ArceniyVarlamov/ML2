# Top-1 Plan Infrastructure (Implemented)

This document summarizes the new code-level capabilities added to support the long-term Top-1 strategy.

## 1) `cross_hypothesis_ensemble.py` additions

### Source manifest
You can now pass sources via JSON/YAML manifest:

```json
{
  "sources": [
    {
      "label": "H0_blend",
      "oof_path": "artifacts_fast_base/ensemble/blend_oof.parquet",
      "test_path": "artifacts_fast_base/ensemble/blend_test.parquet"
    },
    {
      "label": "H4_xgb",
      "oof_path": "artifacts_fast_h4_xgb_gpu4050/base/xgboost_oof.parquet",
      "test_path": "artifacts_fast_h4_xgb_gpu4050/base/xgboost_test.parquet"
    }
  ]
}
```

Usage:

```bash
python cross_hypothesis_ensemble.py \
  --train-target Data/Main/train_target.parquet \
  --sample-submit Data/Main/sample_submit.parquet \
  --source-manifest path/to/sources.json \
  --mode top2_weighted \
  --out-parquet out.parquet \
  --out-report out.json
```

### Partial sources (specialists)
- `--allow-partial-sources`
- Allows sources that only contain a subset of targets (useful for `H8_rare`, `H8_hard`).
- At least one source must cover every target.

### Duplicate source protection
- `--check-duplicates`
- Detects duplicate source content by SHA256 of `(oof parquet, test parquet)` and fails early.

### Candidate source filtering per target
- `--min-delta-auc`: keep only sources within AUC gap from best source per target
- `--max-sources-per-target`: cap candidate pool size per target

### Richer reports
`out-report` now includes:
- `targets[target].candidate_pool`
- `source_usage` summary (counts, weight sums, avg weight when selected)
- final submission SHA256 (`out_parquet_sha256`)

### Optional experiment log
- `--experiment-log path.jsonl`

## 2) `top1_pipeline.py` additions

### Target subsets / target groups (foundation for H8 specialists)
Config supports:

```json
{
  "targets": {
    "include": ["target_2_8", "target_6_5"],
    "exclude": ["target_3_1"],
    "target_groups_file": "configs/targets/groups_h8.json",
    "groups": ["rare", "hard"]
  }
}
```

Notes:
- `include`/`exclude` can be lists or comma-separated strings.
- `groups` unions targets from the group file.
- Filtering is applied consistently in `run-bases`, `blend`, `stack`.

### Group-specific model overrides (H8)
Inside each model config:

```json
{
  "models": {
    "xgboost": {
      "enabled": true,
      "params": { "n_estimators": 2000, "max_depth": 8 },
      "model_overrides_by_target_group": {
        "rare": { "max_depth": 10, "learning_rate": 0.03 },
        "hard": { "min_child_weight": 5 }
      },
      "params_by_target": {
        "target_2_8": { "max_depth": 4 }
      }
    }
  }
}
```

Order of application:
1. `params`
2. `params_by_target[target]`
3. `model_overrides_by_target_group[group]` (for all matched groups)

### Feature set controls (FS_drift_pruned / curated sets)
Config supports:

```json
{
  "features": {
    "keep_extra_file": "artifacts_analysis/keep_extra_cols.json",
    "drop_columns_file": "artifacts_analysis/drop_cols.json"
  }
}
```

Behavior:
- `keep_extra_file`: restricts extra feature candidate pool before ranking/selection
- `drop_columns_file`: removes columns from final feature set (and raw extra columns are also filtered early)

### `submit` fallback source (specialist outputs)
For partial prediction files:

```bash
python top1_pipeline.py --config cfg.json submit \
  --source artifacts_h8_rare/ensemble/blend_test.parquet \
  --fallback-source artifacts_cross_hyp/submissions/current_champion.parquet \
  --output artifacts_h8_rare/submissions/sub_h8_rare_with_fallback.parquet
```

### `write-meta` command (meta standardization)
Writes `meta/columns.json`, `meta/fold_ids.npy`, and `meta/config_snapshot.json` without training:

```bash
python top1_pipeline.py --config cfg.json write-meta
```

Useful when imported artifacts are missing metadata and you need standardized meta for the same `output_dir`.

### Experiment journal (JSONL)
If `tracking.experiment_log` or `tracking.experiment_log_path` is set in config, pipeline events append JSONL records.

If not set, defaults to:

```text
<output_dir>/scores/experiment_log.jsonl
```

Logged events include: `base_model`, `blend`, `stack`, `advval`, `submit`, `write_meta`.

## 3) Target group helper utility

New utility:

```bash
python build_target_groups.py \
  --train-target Data/Main/train_target.parquet \
  --scores-json artifacts_fast_h3_quantize_te/scores/blend_scores.json \
  --rare-top-k 10 \
  --hard-top-k 10 \
  --out configs/targets/groups_h8.json
```

Outputs `rare`, `hard`, and `common` groups for specialist configs.
