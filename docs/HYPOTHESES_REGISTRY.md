# Hypotheses & Strategy Registry (Living Document)

Last updated: `2026-02-23` (manual sync from experiment history and current repo state)

## Purpose

This is the canonical registry of:

1. Hypotheses (`H*`, `FS_*`, `H*_source_expansion`, etc.)
2. Strategy decisions (what is currently included/excluded in the champion)
3. Evidence (OOF + Public LB)
4. Status (planned / infra-ready / tested / frozen)
5. Next actions

Rule: **do not delete old ideas**. Change status + add evidence instead.

## Current State (Ground Truth)

- Current best reported Public LB: **`0.83960751`**
- Current champion submission:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/artifacts_cross_hyp/submissions/sub_H0_H3_H4_H4xgb_H5_H6_top2w.parquet`
- Current top-1 leaderboard score (reported): **`0.8520496542`**
- Gap to top-1: **~`0.01244`**

### Current Champion Core (conceptual)

- `H0_blend`
- `H3_blend`
- `H4_blend`
- `H4_xgb` (top-level source, separate from H4_blend)
- `H5_stack`
- `H6_blend`
- Top-level ensemble mode: `top2_weighted`

## Status Legend

- `PROVEN+`: confirmed useful on Public LB in current/near-current strong core
- `CONDITIONAL+`: useful in some setups or as source, but not in current champion core
- `NEUTRAL`: no clear signal yet
- `NEG_IN_CORE`: tested in strong core and currently degrades score
- `INFRA_READY`: code support implemented, experiment not yet run
- `PLANNED`: idea captured, not implemented yet
- `FROZEN`: deprioritized for now (can be revisited later)

## A. Tested Core Hypotheses (H0–H6)

### H0 — Baseline (CatBoost + LightGBM, fast core)

- Status: `CONDITIONAL+` (foundation source)
- Role: baseline source and reference point
- Config:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/top1_fast_base.json`
- OOF (fast):
  - `catboost`: `0.8223836431534116`
  - `lightgbm`: `0.8139601335201387`
  - `blend`: `0.8278552299284931`
  - `stack`: `0.8203885041064363`
- Public LB:
  - `sub_H0_blend_float64.parquet` -> `0.8328969939`
- Decision:
  - Keep as stable source (`H0_blend`) in many top-level ensembles

### H1 — Quantize Lite (wave/discretized num-features)

- Status: `NEG_IN_CORE` (currently) / `CONDITIONAL+` historically
- Role: feature-set variant exploiting “wave”/discrete numeric patterns
- Config:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/top1_fast_h1_quantize_lite.json`
- OOF (fast):
  - `blend`: `0.8269786886737707`
  - `stack`: `0.8209922536670091`
- Public LB:
  - `sub_H1q_lite_blend_float64.parquet` -> `0.8329185124`
  - `sub_H0_H1_targetwise_top2w.parquet` -> `0.8336840953`
- In current H4-heavy champion core:
  - `sub_H0_H1_H3_H4_H5_H6_top2w.parquet` -> `0.8394881075` (worse than champion)
  - `sub_H0_H1_H3_H4_H5_top2w.parquet` -> `0.8394796517` (worse than champion)
- Decision:
  - Do not include in current champion core by default
  - Keep as reusable source for future combinations

### H2 — TE (OOF global target encoding)

- Status: `NEG_IN_CORE` (currently) / `CONDITIONAL+` historically
- Role: encoding-based feature-set variant
- Config:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/top1_fast_h2_te.json`
- OOF (fast):
  - `blend`: `0.8276754557914803`
  - `stack`: `0.8215034197221238`
- Public LB:
  - `sub_H2_blend_float64.parquet` -> `0.833205326466685`
  - `H2 stack` reported -> `0.8307479606797255` (bad)
- In strong H4-core:
  - `sub_H0_H2_H3_H4_H5_top2w.parquet` -> `0.8394605682108843` (below champion)
- Decision:
  - Exclude from current champion core
  - Keep `H2_blend` as optional source for future ablations only

### H3 — Quantize + TE

- Status: `PROVEN+`
- Role: strong composite feature-set source
- Config:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/top1_fast_h3_quantize_te.json`
- OOF (fast):
  - `blend`: `0.828868080017614`
  - `stack`: `0.8234932806186968`
- Public LB:
  - `sub_H3_blend_float64.parquet` -> `0.8345133097`
- Decision:
  - Keep in champion core as `H3_blend`

### H4 — XGBoost added (GPU) + H4 blend/stack

- Status: `PROVEN+` (major boost)
- Role: diversity source (main boost came here)
- Config:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/top1_fast_h4_xgb_gpu4050.json`
- OOF (fast):
  - `blend`: `0.8338344665520321`
  - `stack`: `0.819015210622143`
- Public LB:
  - `sub_H4_blend_float64_fix.parquet` -> `0.8385766846`
- Important notes:
  - Old `sub_H4_blend_float64.parquet` was mislabeled and actually contained `float32` (rejected by platform)
  - `H4_stack` is not competitive vs `H4_blend`
- Decision:
  - Keep `H4_blend` as core source
  - Expand H4 into top-level base sources (see Source Expansion section)

### H5 — Cross-target stack

- Status: `PROVEN+` (as source)
- Role: meta-source using correlations between targets
- Config:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/top1_fast_h5_crossstack.json`
- OOF (fast):
  - `blend`: `0.8278552299284931` (same baseline-like blend)
  - `stack`: `0.8244874661696919` (this is the useful output)
- Public LB (historically in combinations):
  - `sub_H0_H5_targetwise_top2w.parquet` -> `0.8349128718658754`
  - `sub_H0_H1_H5_targetwise_top2w.parquet` -> `0.8350706554364418`
- Decision:
  - Keep `H5_stack` in champion core

### H6 — Target-aware feature selection (MI), ultrasafe

- Status: `PROVEN+` as micro-source / `NEG` as solo
- Role: weak solo source, but occasionally useful per-target in top-level ensemble
- Config:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/top1_fast_h6_targetaware_ultrasafe.json`
- OOF (fast, ultrasafe):
  - `blend`: `0.8086083771667621`
  - `stack`: `0.8043279866988698`
- Public LB effect (in strong core):
  - `sub_H0_H3_H4_H5_top2w.parquet` -> `0.8394929622991547`
  - `sub_H0_H3_H4_H5_H6_top2w.parquet` -> `0.839501418`
  - delta: tiny positive
- Decision:
  - Keep as optional micro-signal source in champion core
  - Do not use as solo candidate

## B. Cross-Hypothesis / Top-Level Ensemble Methods

### `winner` (per-target best source)

- Status: `CONDITIONAL+` (historically useful, currently not best)
- Files historically:
  - `sub_H0_H1_H5_targetwise_winner.parquet`
  - `sub_H0_H3_H4_H5_H6_winner.parquet`
- Decision:
  - Keep for diagnostics / sanity checks
  - Not default for current champion builds

### `top2_weighted` (per-target top-2 sources, fixed weights)

- Status: `PROVEN+` (current default)
- Role:
  - Robust, simple, currently strongest on Public LB in H4-core
- Decision:
  - Default mode for top-level ensemble

### `optimize_weights` (SLSQP + L2 regularization)

- Status: `NEG_IN_CORE` (current H4-core) / `CONDITIONAL+` infra-wise
- Role:
  - Richer weight search, but currently loses to `top2_weighted` on Public LB
- Evidence:
  - `sub_H0_H3_H4_H5_optw.parquet` -> `0.8376511761985194` (below top2w)
- Decision:
  - Keep implementation
  - Do not prioritize for current core; revisit with richer source library / A100 sources

## C. Source Expansion (Top-Level) — Current Winning Direction

### H4 source decompression: `H4_xgb` as separate top-level source

- Status: `PROVEN+` (major strategic validation)
- Idea:
  - Keep `H4_blend`, but also expose `H4_xgb` separately to top-level ensemble
  - Top-level can choose pure XGB signal where helpful
- Evidence:
  - `sub_H0_H3_H4_H4xgb_H5_H6_top2w.parquet` -> **`0.83960751`** (current best)
  - `sub_H0_H1_H3_H4_H4xgb_H5_H6_top2w.parquet` -> `0.8395941994`
- Important note:
  - OOF did not fully predict this LB gain (Public LB improved more than OOF suggested)
- Decision:
  - This is now a core strategic direction

### H4 source decompression: `H4_cat`, `H4_lgb` (planned next)

- Status: `INFRA_READY`
- Idea:
  - Add `H4_cat` and `H4_lgb` as extra top-level sources, not only `H4_blend`
- Why:
  - Same logic as `H4_xgb` success: uncompress signal for per-target top-level selection
- Current evidence:
  - `H4_xgb` already validated the pattern
- Decision:
  - High priority next source-expansion line

## D. Feature-Set Program (Beyond H0–H6)

These are not “one true feature selection”, but separate source-generating branches.

### FS_core

- Status: `PROVEN+`
- Definition:
  - Current champion source library around `H0/H3/H4/H4_xgb/H5/H6`

### FS_pcat (pseudo-categorical float columns as categories)

- Status: `INFRA_READY`
- Idea:
  - Use discovered pseudo-categorical float columns as categorical features
- Assets:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/artifacts_analysis/pseudo_categorical.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/artifacts_analysis/pseudo_categorical_top200.json`
- Infra:
  - Supported by `top1_pipeline.py` (`pseudo_cat_cols_file`) and feature-set controls
- Decision:
  - High-priority feature-set branch (especially as new source, not necessarily solo)

### FS_drift_pruned (advval-based pruning)

- Status: `INFRA_READY`
- Idea:
  - Use `advval` top drift features to build `drop_columns_file` and test a drift-pruned feature set
- Infra:
  - `advval` already saves `top_drift_features`
  - `features.drop_columns_file` now supported in `top1_pipeline.py`
- Decision:
  - High-priority feature-set branch

### FS_targetaware_stronger (stronger H6, not ultrasafe)

- Status: `PLANNED`
- Idea:
  - Move stronger target-aware MI selection to stronger hardware (or A100)
- Notes:
  - Mac `ultrasafe` works but is intentionally constrained
- Decision:
  - Execute after stronger compute is available

### FS_blocks (`700+700+700` extra blocks)

- Status: `INFRA_READY`
- Idea:
  - Blockwise extra-feature diagnostics and combination
- Assets/configs:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/top1_fast_h6_block1_700_ultrasafe.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/top1_fast_h6_block2_700_ultrasafe.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/top1_fast_h6_block3_700_ultrasafe.json`
- Decision:
  - Diagnostic branch (good for information, not necessarily immediate champion boost)

### Feature audit (OOF Permutation + Null Importance + stability)

- Status: `PLANNED`
- Role:
  - Gold-standard-ish audit for “signal vs noise” (better than variance/MI alone)
- Decision:
  - Medium priority; build once and use repeatedly for curated feature sets

## E. Target Specialists (H8) — Macro AUC Specialists

### H8_rare (rare-target specialists)

- Status: `INFRA_READY` (pipeline support added; configs/training not yet done)
- Goal:
  - Improve rare targets that disproportionately hurt Macro AUC
- Infra now available:
  - target groups file
  - target subset training
  - group-specific parameter overrides
  - fallback submit for partial target outputs
- Decision:
  - High priority next “new training” branch

### H8_hard (low-AUC target specialists)

- Status: `INFRA_READY`
- Goal:
  - Improve hardest targets by current OOF/LB metrics
- Decision:
  - High priority next “new training” branch (parallel to H8_rare)

### H8_common (common targets baseline or separate tuning)

- Status: `PLANNED`
- Decision:
  - Lower priority than `rare/hard`; use if needed for balancing specialists

## F. A100 Branches (Heavy Sources) — Planned

### H9_full_xgb (GPU full/near-full extra features)

- Status: `PLANNED`
- Goal:
  - Full-power XGB source without fast-mode top-k constraints
- Priority:
  - Highest A100 training priority

### H_LAMA (LightAutoML source)

- Status: `PLANNED`
- Goal:
  - Black-box orthogonal source for top-level ensemble
- Important decision:
  - Use only as source, not replacement for current architecture

### H7_mlp (PyTorch MLP source)

- Status: `PLANNED`
- Goal:
  - Orthogonal neural source (embeddings + numeric path)
- Priority:
  - Below `H9_full_xgb` and `H_LAMA`

### Strong H6 on A100/strong CPU

- Status: `PLANNED`
- Goal:
  - Re-run target-aware MI selection with stronger limits than `ultrasafe`

## G. Dynamic HPO Strategy

### Full per-target HPO on all 41 targets

- Status: `FROZEN` (for now)
- Reason:
  - Too expensive vs likely ROI right now
  - Current bigger gains come from source expansion and specialists

### Narrow HPO for specialists (`rare/hard`)

- Status: `PLANNED`
- Goal:
  - Tune only selected targets and only selected models (`xgboost`, `catboost` first)
- Decision:
  - Allowed after H8 baseline specialists are in place

## H. EDA / Analytics Ideas (Only Those Tied to Score)

### Adversarial validation -> drift-pruned feature sets

- Status: `INFRA_READY`
- Decision:
  - Use for `FS_drift_pruned`

### Pseudo-categorical float detection

- Status: `PROVEN_USEFUL_INFRA` (source candidates prepared)
- Assets:
  - `artifacts_analysis/pseudo_categorical*.json`
- Decision:
  - Convert into actual `FS_pcat` experiments

### Target correlation analysis for H5 / specialists

- Status: `READY_DATA`
- Asset:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/artifacts_analysis/target_corr_report.json`
- Decision:
  - Use to improve cross-target neighbor selection and H8 design

### Visual EDA over 2000+ anonymous columns

- Status: `FROZEN`
- Reason:
  - Low ROI vs automated EDA / signal-driven reports

## I. Negative / Deprioritized Findings (Do Not Repeat Blindly)

1. `optimize_weights` on current H4-heavy core underperformed `top2_weighted` on Public LB.
2. `H2_stack` underperformed `H2_blend` significantly.
3. `H1` and `H2` reinserted into current best H4-core did not improve current best LB.
4. `H6` as solo source is weak (keep only as top-level micro-source).
5. `sub_H0_H2_H3_H4_H5_H6_top2w.parquet` is a duplicate of `sub_H0_H2_H3_H4_H5_top2w.parquet` (same content).

## J. Public LB History (Key Anchor Points)

These are reported results used in decision-making. Keep appending here.

### Singles / early combinations

- `sub_H0_blend_float64.parquet` -> `0.8328969939`
- `sub_H1q_lite_blend_float64.parquet` -> `0.8329185124`
- `sub_H2_blend_float64.parquet` -> `0.833205326466685`
- `sub_H3_blend_float64.parquet` -> `0.8345133097`
- `sub_H0_H1_targetwise_top2w.parquet` -> `0.8336840953`
- `sub_H0_H5_targetwise_top2w.parquet` -> `0.8349128718658754`
- `sub_H0_H1_H5_targetwise_top2w.parquet` -> `0.8350706554364418`

### H4-era (major jump)

- `sub_H0_H3_H4_H5_top2w.parquet` -> `0.8394929622991547`
- `sub_H0_H3_H4_H5_H6_top2w.parquet` -> `0.839501418`
- `sub_H0_H2_H3_H4_H5_top2w.parquet` -> `0.8394605682108843`
- `sub_H0_H3_H4_H5_optw.parquet` -> `0.8376511761985194`
- `sub_H4_blend_float64_fix.parquet` -> `0.8385766846`

### H4 source expansion (top-level `H4_xgb` separate)

- `sub_H0_H3_H4_H4xgb_H5_H6_top2w.parquet` -> **`0.83960751`** (current best)
- `sub_H0_H1_H3_H4_H4xgb_H5_H6_top2w.parquet` -> `0.8395941994`

## K. Current Priority Ranking (Living, Strategy-Level)

### Tier 1 — Highest ROI right now

1. **Top-level source expansion around H4**
   - `H4_cat`, `H4_lgb`, combinations with `H4_xgb`
2. **H8 specialists (`rare`, `hard`)**
3. **FS_pcat** and **FS_drift_pruned** as new sources

### Tier 2 — Strong next wave (especially once A100 arrives)

1. `H9_full_xgb`
2. `H_LAMA`
3. Stronger `H6` (non-ultrasafe)
4. `H7_mlp`

### Tier 3 — Narrow tuning / refinement

1. Specialist-only HPO (`rare/hard`)
2. `optimize_weights` revisit (only after source library grows significantly)

## L. Maintenance Protocol (How We Keep This File Alive)

After every meaningful experiment batch:

1. Update **Current State** if champion changed.
2. Append Public LB results to **Section J**.
3. Update status in **Sections A–H** (`PROVEN+`, `NEG_IN_CORE`, etc.).
4. Add a short decision line (keep / exclude / revisit).
5. If a new idea appears, add it to the correct section with `PLANNED`.

Recommended rule:

- If an idea is tested 2–3 times without signal, move it to `FROZEN`.
- If an idea is weak solo but useful in ensemble, explicitly mark it `source-only`.

## M. Pointers to Supporting Infrastructure

- Plan/infrastructure notes:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/docs/TOP1_PLAN_INFRA.md`
- Main training pipeline:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/top1_pipeline.py`
- Top-level cross-hypothesis ensemble:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/cross_hypothesis_ensemble.py`
- Target group generator for H8:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/build_target_groups.py`

