# Hypotheses & Strategy Registry (Living Document)

Last updated: `2026-02-24` (includes H8 specialist LB gains, A100 readiness package, and XGBoost HPO framework infra)

## Purpose

This is the canonical registry of:

1. Hypotheses (`H*`, `FS_*`, `H*_source_expansion`, etc.)
2. Strategy decisions (what is currently included/excluded in the champion)
3. Evidence (OOF + Public LB)
4. Status (planned / infra-ready / tested / frozen)
5. Next actions

Rule: **do not delete old ideas**. Change status + add evidence instead.

## Current State (Ground Truth)

- Current best reported Public LB: **`0.840359537417668`**
- Current champion submission:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/artifacts_cross_hyp/submissions/sub_CHAMP_plus_H8rare_blend_H8hard_blend_top2w.parquet`
- Current top-1 leaderboard score (reported): **`0.8520496542`**
- Gap to top-1: **~`0.01244`**

### Current Champion Core (conceptual)

- `H0_blend`
- `H3_blend`
- `H4_blend`
- `H4_xgb` (top-level source, separate from H4_blend)
- `H5_stack`
- `H6_blend`
- `H8_rare_blend` (partial source; rare targets only)
- `H8_hard_blend` (partial source; hard targets only)
- Top-level ensemble mode: `top2_weighted`

## Status Legend

- `PROVEN+`: confirmed useful on Public LB in current/near-current strong core
- `CONDITIONAL+`: useful in some setups or as source, but not in current champion core
- `NEUTRAL`: no clear signal yet
- `NEG_IN_CORE`: tested in strong core and currently degrades score
- `INFRA_READY`: code support implemented, experiment not yet run
- `RUNNING`: experiment currently running / awaiting results
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

### `top3_weighted` (per-target top-3 sources, fixed weights)

- Status: `INFRA_READY`
- Role:
  - Broader fixed-weight routing without SLSQP overfit risk
- Implementation:
  - `cross_hypothesis_ensemble.py --mode top3_weighted`
  - configurable `--top3-weights` (default normalized `0.45/0.35/0.20`)
- Decision:
  - High-priority no-train mode to test as source library grows (especially after H8 and A100 sources)

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

- Status: `FROZEN` (for current `top2w` H4-heavy core)
- Idea:
  - Add `H4_cat` and `H4_lgb` as extra top-level sources, not only `H4_blend`
- Why:
  - Same logic as `H4_xgb` success: uncompress signal for per-target top-level selection
- Current evidence:
  - `H4_xgb` validated the pattern
  - `H4_cat`, `H4_lgb`, `H4_cat+H4_lgb` produced exact duplicate outputs vs existing non-H4xgb champion core
  - `H4_cat+H4_lgb+H4_xgb` produced exact duplicate of current champion (`H4_xgb` already captures the useful extra signal)
- Decision:
  - Closed for now in current `top2w` core
  - Revisit only if top-level ensemble mode changes (e.g., `top3_weighted`, correlation-aware selection) or source library changes substantially

### H3 source decompression: `H3_cat`, `H3_lgb`

- Status: `FROZEN` (for current `top2w` H4-heavy core)
- Idea:
  - Add H3 base-model sources separately (`H3_cat`, `H3_lgb`) on top of `H3_blend`
- Evidence:
  - `H3_cat`, `H3_lgb`, `H3_cat+H3_lgb` produced exact duplicates of the current champion in tested `top2w` H4-heavy core
- Decision:
  - Closed for now in current top-level mode/core
  - Revisit only if top-level selection policy changes

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

- Status: `INFRA_READY`
- Role:
  - Gold-standard-ish audit for “signal vs noise” (better than variance/MI alone)
- Current infra:
  - `scripts/feature_audit_oof.py` (resumable per-target OOF permutation + null importance)
  - `configs/audit/feature_audit_xgb_rarehard_v1.json`
  - `configs/audit/feature_audit_xgb_all41_v1.json`
  - `configs/campaigns/a100_feature_audit_xgb_v1.json`
- Decision:
  - High priority A100 preparation layer (anti-noise); run before aggressive feature pruning decisions

## E. Target Specialists (H8) — Macro AUC Specialists

### H8_rare (rare-target specialists)

- Status: `PROVEN+` (as partial source)
- Goal:
  - Improve rare targets that disproportionately hurt Macro AUC
- Infra now available:
  - target groups file
  - target subset training
  - group-specific parameter overrides
  - fallback submit for partial target outputs
- Decision:
  - Keep as partial source (`H8_rare_blend`) for champion builds
  - `H8_rare_stack` currently redundant/duplicate in tested top-level setup

### H8_hard (low-AUC target specialists)

- Status: `PROVEN+` (as partial source)
- Goal:
  - Improve hardest targets by current OOF/LB metrics
- Decision:
  - Keep as partial source (`H8_hard_blend`) for champion builds
  - `H8_hard_stack` lower priority than blend

### H8_common (common targets baseline or separate tuning)

- Status: `PLANNED`
- Decision:
  - Lower priority than `rare/hard`; use if needed for balancing specialists

## F. A100 Branches (Heavy Sources) — Planned

### H9_full_xgb (GPU full/near-full extra features)

- Status: `INFRA_READY`
- Goal:
  - Full-power XGB source without fast-mode top-k constraints
- Priority:
  - Highest A100 training priority

### H4_DART / H_DART (LGBM/XGB DART diversity source)

- Status: `INFRA_READY`
- Goal:
  - Add lower-correlation DART predictions as a new source family
- Config:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/top1_a100_h4_dart_xgb_lgb_v1.json`
- Decision:
  - High-priority A100 diversity branch (cheap compared to MTL/AutoML)

### H_LAMA (LightAutoML source)

- Status: `INFRA_READY` (runner/campaign) / `PLANNED` (execution)
- Goal:
  - Black-box orthogonal source for top-level ensemble
- Important decision:
  - Use only as source, not replacement for current architecture
- Note:
  - Mostly CPU/RAM-bound in practice; A100 availability helps Colab runtime access/stability, but GPU itself is not the key accelerator here
- Current infra:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/scripts/lama_source.py`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/lama/top1_a100_hlama_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/campaigns/a100_lama_sources_v1.json`

### H7_mlp (PyTorch MLP source)

- Status: `INFRA_READY` (runner/campaign) / `PLANNED` (execution)
- Goal:
  - Orthogonal neural source (embeddings + numeric path)
- Priority:
  - Below `H9_full_xgb` and `H_LAMA`
- Current infra:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/scripts/h7_mtl_source.py`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/h7/top1_a100_h7_mtl_source_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/campaigns/a100_h7_mtl_source_v1.json`

### Strong H6 on A100/strong CPU

- Status: `INFRA_READY`
- Goal:
  - Re-run target-aware MI selection with stronger limits than `ultrasafe`

## G. Dynamic HPO Strategy

### Full per-target HPO on all 41 targets

- Status: `INFRA_READY` (framework) / `PLANNED` (execution)
- Goal:
  - Use the same XGBoost HPO framework as specialists, but with `scope=all41`
- Current infra:
  - `scripts/hpo_xgb_targets.py`
  - `configs/hpo/xgb_targets_all41_v1.json`
  - `configs/campaigns/a100_hpo_xgb_targets_v1.json`
  - `scripts/hpo_boosting_targets.py`
  - `configs/hpo/cat_targets_all41_v1.json`
  - `configs/hpo/lgb_targets_all41_v1.json`
  - `configs/campaigns/a100_hpo_catlgb_targets_v1.json`
- Decision:
  - Execute after/alongside `H9_full_xgb` when A100 window is stable enough for long campaigns

### Narrow HPO for specialists (`rare/hard`)

- Status: `INFRA_READY`
- Goal:
  - Tune only selected targets and only selected models (`xgboost`, `catboost` first)
- Current infra:
  - `scripts/hpo_xgb_targets.py` (resumable search + export tuned config + source training)
  - `configs/hpo/xgb_targets_rarehard_v1.json`
  - `configs/campaigns/a100_hpo_xgb_targets_v1.json`
  - `scripts/hpo_boosting_targets.py` (generic Cat/LGB/XGB HPO)
  - `configs/hpo/cat_targets_rarehard_v1.json`
  - `configs/hpo/lgb_targets_rarehard_v1.json`
  - `configs/campaigns/a100_hpo_catlgb_targets_v1.json`
- Decision:
  - First HPO execution target once A100 window is stable (after or alongside `H9_full_xgb`)

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
6. `H8_rare_stack` was a duplicate of the pre-H8 champion in the tested top-level setup.

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

### H8 specialists integrated as partial top-level sources

- `sub_CHAMP_plus_H8hard_blend_top2w.parquet` -> `0.839838349`
- `sub_CHAMP_plus_H8rare_blend_top2w.parquet` -> `0.8401286984`
- `sub_CHAMP_plus_H8rare_blend_H8hard_blend_top2w.parquet` -> **`0.840359537417668`** (new current best)

Notes:

- `H8_rare` and `H8_hard` provide additive gains in the current top-level setup (disjoint target groups, partial-source routing works as intended)

## K. Current Priority Ranking (Living, Strategy-Level)

### Tier 1 — Highest ROI right now

1. **H8 specialists v2** (`rare/hard` upgrades: HPO-lite, bagging on rare, source decompression)
2. **`top3_weighted` top-level ensemble mode (fixed weights, no SLSQP)**
3. **H_DART sources (`H4_dart_*`, later `H8_dart_*` if useful)**
4. **FS_pcat** as new source
5. **FS_drift_pruned** as source (only if advval signals drift; currently lower than FS_pcat)

### Tier 2 — Strong next wave (especially once A100 arrives)

1. `H9_full_xgb`
2. `H8` specialists + narrow HPO (`rare/hard`, `xgboost` first)
3. `H_LAMA` / linear-source family (GLM baseline first, H2O GLM optional heavier implementation)
4. Stronger `H6` (non-ultrasafe)
5. `H7_mlp` / `H7_MTL`

### Tier 3 — Narrow tuning / refinement

1. `H5_Smart` (cross-target engineered interactions on OOF preds)
2. `optimize_weights` revisit (only after source library grows significantly)
3. `Pseudo-labeling` (final-stage only)

## K2. Why This Priority Ranking Looks Like This (Decision Rule)

To avoid confusion from “too many ideas”, every idea is ranked by four factors:

1. **Expected upside** (can it plausibly move LB by `+0.000x` to `+0.00x`?)
2. **Time-to-signal** (how fast we can get a trustworthy OOF/LB answer)
3. **Integration friction** (how hard to produce clean OOF/test sources for top-level ensemble)
4. **Risk of false progress** (OOF-overfit, duplicates, unstable training, operational overhead)

In practice, we prioritize ideas that produce **new orthogonal sources quickly** and plug directly into the current winning architecture.

Examples:

- `H4_xgb` source expansion ranked high and paid off immediately (`PROVEN+`)
- `H4_cat/H4_lgb` and `H3_cat/H3_lgb` were cheap to test and are now closed (duplicates)
- `optimize_weights` is implemented but de-prioritized because Public LB already showed underperformance vs `top2_weighted`
- `H8` specialists are high priority because Macro AUC heavily weights weak targets

### Clarification on “Why are we cautious about H2O / heavy ideas?”

The caution is **not** “H2O is bad” and **not** “it cannot finish in a day on Colab”.

What the caution actually means:

1. **A100 GPU does not directly accelerate H2O GLM much**
   - GLM/ElasticNet workloads are primarily CPU/RAM-bound, not GPU-bound
2. **Runtime cost is not only training**
   - OOF extraction, 41-target orchestration, format normalization, and integration into top-level ensemble also cost time
3. **We compare ideas by ROI in the current architecture**
   - A no-train change like `top3_weighted` or a DART source may give signal faster than wiring up H2O

Decision implication:

- `H2O GLM` (or any linear-source family) is **not rejected**
- It is a **valid planned source**, but usually after:
  - `H8 baseline specialists`
  - `H_DART`
  - `top3_weighted`
  - `H9_full_xgb`

If strong compute/RAM and engineering bandwidth are available, H2O/GLM can absolutely be promoted earlier.

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
- A100 campaign runner:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/scripts/a100_campaign.py`
- XGBoost HPO framework (resumable; rare/hard + all41 scopes):
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/scripts/hpo_xgb_targets.py`
- XGBoost HPO configs:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/hpo/xgb_targets_rarehard_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/hpo/xgb_targets_all41_v1.json`
- XGBoost HPO campaign spec:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/campaigns/a100_hpo_xgb_targets_v1.json`
- Generic boosting HPO framework (`catboost` / `lightgbm` / `xgboost`):
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/scripts/hpo_boosting_targets.py`
- Cat/LGB HPO configs/campaign:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/hpo/cat_targets_rarehard_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/hpo/cat_targets_all41_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/hpo/lgb_targets_rarehard_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/hpo/lgb_targets_all41_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/campaigns/a100_hpo_catlgb_targets_v1.json`
- Feature audit pipeline (OOF permutation + null):
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/scripts/feature_audit_oof.py`
- Feature audit configs/campaign:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/audit/feature_audit_xgb_rarehard_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/audit/feature_audit_xgb_all41_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/campaigns/a100_feature_audit_xgb_v1.json`
- H8 XGB bagging specialists runner:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/scripts/h8_xgb_bagging.py`
- H8 bagging configs/campaign:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/h8/top1_h8_rare_xgb_bagging_a100_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/h8/top1_h8_hard_xgb_bagging_a100_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/campaigns/a100_h8_xgb_bagging_v1.json`
- A100 runbook:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/docs/A100_READINESS_RUNBOOK.md`
- Linear source runner/campaign:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/scripts/linear_source.py`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/linear/top1_a100_hlinear_sgd_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/campaigns/a100_linear_sources_v1.json`
- LightAutoML source runner/campaign:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/scripts/lama_source.py`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/lama/top1_a100_hlama_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/campaigns/a100_lama_sources_v1.json`
- H5 Smart source runner/campaign:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/scripts/h5_smart_source.py`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/h5/top1_h5_smart_v2_source.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/campaigns/a100_h5_smart_v2.json`
- H7 MTL source runner/campaign:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/scripts/h7_mtl_source.py`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/h7/top1_a100_h7_mtl_source_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/campaigns/a100_h7_mtl_source_v1.json`

## N. Imported Ideas Backlog (from `/Users/arceniy/Downloads/ML_Гипотезы_План.md`)

These are explicitly captured so they do not get lost. Some overlap with existing registry items.

### H7_MTL (multi-task neural network, 41 heads)

- Status: `INFRA_READY`
- Maps to:
  - `H7_mlp` / `H7_MTL` A100 branch
- Rationale:
  - Potentially strong orthogonal source, especially for rare targets via shared representation
- Current priority:
  - High potential, but below `H9_full_xgb` and `H8` specialist/HPO lines due to implementation/training complexity
- Current infra:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/scripts/h7_mtl_source.py`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/h7/top1_a100_h7_mtl_source_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/campaigns/a100_h7_mtl_source_v1.json`
- Note:
  - Local smoke is CLI/compile-only unless `torch` is installed; runtime validation is expected on A100.

### H_Linear (GLM / ElasticNet source; H2O optional)

- Status: `INFRA_READY` (sklearn linear source) / `PLANNED` (H2O GLM variant)
- Rationale:
  - Linear source may capture global trends trees approximate poorly
- Clarification:
  - The idea is high-quality; H2O is one implementation option, not mandatory
  - A lightweight linear-source baseline can be tested before H2O
- Current infra:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/scripts/linear_source.py`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/linear/top1_a100_hlinear_sgd_v1.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/campaigns/a100_linear_sources_v1.json`

### H8_Rare_Bagging (undersampling bagging for rare targets)

- Status: `INFRA_READY`
- Maps to:
  - `H8` family v2 (after baseline `H8_rare`/`H8_hard`)
- Rationale:
  - Strong macro-AUC-specific idea; likely useful where class weighting alone is insufficient
- Current infra:
  - `scripts/h8_xgb_bagging.py` (partial-source, resumable target-level XGB bagging)
  - `configs/h8/top1_h8_rare_xgb_bagging_a100_v1.json`
  - `configs/h8/top1_h8_hard_xgb_bagging_a100_v1.json`
  - `configs/campaigns/a100_h8_xgb_bagging_v1.json`
- Risk:
  - Higher training cost than standard H8 specialists; monitor stability and duplicate-source risk

### H_DART (LGBM/XGB DART sources)

- Status: `INFRA_READY`
- Rationale:
  - Likely lower solo score but potentially strong diversity source for top-level ensemble
- Current priority:
  - High (cheap new source family, directly compatible with current architecture)

### H4_Unpack (expose strong solo sources instead of only H4_blend)

- Status: `PROVEN+`
- Evidence:
  - `H4_xgb` as top-level source is already part of current champion
- Notes:
  - `H4_cat/H4_lgb` tested and currently redundant in the tested `top2w` core

### Top-3 Weighted (fixed-weight top-3 per target)

- Status: `INFRA_READY`
- Rationale:
  - Middle ground between robust `top2_weighted` and overfitting-prone `optimize_weights`
- Current priority:
  - High no-train experiment candidate

### H5_Smart (cross-target interactions on OOF preds)

- Status: `INFRA_READY`
- Rationale:
  - Can extend H5 using correlation-informed pairs and engineered interactions
- Note:
  - Must use OOF predictions to avoid leakage
- Current infra:
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/scripts/h5_smart_source.py`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/h5/top1_h5_smart_v2_source.json`
  - `/Users/arceniy/Documents/Projects/Data ML Hack 2/configs/campaigns/a100_h5_smart_v2.json`

### Pseudo-Labeling (late-stage)

- Status: `PLANNED`
- Rationale:
  - Potential late boost after strong champion is established
- Current priority:
  - Final-stage only (higher risk, lower interpretability)
