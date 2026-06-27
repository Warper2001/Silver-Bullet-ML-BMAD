# Pre-Registration: Restore YANK's ML filter (ml0.50) on the live combine

**Date:** 2026-06-27 (committed BEFORE the live config change / restart)
**Author:** Alex
**Type:** Config restoration + change record (not a new optimization).
**Cause (documented):** `_bmad-output/bug_yank_ml_filter_silently_disabled_2026-06-27.md`.
**Fix branch:** `fix/yank-ml-feature-cols` (commit `8080495`).

## Why this prereg exists

YANK's sealed config specifies the ML meta-label filter at threshold **0.50**
(`ml0.50`, S25 lineage; validated in the 2026 OOS holdout: ml0.50 PF 1.60 vs
no-ML 1.32). A feature-schema bug silently disabled that filter on the **real
Topstep combine from 2026-06-17**, so live has been running **no-ML** — an
*unvalidated* config — without anyone choosing that. Per program discipline, a
change to a sealed live config requires a documented cause (the bug report) and a
pre-registration. This restores live to the validated config; it is **not** a new
parameter search and tunes nothing.

## The change (frozen)

- **What changes:** the ML filter begins functioning again (returns the model's
  real P(success) instead of a `1.0` fail-open). After the fix, live YANK runs:
  - model `models/xgboost/tier2_meta_labeling_model.pkl` (18-feature pipeline),
  - threshold **0.50** (`models/xgboost/tier2_threshold.json`),
  - all other strategy parameters UNCHANGED (bearish-only, SL2/TP8, gap≥0.25 ATR,
    H1 sweep + M15 CHoCH, etc. — the sealed S25/ml0.50 config).
- **What does NOT change:** no strategy parameter, no model retrain, no threshold
  change. Only the filter is restored to operating per its sealed spec.
- **Defensive guards shipped with it:** `predict_proba` fails CLOSED on error;
  `__init__` aborts on a FEATURE_COLS↔model schema mismatch. These prevent a
  silent recurrence and do not alter the validated behavior when the schema matches.

## Divergence handling (the load-bearing honesty)

- **Divergence window:** 2026-06-17 (Phase-D combine restart) → the fix deploy date.
  During this window, live YANK on combine acct 23884932 ran **effectively no-ML**.
- **Commitment:** all live MNQ performance in that window is labeled **no-ML** in
  every downstream analysis — including the floor-monitor PF history, any live-vs-
  backtest reconciliation, and the YANK performance record. It must NOT be pooled
  with ml0.50 live data as if it were the sealed config.
- This window is a known artifact, not evidence for or against either config.

## What this authorizes (and what it does not)

- **Authorizes:** committing this note, then deploying `fix/yank-ml-feature-cols`
  (merge → restart `trader-yank` → verify real probas in `tier2_filter_log.csv`).
- **Does NOT authorize:** any change to which model, threshold, or strategy
  parameter YANK runs. If a deliberate move to no-ML (or any reconfig) is later
  desired, that is a SEPARATE pre-registration with its own decision rule.

## Verification gate (must hold post-deploy, else roll back)

1. Boot: no `ML schema mismatch` RuntimeError (the startup guard passes).
2. Next ≥3 signals log **real probas in ~0.37–0.71**, not `1.0`, in
   `logs/tier2_filter_log.csv`.
3. First session shows no anomalous behavior; trade frequency drops as expected
   (the filter now filters).
If (1) or (2) fail, restart on the prior commit and re-investigate before retrying.

## Optional companion (NOT part of this restoration; pre-register separately if pursued)

The accidental 06-17→fix no-ML window + the model's weak discrimination (AUC ~0.52)
make a deliberate, pre-registered **ml0.50 vs no-ML on YANK** re-examination
worthwhile. That is a future research item with its own seal and decision rule —
explicitly out of scope here. This note only restores the sealed config.
