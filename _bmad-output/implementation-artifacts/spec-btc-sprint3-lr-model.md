---
title: 'BTC Silver Bullet Sprint 3 — L2 Logistic Regression (6 features)'
type: 'feature'
created: '2026-05-10'
status: 'done'
route: 'one-shot'
context:
  - '{project-root}/_bmad-output/planning-artifacts/btc-silver-bullet-improvement-decision-framework.md'
---

## Intent

**Problem:** XGBoost on N=218 produces train AUC 0.99 vs CV AUC 0.47 — a gap of 0.52. The model memorizes the training set and cannot generalize. The decision framework target for Sprint 3 is: train/CV AUC gap < 0.10.

**Approach:** Replace `XGBClassifier` with `Pipeline([StandardScaler, LogisticRegression(C=0.1)])` in `train_btc_ml.py`, reduce features from 16 to 6 (`LR_FEATURE_COLS`), and print a Sprint 3 gate verdict at end of `main()`.

## Suggested Review Order

**Model swap and feature reduction**

- Import change: xgboost → sklearn LR/Pipeline/StandardScaler; `LR_FEATURE_COLS` constant (6 features)
  [`train_btc_ml.py:20`](../../train_btc_ml.py#L20) · [`train_btc_ml.py:318`](../../train_btc_ml.py#L318)

- CV fold pipeline: `StandardScaler + LR(C=0.1)` replacing XGBClassifier + scale_pos_weight
  [`train_btc_ml.py:563`](../../train_btc_ml.py#L563)

- Final model pipeline and `coef_`-based feature importance
  [`train_btc_ml.py:614`](../../train_btc_ml.py#L614) · [`train_btc_ml.py:666`](../../train_btc_ml.py#L666)

**Sprint 3 gate verdict**

- Gate 1 (AUC gap < 0.10), Gate 2 (CV AUC ≥ 0.54), advisory (holdout PF vs 1.479 baseline); zero-trade holdout handled explicitly
  [`train_btc_ml.py:718`](../../train_btc_ml.py#L718)

**Deferred work**

- Pre-existing issues carried forward: `atr14_100` synthetic fallback at fill_idx 100–113, `mss_to_fvg_bars` negative values, unvalidated threshold in `threshold.json`
  [`deferred-work.md`](deferred-work.md)
