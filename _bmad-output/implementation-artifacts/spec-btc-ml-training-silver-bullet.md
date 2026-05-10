---
title: 'BTC Silver Bullet ML Training Pipeline'
type: 'feature'
created: '2026-05-10'
status: 'done'
baseline_commit: '0579f76'
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The optimized BTC Silver Bullet (NY AM only, stop=0.75×gap, uncapped) achieves train PF 1.730 but degrades to holdout PF 1.073 — suggesting in-sample overfitting and a need to filter out low-probability setups on unseen data.

**Approach:** Build `train_btc_ml.py` — run the optimized backtest once to collect all trades with fill bar indices, extract 11 no-lookahead features at each fill bar, train XGBoost with TimeSeriesSplit(5) CV on the train set, tune the probability threshold to maximize CV OOS profit factor, then evaluate on the holdout (Nov 2025 – May 2026). Save model artifacts to `models/btc_xgb/`.

## Boundaries & Constraints

**Always:**
- Import `load_csv_as_bars`, `detect_swing_points`, `detect_mss_events`, `detect_fvg_setups`, `detect_confluence`, `round_tick`, `BTC_TICK`, `BTC_CONTRACT_VALUE`, `COMMISSION`, `LIMIT_CANCEL_BARS`, `_load_kill_zones` from `backtest_btc_silver_bullet.py`
- Import `filter_setups_by_zones`, `execute_param` from `optimize_btc_silver_bullet.py`
- Fixed optimized config: NY AM 09:00–10:00 CDT only (start_h=9, start_m=0, end_h=10, end_m=0), `stop_mult=0.75`, `target_cap_rr=0`
- Train/holdout split: `SPLIT_TS = datetime(2025, 11, 8, tzinfo=timezone.utc)` — same as optimizer
- All features computed from `bars[fill_bar_idx]` and earlier bars only — no lookahead
- TimeSeriesSplit(n_splits=5) for CV — never random k-fold
- Threshold sweep: `np.arange(0.30, 0.71, 0.05)` over CV OOS predictions; pick threshold that maximizes OOS profit factor (min 10 filtered trades per fold required)
- Output artifacts: `models/btc_xgb/model.joblib`, `models/btc_xgb/threshold.json`, `models/btc_xgb/feature_importance.csv`
- Label: `1 if trade["pnl"] > 0 else 0`
- Recover `fill_bar_idx` from `entry_time`: build `ts_to_idx = {bar.timestamp.isoformat(): i for i, bar in enumerate(bars)}` — no trade dict modification needed

**Ask First:**
- If train set has fewer than 100 trades: HALT before proceeding — config may be misconfigured
- If CV mean AUC < 0.52: HALT and report — model lacks discriminative power; proceed only with explicit user confirmation

**Never:**
- Touch holdout data during training or threshold tuning — holdout is used exactly once after threshold selection
- Modify `backtest_btc_silver_bullet.py`, `optimize_btc_silver_bullet.py`, or any `src/` file
- Use random CV splits or stratified splits — temporal order must be preserved

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Happy path | ≥100 train trades, AUC ≥ 0.52 | Model artifacts written; holdout report printed | — |
| Train trades < 100 | Misconfigured kill zone or split | HALT with clear message | Abort |
| CV AUC < 0.52 | Features not discriminating | HALT with AUC value; ask confirmation | Ask first |
| 0 holdout trades pass filter | Threshold too aggressive or overfit | Print "No holdout trades passed filter (threshold=X.XX)" without crash | Log warning |
| fill_bar_idx < 50 | Early dataset edge | Skip trade from feature set (can't compute 50-bar slope) | Log skip |

</frozen-after-approval>

## Code Map

- `backtest_btc_silver_bullet.py` — imports: `load_csv_as_bars`, `detect_swing_points`, `detect_mss_events`, `detect_fvg_setups`, `detect_confluence`, `round_tick`, `BTC_CONTRACT_VALUE`, `COMMISSION`, `LIMIT_CANCEL_BARS`, `MIN_RR`, `POSITION_SIZE`, `MAX_HOLD_BARS`, `_cdt_date`, `_find_next_liquidity_pool`
- `optimize_btc_silver_bullet.py` — imports: `filter_setups_by_zones` only; `execute_param` is NOT used directly — it returns only 4 fields (pnl, exit_reason, killzone_window, bars_held) which is insufficient for feature extraction. `execute_for_ml()` is defined locally in `train_btc_ml.py` with the same logic but returns full trade metadata (fill_bar_idx, entry_price, stop_loss, gap_size, direction, rr_target)
- `data/kraken/PF_XBTUSD_1min.csv` — source data; 789k rows
- `models/btc_xgb/` — output directory (create if missing)
- `data/reports/` — output directory for text report

## Tasks & Acceptance

**Execution:**
- [x] `train_btc_ml.py` — CREATE: load bars; run detection once on all bars; apply NY AM 09:00–10:00 CDT filter; run `execute_param(stop_mult=0.75, target_cap_rr=0)` on train setups; recover fill_bar_idx per trade; extract 11 features (below); split train/holdout by SPLIT_TS; train XGBoost on train features; TimeSeriesSplit(5) CV for AUC + threshold tuning (maximize OOS PF); save model + threshold + feature_importance; apply to holdout; print comparison report
- [x] `models/btc_xgb/` — CREATE directory and write model artifacts

**Acceptance Criteria:**
- Given the full CSV, when the script runs, then detection runs exactly once and all train trades are collected before any ML step
- Given ≥100 train trades and AUC ≥ 0.52, when script completes, then `models/btc_xgb/model.joblib`, `threshold.json`, and `feature_importance.csv` all exist
- Given threshold.json, when loaded, then it contains `{"threshold": X.XX, "cv_mean_auc": X.XXXX}`
- Given the holdout evaluation, when printed, then output clearly labels "HOLDOUT (out-of-sample)" with unfiltered PF, filtered PF, unfiltered trades, filtered trades, win rate, and P&L for both
- Given all features, when computed, then no feature uses any bar at index > fill_bar_idx (verified by code inspection)

## Design Notes

**Feature set (11 features):**

```python
def extract_features(trade: dict, bars: list, fill_idx: int, swings: list) -> dict | None:
    if fill_idx < 50: return None  # insufficient history
    stop_dist = abs(trade["entry_price"] - trade["stop_loss"])
    gap_size = stop_dist / 1.25  # inverse of stop_mult=0.75 relationship
    atr14 = compute_atr14(bars, fill_idx)
    closes = [bars[j].close for j in range(fill_idx - 49, fill_idx + 1)]
    slope = np.polyfit(range(50), closes, 1)[0] / bars[fill_idx].close
    cdt = bars[fill_idx].timestamp + timedelta(hours=-5)
    return {
        "fvg_gap_size":      gap_size,
        "stop_distance":     stop_dist,
        "rr_at_entry":       trade["rr_target"],
        "direction":         1 if trade["direction"] == "bullish" else 0,
        "atr14_rel_gap":     atr14 / gap_size if gap_size > 0 else 0,
        "momentum_20":       (bars[fill_idx].close - bars[fill_idx-20].close) / bars[fill_idx-20].close,
        "trend_slope_50":    slope,
        "session_time_frac": (cdt.hour - 9 + cdt.minute / 60),  # 0.0=09:00, 1.0=10:00
        "day_of_week":       cdt.weekday(),
        "swing_dist_atr":    nearest_swing_dist(bars[fill_idx].close, swings, fill_idx) / atr14,
        "bull_bar_proxy_20": sum(1 for j in range(fill_idx-20, fill_idx) if bars[j].close > bars[j].open) / 20,
    }
```

**Threshold optimization over CV folds:**
```python
best_threshold, best_pf = 0.5, 0.0
for threshold in np.arange(0.30, 0.71, 0.05):
    oos_preds = [p for fold in cv_oos_predictions for p in fold]
    oos_labels = [l for fold in cv_oos_labels for l in fold]
    filtered_pnls = [pnl for p, l, pnl in zip(oos_preds, oos_labels, oos_pnls) if p >= threshold]
    pf = sum(x for x in filtered_pnls if x > 0) / abs(sum(x for x in filtered_pnls if x < 0) + 1e-9)
    if pf > best_pf and len(filtered_pnls) >= 10 * 5:
        best_pf, best_threshold = pf, threshold
```

**XGBoost params:** `n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='auc', use_label_encoder=False`

## Verification

**Commands:**
- `.venv/bin/python train_btc_ml.py` — expected: completes in <15 min; prints AUC, CV PF, holdout comparison table; no crashes
- `ls models/btc_xgb/` — expected: `model.joblib`, `threshold.json`, `feature_importance.csv`
- `python -c "import json; d=json.load(open('models/btc_xgb/threshold.json')); print(d)"` — expected: dict with `threshold` and `cv_mean_auc` keys

## Suggested Review Order

**Execution & data flow**

- Entry point: detection once, then split and execute train/holdout separately
  [`train_btc_ml.py:332`](../../train_btc_ml.py#L332)

- execute_for_ml: same logic as execute_param but returns fill_bar_idx + full metadata
  [`train_btc_ml.py:68`](../../train_btc_ml.py#L68)

**Feature extraction**

- extract_features: all 11 features at fill_bar_idx, no lookahead
  [`train_btc_ml.py:206`](../../train_btc_ml.py#L206)

- compute_atr14: standard ATR(14) using true range, fallback for early bars
  [`train_btc_ml.py:189`](../../train_btc_ml.py#L189)

**Model training & threshold tuning**

- TimeSeriesSplit(5) CV loop: fold AUC + pooled OOS probabilities for threshold selection
  [`train_btc_ml.py:404`](../../train_btc_ml.py#L404)

- tune_threshold: sweeps 0.30–0.70, picks max-PF threshold, warns on fallback
  [`train_btc_ml.py:296`](../../train_btc_ml.py#L296)

**Holdout evaluation & report**

- Holdout feature extraction and ML filter application (clear OOS labeling)
  [`train_btc_ml.py:463`](../../train_btc_ml.py#L463)

- Artifact save: model.joblib, threshold.json, feature_importance.csv
  [`train_btc_ml.py:494`](../../train_btc_ml.py#L494)
