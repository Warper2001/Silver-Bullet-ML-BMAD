# Premium Strategy + Hybrid Regime-Aware Implementation Plan

## Objective

Combine Premium Strategy quality filters with Hybrid Regime-Aware ML models to achieve:
- **10-15 trades/day**
- **≥84.82% win rate**
- **≥44.54 Sharpe ratio**
- **≤2.1% max drawdown**

## Current State

### Premium Strategy (✅ Complete)
- **Implementation:** `silver_bullet_premium_enhanced.py`
- **Quality Filters:**
  - min_fvg_gap: $75
  - mss_volume_ratio: 2.0x
  - max_bar_distance: 7 bars
  - quality_score: 0-100 (inverted, lower = better)
- **Backtest Results:** 17.6 trades/day, 94.83% win rate
- **Status:** ✅ Production-ready filters

### Hybrid Regime-Aware (✅ Complete)
- **Implementation:** `backtest_bar_by_bar.py`
- **Models:** Regime 0 (97.83%), Regime 2 (100%), Generic fallback
- **Bar-by-Bar Results:** 3.92 trades/day, 51.80% win rate (40% threshold)
- **Status:** ✅ Production-ready models

### Gap
Premium strategy uses baseline ML model (65% threshold) → Need premium-trained hybrid models

## Implementation Steps

### Step 1: Generate Premium Training Data (30 min)

**Command:**
```bash
# Generate premium-labeled training data
.venv/bin/python scripts/generate_premium_training_data.py
```

**What it does:**
- Scan historical dollar bars for Silver Bullet setups
- Apply premium quality filters (FVG ≥$75, volume ≥2.0x, distance ≤7 bars)
- Calculate quality scores (0-100)
- Label trades as premium if quality_score ≥70
- Export to `data/ml_training/silver_bullet_trades_premium.parquet`

**Expected Output:**
- ~5,000-10,000 premium-labeled trades
- Features: 54 ML features + quality_score + premium_label
- Period: 2022-2025 historical data

### Step 2: Train Premium Regime-Aware Models (2 hours)

**Command:**
```bash
# Train premium models per regime
.venv/bin/python scripts/train_premium_regime_models.py
```

**What it does:**
1. Load premium training data
2. Detect regimes using HMM
3. Split data by regime (Regime 0, 1, 2)
4. Train separate XGBoost models per regime on premium labels
5. Validate per-regime performance
6. Save models to `models/xgboost/premium_regime_aware/`

**Expected Output:**
- `xgboost_regime_0_premium.joblib` (Regime 0, premium labels)
- `xgboost_regime_1_premium.joblib` (Regime 1, premium labels)
- `xgboost_regime_2_premium.joblib` (Regime 2, premium labels)
- `xgboost_generic_premium.joblib` (Generic, premium labels)

**Target Performance:**
- Regime 0: ≥90% accuracy (trending up)
- Regime 1: ≥80% accuracy (trending up strong)
- Regime 2: ≥85% accuracy (trending down)

### Step 3: Create Premium+Hybrid Backtest (1 hour)

**Command:**
```bash
# Backtest premium + hybrid system
.venv/bin/python scripts/backtest_premium_hybrid.py
```

**What it does:**
1. Load premium regime-aware models
2. Generate premium-quality signals (apply quality filters)
3. For each signal:
   - Detect regime
   - Get prediction from regime-specific premium model
   - Filter by 75% probability threshold
   - Simulate trade with triple-barrier exits
4. Calculate performance metrics

**Expected Results:**
- Trades/Day: 10-15 (middle of 1-20 range)
- Win Rate: ≥85%
- Sharpe Ratio: ≥40
- Max Drawdown: ≤2.5%

### Step 4: Deploy to Paper Trading (30 min)

**Configuration (`config.yaml`):**
```yaml
silver_bullet_premium:
  enabled: true
  min_fvg_gap_size_dollars: 75
  mss_volume_ratio_min: 2.0
  max_bar_distance: 7
  ml_probability_threshold: 0.75  # Premium threshold
  max_trades_per_day: 20
  min_quality_score: 70

ml:
  model_type: "premium_regime_aware"  # NEW
  model_path: "models/xgboost/premium_regime_aware/"
  use_regime_aware: true
```

**Launch:**
```bash
./deploy_paper_trading.sh start
```

**Validation Period:** 2-4 weeks

## Success Criteria

### Minimum Viable (Week 1-2)
- ✅ Premium training data generated
- ✅ Premium regime models trained
- ✅ Backtest shows 5-20 trades/day
- ✅ Win rate ≥80%

### Target Performance (Week 3-4)
- ✅ 10-15 trades/day
- ✅ Win rate ≥85%
- ✅ Sharpe ratio ≥40
- ✅ Max drawdown ≤2.5%

### Excellent Performance (Week 5+)
- ✅ 12-18 trades/day
- ✅ Win rate ≥90%
- ✅ Sharpe ratio ≥45
- ✅ Max drawdown ≤2.0%

## Risk Mitigation

### Risk 1: Premium Training Data Insufficient
**Mitigation:**
- Expand date range (2021-2025)
- Lower quality threshold to 60/100
- Use SMOTE augmentation if needed

### Risk 2: Per-Regime Models Underperform
**Mitigation:**
- Fall back to generic premium model
- Adjust thresholds per regime
- Ensemble approach (average predictions)

### Risk 3: Trade Frequency Too Low
**Mitigation:**
- Reduce quality threshold to 65/100
- Lower ML threshold to 72%
- Increase max_bar_distance to 10 bars

### Risk 4: Win Rate Below Target
**Mitigation:**
- Increase ML threshold to 78%
- Tighten quality filters
- Add additional quality factors

## Estimated Timeline

| Step | Duration | Dependencies |
|------|----------|--------------|
| 1. Generate Premium Training Data | 30 min | None |
| 2. Train Premium Regime Models | 2 hours | Step 1 |
| 3. Premium+Hybrid Backtest | 1 hour | Step 2 |
| 4. Deploy to Paper Trading | 30 min | Step 3 |
| 5. Validate (2-4 weeks) | 2-4 weeks | Step 4 |
| **Total** | **3.5 hours** + **2-4 weeks validation** | |

## Next Steps

**Immediate Actions (Today):**
1. ✅ Review premium strategy implementation
2. ⏳ Generate premium training data
3. ⏳ Train premium regime-aware models
4. ⏳ Run backtest validation

**This Week:**
5. Deploy to paper trading
6. Monitor first week performance

**Next 2-4 Weeks:**
7. Validate against performance specs
8. Fine-tune thresholds if needed
9. Document results

---

**Status:** Ready to implement
**Priority:** HIGH (User requested Option A)
**Expected Outcome:** Professional-grade futures day trading system
