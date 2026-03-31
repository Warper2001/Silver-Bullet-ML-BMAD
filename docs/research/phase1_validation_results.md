# Phase 1 Validation Results: 3:1 Risk-Reward Adjustment

**Date**: 2026-03-31
**Status**: ❌ Phase 1 Complete - Results Below Expectations

## Summary

Updated the strategy from 2:1 to 3:1 risk-reward ratio to lower the breakeven win rate from 33% → 25%. However, validation results show this change alone is **insufficient** to make the strategy viable.

## Changes Made

### Position Sizer (`src/risk/position_sizer.py`)
- ✅ Added `risk_reward_ratio` parameter (default: 3.0)
- ✅ Added `take_profit` field to `PositionSizeResult`
- ✅ Added `calculate_take_profit_distance()` method
- ✅ Updated `DEFAULT_ATR_MULTIPLIER` from 1.2 → 0.3
- ✅ Updated `DEFAULT_RISK_REWARD_RATIO` to 3.0

### Live Trading Script (`live_paper_trading_optimized.py`)
- ✅ Updated parameters: TP 0.4% → 0.9%, SL 0.2% → 0.3%
- ✅ Added `risk_reward_ratio = 3.0` configuration
- ✅ Updated position sizer initialization with 3:1 R:R
- ✅ Updated logging to reflect hybrid approach

### Model Parameters (`models/xgboost/30_minute/sb_params_optimized.json`)
- ✅ Updated `take_profit_pct`: 0.4 → 0.9
- ✅ Updated `stop_loss_pct`: 0.2 → 0.3
- ✅ Added `risk_reward_ratio`: 3.0

## Walk-Forward Validation Results

### Test Configuration
- **Risk-Reward**: 3:1 (TP 0.9%, SL 0.3%)
- **Filters**: Daily bias, Volatility (ATR% ≥ 0.3%)
- **Periods Tested**:
  - Mar-May 2025 (3 months)
  - Sep-Nov 2025 (3 months)
  - Dec-Feb 2026 (3 months)

### Performance Metrics

| Period | Win Rate | Return | Sharpe | Max DD | Trades | Status |
|--------|----------|--------|--------|--------|--------|--------|
| **Mar-May 2025** | **33.94%** | **+9.75%** | **0.69** | -32.11% | 221 | ✅ Profitable |
| **Sep-Nov 2025** | 13.95% | -7.17% | -4.78 | -12.72% | 43 | ❌ Losing |
| **Dec-Feb 2026** | 9.43% | -12.46% | -9.05 | -15.64% | 53 | ❌ Losing |
| **Average** | **19.11%** | **-3.30%** | **-4.38** | -20.16% | 106 | ❌ Below Breakeven |

### Key Findings

1. **Win Rate Decreased**: 19.11% at 3:1 vs 24.5% at 2:1
   - Expected: Win rate to stay similar
   - Actual: Win rate decreased by 5.4 percentage points
   - **Result**: Still below 25% breakeven

2. **Performance Decay Over Time**:
   - Mar-May 2025: 33.94% win rate (profitable)
   - Sep-Nov 2025: 13.95% win rate (terrible)
   - Dec-Feb 2026: 9.43% win rate (disastrous)
   - **Trend**: Strategy performance is degrading

3. **High Variability**:
   - Std dev: 10.65%
   - Best period: 33.94%
   - Worst period: 9.43%
   - **Spread**: 24.5 percentage points

4. **Only One Profitable Period**:
   - Mar-May 2025: +9.75% return (221 trades)
   - Other periods: Losing money

## Comparison: 2:1 vs 3:1 Risk-Reward

| Metric | 2:1 R:R | 3:1 R:R | Change |
|--------|---------|---------|--------|
| **Avg Win Rate** | 24.5% | 19.11% | -5.4% |
| **Breakeven WR** | 33% | 25% | -8% |
| **Above Breakeven?** | ❌ No | ❌ No | Still losing |
| **Avg Return** | -3.2% | -3.3% | -0.1% |
| **Avg Sharpe** | -5.52 | -4.38 | +1.14 |
| **Std Dev** | 12.44% | 10.65% | -1.8% |

**Conclusion**: Changing to 3:1 risk-reward **did not improve** strategy performance. The win rate decreased more than the breakeven threshold dropped.

## Root Cause Analysis

### Why Did 3:1 R:R Fail?

1. **Wider Stops = More Losses**
   - SL widened from 0.2% → 0.3%
   - Wider stops get hit more often
   - Result: Lower win rate (24.5% → 19.11%)

2. **Model is Outdated**
   - Trained on 2024 data
   - Market conditions changed in 2025-2026
   - Patterns no longer work reliably

3. **Performance Decay is the Real Problem**
   - Mar-May 2025: 33.94% (good)
   - Dec-Feb 2026: 9.43% (terrible)
   - **Issue**: Model age > 90 days = degraded performance

## Next Steps

### Phase 2: Model Retraining (REQUIRED)

**Priority**: ⭐⭐⭐⭐⭐ Critical

The 3:1 R:R change alone is insufficient. We **must** retrain the model on recent 2025-2026 data to address the performance decay.

**Actions**:
1. Train new model on 2025-2026 data only (skip 2024)
2. Use walk-forward validation for testing
3. Apply regularization to prevent overfitting:
   - max_depth: 4
   - learning_rate: 0.05
   - reg_lambda: 1.0
   - reg_alpha: 0.1
4. Target realistic win rate: 30-40%
5. Save as `model_v2.joblib`

**Expected Outcome**:
- Win rate: 30-40% (realistic)
- Performance: More consistent across periods
- Decay: Reduced through monthly retraining

### Phase 3: Filter Enhancement (After Phase 2)

Once the model is retrained, we can add:
- Quality scoring system (confidence × confluence × bias × volatility)
- Minimum quality threshold: 60
- Trade frequency reduction: ~100 → ~50/month
- Expected: +5-10% win rate improvement

## Recommendation

### Option 1: Stop Here ⭐⭐⭐⭐⭐
- **Reason**: 3:1 R:R didn't work, strategy still unprofitable
- **Action**: Abandon strategy, cut losses
- **Outcome**: Preserve capital, find better strategy

### Option 2: Continue to Phase 2 ⭐⭐⭐
- **Reason**: One period (Mar-May 2025) showed 33.94% win rate and profitability
- **Action**: Retrain model on 2025-2026 data
- **Risk**: May still lose money if performance continues to decay
- **Effort**: 2-3 hours development + 1 week testing

### Option 3: Paper Trade Only ⭐⭐⭐⭐
- **Action**: Continue with paper trading (SIM account only)
- **Purpose**: Collect more data on 3:1 R:R performance
- **Duration**: 3-6 months minimum
- **Risk**: No real money at risk

## Acceptance Criteria

### Phase 1 Target (Not Met)
- ❌ Walk-forward win rate ≥ 25%
- ❌ Positive expected value
- ❌ Sharpe ratio ≥ 0
- ✅ Max drawdown < -25% (just barely)

### Revised Expectations
Based on validation results, **3:1 R:R alone is not sufficient**. We should either:
1. **Abandon the strategy** (recommended)
2. **Retrain the model** and retest (high risk, uncertain outcome)

## Files Modified

1. `src/risk/position_sizer.py` - Added 3:1 R:R support
2. `live_paper_trading_optimized.py` - Updated to use 3:1 R:R
3. `models/xgboost/30_minute/sb_params_optimized.json` - Updated parameters
4. `quick_validation.py` - Updated to test 3:1 R:R

## Conclusion

**Phase 1 Status**: ❌ **FAILED**

The 3:1 risk-reward adjustment did not achieve the desired results. The win rate decreased from 24.5% to 19.11%, and the strategy remains unprofitable.

**Critical Finding**: The problem is not the risk-reward ratio - it's the **outdated model** trained on 2024 data. The market has changed, and the patterns no longer work reliably.

**Recommendation**: Either abandon the strategy or proceed to Phase 2 (model retraining) with low expectations.
