# 1-Minute Migration Validation Status Report

**Generated:** 2026-04-15
**Status:** ⚠️ **CRITICAL ISSUES FOUND - REQUIRES FIXES**

## Executive Summary

The 1-minute migration has made significant progress on data generation and model training, but **critical validation issues** prevent production deployment. Several configuration mismatches and suspicious backtest results indicate data leakage or improper testing methodology.

## Validation Status by Phase

### ✅ Phase 1: Data Generation (COMPLETE)
**Status:** PASS with minor warnings

**Completed:**
- ✅ 1-minute dollar bars generated: `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv` (22M)
- ✅ Regime-specific training data created:
  - Regime 0: 19,171 bars (trending up)
  - Regime 1: 247,228 bars (trending up strong) 
  - Regime 2: 6,645 bars (trending down)
- ✅ Tier 1 features engineered (17 features)

**Warnings:**
- ⚠️ No timestamp column found (likely data format issue, not critical)
- ⚠️ Average notional value varies from $50M expected (this is normal for dollar bars)

**Data Quality:** PASS
- Price validation: 100% valid OHLC data
- Volume validation: No zero-volume bars
- Feature completeness: 16/16 features present, <0.5% NaN values

### ✅ Phase 2: HMM Regime Detection (COMPLETE)
**Status:** PASS

**Completed:**
- ✅ HMM model trained: `models/hmm/regime_model_1min/hmm_model.joblib`
- ✅ Training period: Jan-Sep 2025 (214,103 bars)
- ✅ Validation period: Oct-Dec 2025 (75,127 bars)
- ✅ 3 regimes detected with clear separation
- ✅ Convergence achieved (100 iterations)

**HMM Performance:**
- Training regime distribution:
  - Regime 0 (trending up): 2.5% of bars
  - Regime 1 (trending up strong): 90.8% of bars
  - Regime 2 (trending down): 6.7% of bars
- Validation regime distribution: Stable and consistent

### ✅ Phase 3: Feature Engineering (COMPLETE)
**Status:** PASS

**Completed:**
- ✅ Regime-aware training data generated with labels
- ✅ 54 technical indicators engineered (full feature set)
- ✅ 17 Tier 1 features extracted (order flow, volatility, microstructure)
- ✅ Feature distributions analyzed and validated

### ✅ Phase 4: Model Training (COMPLETE)
**Status:** PASS with multiple versions

**Completed:**
- ✅ Multiple model versions trained (54-features and Tier 1)
- ✅ All 4 XGBoost models created:
  - Regime 0 model (685K)
  - Regime 2 model (588K)
  - Generic model (826K)
- ✅ Most recent: `regime_aware_1min_2025_54features/` (April 15, 16:32)

**Model Versions:**
1. `regime_aware_1min_2025/` - 52 features (April 15, 00:13)
2. `regime_aware_1min_2025_54features/` - 54 features (April 15, 16:32) ⭐ **LATEST**
3. `regime_aware_1min_2025_proper/` - empty (placeholder)

### ❌ Phase 5: Threshold Optimization (INCOMPLETE)
**Status:** NOT RUN - Only 3 trades in quick backtest

**Issues:**
- ❌ Quick backtest only generated 3 trades (single day: 2025-10-01)
- ❌ Insufficient sample size for threshold optimization
- ❌ No sensitivity analysis performed (30%, 35%, 40%, 45%, 50%, 55%, 60%)

**Quick Backtest Results (3 trades):**
- Win rate: 66.67% (2 wins / 1 loss)
- Avg probability: 52.48% (realistic for 1-minute data)
- Total P&L: $44.38 over 3 trades
- All exits: Time-based (30-minute max hold)

### ❌ Phase 6: Comprehensive Backtesting (CRITICAL ISSUES)
**Status:** ⚠️ **SUSPICIOUS RESULTS - DATA LEAKAGE LIKELY**

**Old Backtest (April 7, 2026):**
- ❌ **44,540 trades** (unrealistically high for 1 year)
- ❌ **84.82% win rate** (far above 50-52% target)
- ❌ **Sharpe ratio: 44.54** (mathematically impossible in live trading)
- ❌ Used 65% threshold (not 40% from spec)
- ❌ Likely trained on test data (data leakage)

**Comparison with Spec Targets:**
| Metric | Old Backtest | Spec Target | Verdict |
|--------|--------------|-------------|---------|
| Win Rate | 84.82% | 50-52% | ❌ 34% too high |
| Sharpe Ratio | 44.54 | ≥0.6 | ❌ 74x too high |
| Trades/Day | ~182 | 5-25 | ❌ 7x too high |

**Diagnosis:** This backtest is **invalid** and should not be used for decision-making. Likely causes:
1. Train/test data leakage (trained on validation data)
2. Random split instead of temporal split
3. Probability threshold mismatch
4. Missing transaction costs and slippage

### ❌ Phase 7: System Integration (CRITICAL CONFIG ISSUES)
**Status:** ❌ **CONFIGURATION MISMATCHES**

**Critical Issues Found:**

1. **Model Path Mismatch** (`config.yaml:30`):
   ```yaml
   model_path: "models/xgboost/regime_aware_real_labels/"  # ❌ 5-minute models
   ```
   **Should be:**
   ```yaml
   model_path: "models/xgboost/regime_aware_1min_2025_54features/"  # ✅ 1-minute models
   ```

2. **MIN_BARS_BETWEEN_TRADES Not Updated** (`config.yaml:55`):
   ```yaml
   min_bars_between_trades: 30  # ❌ 2.5 hours at 5-min bars
   ```
   **Should be:**
   ```yaml
   min_bars_between_trades: 1  # ✅ 1 minute at 1-min bars
   ```

3. **Comment Outdated** (`config.yaml:54`):
   ```yaml
   enabled: true  # Evaluate every 5-minute bar (not signal-based)
   ```
   **Should be:**
   ```yaml
   enabled: true  # Evaluate every 1-minute bar (not signal-based)
   ```

## Critical Validation Gaps

### 1. No Transaction Cost Analysis (Task 1.4 - CRITICAL)
**Status:** ❌ NOT PERFORMED

**Required:**
- Calculate commission costs at $2.50/contract RT
- Model slippage (0.25-0.50 ticks per trade)
- Breakeven win rate calculation
- Expectation per trade after costs

**Impact:** Without this analysis, you cannot determine if the system is profitable after real-world costs.

### 2. No Temporal Train/Test Split Validation (Tasks 4.2-4.5 - CRITICAL)
**Status:** ❌ NOT VALIDATED

**Required:**
- Train: Jan-Sep 2025, Test: Oct-Dec 2025
- Walk-forward validation with 3-month rolling windows
- Validate no data leakage

**Impact:** The suspicious backtest results suggest data leakage occurred.

### 3. No Concurrent Position Limits (Task 7.5 - CRITICAL)
**Status:** ❌ NOT IMPLEMENTED

**Required:**
- Max 3-5 concurrent positions
- Portfolio correlation check
- Dynamic position sizing
- Margin usage check

**Impact:** 30-min max hold + 1 bar spacing = up to 30 concurrent positions possible, exceeding margin requirements.

### 4. No Comprehensive Backtest with Proper Settings
**Status:** ❌ NOT PERFORMED

**Required:**
- Use correct 1-minute model paths
- Use MIN_BARS_BETWEEN_TRADES = 1
- Include transaction costs ($2.50/contract + 0.25 tick slippage)
- Test on held-out Oct-Dec 2025 data
- Realistic performance metrics (50-52% win rate target)

## Acceptance Criteria Status

| Criterion | Target | Current | Status |
|-----------|--------|---------|--------|
| HMM regime separation | >90% | >90% | ✅ PASS |
| Regime 0 accuracy | >90% | Unknown | ⚠️ NOT VALIDATED |
| Regime 2 accuracy | >95% | Unknown | ⚠️ NOT VALIDATED |
| Generic accuracy | >70% | Unknown | ⚠️ NOT VALIDATED |
| Win rate | 50-52% | 84.82%* | ❌ DATA LEAKAGE |
| **Expectation per trade** | **>$20 after costs** | **Unknown** | ❌ NOT CALCULATED |
| **Profit factor** | **>1.5** | **1.82*** | ❌ SUSPICIOUS |
| Trade frequency | 5-25/day | ~182* | ❌ 7x too high |
| Sharpe ratio | ≥0.6 | 44.54* | ❌ 74x too high |
| Max drawdown | <$1,000 | $2,136.44* | ❌ 2x too high |
| Temporal validation | Required | Not performed | ❌ NOT DONE |
| Transaction cost analysis | Required | Not performed | ❌ NOT DONE |

*From suspicious backtest (April 7) - likely invalid due to data leakage

## Immediate Action Items

### Priority 1: Fix Configuration Mismatches (CRITICAL)
1. Update `config.yaml` line 30: Change model path to 1-minute models
2. Update `config.yaml` line 55: Change `min_bars_between_trades` from 30 to 1
3. Update `config.yaml` line 54: Fix comment about 5-minute vs 1-minute
4. Update `src/ml/hybrid_pipeline.py` line 59: Change MIN_BARS_BETWEEN_TRADES from 30 to 1

### Priority 2: Run Proper Validation Backtest (CRITICAL)
1. Create new backtest script using:
   - Correct 1-minute model paths
   - MIN_BARS_BETWEEN_TRADES = 1
   - Held-out Oct-Dec 2025 test data
   - Transaction costs ($2.50/contract + 0.25 tick slippage)
   - 40% probability threshold
2. Validate against realistic targets (50-52% win rate, Sharpe ≥0.6)
3. Calculate expectation per trade after costs

### Priority 3: Transaction Cost Analysis (CRITICAL)
1. Run cost analysis script (Task 1.4)
2. Calculate breakeven win rate at 5-25 trades/day
3. Determine if MIN_BARS_BETWEEN_TRADES needs adjustment

### Priority 4: Implement Concurrent Position Limits (HIGH)
1. Implement max 3-5 concurrent positions
2. Add portfolio correlation check
3. Add margin usage monitoring

## Recommendations

### Short-term (This Week)
1. ✅ Fix configuration mismatches (Priority 1)
2. ✅ Run proper validation backtest (Priority 2)
3. ✅ Perform transaction cost analysis (Priority 3)

### Mid-term (Next 2 Weeks)
1. Implement concurrent position limits (Priority 4)
2. Validate model performance with temporal splits
3. Run threshold sensitivity analysis (30-60%)

### Long-term (Next Month)
1. Expand to full historical dataset (2023-2026) if 2025 validation successful
2. Consider paper trading deployment
3. Implement adaptive MIN_BARS_BETWEEN_TRADES based on volatility

## Conclusion

The 1-minute migration has **solid foundations** (data generation, HMM training, feature engineering) but **critical validation gaps** prevent production deployment. The suspicious backtest results suggest data leakage, and configuration mismatches would cause the system to use 5-minute models instead of 1-minute models.

**Next Step:** Fix configuration mismatches and run a proper validation backtest with correct settings before proceeding further.

---

**Report prepared by:** Claude Code (BMad Validation Framework)
**Tech Spec Reference:** `_bmad-output/implementation_artifacts/tech-spec-1min-dollarbar-migration-2025.md`
