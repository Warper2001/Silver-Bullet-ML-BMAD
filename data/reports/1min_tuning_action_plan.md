# 1-Minute Strategy Tuning Action Plan

**Date:** 2026-04-16
**Status:** 🎯 **READY FOR EXECUTION**
**Approach:** Hybrid Path E + C (Training Methodology Fix + Exit Parameter Optimization)

---

## Executive Summary

This action plan implements a **phased approach** to fix the 1-minute strategy's severe overfitting issues while exploring quick wins through exit parameter optimization.

**Problem:** Tier1 models severely overfitted - only 3 trades in 3 months (Oct-Dec 2025), 33.3% win rate vs expected 93.8%

**Root Cause:** Data leakage in training, temporal mismatch, insufficient validation

**Solution:** Fix training methodology (Path E) + optimize exits (Path C) with go/no-go decision points

**Timeline:** 10 weeks total, with 2 major decision gates

---

## Phase 1: Quick Wins & Foundation Setup (Week 1)

**Goal:** Test exit parameter improvements while building validation infrastructure

### Task 1.1: Exit Parameter Grid Search
**Timeline:** Days 1-3
**Owner:** ML Engineer
**Deliverable:** `data/reports/exit_parameter_optimization_1min.md`

**Actions:**
1. Create script: `scripts/optimize_exit_parameters_1min.py`
2. Test parameter combinations:
   - Stop Loss: 0.15%, 0.20%, 0.25%, 0.30%, 0.35%, 0.40%
   - Take Profit: 0.20%, 0.25%, 0.30%, 0.35%, 0.40%
   - Max Hold: 15min, 30min, 45min, 60min, 90min
3. Run on Oct-Dec 2025 validation data using Tier1 models
4. Metrics to optimize: Sharpe ratio, expectation/trade, win rate
5. Document best 3 parameter sets

**Success Criteria:**
- ✅ Test at least 50 parameter combinations
- ✅ Identify parameters that improve Sharpe ratio by ≥20%
- ⚠️ If no improvement: proceed anyway, focus on Path E

### Task 1.2: Validation Framework Infrastructure
**Timeline:** Days 4-7
**Owner:** ML Engineer + Data Engineer
**Deliverable:** `src/ml/validation_framework.py`

**Actions:**
1. Create `TemporalSplitValidator` class:
   ```python
   class TemporalSplitValidator:
       def __init__(self, train_start, train_end, val_start, val_end):
           # Ensure strict temporal separation

       def validate_no_leakage(self, train_data, val_data):
           # Check for temporal overlap
           # Check for data leakage

       def walk_forward_validation(self, model, data, window_months=3):
           # Rolling window validation
   ```

2. Create `DataLeakageDetector` class:
   - Detect temporal overlap
   - Detect feature leakage (future information in features)
   - Detect target leakage (labels contaminated by future)
   - Generate audit reports

3. Create `PerformanceValidator` class:
   - Calculate realistic metrics (win rate, Sharpe, etc.)
   - Include transaction costs in all calculations
   - Generate validation reports with pass/fail criteria

**Success Criteria:**
- ✅ All validation classes implemented
- ✅ Unit tests passing (≥80% coverage)
- ✅ Documentation complete

### Task 1.3: Data Period Definition
**Timeline:** Day 7
**Owner:** Data Engineer
**Deliverable:** `data/ml_training/data_periods_1min.yaml`

**Actions:**
1. Define temporal splits:
   ```yaml
   training:
     start: "2025-01-01"
     end: "2025-09-30"
     bars: 214,103  # 9 months of 1-min data

   validation:
     start: "2025-10-01"
     end: "2025-12-31"
     bars: 75,127  # 3 months - LOCKED, never for training

   test:
     start: "2026-01-01"
     end: "2026-03-31"
     bars: ~75,000  # Future data only
   ```

2. Lock validation data:
   - Create checksum file for validation data
   - Add validation checks to prevent training on it
   - Document in README

**Success Criteria:**
- ✅ Data periods clearly defined and documented
- ✅ Validation data locked with checksums
- ✅ Automated checks prevent leakage

---

## Phase 2: Methodology Foundation (Weeks 2-4)

**Goal:** Establish proper training methodology with temporal splits and realistic targets

### Task 2.1: Realistic Performance Target Definition
**Timeline:** Week 2, Days 1-2
**Owner:** Quantitative Analyst + ML Engineer
**Deliverable:** `data/reports/realistic_performance_targets_1min.md`

**Actions:**
1. Analyze 5-minute baseline performance:
   - Win rate: 51.80%
   - Trades/day: 3.92
   - Expectation/trade: ~$50

2. Adjust for 1-minute characteristics:
   - Higher noise-to-signal ratio → lower win rate (45-50% target)
   - More trading opportunities → higher frequency (5-25 trades/day target)
   - Higher transaction costs → lower expectation/trade ($20-40 target)

3. Define target metrics:
   ```yaml
   primary_targets:
     win_rate: {min: 45, max: 55, unit: percent}
     trades_per_day: {min: 5, max: 25, unit: trades}
     expectation_per_trade: {min: 20, unit: dollars}

   secondary_targets:
     sharpe_ratio: {min: 0.6, max: 1.5}
     profit_factor: {min: 1.3}
     max_drawdown: {max: 1000, unit: dollars}

   red_flags:
     win_rate: "> 70%"  # Unrealistic for 1-min data
     sharpe_ratio: "> 3.0"  # Indicates overfitting
     trades_per_day: "> 30"  # Too frequent, likely noise
   ```

**Success Criteria:**
- ✅ Document with rationale for each target
- ✅ Approved by quantitative analyst
- ✅ Integrated into validation framework

### Task 2.2: Training Data Preparation
**Timeline:** Week 2, Days 3-5
**Owner:** Data Engineer
**Deliverable:** Training data files and metadata

**Actions:**
1. Extract training data: Jan-Sep 2025 ONLY
2. Generate features:
   - HMM features for regime detection
   - Tier1 order flow features
   - Triple-barrier labels with proper temporal alignment
3. Create temporal train/test split:
   - Train: Jan-Aug 2025 (8 months, ~190K bars)
   - Test: September 2025 (1 month, ~24K bars)
   - Validation: Oct-Dec 2025 (LOCKED, separate)
4. Generate metadata:
   - Feature statistics (mean, std, min, max)
   - Label distribution (win rate by regime)
   - Data quality report

**Success Criteria:**
- ✅ Training data: Jan-Sep 2025 ONLY
- ✅ No temporal overlap with validation
- ✅ Data quality >99.9% completeness
- ✅ Metadata documented

### Task 2.3: Regularization & Model Architecture
**Timeline:** Week 3
**Owner:** ML Engineer
**Deliverable:** Updated model training script with regularization

**Actions:**
1. Analyze current Tier1 model overfitting:
   - Check feature importance distribution
   - Analyze training vs validation performance gap
   - Identify overfitting patterns

2. Add regularization to XGBoost:
   ```python
   xgb_params = {
       'max_depth': 4,  # Reduced from default (6)
       'min_child_weight': 5,  # Increased from 1
       'gamma': 0.1,  # Minimum loss reduction
       'subsample': 0.8,  # Prevent overfitting
       'colsample_bytree': 0.8,  # Feature sampling
       'reg_alpha': 0.1,  # L1 regularization
       'reg_lambda': 1.0,  # L2 regularization
       'learning_rate': 0.01,  # Slower learning
       'n_estimators': 500,  # More trees with slower learning
   }
   ```

3. Implement early stopping:
   - Use September 2025 as early stopping set
   - Patience: 50 rounds
   - Monitor validation performance

4. Feature selection:
   - Start with all 17 Tier1 features
   - Remove features with near-zero importance
   - Test reduced feature sets (8, 10, 12 features)

**Success Criteria:**
- ✅ Regularization parameters documented
- ✅ Early stopping implemented
- ✅ Feature importance analysis complete
- ✅ Training script ready

### Task 2.4: Model Retraining v1 (Initial Fix)
**Timeline:** Week 4
**Owner:** ML Engineer
**Deliverable:** Retrained models with proper methodology

**Actions:**
1. Train regime-specific models:
   - Regime 0 model on Jan-Sep 2025 data
   - Regime 1 model on Jan-Sep 2025 data
   - Regime 2 model on Jan-Sep 2025 data
2. Use proper temporal split:
   - Train: Jan-Aug 2025
   - Early stopping: September 2025
3. Apply regularization and early stopping
4. Save to: `models/xgboost/regime_aware_tier1_v2/`
5. Generate metadata:
   - Training/test performance
   - Feature importance
   - Confusion matrices

**Success Criteria:**
- ✅ Models trained on Jan-Sep 2025 ONLY
- ✅ No data leakage
- ✅ Training accuracy: 65-75% (realistic)
- ✅ Test accuracy: 60-70% (close to training, no overfitting)

---

## Phase 3: Model Validation & Iteration (Weeks 5-8)

**Goal:** Validate retrained models on held-out data, iterate if needed

### Task 3.1: Validation on Oct-Dec 2025 (CRITICAL)
**Timeline:** Week 5, Days 1-3
**Owner:** ML Engineer + QA Engineer
**Deliverable:** `data/reports/model_v2_validation_octdec2025.md`

**Actions:**
1. Run complete backtest on Oct-Dec 2025:
   - Use optimized exit parameters from Task 1.1
   - Test on LOCKED validation data
   - Full transaction costs included

2. Generate validation report:
   ```yaml
   period: "Oct-Dec 2025"
   trades_count: actual
   win_rate: actual vs target (45-55%)
   trades_per_day: actual vs target (5-25)
   expectation_per_trade: actual vs target (≥$20)
   sharpe_ratio: actual vs target (≥0.6)
   profit_factor: actual vs target (≥1.3)
   max_drawdown: actual vs target (<$1000)

   trade_distribution:  # Check for temporal clustering
     oct_2025: count
     nov_2025: count
     dec_2025: count
   ```

3. Analyze results:
   - Is performance realistic? (win rate 45-55%, not 90%+)
   - Are trades distributed across all 3 months? (not just October 1st)
   - Do we have sufficient trade frequency? (5-25/day)

**Success Criteria:**
- ✅ Complete validation report
- ✅ Metrics documented
- ✅ No temporal clustering (trades across all months)

### Task 3.2: Go/No-Go Decision Gate 1
**Timeline:** Week 5, Day 4
**Owner:** Project Lead + Quantitative Analyst
**Deliverable:** Go/No-Go decision document

**Decision Criteria:**

**GO (Proceed to Phase 4):**
- ✅ Win rate: 45-55% (realistic)
- ✅ Trades/day: 5-25 (sufficient frequency)
- ✅ Expectation/trade: ≥$20
- ✅ Trades distributed across Oct-Dec (no clustering)
- ✅ Sharpe ratio: ≥0.6
- ✅ Max drawdown: <$1,000

**NO-GO (Iterate or Pivot):**
- ❌ Win rate <45% (poor signal)
- ❌ Trades/day <5 (insufficient frequency)
- ❌ Expectation/trade <$20 (losing after costs)
- ❌ Trades clustered in single month (overfitting)
- ❌ Sharpe ratio <0.6 (poor risk-adjusted returns)

**If NO-GO, choose path:**
- **Option A:** Iterate with different features (2 more weeks)
- **Option B:** Pivot to 3-minute timeframe (4-6 weeks)
- **Option C:** Revert to 5-minute system

**Success Criteria:**
- ✅ Clear decision made
- ✅ Next steps defined

### Task 3.3 (Conditional): Feature Engineering Iteration
**Timeline:** Weeks 6-7 (ONLY if Go/No-Go = Option A)
**Owner:** ML Engineer
**Deliverable:** Improved model v3

**Actions:**
1. Feature analysis:
   - Remove low-importance features
   - Add robust features (if any)
   - Test feature combinations: 8, 10, 12 features

2. Model iteration:
   - Train models with reduced feature set
   - Use same temporal splits (Jan-Sep)
   - Apply stronger regularization if needed

3. Validation:
   - Test on Oct-Dec 2025
   - Compare with v2 results

**Success Criteria:**
- ✅ Improved metrics vs v2
- ✅ Pass Go/No-Go criteria

### Task 3.4 (Conditional): 3-Minute Timeframe Pivot
**Timeline:** Weeks 6-10 (ONLY if Go/No-Go = Option B)
**Owner:** ML Engineer + Data Engineer
**Deliverable:** 3-minute trading system

**Actions:**
1. Generate 3-minute dollar bars from 2025 data
2. Train HMM regime detector on 3-minute data
3. Train Tier1 models on 3-minute data (Jan-Sep)
4. Validate on Oct-Dec 2025
5. Compare with 5-minute baseline

**Success Criteria:**
- ✅ Win rate: 48-55%
- ✅ Trades/day: 3-15
- ✅ Expectation/trade: ≥$30

---

## Phase 4: Walk-Forward Validation & Deployment (Weeks 9-10)

**Goal:** Final validation before production deployment

### Task 4.1: Walk-Forward Validation
**Timeline:** Week 9
**Owner:** ML Engineer
**Deliverable:** `data/reports/walk_forward_validation_1min.md`

**Actions:**
1. Implement rolling window validation:
   - Window 1: Jan-Mar train, Apr test
   - Window 2: Apr-Jun train, Jul test
   - Window 3: Jul-Sep train, Oct test
   - Window 4: Oct-Dec (validation period)

2. Test consistency:
   - Do models perform consistently across windows?
   - Any regime-specific issues?
   - Any temporal degradation?

3. Generate walk-forward report:
   - Performance by window
   - Consistency metrics
   - Robustness assessment

**Success Criteria:**
- ✅ All windows show positive expectation
- ✅ No severe performance degradation
- ✅ Win rate 45-55% across windows

### Task 4.2: Stress Testing
**Timeline:** Week 10, Days 1-2
**Owner:** QA Engineer
**Deliverable:** `data/reports/stress_testing_1min.md`

**Actions:**
1. Test on different market conditions:
   - High volatility periods
   - Low volatility periods
   - Trending markets
   - Ranging markets

2. Black swan scenarios:
   - What if win rate drops to 40%?
   - What if trades/day drops to 3?
   - What if transaction costs increase 50%?

3. Generate stress test report:
   - Best case, worst case, expected case
   - Break-even analysis
   - Risk assessment

**Success Criteria:**
- ✅ Stress scenarios documented
- ✅ System survives worst-case scenarios
- ✅ Risk mitigation strategies defined

### Task 4.3: Final Go/No-Go Decision
**Timeline:** Week 10, Day 3
**Owner:** Project Lead
**Deliverable:** Final deployment decision

**Decision Checklist:**

**✅ READY FOR DEPLOYMENT:**
- [ ] Training methodology fixed (proper temporal splits)
- [ ] Validated on held-out Oct-Dec 2025 data
- [ ] Walk-forward validation passed
- [ ] Stress testing completed
- [ ] Win rate: 45-55%
- [ ] Trades/day: 5-25
- [ ] Expectation/trade: ≥$20
- [ ] Sharpe ratio: ≥0.6
- [ ] Max drawdown: <$1,000
- [ ] Exit parameters optimized
- [ ] Realistic performance targets met
- [ ] No red flags detected

**❌ NOT READY FOR DEPLOYMENT:**
- Revert to 5-minute system
- Document lessons learned
- Archive 1-minute codebase
- Consider if 1-minute timeframe is viable

**Success Criteria:**
- ✅ Clear deployment decision
- ✅ Deployment plan ready (if GO)
- ✅ Rollback plan ready (if needed)

### Task 4.4: Paper Trading Deployment (Conditional)
**Timeline:** Week 10, Days 4-7 (ONLY if Final Go/No-Go = GO)
**Owner:** DevOps Engineer
**Deliverable:** Paper trading system running

**Actions:**
1. Update config.yaml with v2 (or v3) model paths
2. Deploy to paper trading environment
3. Monitor for 4 weeks:
   - Daily performance reports
   - Weekly analysis
   - Compare expected vs actual performance
4. Decision point: Scale to live or rollback

**Success Criteria:**
- ✅ Paper trading deployed without issues
- ✅ Monitoring systems active
- ✅ Performance within 20% of expected

---

## Risk Mitigation

### High-Risk Areas

**1. 1-Minute Data May Not Have Sufficient Signal**
- **Risk:** After all fixes, win rate still <45%
- **Mitigation:** Go/No-Go decision gates at weeks 5 and 10
- **Fallback:** Pivot to 3-minute or revert to 5-minute

**2. Overfitting Persists Despite Fixes**
- **Risk:** Models still overfit to training period
- **Mitigation:** Strong regularization, early stopping, held-out validation
- **Fallback:** Reduce model complexity, try 3-minute timeframe

**3. Transaction Costs Eat Profits**
- **Risk:** High trade frequency makes system unprofitable after costs
- **Mitigation:** Include costs in all analysis, optimize for expectation/trade
- **Fallback:** Increase minimum probability threshold to reduce frequency

**4. Time and Resource Overrun**
- **Risk:** 10-week plan extends to 12-14 weeks
- **Mitigation:** Weekly progress reviews, hard decision gates
- **Fallback:** Revert to 5-minute system at any decision gate

---

## Success Metrics

### Phase Completion Criteria

**Phase 1 (Week 1):**
- ✅ Exit parameter optimization complete
- ✅ Validation framework infrastructure built
- ✅ Data periods defined and locked

**Phase 2 (Weeks 2-4):**
- ✅ Realistic targets defined
- ✅ Training data prepared (Jan-Sep 2025)
- ✅ Model v2 retrained with proper methodology

**Phase 3 (Weeks 5-8):**
- ✅ Validation on Oct-Dec 2025 complete
- ✅ Go/No-Go decision 1 made
- ✅ Either improved model v3 OR pivot to 3-minute OR revert

**Phase 4 (Weeks 9-10):**
- ✅ Walk-forward validation complete
- ✅ Stress testing complete
- ✅ Final deployment decision made
- ✅ Either deployed to paper trading OR reverted to 5-minute

### Overall Success

**✅ PROJECT SUCCESS:**
- Deployed 1-minute system to paper trading
- Validated performance on held-out data
- Realistic performance metrics (win rate 45-55%, trades 5-25/day)
- System robust across market conditions

**❌ PROJECT FAILED (but learned):**
- 1-minute timeframe not viable for this strategy
- Reverted to 5-minute system
- Lessons learned documented
- 3-minute timeframe identified as alternative

---

## Resource Requirements

### Personnel
- **ML Engineer:** Full-time (10 weeks)
- **Data Engineer:** Part-time (2 weeks)
- **QA Engineer:** Part-time (2 weeks)
- **Quantitative Analyst:** Part-time (1 week)
- **Project Lead:** Part-time (ongoing)

### Compute Resources
- **Training:** ~8 hours per model training (3 models × 2 iterations = ~48 hours)
- **Validation:** ~4 hours per backtest (10+ backtests = ~40 hours)
- **Total:** ~100 hours of compute time

### Data Storage
- Training data: ~2 GB (Jan-Sep 2025)
- Validation data: ~0.5 GB (Oct-Dec 2025)
- Models: ~100 MB per model (9 models total = ~900 MB)
- Reports and logs: ~500 MB
- **Total:** ~3.5 GB

---

## Communication Plan

### Weekly Progress Reports
Every Friday, send status update:
- Tasks completed this week
- Tasks planned for next week
- Blockers and risks
- Metrics and KPIs

### Decision Gate Notifications
Immediately after Go/No-Go decisions:
- Decision made
- Rationale
- Next steps
- Impact on timeline

### Final Report
At project completion:
- Executive summary
- Methodology changes
- Validation results
- Deployment decision
- Lessons learned
- Recommendations for future work

---

## Conclusion

This action plan provides a **structured, phased approach** to fixing the 1-minute strategy's fundamental issues while allowing for quick wins and early decision points.

**Key Success Factors:**
1. **Proper temporal splits** - No training on validation data
2. **Realistic expectations** - 45-55% win rate, not 90%+
3. **Rigorous validation** - Held-out Oct-Dec 2025 data only
4. **Clear decision gates** - Go/No-Go at weeks 5 and 10
5. **Fallback options** - 3-minute pivot or 5-minute reversion

**Expected Outcomes:**
- **Best case:** Deployed 1-minute system with 5-25 trades/day, 48-52% win rate
- **Middle case:** 3-minute system with 3-15 trades/day, 50-54% win rate
- **Worst case:** Revert to proven 5-minute system (51.80% win rate, 3.92 trades/day)

**Timeline:** 10 weeks total, with 2 major decision gates
**Investment:** ~$50K-100K in personnel costs
**Potential Payoff:** 2-5x increase in trade frequency while maintaining profitability

---

**Next Steps:**
1. ✅ Review and approve this action plan
2. ✅ Assign personnel and resources
3. ✅ Begin Phase 1, Task 1.1 (Exit Parameter Grid Search)
4. ✅ Schedule weekly progress review meetings

**Prepared By:** Claude Code - Tree of Thoughts Analysis
**Based On:** Investigation reports in `data/reports/1min_*.md`
**Tech Spec:** `_bmad-output/implementation_artifacts/tech-spec-1min-dollarbar-migration-2025.md`
