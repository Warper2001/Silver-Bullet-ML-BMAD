# 1-Minute Dollar Bar Migration Plan

**Objective**: Migrate entire hybrid trading system from 5-minute to 1-minute dollar bars

**Current State**: System trained and optimized for 5-minute dollar bars
**Target State**: System trained and optimized for 1-minute dollar bars

**Key Benefits**:
- 5x more data resolution (1-min vs 5-min)
- More responsive signals
- Better intraday pattern recognition
- Improved risk management with finer granularity

---

## Phase 1: Data Generation & Validation (Days 1-2)

### 1.1 Generate 1-Minute Dollar Bars
- **Input**: Raw MNQ trade data (Dec 2023 - Mar 2026)
- **Output**: `data/processed/dollar_bars/1_minute/mnq_1min_full.csv`
- **Script**: Modify existing dollar bar generation for 1-minute timeframe
- **Target**: ~800K bars (vs ~160K for 5-minute)

### 1.2 Data Quality Validation
- Verify completeness (99.99% target)
- Check for gaps and forward-fill
- Validate OHLC consistency
- Ensure notional values are reasonable

**Deliverables**:
- ✅ Validated 1-minute dollar bar dataset
- ✅ Data quality report

---

## Phase 2: HMM Regime Detection (Days 3-4)

### 2.1 Retrain HMM on 1-Minute Data
- **Current**: HMM trained on 5-minute bars
- **Target**: HMM trained on 1-minute bars (3 regimes)
- **Script**: `train_hmm_regime_detector.py` (modify for 1-min)
- **Output**: `models/hmm/regime_model_1min/hmm_model.joblib`

### 2.2 Regime Feature Engineering
- Adapt HMM features for 1-minute timeframe
- Update lookback periods (shorter windows)
- Validate regime detection accuracy

**Deliverables**:
- ✅ HMM model for 1-minute data
- ✅ Regime validation report

---

## Phase 3: Training Data Generation (Days 5-7)

### 3.1 Generate Silver Bullet Setups
- **Script**: Modify `generate_regime_aware_training_data.py` for 1-min
- **Input**: 1-minute dollar bars + HMM regime predictions
- **Output**: `data/ml_training/regime_aware_1min/`
  - Regime 0 training data
  - Regime 1 training data
  - Regime 2 training data

### 3.2 Feature Engineering
- Adapt 54 features for 1-minute timeframe
- Update rolling windows (shorter periods)
- Recalculate technical indicators

**Deliverables**:
- ✅ Regime-specific training datasets
- ✅ Feature engineering report

---

## Phase 4: Model Training (Days 8-10)

### 4.1 Train Regime-Specific XGBoost Models
- **Script**: `train_regime_models_real_labels.py` (modify for 1-min)
- **Models**:
  - Regime 0 model (trending_up)
  - Regime 1 model (trending_up_strong) - Generic fallback
  - Regime 2 model (trending_down)
  - Generic model (fallback)
- **Output**: `models/xgboost/regime_aware_1min_real_labels/`

### 4.2 Model Validation
- Cross-validation on 1-minute data
- Accuracy targets: Regime 0 >95%, Regime 2 >98%, Generic >75%
- Calibration validation (Brier score <0.15)

**Deliverables**:
- ✅ 4 trained XGBoost models
- ✅ Model validation report

---

## Phase 5: Threshold Optimization (Days 11-12)

### 5.1 Probability Threshold Analysis
- **Script**: `backtest_threshold_sensitivity.py` (modify for 1-min)
- **Test Range**: 30% - 60% thresholds
- **Metrics**: Trade frequency, win rate, Sharpe ratio
- **Target**: 1-20 trades/day with win rate >50%

### 5.2 Parameter Optimization
- Min bars between trades (adjust for 1-min)
- Triple-barrier exits (may need adjustment)
- Position sizing (volatility adaptation)

**Deliverables**:
- ✅ Optimal probability threshold
- ✅ Optimized trading parameters

---

## Phase 6: Backtesting & Validation (Days 13-15)

### 6.1 Comprehensive Backtest
- **Script**: `backtest_bar_by_bar.py` (modify for 1-min)
- **Period**: Dec 2023 - Mar 2026 (28 months)
- **Metrics**:
  - Total trades
  - Win rate
  - Sharpe ratio
  - Max drawdown
  - Annual return

### 6.2 Performance Comparison
- **Compare**: 1-min vs 5-min performance
- **Validate**: Performance improvement or degradation
- **Document**: Key differences and trade-offs

**Deliverables**:
- ✅ Comprehensive backtest report
- ✅ Performance comparison analysis

---

## Phase 7: Configuration Updates (Day 16)

### 7.1 Update System Configuration
- **File**: `config.yaml`
- **Changes**:
  - Timeframe: 5-min → 1-min
  - Min bars between trades: 30 → 150 (2.5 hours in minutes)
  - HMM feature windows: adjust for 1-min
  - Feature engineering: update parameters

### 7.2 Update Code References
- **Files**:
  - `src/ml/hybrid_pipeline.py`
  - `src/ml/features.py`
  - `src/data/orchestrator.py`
- **Changes**: Update any hardcoded 5-minute assumptions

**Deliverables**:
- ✅ Updated configuration files
- ✅ Code changes for 1-minute support

---

## Phase 8: Testing & Deployment (Days 17-18)

### 8.1 Paper Trading Testing
- **Action**: Deploy 1-minute system to paper trading
- **Duration**: 1-2 weeks monitoring
- **Validation**: Compare live performance vs backtest

### 8.2 Production Deployment
- **Action**: Replace 5-minute system with 1-minute system
- **Monitoring**: Enhanced monitoring for 1-minute signals
- **Rollback**: Plan to revert to 5-minute if needed

**Deliverables**:
- ✅ Paper trading validation report
- ✅ Production deployment checklist

---

## Resource Requirements

### Computational
- **CPU**: 8+ cores recommended (for parallel training)
- **RAM**: 16GB+ (1-minute data is 5x larger)
- **Storage**: 10GB+ for models and data
- **Time**: 18 days (can be parallelized to ~12 days)

### Data Storage
- **Current 5-min data**: ~500MB
- **1-min data estimate**: ~2.5GB
- **Models**: 4x larger (more features)

---

## Expected Performance Changes

### Potential Improvements
- **Signal Frequency**: 5-25 trades/day (more responsive)
- **Risk Management**: Better stop-loss timing
- **Pattern Recognition**: Finer pattern detection

### Potential Challenges
- **Noise**: More false signals (need higher threshold)
- **Computational Load**: 5x more data processing
- **Overfitting Risk**: More parameters to tune

### Success Criteria
- ✅ Win rate >50% (vs 51.80% for 5-min)
- ✅ Sharpe ratio >0.7 (vs 0.74 for 5-min)
- ✅ 5-25 trades/day (vs 3.92 for 5-min)
- ✅ Max drawdown <5% (vs 2.78% for 5-min)

---

## Implementation Order

**Critical Path** (must be sequential):
1. 1-min dollar bar generation
2. HMM regime detection retraining
3. Training data generation
4. Model training
5. Threshold optimization
6. Backtesting validation

**Parallel Paths**:
- Configuration updates (can be done alongside Phase 4)
- Documentation updates (can be done throughout)
- Testing infrastructure (can be done alongside Phase 6)

---

## Risk Mitigation

### Technical Risks
- **Data Quality**: 1-min data may have gaps/imperfections
- **Overfitting**: More data points → higher overfitting risk
- **Computational**: May exceed current hardware limits

### Mitigation Strategies
- **Incremental Testing**: Test each phase before proceeding
- **Fallback Plan**: Keep 5-minute system operational
- **Rollback Plan**: Quick revert to 5-minute if needed
- **Resource Planning**: Ensure adequate computational capacity

---

## Next Steps

1. **Confirm**: User approval to proceed with migration
2. **Resource Check**: Verify computational resources available
3. **Data Access**: Confirm access to 1-minute historical data
4. **Timeline**: Set expected completion date
5. **Start**: Begin Phase 1 (Data Generation)

---

**Status**: 🔄 Ready to begin
**Estimated Completion**: 18 days (can be compressed with parallelization)
**Priority**: High (major system enhancement)
