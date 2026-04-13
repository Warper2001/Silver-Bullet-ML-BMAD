# Hybrid Regime-Aware System - Deployment Guide

**Date:** 2026-04-13
**Status:** ✅ Ready for Paper Trading Deployment
**Configuration:** Bar-by-Bar Evaluation with 40% Probability Threshold

---

## Overview

This guide explains how to deploy the hybrid regime-aware trading system to paper trading. The system has been integrated into the existing paper trading infrastructure with the following key changes:

### What Changed?

1. **New Pipeline**: `HybridMLPipeline` replaces the old `MLPipeline`
2. **Regime Detection**: HMM-based regime detection (3 regimes)
3. **Model Selection**: Regime-specific models instead of generic
4. **Threshold**: 40% instead of 65%
5. **Evaluation**: Bar-by-bar instead of signal-based

### Expected Performance

- **Trades per day**: 3.92
- **Win rate**: 51.80%
- **Sharpe ratio**: 0.74
- **Max drawdown**: -2.78%
- **Annual return**: +18-30%

---

## System Architecture

### Pipeline Flow

```
Data Orchestrator → DollarBar → HybridMLPipeline → SignalQueue → Risk → Order
                                              ↓
                                        [Regime Detection]
                                              ↓
                                        [Model Selection]
                                              ↓
                                        [Probability Filter]
                                              ↓
                                        [Trade Frequency Control]
```

### Components

1. **HybridMLPipeline** (`src/ml/hybrid_pipeline.py`)
   - Detects market regime using HMM
   - Selects appropriate model (Regime 0, Regime 2, or Generic)
   - Evaluates every 5-minute bar
   - Applies 40% probability threshold
   - Enforces minimum 30 bars between trades

2. **HMM Regime Detector** (`models/hmm/regime_model/`)
   - 3 regimes: trending_up, trending_up_strong, trending_down
   - Features: returns, volatility, volume, ATR
   - Updated every bar

3. **Regime-Specific Models** (`models/xgboost/regime_aware_real_labels/`)
   - Regime 0 model: 97.83% accuracy
   - Regime 2 model: 100.00% accuracy
   - Generic model: 79.30% accuracy (fallback for Regime 1)

---

## Deployment Steps

### Step 1: Verify Models Exist

```bash
# Check HMM regime detector
ls -la models/hmm/regime_model/

# Check regime-specific models
ls -la models/xgboost/regime_aware_real_labels/

# Expected output:
# - xgboost_regime_0_real_labels.joblib
# - xgboost_regime_1_real_labels.joblib (generic)
# - xgboost_regime_2_real_labels.joblib
```

### Step 2: Update Configuration

Configuration is already updated in `config.yaml`:

```yaml
ml:
  probability_threshold: 0.40  # 40% threshold
  model_type: "regime_aware"
  use_regime_aware: true

  regime_detection:
    enabled: true
    model_path: "models/hmm/regime_model/"

  bar_evaluation:
    enabled: true
    min_bars_between_trades: 30
    probability_threshold: 0.40

  exits:
    take_profit_pct: 0.003  # 0.3%
    stop_loss_pct: 0.002  # 0.2%
    max_hold_minutes: 30
```

### Step 3: Start Paper Trading

```bash
# Make sure you have TradeStation OAuth token
cat .access_token

# Start the system
.venv/bin/python start_paper_trading.py
```

### Step 4: Monitor System

The system will log:
- Bars processed
- Regime changes
- Signals generated
- ML predictions
- Order submissions

Example output:
```
🎯 Signal #1: P(Success)=45.00%, Direction=LONG, Regime=0, Model=Regime_0, Latency=12.34ms
```

---

## Deployment Phases

### Phase 1: Setup & Monitoring (Week 1-2)

**Objective**: Verify system stability without trading

**Steps**:
1. Deploy system with 40% threshold
2. Monitor signal generation rate
3. Validate regime distribution
4. Check probability distribution
5. Verify ML predictions are working
6. **NO actual trading** - just monitor

**Monitoring Commands**:
```bash
# Tail the logs
tail -f logs/paper_trading.log

# Check signal generation rate
grep "Signal #" logs/paper_trading.log | wc -l

# Check regime distribution
grep "Regime=" logs/paper_trading.log | cut -d'=' -f3 | sort | uniq -c
```

**Success Criteria**:
- ✅ System runs without crashes
- ✅ 3-4 signals generated per day
- ✅ Regime distribution reasonable
- ✅ Probability distribution matches backtest

### Phase 2: Small Size Paper Trading (Week 3-4)

**Objective**: Validate performance with minimal risk

**Steps**:
1. Enable paper trading with minimal size
2. Start with 1 MNQ contract (or paper account minimum)
3. Track actual vs expected performance

**Expected Metrics**:
- Trades/day: 3-4
- Win rate: 51-52%
- Sharpe ratio: 0.7-0.8
- Max drawdown: <5%

**Monitoring Commands**:
```bash
# Track win rate
grep "Order submitted" logs/paper_trading.log | wc -l
grep "Position closed" logs/paper_trading.log | grep "profit" | wc -l

# Calculate win rate
# (Winning trades / Total trades) * 100
```

**Success Criteria**:
- ✅ Win rate within 5% of 51.80%
- ✅ Trades/day within range 3-5
- ✅ No critical errors

### Phase 3: Full Validation (Week 5-8)

**Objective**: Normal position size, comprehensive validation

**Steps**:
1. Increase to normal position size
2. Continue monitoring metrics
3. Compare backtest vs live performance
4. Adjust if significant deviation (>20%)

**Target Performance**:
- Trades/day: 3-4
- Win rate: 50-53%
- Sharpe ratio: >0.6
- Monthly return: >1%

---

## Troubleshooting

### Issue: Too Few Trades (< 2/day)

**Possible Causes**:
- Market conditions changed
- Model probabilities too low
- Regime distribution shifted

**Solutions**:
1. Check logs for "Bar filtered" messages
2. Verify HMM regime detector working
3. Check feature engineering pipeline
4. Consider lowering threshold to 38%

**Debug Commands**:
```bash
# Check probability distribution
grep "P(Success)=" logs/paper_trading.log | awk '{print $2}' | sort -n

# Check regime distribution
grep "Regime=" logs/paper_trading.log | cut -d'=' -f3 | sort | uniq -c
```

### Issue: Too Many Trades (> 8/day)

**Possible Causes**:
- Threshold too low
- Model overconfident
- High volatility period

**Solutions**:
1. Raise threshold to 42-43%
2. Reduce position size
3. Check model calibration

**Debug Commands**:
```bash
# Count signals per hour
grep "Signal #" logs/paper_trading.log | cut -d' ' -f1 | uniq -c
```

### Issue: Win Rate Below 45%

**Possible Causes**:
- Model degradation
- Regime shift not detected
- Market structure change

**Solutions**:
1. Check drift detection logs
2. Retrain models with recent data
3. Reduce position size temporarily
4. Consider suspending trading

**Debug Commands**:
```bash
# Calculate recent win rate
grep "Position closed" logs/paper_trading.log | tail -100 | grep "profit" | wc -l
```

### Issue: Large Drawdown (> 4%)

**Possible Causes**:
- Multiple consecutive losses
- Large outlier loss
- Stop failures

**Solutions**:
1. **Immediate**: Stop trading
2. Review recent trades
3. Check exit logic execution
4. Verify risk management rules

**Emergency Stop**:
```bash
# Press Ctrl+C in terminal
# Or kill the process
pkill -f start_paper_trading.py
```

---

## Monitoring Dashboard

To visualize system performance, use the Streamlit dashboard:

```bash
# Start dashboard
.venv/bin/streamlit run src/dashboard/streamlit_app.py
```

**Key Metrics to Track**:
- Signal generation rate
- Regime distribution over time
- Probability distribution
- Win rate by regime
- Trade frequency
- Drawdown chart

---

## File Structure

```
Silver-Bullet-ML-BMAD/
├── src/ml/
│   ├── hybrid_pipeline.py          # NEW: Hybrid regime-aware pipeline
│   ├── features.py                 # UPDATED: Added bar-by-bar features
│   └── regime_detection/
│       └── hmm_detector.py         # HMM regime detection
├── models/
│   ├── hmm/regime_model/           # HMM models
│   └── xgboost/regime_aware_real_labels/  # Regime-specific models
├── config.yaml                     # UPDATED: Hybrid configuration
├── start_paper_trading.py          # UPDATED: Uses HybridMLPipeline
└── DEPLOYMENT_CONFIGURATION_40PCT_THRESHOLD.md  # Deployment config
```

---

## Performance Tracking

### Daily Metrics

Track these metrics every day:

```bash
# Create daily report
cat > daily_report.txt << EOF
# Daily Trading Report - $(date +%Y-%m-%d)

## Signals Generated
$(grep "Signal #" logs/paper_trading.log | wc -l)

## Win Rate
$(grep "Position closed" logs/paper_trading.log | grep "profit" | wc -l) / $(grep "Position closed" logs/paper_trading.log | wc -l)

## Regime Distribution
$(grep "Regime=" logs/paper_trading.log | cut -d'=' -f3 | sort | uniq -c)

## Avg Probability
$(grep "P(Success)=" logs/paper_trading.log | awk '{print $2}' | sed 's/%//' | awk '{sum+=$1; count++} END {print sum/count"%"}')
EOF
```

### Weekly Review

At the end of each week, review:

1. **Trade Count**: Did we get 20-25 trades?
2. **Win Rate**: Is it 51-52%?
3. **Sharpe Ratio**: Is it >0.6?
4. **Drawdown**: Is it <5%?

Create weekly report:
```bash
grep "Signal #" logs/paper_trading.log | wc -l
grep "Position closed" logs/paper_trading.log | grep "profit" | wc -l
```

---

## Safety Checks

### Pre-Flight Checklist

Before starting paper trading:

- [ ] HMM model exists in `models/hmm/regime_model/`
- [ ] Regime-specific models exist in `models/xgboost/regime_aware_real_labels/`
- [ ] Config.yaml has probability_threshold: 0.40
- [ ] TradeStation OAuth token valid (`.access_token`)
- [ ] Risk management enabled (8 safety layers)
- [ ] Log directory exists (`logs/`)
- [ ] Data directory exists (`data/processed/dollar_bars/`)

### Runtime Monitoring

Monitor these metrics while system is running:

1. **Memory Usage**: Should be stable
2. **CPU Usage**: Should be <50%
3. **Queue Depth**: Should not fill up
4. **Signal Rate**: Should be 3-4 per day
5. **Regime Stability**: Should not flip-flop

---

## Next Steps

1. ✅ Review this deployment guide
2. ✅ Verify all models exist
3. ✅ Update configuration (done)
4. ⏳ Start paper trading (Week 1-2)
5. ⏳ Monitor for 4-8 weeks
6. ⏳ Validate performance
7. ⏳ Go-live decision

---

## Support

If you encounter issues:

1. Check logs: `tail -f logs/paper_trading.log`
2. Review this guide's troubleshooting section
3. Consult deployment configuration: `DEPLOYMENT_CONFIGURATION_40PCT_THRESHOLD.md`
4. Review backtest results: Check backtest output for expected behavior

---

**Deployment Date**: 2026-04-13
**Validation Period**: 4-8 weeks (paper trading)
**Go-Live Decision**: After successful paper trading validation

**Status**: ✅ Ready for Deployment
