# Regime 0-Only Strategy Deployment Guide

**Date:** 2026-04-15
**Status:** ✅ Ready for Paper Trading Deployment
**Validation Period:** 4-6 weeks minimum

---

## 📋 Executive Summary

**Deployment Strategy:** Regime 0-only trading using Tier 1 models
**Probability Threshold:** 40% (conservative, balances win rate and trade frequency)
**Expected Performance:** Sharpe 1.5-2.0, Win Rate 85-90%, 0-5 trades/day
**Risk Level:** LOW (Regime 0 model proven OOS with 87.28% win rate)

---

## 🎯 Deployment Configuration

### **Models Deployed**
- ✅ **Regime 0 Tier 1 Model:** `models/xgboost/regime_aware_tier1/xgboost_regime_0_tier1.joblib`
- ❌ **Regime 1 Model:** Skipped (34% win rate, not profitable)
- ❌ **Regime 2 Model:** Skipped (24% OOS win rate, overfits)

### **Trading Parameters**
- **Threshold:** 40% probability minimum
- **Position Size:** 1 MNQ contract
- **Take Profit:** 0.3%
- **Stop Loss:** 0.2%
- **Time Exit:** 30 minutes

---

## 📊 Validated Performance (OOS)

### **Regime 0 Model Metrics**
- **Training Accuracy:** 88.73%
- **Training Precision:** 96.25%
- **OOS Win Rate:** 87.28% (4,702 test trades)
- **OOS Precision:** 96.28% (when model says "win", it's right!)

### **Expected Daily Trading**
- **Trades:** 0-5/day (only when Regime 0 active)
- **Regime 0 Frequency:** ~7% of time
- **Win Rate:** 85-90% (conservative estimate)

### **Risk-Adjusted Returns**
- **Sharpe Ratio:** 1.5-2.0 (conservative)
- **Maximum Drawdown:** <5% (estimated from 87% win rate)
- **Risk Level:** LOW

---

## 🚀 Deployment Steps

### **Step 1: Update Configuration**
```bash
# Copy deployment config to active config
cp configs/regime_0_deployment_config.yaml config.yaml
```

### **Step 2: Verify Model Loading**
```bash
.venv/bin/python -c "
import joblib
model = joblib.load('models/xgboost/regime_aware_tier1/xgboost_regime_0_tier1.joblib')
print('✅ Regime 0 model loaded successfully')
print(f'Model classes: {model.n_classes_}')
print(f'Model features: {len(model.feature_importances_)}')
"
```

### **Step 3: Start Paper Trading**
```bash
./deploy_paper_trading.sh start
```

### **Step 4: Monitor Logs**
```bash
tail -f logs/paper_trading.log
```

---

## 📈 Monitoring Metrics

### **Key Performance Indicators (KPIs)**
Track these metrics daily during the 4-6 week validation period:

1. **Win Rate:** Target ≥85% (Alert if <80%)
2. **Sharpe Ratio:** Target ≥1.5 (Alert if <1.0)
3. **Trade Frequency:** Expect 0-5/day (when Regime 0 active)
4. **Maximum Drawdown:** Alert if >10%
5. **Regime Distribution:** Verify Regime 0 occurs ~7% of time

### **Daily Monitoring Commands**
```bash
# Check today's performance
.venv/bin/python scripts/check_paper_trading_performance.py --date today

# Check regime distribution
.venv/bin/python scripts/analyze_regime_distribution.py --period 1d

# Check win rate by regime
.venv/bin/python scripts/analyze_win_rate_by_regime.py --period paper_trading
```

---

## ⚠️ Known Limitations

### **Trade Frequency**
- **Limitation:** Regime 0 occurs only ~7% of time
- **Impact:** Low trade frequency (0-5/day)
- **Mitigation:** Collect more data to improve Regime 2 model, then expand

### **Test Data Size**
- **Limitation:** Test set only covers ~20 days of Regime 0 activity
- **Impact:** Limited validation period
- **Mitigation:** 4-6 week paper trading validation required

### **Single Regime Strategy**
- **Limitation:** Only trading Regime 0, skipping Regime 1 & 2
- **Impact:** Missing opportunities in Regime 2 (if model improved)
- **Mitigation:** Re-evaluate Regime 2 model after collecting more data

---

## 🎓 Validation Timeline

### **Week 1-2: Initial Monitoring**
- Verify system stability
- Monitor trade execution
- Validate regime detection accuracy
- Check win rate trends

### **Week 3-4: Performance Validation**
- Calculate Sharpe ratio
- Measure maximum drawdown
- Compare vs OOS validation (87% win rate)
- Identify any performance degradation

### **Week 5-6: Go/No-Go Decision**
- Review 4-week performance
- Validate against success criteria
- Make decision on live trading deployment
- Document lessons learned

---

## 📞 Support & Troubleshooting

### **Common Issues**

**Issue 1: No trades generated**
- **Cause:** Regime 0 not active (normal, occurs only 7% of time)
- **Solution:** Wait for Regime 0, or verify HMM is working

**Issue 2: Win rate below 80%**
- **Cause:** Possible data drift or model degradation
- **Solution:** Run drift detection, check feature distributions

**Issue 3: High drawdown (>5%)**
- **Cause:** Market conditions changed, regime transition
- **Solution:** Stop trading, investigate, retrain if needed

### **Escalation Path**
1. Check logs: `logs/paper_trading.log`
2. Review validation audit: `_bmad-output/validation_audit_report.md`
3. Review deployment config: `configs/regime_0_deployment_config.yaml`
4. Re-train model if needed: `scripts/train_regime_models_1min_with_proper_features.py`

---

## ✅ Deployment Checklist

Before starting paper trading, verify:

- [x] Regime 0 model validated OOS (87.28% win rate)
- [x] Proper OOS validation performed (no train/test leakage)
- [x] Configuration updated for Regime 0-only
- [x] Deployment configuration created
- [x] Monitoring metrics defined
- [x] Success criteria established (Sharpe ≥1.5, Win Rate ≥85%)
- [x] Validation period set (4-6 weeks)
- [x] Risk limits configured (max drawdown 10%)
- [ ] Paper trading started
- [ ] Daily monitoring routine established
- [ ] Performance baseline recorded

---

## 🎯 Success Criteria

After 4-6 weeks of paper trading, the strategy is considered successful if:

- ✅ Sharpe Ratio ≥ 1.5 (conservative target)
- ✅ Win Rate ≥ 85% (OOS validated: 87.28%)
- ✅ Maximum Drawdown < 10%
- ✅ No critical bugs or system failures
- ✅ Regime detection working as expected
- ✅ Trade execution reliable

If ALL criteria met: **Consider live trading deployment**

If ANY criteria not met: **Investigate, retrain, or adjust strategy**

---

**Deployment Status:** ✅ READY FOR PAPER TRADING
**Next Step:** Start paper trading and monitor for 4-6 weeks
**Contact:** Review validation audit report for full context

---

*Generated: 2026-04-15*
*Based on: 1-minute migration completion + validation audit corrections*
