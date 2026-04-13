# Extended Paper Trading Plan

**Calibrated ML Model Validation - 4-8 Week Plan**
**Start Date:** 2026-04-13
**End Date:** 2026-06-13 (8 weeks)
**Model:** Calibrated XGBoost from Epic 5 Phase 1

---

## Executive Summary

This document outlines the comprehensive plan for extended paper trading validation using the newly calibrated ML model from Epic 5. The calibrated model addresses the critical March 2025 overconfidence failure and is ready for real-world validation.

**Objective:** Validate that the calibrated ML model performs safely and profitably in live market conditions over 4-8 weeks of paper trading.

**Success Criteria:**
- ✅ Model remains calibrated (Brier score < 0.15)
- ✅ Probability predictions match actual outcomes (< 5% error)
- ✅ Win rate > 55%
- ✅ Maximum drawdown < 12%
- ✅ No recurrence of March 2025 failure mode

---

## 1. Calibrated Model Specifications

### Model Details

**Base Model:** XGBoost Classifier
**Calibration Method:** Platt Scaling / Isotonic Regression
**Training Dataset:** 2-year MNQ futures data
**Validation:** Historical validation with regime-specific testing

**Performance Metrics (Historical):**

| Metric | Uncalibrated | Calibrated | Target | Status |
|--------|--------------|------------|--------|--------|
| Brier Score | 0.28 | 0.14 | < 0.15 | ✅ PASS |
| Probability Match | 15% error | 1% error | < 5% | ✅ PASS |
| Win Rate | 60.7% | 62.0% | > 55% | ✅ PASS |
| March 2025 Loss | -8.56% | Prevented | N/A | ✅ PASS |

**Key Improvements:**
1. Overconfidence eliminated (99.25% → 62% predictions in ranging markets)
2. Probability-reality alignment (1% error vs 15% error)
3. March 2025 failure case would be prevented
4. Regime-specific calibration (trending vs ranging)

---

## 2. Deployment Configuration

### System Components

**1. Calibrated ML Model**
- Path: `data/models/xgboost/1_minute/model_calibrated.joblib`
- Metadata: `data/models/xgboost/1_minute/metadata_calibrated.json`
- Calibration: Loaded automatically with model

**2. Safety System (8 Risk Layers)**
- All active and validated
- Risk validation BEFORE every order
- No safety bypasses possible

**3. Paper Trading Infrastructure**
- Deployment: `./deploy_paper_trading.sh start`
- Monitoring: Streamlit dashboard
- Logging: `logs/paper_trading.log`
- CSV Audit Trails: All risk events tracked

**4. Monitoring & Reporting**
- Daily Performance Reports: Auto-generated
- Weekly Calibration Validation: Brier score tracking
- Monthly Comprehensive Reports: Full analysis

---

## 3. Weekly Plan (8 Weeks)

### Week 1-2: Initial Deployment & Stabilization

**Goals:**
- Deploy calibrated model to paper trading
- Verify system stability
- Validate initial calibration metrics

**Activities:**
- Day 1-2: Deploy calibrated model, verify all components
- Day 3-7: Monitor system stability, track all trades
- Day 8-14: Collect initial performance data

**Monitoring Focus:**
- Brier score (target: < 0.15)
- Probability prediction accuracy (target: < 5% error)
- System uptime (target: > 99%)
- Order success rate (target: > 95%)

**Success Criteria:**
- ✅ System runs 7 days without critical errors
- ✅ At least 20 trades executed
- ✅ Brier score < 0.15 maintained
- ✅ Probability error < 5%

**Deliverables:**
- Week 1-2 Performance Report
- Initial calibration validation results

---

### Week 3-4: Performance Tracking & Calibration Monitoring

**Goals:**
- Track calibrated vs uncalibrated comparison
- Validate calibration in different market conditions
- Monitor for March 2025-like conditions (ranging markets)

**Activities:**
- Daily: Track calibrated model predictions
- Weekly: Compare actual vs predicted win rates
- Monitor: Regime detection (trending vs ranging)
- Validate: Brier score stability

**Key Metrics:**
- Calibrated Win Rate vs Predicted
- Uncalibrated Win Rate (baseline comparison)
- Brier Score over time
- Probability prediction error
- Regime-specific performance

**Success Criteria:**
- ✅ Calibrated win rate matches predictions (±5%)
- ✅ Brier score remains < 0.15
- ✅ No overconfidence detected
- ✅ March 2025 conditions handled correctly

**Deliverables:**
- Week 3-4 Performance Comparison Report
- Calibration Stability Analysis
- Regime Performance Breakdown

---

### Week 5-6: Extended Validation & Stress Testing

**Goals:**
- Collect 4+ weeks of performance data
- Validate against backtest expectations
- Stress test different market conditions

**Activities:**
- Continuous paper trading execution
- Daily performance tracking
- Weekly comprehensive analysis
- Stress test: High volatility periods
- Stress test: Ranging market conditions

**Validation Metrics:**
- Total Trades: > 100 (target)
- Win Rate: > 55%
- Profit Factor: > 1.5
- Maximum Drawdown: < 12%
- Sharpe Ratio: > 2.0

**Calibration Metrics:**
- Brier Score: < 0.15 (critical)
- Probability Error: < 5% (critical)
- No Model Drift: Stable predictions
- March 2025: No similar failure

**Deliverables:**
- Week 5-6 Performance Report
- Backtest vs Paper Trading Comparison
- Stress Test Results
- Model Drift Analysis

---

### Week 7-8: Go/No-Go Decision & Final Reporting

**Goals:**
- Review all 8-week performance data
- Validate against success criteria
- Make final live trading decision
- Generate deployment report

**Activities:**
- Comprehensive performance analysis
- Calibration validation review
- Risk assessment
- Go/No-Go decision framework
- Final deployment report

**Decision Framework:**

| Criterion | Target | Weight | Status |
|-----------|--------|--------|--------|
| Brier Score | < 0.15 | CRITICAL | TBD |
| Probability Match | < 5% | CRITICAL | TBD |
| Win Rate | > 55% | HIGH | TBD |
| Max Drawdown | < 12% | HIGH | TBD |
| March 2025 Recurrence | 0 | CRITICAL | TBD |
| System Stability | > 99% | MEDIUM | TBD |
| Total Trades | > 100 | LOW | TBD |

**Go/No-Go Decision:**
- **PROCEED:** All CRITICAL + 2+ HIGH criteria met
- **CONDITIONAL:** CRITICAL met but need additional monitoring
- **DO NOT PROCEED:** Any CRITICAL criterion failed

**Deliverables:**
- Week 7-8 Final Performance Report
- Go/No-Go Decision Document
- Live Trading Deployment Plan (if approved)
- Risk Assessment Report

---

## 4. Success Metrics & Thresholds

### Critical Success Factors (Must Pass)

**1. Calibration Quality:**
- Brier Score: < 0.15 (must maintain)
- Probability Error: < 5% (must maintain)
- **Failure Action:** Stop trading, investigate model drift

**2. Safety System:**
- Daily Loss Limit: 0 breaches (critical)
- Max Drawdown: < 12% (critical)
- Emergency Stop: Not triggered (critical)
- **Failure Action:** Immediate system halt

**3. March 2025 Failure Prevention:**
- No ranging market overconfidence
- Probability predictions < 70% in ranging markets
- Win rate in ranging markets > 40%
- **Failure Action:** Revert to uncalibrated model

### Important Success Factors (Should Pass)

**4. Performance:**
- Win Rate: > 55%
- Profit Factor: > 1.5
- Sharpe Ratio: > 2.0
- **Action:** Monitor and optimize if below target

**5. Operational:**
- System Uptime: > 99%
- Order Success Rate: > 95%
- Data Completeness: > 99.9%
- **Action:** Investigate and fix issues

### Nice-to-Have Success Factors

**6. Additional Metrics:**
- Total Trades: > 100
- Average Win/Loss Ratio: > 1.0
- Regime Robustness: Consistent across market types
- **Action:** Track but not critical

---

## 5. Risk Management

### Known Risks

**1. Model Drift Risk** (Medium)
- **Description:** Model performance degrades over time
- **Mitigation:** Weekly retraining, drift monitoring
- **Contingency:** Stop trading, retrain model

**2. Regime Change Risk** (Medium)
- **Description:** Market conditions change dramatically
- **Mitigation:** Regime-specific calibration already implemented
- **Contingency:** Monitor regime performance, adjust if needed

**3. Data Quality Risk** (Low)
- **Description:** Poor data quality affects predictions
- **Mitigation:** Gap detection, data validation
- **Contingency:** Pause trading until data quality improves

**4. Technical Risk** (Low)
- **Description:** System failures, API issues
- **Mitigation:** Robust error handling, retry logic
- **Contingency:** Manual intervention, system restart

### Risk Monitoring

**Daily Risk Checks:**
- [ ] Brier score < 0.15
- [ ] Probability error < 5%
- [ ] No safety limit breaches
- [ ] System uptime > 99%
- [ ] Data completeness > 99.9%

**Weekly Risk Reviews:**
- [ ] Model drift analysis
- [ ] Regime performance check
- [ ] Drawdown trend analysis
- [ ] Overconfidence check
- [ ] Compare vs backtest expectations

**Immediate Stop Conditions:**
- Brier score > 0.20 (calibration broken)
- Probability error > 10%
- Daily loss limit breached
- Max drawdown > 15%
- Emergency stop triggered
- March 2025 conditions detected (overconfidence in ranging market)

---

## 6. Monitoring & Reporting

### Daily Reports (Automated)

**Contents:**
- Trades executed today
- Win rate today
- P&L today
- Brier score today
- Probability predictions vs actual outcomes
- Risk events (any limit breaches)
- System uptime

**Format:** CSV + Dashboard

### Weekly Reports (Automated)

**Contents:**
- Week summary (trades, P&L, win rate)
- Calibration metrics (Brier score, probability error)
- Regime performance breakdown
- Comparison to backtest expectations
- Risk assessment
- Recommendations

**Format:** Markdown + Email

### Monthly Reports (Comprehensive)

**Contents:**
- Full performance analysis
- Calibration validation results
- Regime-specific analysis
- Backtest vs Paper Trading comparison
- Risk assessment
- Go/No-Go recommendation
- Next steps

**Format:** PDF + Dashboard

---

## 7. Contingency Plans

### If Calibration Fails (Brier Score > 0.15)

**Immediate Actions:**
1. Stop all trading immediately
2. Investigate cause (model drift? regime change?)
3. Review Epic 5 Phase 2 (drift detection)
4. Decision: Recalibrate model or revert to baseline

### If Safety Breach Occurs

**Immediate Actions:**
1. Emergency stop triggered automatically
2. Review breach details
3. System halt until investigation complete
4. Fix issue before resuming

### If March 2025 Conditions Recur

**Detection:**
- Ranging market detected
- Model overconfidence (> 70% predictions)
- Win rate drops < 40% in ranging markets

**Actions:**
1. Reduce position sizes
2. Increase monitoring frequency
3. Consider regime-specific model
4. Validate calibration effectiveness

---

## 8. Resources & Schedule

### Time Commitment

**Duration:** 8 weeks (56 days)
**Start Date:** 2026-04-13
**End Date:** 2026-06-13

**Milestones:**
- Week 2: Initial stabilization complete
- Week 4: Calibration monitoring complete
- Week 6: Extended validation complete
- Week 8: Go/No-Go decision made

### System Resources

**Required:**
- TradeStation SIM Account (active)
- Calibrated Model (Epic 5 deliverable)
- Paper Trading Infrastructure (Stories 4.1, 4.6)
- Monitoring Dashboard (Story 4.3)
- Risk Management (Story 4.2)

**Optional:**
- Epic 5 Phase 2 (Drift Detection) - can start Week 4
- Epic 5 Phase 3 (Regime-Aware Models) - can start Week 6

---

## 9. Success Definition

### Minimum Viable Success (4 weeks)

**Requirements:**
- System runs stable for 4 weeks
- At least 50 trades executed
- Brier score maintained < 0.15
- Probability error maintained < 5%
- No safety breaches
- Win rate > 50%

**Outcome:** Continue to Week 8 if met, else investigate and decide.

### Complete Success (8 weeks)

**Requirements:**
- All Minimum Viable Success criteria met
- Win rate > 55%
- Profit factor > 1.5
- Max drawdown < 12%
- Sharpe ratio > 2.0
- > 100 trades executed
- Comprehensive validation complete

**Outcome:** APPROVE for live trading deployment.

### Failure Criteria

**Any of the following:**
- Brier score > 0.20 (calibration broken)
- Probability error > 10%
- Max drawdown > 15%
- Daily loss limit breached
- March 2025 failure recurs (overconfidence in ranging markets)
- System uptime < 95%

**Outcome:** STOP trading, investigate, recalibrate or fix issues before retrying.

---

## 10. Post-Validation Next Steps

### If Go Decision (Approved for Live Trading)

**Actions:**
1. Review Story 4.9 (Go/No-Go Decision for Live Trading)
2. Complete Epic 4 remaining stories (4-8, 4-9)
3. Implement Epic 5 Phase 2 (Drift Detection)
4. Live trading deployment planning

### If No-Go Decision (Not Approved)

**Actions:**
1. Root cause analysis of failure
2. Model recalibration (Epic 5 Phase 2)
3. Extended validation (another 4-8 weeks)
4. Or pivot to alternative approach

### If Conditional Go (Approved with Monitoring)

**Actions:**
1. Proceed with increased monitoring
2. Start Epic 5 Phase 2 (Drift Detection)
3. Implement regime-aware models (Epic 5 Phase 3)
4. Review status after 4 more weeks

---

## Appendix: Validation Checklists

### Daily Validation Checklist

- [ ] System running without errors
- [ ] Brier score calculated and < 0.15
- [ ] Probability error < 5%
- [ ] No safety limit breaches
- [ ] Data completeness > 99.9%
- [ ] Daily report generated

### Weekly Validation Checklist

- [ ] 7-day summary report generated
- [ ] Brier score trend analyzed (stable?)
- [ ] Regime performance reviewed
- [ ] Model drift check performed
- [ ] Comparison to backtest expectations
- [ ] Risk assessment updated
- [ ] Recommendations documented

### Go/No-Go Checklist (Week 8)

**Critical (Must Pass):**
- [ ] Brier Score < 0.15 (8-week average)
- [ ] Probability Error < 5% (8-week average)
- [ ] March 2025: No ranging market failure

**Important (Should Pass):**
- [ ] Win Rate > 55%
- [ ] Max Drawdown < 12%
- [ ] System Uptime > 99%

**Nice-to-Have:**
- [ ] Total Trades > 100
- [ ] Profit Factor > 1.5
- [ ] Sharpe Ratio > 2.0

---

**Plan Created:** 2026-04-12
**Plan Status:** READY FOR EXECUTION
**Next Action:** Begin Week 1-2 deployment and stabilization

*End of Plan*
