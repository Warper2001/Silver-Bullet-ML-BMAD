# Historical Validation Report

**Generated:** 2026-04-12T00:52:05.264480

## Summary

This report compares the performance of uncalibrated vs calibrated models on historical MNQ data.

## Metrics

### Uncalibrated Model
- **Win Rate:** 47.56%
- **Mean Predicted Probability:** 60.00%
- **Brier Score:** 0.2649
- **Trade Count:** 2479

### Calibrated Model
- **Win Rate:** 47.56%
- **Mean Predicted Probability:** 55.00%
- **Brier Score:** 0.2549
- **Trade Count:** 2479

## Comparison

### Improvements
- **Win Rate Change:** +0.00%
- **Brier Score Improvement:** 0.0099
- **Probability Match (Uncalibrated):** 0.1244
- **Probability Match (Calibrated):** 0.0744

## Success Criteria Validation

| Criterion | Target | Uncalibrated | Calibrated | Status |
|-----------|--------|--------------|------------|--------|
| Brier Score | < 0.15 | 0.2649 | 0.2549 | ❌ FAIL |
| Probability Match | < 0.05 | 0.1244 | 0.0744 | ❌ FAIL |

## Recommendation

⚠️ **ITERATE** on calibration before Phase 2

---

*End of Report*
