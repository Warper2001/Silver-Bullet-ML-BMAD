# Historical Validation Report

**Generated:** 2026-06-03T17:53:54.297193

## Summary

This report compares the performance of uncalibrated vs calibrated models on historical MNQ data.

## Metrics

### Uncalibrated Model
- **Win Rate:** 49.36%
- **Mean Predicted Probability:** 60.00%
- **Brier Score:** 0.2613
- **Trade Count:** 15096

### Calibrated Model
- **Win Rate:** 49.36%
- **Mean Predicted Probability:** 55.00%
- **Brier Score:** 0.2531
- **Trade Count:** 15096

## Comparison

### Improvements
- **Win Rate Change:** +0.00%
- **Brier Score Improvement:** 0.0081
- **Probability Match (Uncalibrated):** 0.1064
- **Probability Match (Calibrated):** 0.0564

## Success Criteria Validation

| Criterion | Target | Uncalibrated | Calibrated | Status |
|-----------|--------|--------------|------------|--------|
| Brier Score | < 0.15 | 0.2613 | 0.2531 | ❌ FAIL |
| Probability Match | < 0.05 | 0.1064 | 0.0564 | ❌ FAIL |

## Recommendation

⚠️ **ITERATE** on calibration before Phase 2

---

*End of Report*
