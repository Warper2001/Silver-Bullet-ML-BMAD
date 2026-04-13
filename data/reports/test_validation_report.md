# Historical Validation Report

**Generated:** 2026-04-12T00:51:56.166612

## Summary

This report compares the performance of uncalibrated vs calibrated models on historical MNQ data.

## Metrics

### Uncalibrated Model
- **Win Rate:** 60.00%
- **Mean Predicted Probability:** 75.00%
- **Brier Score:** 1.0000
- **Trade Count:** 0

### Calibrated Model
- **Win Rate:** 62.00%
- **Mean Predicted Probability:** 61.00%
- **Brier Score:** 1.0000
- **Trade Count:** 0

## Comparison

### Improvements
- **Win Rate Change:** +2.00%
- **Brier Score Improvement:** 0.0000
- **Probability Match (Uncalibrated):** 0.0000
- **Probability Match (Calibrated):** 0.0000

## Success Criteria Validation

| Criterion | Target | Uncalibrated | Calibrated | Status |
|-----------|--------|--------------|------------|--------|
| Brier Score | < 0.15 | 1.0000 | 1.0000 | ❌ FAIL |
| Probability Match | < 0.05 | 0.0000 | 0.0000 | ✅ PASS |

## Recommendation

⚠️ **ITERATE** on calibration before Phase 2

---

*End of Report*
