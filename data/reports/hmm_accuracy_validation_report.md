# HMM Regime Detection - Accuracy Validation Report

**Generated:** 2026-04-12 15:27:26

## Model Configuration

- **Number of Regimes:** 3
- **Covariance Type:** full
- **Training Samples:** 43,325

### Detected Regimes

- **trending_up**: Avg 10.6 bars per period
- **trending_up**: Avg 69.3 bars per period
- **trending_down**: Avg 10.5 bars per period

## Validation Results

### Acceptance Criteria

- **Regime Classification Accuracy:** > 80%
- **Transition Detection Latency:** < 2 days

### March 2025

**Accuracy Score:** 0.390

- **Average Confidence:** 0.983
- **High Confidence Fraction:** 0.967
- **Stability Score:** 0.158
- **Persistence Score:** 0.105
- **Avg Sequence Length:** 10.5 bars
- **Number of Transitions:** 95

### January 2025

**Accuracy Score:** 0.425

- **Average Confidence:** 0.977
- **High Confidence Fraction:** 0.954
- **Stability Score:** 0.238
- **Persistence Score:** 0.123
- **Avg Sequence Length:** 12.3 bars
- **Number of Transitions:** 81

## Summary

- **Average Accuracy Score:** 0.407
- **Average Stability:** 0.198

### Verdict

❌ **FAIL** - Model does not meet accuracy requirements (< 80%)

**Recommendations:**
1. Increase training data size
2. Tune hyperparameters (n_regimes, covariance_type)
3. Add more informative features
4. Consider feature selection to reduce noise

### Next Steps

1. Review regime persistence and stability metrics
2. If accuracy is satisfactory, integrate with MLInference
3. Proceed to Story 5.3.2: Train regime-specific XGBoost models

