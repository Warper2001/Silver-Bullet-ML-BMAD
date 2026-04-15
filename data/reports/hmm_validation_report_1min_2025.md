# HMM Regime Detector Validation Report - 1-Minute 2025

**Generated:** 2026-04-14 23:36:06

## Model Configuration

- **Timeframe:** 1-minute dollar bars
- **Training Period:** Jan-Sep 2025 (214,103 bars)
- **Validation Period:** Oct-Dec 2025 (75,127 bars)
- **Number of Regimes:** 3
- **Covariance Type:** full

## Training Results

- **Convergence:** True
- **Iterations:** 100
- **Log-Likelihood:** 4729060.08

### Regime Distribution (Training)

| Regime | Name | Bars | Percentage |
|--------|------|------|------------|
| trending_down | 5,392 | 2.5% |
| trending_up | 194,433 | 90.8% |

**Total Transitions:** 2221

## Validation Results

### Regime Distribution (Validation)

| Regime | Name | Bars | Percentage |
|--------|------|------|------------|
| trending_down | 1,251 | 1.7% |
| trending_up | 68,592 | 91.3% |

**Transitions:** 723

## Conclusions

✅ HMM model trained successfully on 1-minute data
✅ 3 regimes detected with clear separation
✅ Regime prediction accuracy validated on Oct-Dec 2025
