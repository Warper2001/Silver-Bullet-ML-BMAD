# HMM Regime Detection - Accuracy Validation Report

**Generated:** 2026-04-12 15:49:29

## Executive Summary

- **Average Confidence:** 0.979
- **Average Stability:** 0.222
- **Average Duration:** 10.8 bars

### Quality Assessment

✅ **HIGH CONFIDENCE** - Regime detection is confident (> 0.8)

⚠️ **MODERATE STABILITY** - Regime predictions change frequently

✅ **REASONABLE PERSISTENCE** - Regimes last long enough (> 10 bars)

## Model Configuration

- **Number of Regimes:** 3
- **Covariance Type:** full
- **Training Samples:** 43,325
- **BIC Score:** 1068650.09

### Detected Regimes

- **trending_up**: Avg 10.6 bars per period
- **trending_up**: Avg 69.3 bars per period
- **trending_down**: Avg 10.5 bars per period

## Validation Results

### February 2025

**Bars:** 4,873

#### Quality Metrics

- **Avg Confidence:** 0.977
- **High Confidence Fraction:** 0.951
- **Stability Score:** 0.219
- **Avg Duration:** 10.9 bars
- **Transitions:** 447

#### Clustering Quality

- **Silhouette Score:** 0.118
- **Inertia:** 50335.63

#### Regime Distribution

| Regime | Count | Percentage |
|--------|-------|------------|
| trending_up | 1,595 | 32.7% |
| trending_down | 1,552 | 31.8% |

### March 2025

**Bars:** 2,506

#### Quality Metrics

- **Avg Confidence:** 0.986
- **High Confidence Fraction:** 0.973
- **Stability Score:** 0.221
- **Avg Duration:** 10.9 bars
- **Transitions:** 228

#### Clustering Quality

- **Silhouette Score:** 0.292
- **Inertia:** 21706.51

#### Regime Distribution

| Regime | Count | Percentage |
|--------|-------|------------|
| trending_up | 173 | 6.9% |
| trending_down | 1,117 | 44.6% |

### January 2025

**Bars:** 4,796

#### Quality Metrics

- **Avg Confidence:** 0.980
- **High Confidence Fraction:** 0.961
- **Stability Score:** 0.218
- **Avg Duration:** 11.5 bars
- **Transitions:** 416

#### Clustering Quality

- **Silhouette Score:** 0.082
- **Inertia:** 51767.96

#### Regime Distribution

| Regime | Count | Percentage |
|--------|-------|------------|
| trending_up | 1,746 | 36.4% |
| trending_down | 1,420 | 29.6% |

### October 2024

**Bars:** 3,867

#### Quality Metrics

- **Avg Confidence:** 0.976
- **High Confidence Fraction:** 0.952
- **Stability Score:** 0.229
- **Avg Duration:** 10.1 bars
- **Transitions:** 383

#### Clustering Quality

- **Silhouette Score:** 0.077
- **Inertia:** 41470.61

#### Regime Distribution

| Regime | Count | Percentage |
|--------|-------|------------|
| trending_up | 1,460 | 37.8% |
| trending_down | 1,151 | 29.8% |

## Conclusions

### Quality Assessment

❌ **LOW QUALITY** - Regime detection needs improvement

### Recommendations

2. **Improve stability** - Consider using fewer regimes or different covariance
4. **Validate with business logic** - Confirm regimes make sense from trading perspective
5. **Monitor in production** - Track regime stability and model performance over time

### Next Steps

1. Review regime characteristics and assign meaningful labels
2. Validate regime-specific model performance (Story 5.3.5)
3. Complete historical validation (Story 5.3.6)

