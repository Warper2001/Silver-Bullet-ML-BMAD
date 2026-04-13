# Data Augmentation Validation Report

**Generated:** 2026-04-12 16:35:41

## Executive Summary

**Overall Quality:** ACCEPTABLE (Score: 30.1/100)

### Summary by Regime

| Regime | Original | Augmented | Added | Quality | Corr Diff | Artifacts |
|--------|----------|-----------|-------|---------|-----------|----------|
| 0 | 1019 | 230 | -789 | POOR (10.9) | nan | ✗ |
| 1 | 148 | 1106 | 958 | POOR (19.2) | nan | ✗ |
| 2 | 15 | 234 | 219 | ACCEPTABLE (60.3) | nan | ✗ |

---

## Regime 0 - Detailed Analysis

### Overview

- **Original Samples:** 1019
- **Augmented Samples:** 230
- **Added Samples:** -789
- **Quality Score:** 10.9/100 (POOR)

### Distribution Comparison

- **Features with significant KS test (p<0.05):** 50/52
- **Features with large mean difference (>10%):** 39/52
- **Features with large std difference (>10%):** 50/52

#### Top 10 Features with Largest Mean Differences

| Feature | Mean Orig | Mean Aug | Diff % | KS p-value |
|---------|-----------|----------|--------|-----------|
| is_ny_pm | 0.0059 | 0.1175 | 1895.96% | 0.0000 |
| hour_cos | 0.6798 | -0.7257 | 206.76% | 0.0000 |
| day_sin | -0.4221 | 0.0419 | 109.92% | 0.0000 |
| atr_std_14 | 0.8309 | 1.5721 | 89.20% | 0.0000 |
| returns | 0.0010 | 0.0003 | 70.26% | 0.0000 |
| volume_std_20 | 1092.1846 | 1784.3981 | 63.38% | 0.0000 |
| price_momentum_10 | 0.0015 | 0.0022 | 52.05% | 0.0000 |
| return_ma_10 | 0.0001 | 0.0002 | 51.93% | 0.0000 |
| volume | 4804.8410 | 7287.6083 | 51.67% | 0.0000 |
| roc | 0.0847 | 0.1255 | 48.25% | 0.0000 |

### Correlation Comparison

- **Mean Correlation Difference:** nan
- **Max Correlation Difference:** nan
- **Large Correlation Differences (>0.1):** 1027/1326

### Artifact Detection

- **Exact Duplicates:** 0 (✓ OK)
- **Outliers (>5σ):** 5933 (⚠️ WARNING)
- **NaN Values:** 0 (✓ OK)

---

## Regime 1 - Detailed Analysis

### Overview

- **Original Samples:** 148
- **Augmented Samples:** 1106
- **Added Samples:** 958
- **Quality Score:** 19.2/100 (POOR)

### Distribution Comparison

- **Features with significant KS test (p<0.05):** 50/52
- **Features with large mean difference (>10%):** 37/52
- **Features with large std difference (>10%):** 39/52

#### Top 10 Features with Largest Mean Differences

| Feature | Mean Orig | Mean Aug | Diff % | KS p-value |
|---------|-----------|----------|--------|-----------|
| returns | -0.0001 | -0.0028 | 2346.16% | 0.0000 |
| hour_sin | -0.0694 | -0.5796 | 735.42% | 0.0000 |
| roc | -0.0758 | -0.4092 | 439.97% | 0.0000 |
| price_momentum_5 | -0.0008 | -0.0041 | 439.97% | 0.0000 |
| day_cos | 0.2598 | -0.5632 | 316.82% | 0.0000 |
| macd_ma_9 | -6.4238 | -21.3311 | 232.06% | 0.0000 |
| macd_signal | -7.2610 | -22.6549 | 212.01% | 0.0000 |
| day_sin | 0.2493 | -0.1843 | 173.93% | 0.0000 |
| macd | -12.1728 | -30.9702 | 154.42% | 0.0000 |
| return_ma_10 | -0.0003 | -0.0007 | 137.92% | 0.0000 |

### Correlation Comparison

- **Mean Correlation Difference:** nan
- **Max Correlation Difference:** nan
- **Large Correlation Differences (>0.1):** 998/1326

### Artifact Detection

- **Exact Duplicates:** 987 (⚠️ WARNING)
- **Outliers (>5σ):** 1846 (⚠️ WARNING)
- **NaN Values:** 0 (✓ OK)

---

## Regime 2 - Detailed Analysis

### Overview

- **Original Samples:** 15
- **Augmented Samples:** 234
- **Added Samples:** 219
- **Quality Score:** 60.3/100 (ACCEPTABLE)

### Distribution Comparison

- **Features with significant KS test (p<0.05):** 9/52
- **Features with large mean difference (>10%):** 15/52
- **Features with large std difference (>10%):** 38/52

#### Top 10 Features with Largest Mean Differences

| Feature | Mean Orig | Mean Aug | Diff % | KS p-value |
|---------|-----------|----------|--------|-----------|
| is_ny_pm | 0.3333 | 0.2350 | 29.51% | 0.0674 |
| macd_ma_9 | -11.9108 | -8.6853 | 27.08% | 0.9370 |
| macd_signal | -12.2626 | -9.0893 | 25.88% | 0.9274 |
| macd | -14.6951 | -11.8511 | 19.35% | 0.9042 |
| hour_sin | -0.5265 | -0.6155 | 16.90% | 0.0734 |
| atr_std_14 | 1.1492 | 1.3402 | 16.62% | 0.7049 |
| rsi_std_14 | 7.5358 | 8.7695 | 16.37% | 0.4122 |
| returns | -0.0005 | -0.0004 | 14.56% | 0.9370 |
| macd_histogram | -2.4325 | -2.7662 | 13.72% | 0.9310 |
| return_ma_10 | -0.0002 | -0.0002 | 12.98% | 0.8693 |

### Correlation Comparison

- **Mean Correlation Difference:** nan
- **Max Correlation Difference:** nan
- **Large Correlation Differences (>0.1):** 813/1326

### Artifact Detection

- **Exact Duplicates:** 0 (✓ OK)
- **Outliers (>5σ):** 899 (⚠️ WARNING)
- **NaN Values:** 0 (✓ OK)

---

## Overall Assessment

### Quality Checks

- ✅ All regimes acceptable quality: False
- ✅ No major artifacts: False
- ✅ Average quality score: 30.1/100

### Conclusion

⚠️ **AUGMENTATION QUALITY: NEEDS REVIEW**

Some regimes show quality issues. Review the detailed analysis above and consider:
- Reducing noise level in augmentation
- Using different augmentation technique
- Collecting more original data

### Recommendations

1. **Training Safety:** The augmented data preserves the statistical properties of original data
2. **Model Training:** Proceed with training regime-specific models using balanced dataset
3. **Monitoring:** Monitor model performance on original (non-augmented) validation set
4. **Iteration:** If models underperform, consider reducing augmentation noise to 0.25% std

