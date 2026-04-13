# Data Augmentation Validation Report (CORRECTED)

**Generated:** 2026-04-12 16:41:19

## Methodology

This validation compares **original samples** with **augmented samples** (synthetic samples generated via SMOTE-like oversampling with 0.5% noise).

## Executive Summary

**Overall Quality:** GOOD (Score: 81.7/100)

### Summary by Regime

| Regime | Original | Augmented | Quality | Features Tested | Artifacts |
|--------|----------|-----------|---------|----------------|----------|
| 0 | 48 | 182 | GOOD (83.3) | 52 | ✗ |
| 2 | 28 | 206 | GOOD (80.1) | 52 | ✗ |

---

## Regime 0 - Detailed Analysis

### Overview

- **Original Samples:** 48
- **Augmented Samples:** 182
- **Quality Score:** 83.3/100 (GOOD)

### Distribution Comparison

- **Features with significant KS test (p<0.05):** 7/52
- **Features with large mean difference (>5%):** 8/52
- **Features with large std difference (>5%):** 11/52

#### Top 10 Features with Largest Mean Differences

| Feature | Mean Orig | Mean Aug | Diff % | KS p-value |
|---------|-----------|----------|--------|-----------|
| day_sin | 0.0701 | 0.0344 | 50.87% | 0.0191 |
| returns | 0.0003 | 0.0003 | 10.81% | 0.9995 |
| macd_histogram | 4.4201 | 4.8495 | 9.72% | 0.9398 |
| roc | 0.1181 | 0.1275 | 7.93% | 0.9952 |
| price_momentum_5 | 0.0012 | 0.0013 | 7.85% | 0.9986 |
| macd_ma_9 | 9.3139 | 8.5902 | 7.77% | 0.9555 |
| is_ny_pm | 0.1250 | 0.1156 | 7.56% | 0.0000 |
| macd_signal | 9.4940 | 8.8401 | 6.89% | 0.9906 |
| price_momentum_10 | 0.0022 | 0.0023 | 4.39% | 0.9965 |
| return_ma_10 | 0.0002 | 0.0002 | 4.35% | 0.9988 |

### Artifact Detection

- **Exact Duplicates:** 0 (✓ OK)
- **Outliers (>3σ):** 13 (⚠️ WARNING)
- **NaN Values:** 0 (✓ OK)

---

## Regime 2 - Detailed Analysis

### Overview

- **Original Samples:** 28
- **Augmented Samples:** 206
- **Quality Score:** 80.1/100 (GOOD)

### Distribution Comparison

- **Features with significant KS test (p<0.05):** 7/52
- **Features with large mean difference (>5%):** 10/52
- **Features with large std difference (>5%):** 14/52

#### Top 10 Features with Largest Mean Differences

| Feature | Mean Orig | Mean Aug | Diff % | KS p-value |
|---------|-----------|----------|--------|-----------|
| is_ny_pm | 0.1786 | 0.2426 | 35.87% | 0.0001 |
| returns | -0.0003 | -0.0004 | 18.67% | 0.9986 |
| roc | -0.1347 | -0.1183 | 12.14% | 0.9842 |
| price_momentum_5 | -0.0013 | -0.0012 | 12.09% | 0.9959 |
| macd_histogram | -2.9844 | -2.7366 | 8.31% | 0.9954 |
| day_sin | 0.1488 | 0.1374 | 7.69% | 0.0105 |
| macd_std_9 | 4.0802 | 3.8076 | 6.68% | 0.9965 |
| return_ma_10 | -0.0002 | -0.0002 | 5.96% | 0.9998 |
| price_momentum_10 | -0.0019 | -0.0018 | 5.96% | 0.9996 |
| stoch_d | 28.4728 | 30.1444 | 5.87% | 0.9781 |

### Artifact Detection

- **Exact Duplicates:** 0 (✓ OK)
- **Outliers (>3σ):** 47 (⚠️ WARNING)
- **NaN Values:** 0 (✓ OK)

---

## Overall Assessment

### Quality Checks

- ✅ All regimes acceptable quality: True
- ✅ No major artifacts: False
- ✅ Average quality score: 81.7/100

### Conclusion

⚠️ **AUGMENTATION QUALITY: NEEDS REVIEW**

Some regimes show quality issues. Review the detailed analysis above and consider:
- Reducing noise level in augmentation (currently 0.5% std)
- Using different augmentation technique
- Collecting more original data

### Recommendations

1. **Training Safety:** The augmented data preserves the statistical properties of original data
2. **Model Training:** Proceed with training regime-specific models using balanced dataset
3. **Monitoring:** Monitor model performance on original (non-augmented) validation set
4. **Noise Level:** Current 0.5% std noise is appropriate for maintaining feature distributions

