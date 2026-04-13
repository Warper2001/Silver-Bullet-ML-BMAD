# Regime-Specific Model Comparison Report

**Generated:** 2026-04-12 15:36:23

## Summary

This report compares the performance of a generic XGBoost model vs. regime-specific XGBoost models trained on data subset by HMM-detected market regime.

## Generic Model (Baseline)

- **Samples:** 43,325
- **Accuracy:** 0.5421
- **Precision:** 0.5487
- **Recall:** 0.6346
- **F1 Score:** 0.5886
- **ROC-AUC:** 0.5576

## Regime-Specific Models

| Regime | Samples | Accuracy | Precision | Recall | F1 | ROC-AUC | Improvement |
|--------|---------|----------|-----------|--------|-----|---------|-------------|
| trending_up | 20,220 | 0.5462 | 0.5579 | 0.6768 | 0.6116 | 0.5647 | +0.8% |
| trending_up | 4,618 | 0.6039 | 0.6148 | 0.6529 | 0.6333 | 0.6446 | +11.4% |
| trending_down | 18,487 | 0.5479 | 0.5501 | 0.5364 | 0.5432 | 0.5655 | +1.1% |

## Top Features by Regime

### trending_up

1. **volatility_20**: 0.1270
2. **atr_norm**: 0.1212
3. **volatility_10**: 0.1152
4. **rsi**: 0.1143
5. **trend_strength**: 0.1106

### trending_up

1. **volatility_20**: 0.1217
2. **volatility_10**: 0.1198
3. **rsi**: 0.1175
4. **trend_strength**: 0.1142
5. **atr_norm**: 0.1136

### trending_down

1. **volatility_10**: 0.1186
2. **rsi**: 0.1176
3. **volatility_20**: 0.1162
4. **atr_norm**: 0.1160
5. **returns_10**: 0.1120

## Conclusions

**Average Accuracy Improvement:** +4.4%

✅ Regime-specific models show **improved performance** over the generic model.

### Next Steps

1. Use actual Silver Bullet signal labels instead of synthetic labels
2. Increase training data size per regime
3. Tune hyperparameters separately for each regime
4. Test regime-aware predictions in backtesting

