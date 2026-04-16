# 1-Minute Model Training Data

**Status:** 🔒 **LOCKED** - Validation data protected with checksum
**Last Updated:** 2026-04-16
**Owner:** 1-Minute Strategy Tuning Project

---

## 🚨 CRITICAL WARNINGS

⚠️ **NEVER train models on validation data (Oct-Dec 2025)**
⚠️ **ALL temporal splits must be strictly enforced**
⚠️ **Any model trained on validation period is INVALID**

---

## Data Periods

### Training Data: Jan-Sep 2025
- **File:** `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv` (filtered)
- **Period:** 2025-01-01 to 2025-09-30
- **Bars:** ~214,103 (9 months)
- **Purpose:** Model training ONLY
- **Split:** Jan-Aug (train), Sep (early stopping)

### Validation Data: Oct-Dec 2025 🔒
- **File:** `data/ml_training/validation_octdec2025_LOCKED.csv`
- **Period:** 2025-10-01 to 2025-12-31
- **Bars:** ~75,127 (3 months)
- **Checksum:** `5bed1cd1adeadc91bb907b6b83d6f600caaef92c936d730a027841cb710dd941`
- **Purpose:** Held-out validation ONLY
- **Status:** LOCKED - NEVER use for training

### Test Data: Jan-Mar 2026
- **Period:** 2026-01-01 to 2026-03-31
- **Status:** Future data - not yet available
- **Purpose:** Final validation before deployment

---

## Validation Checks

### 1. Temporal Separation Check
```python
from src.ml.validation_framework import TemporalSplitValidator

validator = TemporalSplitValidator(
    train_start='2025-01-01',
    train_end='2025-09-30',
    val_start='2025-10-01',
    val_end='2025-12-31'
)

results = validator.validate_no_leakage(train_data, val_data)
assert results['passed'] is True
```

### 2. Data Leakage Check
```python
from src.ml.validation_framework import DataLeakageDetector

detector = DataLeakageDetector()

# Check for temporal leakage
feature_windows = {
    'sma_50': 50,
    'rsi_14': 14,
    # ... other features
}
results = detector.detect_temporal_leakage(df, feature_windows)
assert results['leakage_detected'] is False
```

### 3. Checksum Verification
```bash
# Verify validation data integrity
sha256sum data/ml_training/validation_octdec2025_LOCKED.csv

# Should match:
# 5bed1cd1adeadc91bb907b6b83d6f600caaef92c936d730a027841cb710dd941
```

### 4. Train-on-Validation Prevention
```python
# Automated check in training scripts
def validate_training_period(df):
    """Ensure training data doesn't include validation period."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Check for October-December 2025 dates
    invalid_dates = df[
        (df['timestamp'] >= '2025-10-01') &
        (df['timestamp'] <= '2025-12-31')
    ]

    if len(invalid_dates) > 0:
        raise ValueError(
            f"❌ CRITICAL: Training data includes {len(invalid_dates)} "
            "bars from validation period (Oct-Dec 2025)! "
            "This causes data leakage and invalidates the model."
        )

    print("✅ Training period validated - no validation data present")
```

---

## Realistic Performance Targets

### Primary Metrics (Must Meet)
- **Win Rate:** 45-55% (realistic for 1-minute data)
- **Trades/Day:** 5-25 (sufficient frequency)
- **Expectation/Trade:** ≥$20 (profitable after costs)

### Secondary Metrics (Should Meet)
- **Sharpe Ratio:** ≥0.6 (risk-adjusted returns)
- **Profit Factor:** ≥1.3 (winners vs losers)
- **Max Drawdown:** ≤$1,000 (manageable risk)

### Red Flags (Must Avoid)
- ❌ Win rate >70% (unrealistic, indicates overfitting)
- ❌ Sharpe ratio >3.0 (indicates data leakage)
- ❌ Trades/day >30 (too frequent, likely noise)
- ❌ Zero trades (model broken or too conservative)

---

## Training Procedure

### Step 1: Load Training Data
```python
import pandas as pd

# Load full 2025 data
df = pd.read_csv('data/processed/dollar_bars/1_minute/mnq_1min_2025.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter to Jan-Sep 2025 ONLY
train_df = df[
    (df['timestamp'] >= '2025-01-01') &
    (df['timestamp'] <= '2025-09-30')
].copy()

# Validate
validate_training_period(train_df)

print(f"✅ Training data: {len(train_df):,} bars")
```

### Step 2: Create Temporal Train/Test Split
```python
# Train: Jan-Aug 2025
train_data = train_df[
    train_df['timestamp'] <= '2025-08-31'
]

# Early stopping: Sep 2025
early_stop_data = train_df[
    (train_df['timestamp'] >= '2025-09-01') &
    (train_df['timestamp'] <= '2025-09-30')
]

print(f"✅ Train: {len(train_data):,} bars")
print(f"✅ Early stop: {len(early_stop_data):,} bars")
```

### Step 3: Train Models with Regularization
```python
import xgboost as xgb

params = {
    'max_depth': 4,  # Reduced to prevent overfitting
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'learning_rate': 0.01,
    'n_estimators': 500,
}

model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_early_stop, y_early_stop)],
    early_stopping_rounds=50,
    verbose=False
)
```

### Step 4: Validate on Held-Out Data
```python
# Load LOCKED validation data
val_df = pd.read_csv('data/ml_training/validation_octdec2025_LOCKED.csv')
val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])

# Verify checksum
# (Automated check in training script)

# Run validation
metrics = validate_model(model, val_df)

# Check against targets
assert 45 <= metrics['win_rate'] <= 55, "Win rate outside realistic range"
assert 5 <= metrics['trades_per_day'] <= 25, "Trade frequency issue"
assert metrics['expectation_per_trade'] >= 20, "Not profitable after costs"

print("✅ Model validation passed")
```

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Training on Validation Data
```python
# WRONG - includes October
train_df = df[df['timestamp'] <= '2025-10-15']
```

### ✅ Correct: Stop at September 30
```python
# CORRECT - stops before validation
train_df = df[df['timestamp'] <= '2025-09-30']
```

### ❌ Mistake 2: Using Future Information
```python
# WRONG - features use future data
df['future_return'] = df['close'].shift(-5)  # Look-ahead!
```

### ✅ Correct: Use Only Historical Data
```python
# CORRECT - only uses past data
df['past_return'] = df['close'].pct_change(5)
```

### ❌ Mistake 3: Ignoring Transaction Costs
```python
# WRONG - doesn't include costs
expectation = trades['pnl'].mean()  # Inflated!
```

### ✅ Correct: Include All Costs
```python
# CORRECT - includes costs
cost_per_trade = 2.50 * 5 + 0.50 * 0.25 * 5  # ~$13.75
trades['pnl_after_costs'] = trades['pnl'] - cost_per_trade
expectation = trades['pnl_after_costs'].mean()
```

---

## File Structure

```
data/ml_training/
├── README.md                                # This file
├── data_periods_1min.yaml                   # Configuration
├── validation_octdec2025_LOCKED.csv         # Locked validation data
└── validation_octdec2025_LOCKED.csv.sha256  # Checksum
```

---

## References

- **Investigation Report:** `data/reports/1min_training_methodology_investigation.md`
- **Action Plan:** `data/reports/1min_tuning_action_plan.md`
- **Tech Spec:** `_bmad-output/implementation_artifacts/tech-spec-1min-dollarbar-migration-2025.md`
- **Validation Framework:** `src/ml/validation_framework.py`

---

## Contact

**Project Lead:** 1-Minute Strategy Tuning
**Status:** Active (Phase 1: Week 1)
**Next Review:** End of Week 1 (2026-04-23)

---

**Remember:** The validation process worked correctly by identifying overfitted models. Trust the process, follow the methodology, and validate on held-out data ONLY.
