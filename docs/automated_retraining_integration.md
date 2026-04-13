# Automated Retraining Integration Guide

**Epic 5 Phase 2 - Story 5.2.3 Integration Complete**

This document explains how the automated retraining system is integrated with drift detection to provide fully automated model adaptation.

---

## Overview

The automated retraining system now provides **zero-touch operation**:
1. Drift detection monitors model performance hourly
2. When drift is detected, retraining trigger evaluates conditions
3. If conditions met (severe drift, minimum interval, sufficient data), automated retraining begins
4. New model trained with XGBoost + calibration
5. New model validated against old model
6. If validation passes, new model deployed automatically
7. MLInference loads new model on next inference

**No manual intervention required.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MLInference System                           │
│                                                                   │
│  ┌──────────────┐     ┌───────────────┐     ┌──────────────┐  │
│  │  Drift       │     │  Retraining  │     │  Async       │  │
│  │  Detection   │────▶│  Trigger     │────▶│  Retraining  │  │
│  │  (Hourly)    │     │  (Evaluate)  │     │  (Background) │  │
│  └──────────────┘     └───────────────┘     └──────────────┘  │
│                               │                     │          │
│                               ▼                     ▼          │
│                         Trigger?          Train & Validate    │
│                               │                     │          │
│                               ▼                     ▼          │
│                          All Met?              Deploy?     │
│                               │                     │          │
│                               └──────────┬──────────┘          │
│                                          ▼                     │
│                                    Invalidate Cache           │
│                                          │                     │
│                                          ▼                     │
│                                    New Model Active          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

Add to `config.yaml`:

```yaml
ml:
  # Drift detection (Story 5.2.1)
  drift_detection:
    enabled: true
    check_interval_hours: 1
    rolling_window_hours: 24
    psi:
      bins: 10
      threshold_moderate: 0.2
      threshold_severe: 0.5
    ks_test:
      p_value_threshold: 0.05

  # Automated retraining (Story 5.2.3) - ENABLED
  retraining:
    enabled: true  # Set to true to enable automated retraining
    mode: "auto"  # "auto" or "manual"

    trigger:
      psi_threshold: 0.5  # Severe drift threshold
      ks_p_value_threshold: 0.01  # Highly significant threshold
      min_interval_hours: 24  # Minimum time between retrainings
      min_samples: 1000  # Minimum new samples required

    validation:
      brier_score_max: 0.2  # Maximum Brier score (calibration quality)
      win_rate_min_delta: 0.0  # Minimum win rate change (no degradation)
      feature_stability_threshold: 0.3  # Max feature importance change

    execution:
      async_enabled: true  # Run retraining in background
      timeout_minutes: 60  # Maximum retraining time
      max_concurrent_retrainings: 1  # Prevent overlapping retrainings

    model_versioning:
      backup_count: 5  # Number of old models to keep
      models_dir: "models/xgboost/1_minute"
      metadata_file: "models/model_lineage.json"
```

---

## Usage

### Step 1: Initialize MLInference with Automated Retraining

```python
from src.ml.inference import MLInference
import yaml

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Get retraining configuration
retraining_config = config["ml"]["retraining"]

# Initialize MLInference with automated retraining ENABLED
ml_inference = MLInference(
    model_dir="models/xgboost",
    use_calibration=True,
    enable_automated_retraining=True,  # ENABLE AUTOMATED RETRAINING
    retraining_config=retraining_config
)
```

### Step 2: Initialize Drift Detection

```python
from src.ml.drift_detection import StatisticalDriftDetector

# Load baseline features from model metadata
baseline_features = {...}  # From training data
baseline_predictions = {...}  # From training data

# Initialize drift detector
drift_detector = StatisticalDriftDetector(
    baseline_features=baseline_features,
    baseline_predictions=baseline_predictions,
    feature_names=list(baseline_features.keys())[:10]
)

# Initialize drift detection in MLInference
ml_inference.initialize_drift_detection(
    drift_detector=drift_detector,
    window_hours=24,
    enable_monitoring=True
)
```

### Step 3: Run Periodic Drift Checks

```python
import time
from datetime import datetime, timedelta

# Run drift detection every hour
while True:
    # Check for drift (this will automatically trigger retraining if needed)
    result = ml_inference.check_drift_and_log()

    if result:
        print(f"Drift check completed at {datetime.now()}")
        print(f"  Drift detected: {result['drift_detected']}")
        if result['drift_detected']:
            print(f"  Drifting features: {len(result['drifting_features'])}")
            print(f"  Automated retraining triggered if conditions met")

    # Wait 1 hour before next check
    time.sleep(3600)  # 1 hour = 3600 seconds
```

### Step 4: Monitor Retraining Events

**Retraining decisions are logged to:** `logs/retraining_events/retraining_decisions.csv`

**Columns:**
- `timestamp` - When decision was made
- `trigger` - True/False (was retraining triggered?)
- `justification` - Reason for decision
- `max_psi` - Maximum PSI score
- `ks_p_value` - KS test p-value
- `samples_available` - Number of new samples
- `hours_since_last` - Hours since last retraining

**Example CSV:**
```csv
timestamp,trigger,justification,max_psi,ks_p_value,samples_available,hours_since_last
2026-04-12T14:30:00,True,All checks passed: PSI=0.6500>0.5, 48.0h since last, 5000 samples,0.68,9.7599e-14,5000,48.0
2026-04-13T10:15:00,False,Drift not severe: PSI=0.15, KS p-value=0.12,0.15,0.12,2000,19.75
```

---

## Trigger Conditions

Retraining is **automatically triggered** when **ALL** conditions are met:

1. **Severe Drift**: PSI > 0.5 **OR** KS p-value < 0.01
2. **Minimum Interval**: ≥24 hours since last retraining
3. **Data Availability**: ≥1000 new samples since last training

### Example Scenarios

**Scenario 1: Trigger Retraining**
```
Drift detected: True
Max PSI: 0.65 (> 0.5 threshold) ✅
KS p-value: 9.76e-14 (< 0.01 threshold) ✅
Hours since last: 48 (≥24 hours) ✅
Samples available: 5000 (≥1000) ✅

Decision: TRIGGER RETRAINING ✅
```

**Scenario 2: Skip Retraining (Drift Not Severe)**
```
Drift detected: True
Max PSI: 0.15 (< 0.5 threshold) ❌
KS p-value: 0.12 (> 0.01 threshold) ❌
Hours since last: 48 (≥24 hours) ✅
Samples available: 5000 (≥1000) ✅

Decision: DO NOT TRIGGER - Drift not severe
```

**Scenario 3: Skip Retraining (Too Soon)**
```
Drift detected: True
Max PSI: 0.68 (> 0.5 threshold) ✅
KS p-value: 8.5e-15 (< 0.01 threshold) ✅
Hours since last: 12 (< 24 hours) ❌
Samples available: 5000 (≥1000) ✅

Decision: DO NOT TRIGGER - Too soon since last retraining
```

**Scenario 4: Skip Retraining (Insufficient Data)**
```
Drift detected: True
Max PSI: 0.62 (> 0.5 threshold) ✅
KS p-value: 1.2e-13 (< 0.01 threshold) ✅
Hours since last: 48 (≥24 hours) ✅
Samples available: 500 (< 1000) ❌

Decision: DO NOT TRIGGER - Insufficient data for training
```

---

## Retraining Workflow

When retraining is triggered:

1. **Data Collection** (~5 minutes)
   - Load dollar bars from `data/processed/dollar_bars/1_minute/`
   - Filter to data since last training (minimum 30 days, maximum 90 days)
   - Engineer features using `FeatureEngineer`
   - Create binary labels (close > open)
   - Split train/test (80/20)

2. **Model Training** (~10-20 minutes)
   - Train XGBoost model with proper hyperparameters
   - Apply probability calibration (isotonic regression)
   - Create wrapped model with calibrated predictions

3. **Model Validation** (~2 minutes)
   - Calculate Brier score (< 0.2 required)
   - Calculate win rate delta (≥0.0 required, no degradation)
   - Check prediction distribution
   - Validate against old model performance

4. **Model Deployment** (~1 minute)
   - Save model to `models/xgboost/1_minute/{model_hash}/`
   - Update model lineage metadata
   - **Invalidate MLInference cache**
   - Next inference automatically loads new model

**Total Time:** 20-30 minutes (runs in background, doesn't block inference)

---

## Validation Criteria

New model must pass **ALL** validation checks:

1. **Brier Score < 0.2** (calibration quality)
2. **Win Rate ≥ Old Model** (no degradation)
3. **Prediction Distribution OK** (not degenerate)

If validation fails:
- New model is saved but NOT deployed
- Old model continues to be used
- Failure logged to audit trail
- System continues safely with old model

---

## Safety Features

### 1. Race Condition Prevention
- `asyncio.Lock()` ensures only one retraining at a time
- Multiple concurrent drift detections won't trigger multiple retrainings

### 2. Timeout Protection
- Retraining times out after 60 minutes (configurable)
- Prevents hung tasks from blocking system

### 3. Minimum Interval
- Won't retrain more than once per 24 hours (configurable)
- Prevents excessive retraining and model churn

### 4. Data Availability Check
- Requires 1000+ samples before retraining
- Prevents retraining on insufficient data

### 5. Performance Validation
- New model must match or exceed old model performance
- Prevents deployment of degraded models

### 6. Atomic Deployment
- Model saved and validated BEFORE deployment
- Old model kept as backup
- Rollback capability via `ModelVersioning.rollback_model()`

### 7. Audit Trail
- All trigger decisions logged
- All retraining results logged
- Full traceability of model lineage

---

## Monitoring and Debugging

### Logs

**Drift Detection Logs:** `logs/drift_events/drift_events.csv`
- Every drift check result
- PSI scores, KS p-values
- Drifting features list

**Retraining Decision Logs:** `logs/retraining_events/retraining_decisions.csv`
- Every trigger decision (trigger/skip)
- Justification for decision
- Drift metrics, data availability

**System Logs:** Console output
```python
INFO: RetrainingTrigger initialized: PSI>0.5, KS p-value<0.01
INFO: Evaluating retraining trigger...
INFO: All checks passed: PSI=0.6500>0.5, 48.0h since last, 5000 samples
WARNING: 🚨 RETRAINING TRIGGERED: All checks passed: PSI=0.6500>0.5, 48.0h since last, 5000 samples
INFO: 🚨 Initiating automated retraining...
INFO: Automated retraining task started in background
INFO: Step 1: Collecting new training data...
INFO: Loaded 5000 dollar bars for training
INFO: Step 2: Training new model...
INFO: XGBoost model training completed
INFO: Probability calibration applied (isotonic regression)
INFO: Step 3: Validating model performance...
INFO: ✅ Model validation passed: Brier score=0.1850, win rate delta=+0.0250
INFO: Step 4: Deploying new model...
INFO: Model saved: hash=a1b2c3d4..., file=models/xgboost/1_minute/a1b2c3d4.../model.joblib
INFO: Cleared MLInference model cache
INFO: ✅ Automated retraining completed: old_hash → new_hash
```

### Dashboard

View drift metrics in real-time on the Streamlit dashboard:
- Navigate to "Drift Monitoring" page
- View PSI scores, KS test results
- See historical drift timeline
- Monitor automated retraining events

---

## Troubleshooting

### Issue: Automated Retraining Not Triggering

**Check:**
1. Is `enable_automated_retraining=True`?
2. Is `config.yaml` retraining section present?
3. Is drift detection actually detecting drift?
4. Check `logs/retraining_events/retraining_decisions.csv` for trigger decisions
5. Check system logs for evaluation errors

**Common Causes:**
- Drift not severe enough (PSI < 0.5, KS p-value > 0.01)
- Too soon since last retraining (< 24 hours)
- Insufficient data available (< 1000 samples)

### Issue: Retraining Fails

**Check:**
1. Data quality: Are dollar bars available?
2. Feature engineering: Are features being generated correctly?
3. Model training: Check logs for XGBoost errors
4. Validation: Check Brier score, win rate delta

**Common Causes:**
- Insufficient training data (< 1000 samples)
- Feature engineering failures (NaN, constant features)
- Model overfitting (Brier score too high)
- Calibration failures

### Issue: New Model Not Deployed

**Check:**
1. Did validation pass? (Brier score < 0.2, win rate delta ≥ 0)
2. Was model saved successfully?
3. Was cache invalidation successful? (Check logs for "Cleared MLInference model cache")
4. Is MLInference using the new model? (Check inference logs)

**Common Causes:**
- Validation failed (Brier score too high, win rate degraded)
- Cache invalidation failed
- MLInference not properly initialized with retraining config

---

## Example: End-to-End Workflow

```python
import asyncio
import logging
from datetime import datetime, timedelta

from src.ml.inference import MLInference
from src.ml.drift_detection import StatisticalDriftDetector
import yaml

logging.basicConfig(level=logging.INFO)

async def main():
    # 1. Load configuration
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # 2. Initialize MLInference with automated retraining
    ml_inference = MLInference(
        model_dir="models/xgboost",
        enable_automated_retraining=True,
        retraining_config=config["ml"]["retraining"]
    )

    # 3. Initialize drift detection
    drift_detector = StatisticalDriftDetector(
        baseline_features=baseline_features,
        baseline_predictions=baseline_predictions,
        feature_names=feature_names
    )

    ml_inference.initialize_drift_detection(
        drift_detector=drift_detector,
        window_hours=24
    )

    # 4. Run periodic drift checks (every hour)
    while True:
        # Check for drift (will automatically trigger retraining if needed)
        result = ml_inference.check_drift_and_log()

        if result and result["drift_detected"]:
            print(f"⚠️  Drift detected: {len(result['drifting_features'])} features")
            print(f"   Automated retraining triggered if conditions met")

        # Wait 1 hour before next check
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Summary

✅ **Automated Retraining Integration Complete**

**What's Automated:**
- Drift detection (hourly monitoring)
- Trigger evaluation (automatic decision making)
- Data collection (loads dollar bars automatically)
- Model training (XGBoost + calibration)
- Model validation (performance checks)
- Model deployment (automatic cache invalidation)

**What's NOT Automated:**
- Initial system startup (manual initialization required)
- Configuration changes (manual config file edits)
- Manual override (can disable automated retraining by setting `enabled: false`)

**Production Deployment:**
1. Set `retraining.enabled: true` in config.yaml
2. Initialize MLInference with `enable_automated_retraining=True`
3. Schedule periodic drift checks (cron job, background task)
4. Monitor logs and dashboard
5. System will automatically adapt to regime changes

**Business Value:**
- Eliminates manual monitoring
- Reduces response time from weeks to hours
- Prevents losses from degraded model performance
- Addresses March 2025 failure (delayed retraining)
- Provides zero-touch model adaptation

---

**Last Updated:** 2026-04-12
**Epic:** 5 - ML Training Methodology Overhaul
**Phase:** 2 - Concept Drift Detection
**Stories:** 5.2.1 (Drift Detection), 5.2.2 (Dashboard), 5.2.3 (Retraining Triggers), 5.2.4 (Latency Validation)
