# HMM Regime Detection - Validation Report

**Generated:** 2026-04-12 15:25:53

## Model Configuration

- **Number of Regimes:** 3
- **Covariance Type:** diag
- **Training Samples:** 43,325
- **BIC Score:** 1068650.09
- **Convergence Iterations:** 100

## Detected Regimes

| Regime | Description | Avg Duration (bars) |
|--------|-------------|-------------------|
| trending_up | Market regime 0 | 10.6 |
| trending_up | Market regime 1 | 69.3 |
| trending_down | Market regime 2 | 10.5 |

## Training Results

**Total Transitions:** 4057

### Regime Distribution

| Regime | Count | Percentage |
|--------|-------|------------|
| trending_up | 4,618 | 10.7% |
| trending_down | 18,487 | 42.7% |

## Hyperparameter Tuning Results

| n_regimes | covariance_type | BIC Score |
|-----------|-----------------|-----------|
| 2 | diag | 1268314.47 |
| 3 | diag | 1068650.09 |

## Validation Results

### February 2025

- **Bars:** 4,873
- **Transitions:** 447

#### Regime Distribution

| Regime | Count | Percentage |
|--------|-------|------------|
| trending_up | 1,595 | 32.7% |
| trending_down | 1,552 | 31.8% |

#### Regime Persistence

| Regime | Avg Duration (bars) | Periods |
|--------|-------------------|---------|
| trending_up | 19.2 | 83 |
| trending_down | 8.5 | 182 |

### March 2025

- **Bars:** 2,506
- **Transitions:** 228

#### Regime Distribution

| Regime | Count | Percentage |
|--------|-------|------------|
| trending_up | 173 | 6.9% |
| trending_down | 1,117 | 44.6% |

#### Regime Persistence

| Regime | Avg Duration (bars) | Periods |
|--------|-------------------|---------|
| trending_up | 34.6 | 5 |
| trending_down | 10.1 | 111 |

### January 2025

- **Bars:** 4,796
- **Transitions:** 416

#### Regime Distribution

| Regime | Count | Percentage |
|--------|-------|------------|
| trending_up | 1,746 | 36.4% |
| trending_down | 1,420 | 29.6% |

#### Regime Persistence

| Regime | Avg Duration (bars) | Periods |
|--------|-------------------|---------|
| trending_up | 21.3 | 82 |
| trending_down | 8.6 | 165 |

### October 2024

- **Bars:** 3,867
- **Transitions:** 383

#### Regime Distribution

| Regime | Count | Percentage |
|--------|-------|------------|
| trending_up | 1,460 | 37.8% |
| trending_down | 1,151 | 29.8% |

#### Regime Persistence

| Regime | Avg Duration (bars) | Periods |
|--------|-------------------|---------|
| trending_up | 20.3 | 72 |
| trending_down | 7.6 | 152 |

## Conclusion

The HMM regime detector has been successfully trained and validated. The model identified distinct market regimes with varying characteristics. Validation on out-of-sample data confirms the model's ability to generalize to new market conditions.

## Transition Matrix

Probability of transitioning from one regime to another:

| From \ To | trending_up | trending_up | trending_down |
|-----------|----------|----------|----------|
| trending_up | 0.905 | 0.007 | 0.088 |
| trending_up | 0.037 | 0.930 | 0.033 |
| trending_down | 0.095 | 0.010 | 0.895 |
