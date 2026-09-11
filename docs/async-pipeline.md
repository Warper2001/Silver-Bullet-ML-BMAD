# Infrastructure (`src/`): Async Pipeline

Relocated verbatim from `CLAUDE.md` on 2026-09-11. It is loaded on demand: `AGENTS.md` points here when work touches the async pipeline. As of 2026-09-11 none of this pipeline is wired to a live bot. The live bots are the systemd `trader-*` units, and `AGENTS.md` lists the `src/` modules they do import.

This code exists and has tests, but it is **not connected to the current paper-trading system**. It represents the original architecture design and may be useful for future development.

## Intended Architecture

The `src/` pipeline was designed as an async queue-based system:

1. **Data Pipeline** (`src/data/`) — TradeStation WebSocket → DollarBar transformer → Validator → Gap Detector → Persistence
2. **Detection Pipeline** (`src/detection/`) — MSS, FVG, liquidity sweep pattern detection → `SilverBulletSetup` events
3. **ML Pipeline** (`src/ml/`) — Feature engineering → XGBoost inference → probability filter (default P ≥ 0.65)
4. **Execution Pipeline** (`src/execution/`) — Position sizing, order submission, triple-barrier exits

## Key Infrastructure Components

**ML Pipeline** (`src/ml/pipeline.py`):
- `MLInference`: XGBoost model for predicting trade success probability
- `FeatureEngineer`: Generates features from `SilverBulletSetup` + historical context
- `SignalFilter`: Probability threshold (default 0.65)
- `DriftDetector`: Monitors model degradation
- `WalkForwardOptimizer`: Weekly retraining

Models are stored in `models/xgboost/5_minute/` with joblib serialization.

**Silver Bullet Detection** (`src/detection/silver_bullet_detector.py`):
- Requires confluence of MSS + FVG + Liquidity Sweep within `max_bar_distance` (default 10 bars)
- Maintains event histories (max 50 events each)
- Publishes `SilverBulletSetup` events

**Data Pipeline Orchestrator** (`src/data/orchestrator.py`):
- WebSocket client for TradeStation
- DollarBar transformer ($50M notional threshold, configurable in `config.yaml`)
- Validator (99.99% completeness target)
- GapDetector with forward-fill
- Persistence to `data/processed/dollar_bars/`

**Pydantic Models** (`src/data/models.py`):
- `MarketData`, `DollarBar`, `SilverBulletSetup`
- `FVGEvent`, `MSSEvent`, `LiquiditySweepEvent`

## Infrastructure Configuration

- **`config.yaml`**: System parameters (risk limits, ML thresholds, data completeness targets)
- **`.env`**: TradeStation API credentials (NEVER commit actual credentials)
- **`pyproject.toml`**: Poetry dependencies

Key config values:
- `ml.probability_threshold`: default 0.65
- `risk.daily_loss_limit`: default $500
- `risk.max_drawdown_percent`: default 12%
- `data.dollar_bar_threshold`: default $50,000,000

## Demo / Legacy Scripts

```bash
# Generate test data for src/ pipeline demo
.venv/bin/python generate_test_data.py

# Simple backtest (src/ pipeline demo)
.venv/bin/python simple_backtest.py
```

## Model Training (Infrastructure)

To train/retrain the `src/ml/` models:
```bash
# Generate training data from historical dollar bars
.venv/bin/python generate_ml_training_data.py

# Train XGBoost model with walk-forward optimization
.venv/bin/python train_meta_model.py
```

Models saved to `models/xgboost/5_minute/model.joblib` with:
- `preprocessor.pkl` — Feature preprocessing pipeline
- `metadata.json` — Model performance metrics
- `threshold.json` — Optimal probability threshold
