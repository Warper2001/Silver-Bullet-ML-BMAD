# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Environment

**Always use the `.venv` virtual environment** for Python commands:
```bash
.venv/bin/python <script>.py
.venv/bin/python -m pytest tests/
.venv/bin/pip list
```

The project uses Poetry for dependency management, but scripts should be invoked directly through `.venv/bin/python`.

## Project Overview

Hybrid trading system combining ICT (Inner Circle Trader) discretionary concepts with machine learning meta-labeling for regime-adaptive edge in MNQ (Micro E-mini Nasdaq-100) futures trading.

### Core Architecture

The system uses an **async pipeline architecture** with three main stages:

1. **Data Pipeline** (`src/data/`) - Ingests market data via TradeStation WebSocket, transforms to dollar bars, validates completeness, detects gaps, persists to disk
2. **Detection Pipeline** (`src/detection/`) - Identifies ICT patterns (MSS, FVG, liquidity sweeps) and detects Silver Bullet setups (pattern confluence)
3. **ML Pipeline** (`src/ml/`) - Applies feature engineering, predicts success probability, filters by probability threshold
4. **Execution Pipeline** (`src/execution/`) - Position sizing, order submission, triple-barrier exits, position monitoring

### Key Data Flow

```
TradeStation WebSocket → DollarBar Transformer → Validator → Gap Detector → Persistence
                                                                         ↓
                                              DollarBar → Pattern Detectors → SilverBulletSetup
                                                                         ↓
                                              MLPipeline → Feature Engineering → XGBoost Inference
                                                                         ↓
                                              SignalFilter (P > threshold) → ExecutionPipeline
```

### Important Performance Requirements

- Detection latency: < 100ms from pattern receipt to setup publication
- ML inference latency: < 50ms total (features + inference + filtering)
- Dollar bar threshold: $50M notional value (configurable in `config.yaml`)

## Common Development Commands

### Testing
```bash
# Run all tests
.venv/bin/python -m pytest tests/

# Run specific test file
.venv/bin/python -m pytest tests/integration/test_ml_pipeline_integration.py -v

# Run with coverage
.venv/bin/python -m pytest --cov=src tests/

# Run single test
.venv/bin/python -m pytest tests/integration/test_orchestrator_integration.py::TestPipelineEndToEnd::test_pipeline_initialization -v
```

### Code Quality
```bash
# Format code
poetry run black src/ tests/

# Run linting (flake8 + mypy)
poetry run flake8 src/ tests/
poetry run mypy src/

# Run all via Make
make format
make lint
```

### Paper Trading
```bash
# Start paper trading (requires valid TradeStation credentials)
./deploy_paper_trading.sh start

# Monitor logs
tail -f logs/paper_trading.log

# Check status
./deploy_paper_trading.sh status
```

### Demo Mode (No credentials required)
```bash
# Generate test data
.venv/bin/python generate_test_data.py

# Run backtest
.venv/bin/python simple_backtest.py
```

## Source Code Organization

```
src/
├── cli/              # Command-line interfaces (emergency_stop, etc.)
├── dashboard/        # Streamlit dashboard for monitoring
├── data/             # Data ingestion, auth, websocket, models, validation, persistence
├── detection/        # Pattern detection (MSS, FVG, liquidity sweeps, Silver Bullet)
├── execution/        # Order execution, position management, exits
├── ml/               # Machine learning pipeline (features, inference, drift detection)
├── monitoring/       # Health checks, logging, crash recovery, terminal UI
├── research/         # Backtesting, performance metrics, equity curves
└── risk/             # Risk management (drawdown limits, circuit breakers, position sizing)
```

## Key Components

### ML Pipeline (`src/ml/pipeline.py`)

Integrates all ML components:
- **MLInference**: XGBoost model for predicting trade success probability
- **FeatureEngineer**: Generates features from SilverBulletSetup + historical context
- **SignalFilter**: Applies probability threshold (default 0.65 = 65%)
- **DriftDetector**: Monitors model performance degradation
- **WalkForwardOptimizer**: Weekly retraining with walk-forward validation

Models are stored in `models/xgboost/5_minute/` with joblib serialization (not native XGBoost format due to compatibility issues).

### Silver Bullet Detection (`src/detection/silver_bullet_detector.py`)

The Silver Bullet setup requires **confluence of three ICT patterns**:
1. **MSS** (Market Structure Shift) - Break of market structure
2. **FVG** (Fair Value Gap) - Price imbalance between candles
3. **Liquidity Sweep** - Price sweeping liquidity levels

The detector maintains event histories (max 50 events each) and publishes `SilverBulletSetup` events when patterns align within `max_bar_distance` (default 10 bars).

### Data Pipeline Orchestrator (`src/data/orchestrator.py`)

Coordinates the data flow:
- WebSocket client for TradeStation market data
- DollarBar transformer (aggregates trades to $50M notional bars)
- Validator (checks 99.99% completeness target)
- GapDetector (identifies data gaps with forward-fill)
- Persistence (writes to `data/processed/dollar_bars/`)

### Pydantic Models (`src/data/models.py`)

Key data structures:
- `MarketData`: Real-time quote/trade data from TradeStation
- `DollarBar`: Fixed notional value bar (OHLCV + metadata)
- `SilverBulletSetup`: Detected trading setup with entry, stop-loss, take-profit
- `FVGEvent`, `MSSEvent`, `LiquiditySweepEvent`: Pattern detection events

## Configuration

- **config.yaml**: System parameters (risk limits, ML thresholds, data completeness targets)
- **.env**: TradeStation API credentials (NEVER commit actual credentials)
- **pyproject.toml**: Poetry dependencies and tool configurations

Key configuration values:
- `ml.probability_threshold`: Minimum success probability to take trade (default 0.65)
- `risk.daily_loss_limit`: Daily loss limit in USD (default 500)
- `risk.max_drawdown_percent`: Maximum drawdown before trading halt (default 12%)
- `data.dollar_bar_threshold`: Notional value threshold for dollar bars (default 50000000)

## Type Checking

Many modules have mypy errors disabled via `pyproject.toml` overrides:
- `src.data.*` (config, websocket, models, transformation, tradestation_*)
- `src.ml.*` (all ML modules)

These are known issues from the original development. When adding new code, prefer proper type hints over adding more ignore overrides.

## Testing Structure

```
tests/
├── integration/      # End-to-end pipeline tests
│   ├── test_ml_pipeline_integration.py
│   ├── test_orchestrator_integration.py
│   ├── test_gap_detection_integration.py
│   └── ...
└── unit/            # Isolated component tests
```

Integration tests use mock TradeStationAuth to avoid requiring live credentials. Tests are structured as pytest classes with async test methods using `@pytest.mark.asyncio`.

## Important Design Patterns

1. **Async/Await**: All pipeline components use asyncio for concurrent processing
2. **Queue-based communication**: Components communicate via `asyncio.Queue` with maxsize limits
3. **Pydantic validation**: All data models use Pydantic for runtime validation
4. **Background tasks**: ML pipeline runs weekly retraining and hourly drift monitoring as async background tasks
5. **Immutability**: Logs are written to `logs/` as append-only audit trail

## Model Training

To train/retrain ML models:
```bash
# Generate training data from historical dollar bars
.venv/bin/python generate_ml_training_data.py

# Train XGBoost model with walk-forward optimization
.venv/bin/python train_meta_model.py
```

Models are saved with joblib to `models/xgboost/5_minute/model.joblib` along with:
- `preprocessor.pkl` - Feature preprocessing pipeline
- `metadata.json` - Model performance metrics (win_rate, precision, recall)
- `threshold.json` - Optimal probability threshold
