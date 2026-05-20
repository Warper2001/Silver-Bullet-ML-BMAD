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

## Web Search Tool

Use the native `WebSearch` tool for web searches. BrightData MCP tools (`mcp__brightdata__*`) are available but are not the default preference.

---

## ⚠️ Deployed System vs. Infrastructure — Read This First

**The `src/` directory is research infrastructure, not the live paper-trading system.**

The live paper-trading system is `src/research/tier2_streaming_working.py` (`Tier2StreamingTrader`). It was built iteratively in the `research/` directory and does not use the async pipeline in `src/data/`, `src/detection/`, `src/execution/`, or `src/ml/`.

The `CLAUDE.md` description below is split into two sections:

1. **Deployed System** — what is actually running paper trades right now
2. **Infrastructure (`src/`)** — the async pipeline that exists in code but is not currently wired to the live system

---

## Deployed System: Tier2StreamingTrader

**File:** `src/research/tier2_streaming_working.py`
**Instrument:** MNQ (Micro E-mini Nasdaq-100 futures), symbol `MNQM26`
**Bar type:** 1-minute OHLCV bars via TradeStation HTTP REST polling (not WebSocket, not dollar bars)

### How It Works

```
TradeStation REST API (1-min bars, poll every 60s)
  → Tier2StreamingTrader._poll_and_process()
  → _update_h1_structure()       # resample to H1, detect liquidity sweeps
  → _detect_and_enter()          # check filters, detect FVG, enter trade
  → _advance_active_trade()      # check TP / SL / time-stop each bar
  → TradeStation SIM bracket orders (entry limit + TP limit + SL stop)
```

### Active Filters (as of 2026-05-20)

| Filter | Value | Notes |
|---|---|---|
| Direction | Bearish only | `BEARISH_ONLY = True`; bullish path is dead code |
| H1 sweep required | Must be active | Bearish H1 liquidity sweep within last 6 H1 bars |
| FVG detection | 3-bar bearish FVG on 1-min bars | Gap ≥ 0.5× ATR, ≤ $60, ≥ 15% of H1 ATR |
| Day-of-week | No Tuesdays | Hard-coded return on `weekday() == 1` |
| Volatility regime | ATR gate | Blocks when H1 ATR > 75th pct of 120-bar rolling history |
| ML filter | **Disabled** | `ML_THRESHOLD = 0.0`; always passes through |
| LR regime filter | Optional | Active only if `models/xgboost/lr_regime_config.json` exists |
| Seasonal block | Off by default | Env var `TIER2_BLOCKED_MONTHS` to activate |
| Daily circuit breaker | -$750/day | Halts for rest of day if daily P&L ≤ -$750 |

### Entry / Exit

- **Entry:** Limit order at FVG midpoint (50% of gap)
- **Stop Loss:** 5.0× gap size from entry
- **Take Profit:** 6.0× gap size from entry
- **Time stop (pending):** Cancel if not filled within 240 bars
- **Time stop (active):** Close at market if held 60 bars after fill
- **Position size:** 5 MNQ contracts

### Key Configuration Constants

```python
SL_MULTIPLIER = 5.0
TP_MULTIPLIER = 6.0
ENTRY_PCT = 0.5
MAX_HOLD_BARS = 60      # bars from fill
MAX_PENDING_BARS = 240  # bars waiting for limit fill
CONTRACTS_PER_TRADE = 5
MAX_DAILY_LOSS = -750.0
VOL_REGIME_LOOKBACK = 120
VOL_REGIME_THRESHOLD = 0.75
MIN_GAP_ATR_RATIO = 0.15
BEARISH_ONLY = True
ML_THRESHOLD = 0.0      # disabled
```

### Running the Deployed System

```bash
# Paper trading (requires TradeStation credentials in .access_token)
.venv/bin/python src/research/tier2_streaming_working.py

# Monitor logs
tail -f logs/tier2_streaming_working.log
tail -f logs/tier2_filter_log.csv
```

### Backtesting the Deployed System

```bash
# Full 1-year backtest on local 1-min CSV data
.venv/bin/python backtest_tier2_1year_validation.py

# Today's session replay (for debugging)
.venv/bin/python backtest_tier2_today.py
```

Data files:
- `data/processed/mnq_1min_2025.csv` — 2025 full-year 1-min bars
- `data/processed/mnq_1min_2026_ytd.csv` — 2026 year-to-date 1-min bars

### Methodology Status (2026-05-20)

**All backtest results for this system are under re-validation.** The 1-year honest OOS run (May 2025–May 2026) showed PF ≈ 1.0, Sharpe ≈ 0, WR ≈ 48%. Program C (falsification-first) is in progress:

- Phase 0 (housekeeping): in progress
- Phase 1 (S12 random-entry control + S13 timeframe replication): not started
- Sealed holdout (`data/sealed_holdout/`, 2026-03-01+): not yet established

Do not run new parameter searches or claim performance improvement until Phase 1 is complete. See `_bmad-output/problem-solution-2026-05-20.md` and `_bmad-output/strategy_spec_current.md`.

---

## Infrastructure (`src/`): Async Pipeline

This code exists and has tests, but it is **not connected to the current paper-trading system**. It represents the original architecture design and may be useful for future development.

### Intended Architecture

The `src/` pipeline was designed as an async queue-based system:

1. **Data Pipeline** (`src/data/`) — TradeStation WebSocket → DollarBar transformer → Validator → Gap Detector → Persistence
2. **Detection Pipeline** (`src/detection/`) — MSS, FVG, liquidity sweep pattern detection → `SilverBulletSetup` events
3. **ML Pipeline** (`src/ml/`) — Feature engineering → XGBoost inference → probability filter (default P ≥ 0.65)
4. **Execution Pipeline** (`src/execution/`) — Position sizing, order submission, triple-barrier exits

### Key Infrastructure Components

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

### Infrastructure Configuration

- **`config.yaml`**: System parameters (risk limits, ML thresholds, data completeness targets)
- **`.env`**: TradeStation API credentials (NEVER commit actual credentials)
- **`pyproject.toml`**: Poetry dependencies

Key config values:
- `ml.probability_threshold`: default 0.65
- `risk.daily_loss_limit`: default $500
- `risk.max_drawdown_percent`: default 12%
- `data.dollar_bar_threshold`: default $50,000,000

---

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

### Demo / Legacy Scripts
```bash
# Generate test data for src/ pipeline demo
.venv/bin/python generate_test_data.py

# Simple backtest (src/ pipeline demo)
.venv/bin/python simple_backtest.py
```

---

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
├── research/         # ← DEPLOYED SYSTEM IS HERE (tier2_streaming_working.py)
└── risk/             # Risk management (drawdown limits, circuit breakers, position sizing)
```

---

## Type Checking

Many modules have mypy errors disabled via `pyproject.toml` overrides:
- `src.data.*` (config, websocket, models, transformation, tradestation_*)
- `src.ml.*` (all ML modules)

These are known issues from the original development. When adding new code, prefer proper type hints over adding more ignore overrides.

## Testing Structure

```
tests/
├── integration/      # End-to-end pipeline tests (src/ pipeline)
│   ├── test_ml_pipeline_integration.py
│   ├── test_orchestrator_integration.py
│   ├── test_gap_detection_integration.py
│   └── ...
└── unit/            # Isolated component tests
```

Integration tests use mock `TradeStationAuth` to avoid requiring live credentials. Tests are structured as pytest classes with async test methods using `@pytest.mark.asyncio`.

Note: The test suite covers the `src/` infrastructure pipeline. There are no automated tests for `Tier2StreamingTrader` — it is validated via backtesting scripts.

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

For the deployed Tier2 meta-labeling model (currently disabled):
- Model: `models/xgboost/tier2_meta_labeling_model.pkl`
- Threshold: `models/xgboost/tier2_threshold.json`
- LR regime config: `models/xgboost/lr_regime_config.json`
