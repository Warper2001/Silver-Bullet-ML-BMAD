# Silver Bullet ML - Source Tree Documentation

**Last Updated**: 2026-03-27
**Version**: 0.1.0

---

## Directory Structure

```
silver-bullet-ml-bmad/
├── src/                           # Main source code (33,245 lines, 87 files)
│   ├── cli/                       # Command-line interfaces
│   ├── dashboard/                 # Streamlit monitoring dashboard
│   ├── data/                      # Data collection and management
│   ├── detection/                 # Pattern detection algorithms
│   ├── execution/                 # Trade execution and order management
│   ├── ml/                        # Machine learning pipeline
│   ├── monitoring/                # System monitoring and alerting
│   ├── research/                  # Backtesting and analysis
│   └── risk/                      # Risk management controls
│
├── tests/                         # Test suites (73 test files)
│   ├── unit/                      # Unit tests (50 files)
│   └── integration/               # Integration tests (23 files)
│
├── data/                          # Data storage
│   ├── audit/                     # Immutable audit trail
│   ├── features/                  # Extracted ML features
│   ├── historical/                # Historical price data
│   ├── logs/                      # System logs
│   ├── ml_training/               # ML training datasets
│   ├── models/                    # Trained ML models
│   ├── processed/                 # Processed data (dollar bars, time bars)
│   ├── raw/                       # Raw data from TradeStation
│   ├── reports/                   # Backtest and performance reports
│   └── state/                     # System state persistence
│
├── docs/                          # Documentation
│   ├── research/                  # Research reports and validation
│   ├── ARCHITECTURE.md            # System architecture (this file)
│   ├── SOURCE_TREE.md             # Source tree documentation
│   ├── DEVELOPMENT.md             # Development guide
│   └── project-scan-report.json   # Project scan results
│
├── _bmad/                         # BMAD framework
│   ├── _config/                   # BMAD configuration
│   ├── _memory/                   # BMAD memory files
│   ├── bmm/                       # BMAD module configuration
│   ├── cis/                       # Collaborative innovation system
│   └── core/                      # BMAD core utilities
│
├── _bmad-output/                  # BMAD artifacts
│   ├── implementation-artifacts/  # Implementation documents
│   └── planning-artifacts/        # Planning documents
│
├── .streamlit/                    # Streamlit configuration
├── .venv/                         # Python virtual environment
├── logs/                          # Runtime logs
├── models/                        # Additional model storage
│
├── config.yaml                    # System configuration
├── pyproject.toml                 # Poetry dependencies
├── .env.example                   # Environment variables template
├── .env                           # Actual environment variables (not committed)
├── .gitignore                     # Git ignore rules
├── .pre-commit-config.yaml        # Pre-commit hooks
├── Makefile                       # Development commands
│
├── README.md                      # Project overview
├── QUICK_START.md                 # Quick start guide
├── DATA_COLLECTION_GUIDE.md       # Data collection instructions
├── DATA_COLLECTION_COMPLETE.md    # Data collection status
├── DEPLOYMENT_SUMMARY.md          # Deployment guide
│
├── collect_historical_data.py     # Historical data collection script
├── collect_realtime_data.py       # Real-time data collection script
├── generate_test_data.py          # Test data generation
├── convert_csv_to_hdf5.py         # CSV to HDF5 conversion
├── convert_to_time_bars.py        # Dollar bars to time bars conversion
├── generate_ml_training_data.py   # ML training data generation
│
├── simple_backtest.py             # Simple backtest for testing
├── run_optimized_silver_bullet.py # Optimized baseline backtest
├── run_ml_backtest.py             # ML-enhanced backtest
├── run_meta_labeling_backtest.py  # Meta-labeling A/B test
├── run_enhanced_baseline.py       # Enhanced filters backtest
│
├── train_ml_model.py              # ML model training
├── train_meta_model.py            # Meta-model training
├── test_ml_components.py          # ML component testing
│
├── deploy_demo_mode.sh            # Deploy in demo mode
├── deploy_paper_trading.sh        # Deploy paper trading
└── test_auth.py                   # Test TradeStation authentication
```

---

## Source Code Organization

### `src/cli/` - Command-Line Interfaces

**Purpose**: User-facing CLI commands for system control

**Files**:
- `backtest.py` (9,863 lines) - Comprehensive backtesting CLI
  - Multiple backtest modes (simple, optimized, ML-enhanced)
  - Performance report generation
  - Parameter optimization
  - CSV/JSON output formats

- `emergency_stop.py` (3,399 lines) - Emergency shutdown handler
  - Immediate position closing
  - Order cancellation
  - System state persistence
  - Alert notifications

**Usage**:
```bash
# Run backtest
python -m src.cli.backtest --mode ml --start-date 2025-12-01 --end-date 2026-03-06

# Emergency stop
python -m src.cli.emergency_stop --reason "Manual intervention"
```

---

### `src/dashboard/` - Monitoring Dashboard

**Purpose**: Real-time Streamlit dashboard for system monitoring

**Files**:
- `streamlit_app.py` (1,613 lines) - Main Streamlit application
  - Layout configuration
  - Theme customization
  - Page routing

- `shared_state.py` (1,294 lines) - Dashboard state management
  - Shared state across components
  - Session management
  - Data caching

- `navigation.py` (1,273 lines) - Navigation system
  - Menu structure
  - Page routing
  - URL management

- `theme.py` (753 lines) - Theme configuration
  - Color schemes
  - Typography
  - Component styling

**Features**:
- Real-time P&L chart
- Open positions monitor
- Signal log with probabilities
- System health indicators
- Performance metrics

**Access**:
```bash
streamlit run src/dashboard/streamlit_app.py
```

---

### `src/data/` - Data Layer

**Purpose**: Collect, validate, transform, and store market data

**Files**:
- `tradestation_client.py` - TradeStation API client
  - OAuth 2.0 authentication
  - Historical data download
  - Real-time streaming
  - Error handling and retries

- `tradestation_auth.py` (639 lines) - Authentication manager
  - Token management
  - Refresh logic
  - Credential storage

- `websocket.py` - WebSocket streaming
  - Real-time market data
  - Reconnection logic
  - Heartbeat monitoring

- `validation.py` - Data validation
  - Schema validation
  - Completeness checks (99.99% target)
  - Quality metrics

- `transformation.py` - Data transformation
  - Tick data → Dollar bars
  - Dollar bars → Time bars
  - Feature engineering

- `persistence.py` - Data persistence
  - HDF5 storage management
  - Metadata tracking
  - Query optimization

- `orchestrator.py` - Data pipeline orchestration
  - Pipeline coordination
  - Error recovery
  - Progress tracking

- `models.py` - Data models
  - MSSEvent, FVGEvent, LiquiditySweepEvent
  - SilverBulletSetup, Trade
  - Pydantic validation

- `gap_detection.py` - Gap detection
  - Overnight gap detection
  - Price jump analysis

- `futures_symbols.py` - Futures symbol management
  - MNQ contract specifications
  - Roll date tracking

- `historical_downloader.py` (565 lines) - Historical data downloader
  - Batch download
  - Retry logic
  - Progress reporting

- `cli.py` - Data CLI commands
  - Data collection commands
  - Validation commands
  - Status queries

---

### `src/detection/` - Pattern Detection

**Purpose**: Identify ICT Silver Bullet setups

**Files**:
- `silver_bullet_detection.py` - Silver Bullet detector
  - MSS + FVG + Sweep confluence checking
  - Confidence scoring
  - Priority assignment

- `silver_bullet_detector.py` - Alternative implementation
  - Optimized detection logic
  - Batch processing

- `mss_detector.py` - Market Structure Shift detector
  - Swing point identification
  - Break confirmation
  - Volume analysis

- `fvg_detector.py` / `fvg_detection.py` - Fair Value Gap detector
  - 3-candle pattern identification
  - Gap size calculation
  - Direction validation

- `liquidity_sweep_detector.py` / `liquidity_sweep_detection.py` - Sweep detector
  - Swing level identification
  - Sweep detection
  - Recovery confirmation

- `swing_detection.py` - Swing point detector
  - High/low identification
  - Trend analysis

- `confidence_scorer.py` - Confidence scoring
  - Pattern strength calculation
  - Volume confirmation
  - Time alignment scoring

- `time_window_filter.py` - Time-based filtering
  - Killzone filtering
  - Trading hours enforcement
  - Session detection

- `pipeline.py` - Detection pipeline
  - Multi-pattern coordination
  - Event aggregation
  - Signal generation

---

### `src/execution/` - Execution Layer

**Purpose**: Submit orders and manage positions

**Files**:
- `trade_execution_pipeline.py` (615 lines) - Main execution pipeline
  - Order flow coordination
  - Risk check integration
  - Execution monitoring

- `immutable_audit_trail.py` (798 lines) - Audit trail manager
  - Immutable event logging
  - HDF5 storage
  - Query interface

- `market_order_submitter.py` - Market order execution
  - Fast execution
  - Slippage tracking
  - Fill confirmation

- `limit_order_submitter.py` - Limit order execution
  - Price improvement
  - Partial fill handling
  - Cancellation logic

- `order_type_selector.py` - Order type selection
  - Market vs limit decision
  - Condition evaluation
  - Fallback logic

- `partial_fill_handler.py` (537 lines) - Partial fill handling
  - Fill aggregation
  - Retry logic
  - Cancellation rules

- `position_monitoring_service.py` (528 lines) - Position monitoring
  - Real-time tracking
  - P&L calculation
  - Exit condition checking

- `position_tracker.py` - Position tracking
  - State management
  - Reconciliation
  - Persistence

- `triple_barrier_calculator.py` - Triple barrier calculation
  - Take profit calculation (2.0× risk)
  - Stop loss calculation (FVG edge or 1.0× ATR)
  - Time exit calculation (30 minutes)

- `triple_barrier_monitor.py` - Triple barrier monitoring
  - Barrier breach detection
  - Exit triggering
  - P&L calculation

- `triple_barrier_exit_executor.py` - Exit execution
  - Exit order submission
  - Fill confirmation
  - Position update

- `time_window_filter.py` (1,545 lines) - Time window filtering
  - Trading hours enforcement
  - Session detection
  - Killzone filtering

---

### `src/ml/` - Machine Learning Pipeline

**Purpose**: Train and deploy ML models for signal filtering

**Files**:
- `features.py` (610 lines) - Feature engineering
  - 40+ technical indicators
  - Price features (RSI, MACD, Bollinger Bands, ATR)
  - Volume features (volume surge, relative volume, OBV)
  - Time features (hour-of-day, day-of-week, killzone flags)
  - Pattern features (MSS strength, FVG size, sweep magnitude)
  - Market regime (volatility, trend strength)

- `training_data.py` (522 lines) - Training data pipeline
  - Triple barrier labeling
  - Label generation
  - Data splitting
  - Augmentation

- `xgboost_trainer.py` (508 lines) - XGBoost training
  - Model training
  - Hyperparameter tuning
  - Cross-validation
  - Feature importance

- `inference.py` (514 lines) - ML inference
  - Probability scoring
  - Batch prediction
  - Model loading
  - Feature extraction

- `signal_filter.py` - Signal filtering
  - Probability-based filtering
  - Threshold management
  - Filter statistics

- `walk_forward_optimizer.py` (839 lines) - Walk-forward optimization
  - Rolling window training
  - Performance tracking
  - Parameter adaptation

- `drift_detector.py` - Performance drift detection
  - Model degradation monitoring
  - Retraining triggers
  - Performance metrics

- `pipeline.py` - ML pipeline orchestration
  - Feature extraction → Training → Inference
  - Pipeline serialization
  - Version management

- `pipeline_serializer.py` - Pipeline serialization
  - Model persistence
  - Metadata storage
  - Version control

- `meta_training_data_builder.py` - Meta-labeling dataset builder
  - Signal-to-outcome mapping
  - Feature extraction
  - Label generation

- `label_mapper.py` - Label mapping
  - Signal to trade outcome mapping
  - Binary label generation
  - Outcome tracking

- `signal_feature_extractor.py` - Signal feature extraction
  - Feature extraction at signal time
  - Lookback window management
  - Batch extraction

---

### `src/monitoring/` - System Monitoring

**Purpose**: Real-time health monitoring and alerting

**Files**:
- `monitoring_integration.py` (526 lines) - Monitoring integration
  - Component monitoring
  - Event aggregation
  - Alert routing

- `health_check_manager.py` - Health check manager
  - Component health checks
  - Status tracking
  - Failure detection

- `data_staleness_detector.py` - Data staleness detection
  - Data age monitoring
  - Staleness alerts
  - Quality checks

- `resource_monitor.py` - Resource monitoring
  - CPU usage
  - Memory usage
  - Disk usage

- `crash_recovery.py` - Crash recovery
  - State restoration
  - Position recovery
  - Audit trail replay

- `graceful_shutdown_manager.py` - Graceful shutdown
  - Signal handling
  - Position closing
  - State persistence

- `audit_trail.py` - Audit trail manager
  - Event logging
  - Immutable storage
  - Query interface

- `error_logger.py` - Error logging
  - Error capture
  - Context tracking
  - Alert generation

- `daily_report_generator.py` (518 lines) - Daily report generation
  - Performance summary
  - P&L calculation
  - Trade statistics

- `warning_batcher.py` - Warning batching
  - Alert aggregation
  - Rate limiting
  - Digest generation

- `notification_manager.py` - Notification manager
  - Alert routing
  - Channel management
  - Priority handling

- `notification_integration.py` - Notification integration
  - External notifications
  - Webhook support
  - Email alerts

- `terminal_ui.py` - Terminal UI
  - Live terminal display
  - Real-time updates
  - Interactive controls

- `terminal_layout.py` - Terminal layout
  - Screen layout
  - Component positioning
  - Theme management

- `terminal_theme.py` - Terminal theme
  - Color schemes
  - Styling
  - Display options

- `terminal_events.py` - Terminal event handling
  - User input
  - Keyboard handling
  - Event routing

- `__main__.py` - Monitoring entry point
  - Process startup
  - Configuration
  - Main loop

---

### `src/research/` - Research and Backtesting

**Purpose**: Strategy research and backtesting

**Files**:
- `silver_bullet_backtester.py` (646 lines) - Silver Bullet backtester
  - Pattern-based backtesting
  - Performance calculation
  - Report generation

- `ml_meta_labeling_backtester.py` (561 lines) - ML meta-labeling backtester
  - A/B testing (baseline vs ML-filtered)
  - Performance comparison
  - Threshold optimization

- `historical_data_loader.py` - Historical data loader
  - Data loading
  - Resampling
  - Validation

- `performance_metrics_calculator.py` - Performance metrics
  - Sharpe ratio
  - Win rate
  - Drawdown calculation
  - P&L analysis

- `equity_curve_visualizer.py` - Equity curve visualization
  - Chart generation
  - Performance plotting
  - Comparison charts

- `feature_importance_analyzer.py` - Feature importance analysis
  - Feature ranking
  - Importance plotting
  - Correlation analysis

- `market_regime_analyzer.py` (519 lines) - Market regime analysis
  - Volatility regimes
  - Trend detection
  - Regime classification

- `backtest_report_generator.py` - Backtest report generation
  - HTML reports
  - CSV exports
  - Summary statistics

---

### `src/risk/` - Risk Management

**Purpose**: Capital protection and risk controls

**Files**:
- `risk_orchestrator.py` (595 lines) - Risk orchestration
  - Risk limit enforcement
  - Rule coordination
  - Decision aggregation

- `daily_loss_tracker.py` - Daily loss tracking
  - Daily P&L calculation
  - Limit enforcement
  - Reset logic

- `drawdown_tracker.py` - Drawdown tracking
  - Peak tracking
  - Drawdown calculation
  - Limit enforcement

- `position_sizer.py` - Position sizing
  - Risk-based sizing
  - Contract calculation
  - Limit enforcement

- `position_size_tracker.py` - Position size tracking
  - Exposure monitoring
  - Limit checking
  - Alerting

- `per_trade_risk_limit.py` - Per-trade risk limits
  - Risk per trade calculation
  - Limit enforcement
  - Position sizing integration

- `circuit_breaker_detector.py` - Circuit breaker
  - Failure detection
  - Trigger logic
  - Recovery procedures

- `emergency_stop.py` - Emergency stop
  - Immediate shutdown
  - Position closing
  - Alert generation

- `news_event_filter.py` - News event filtering
  - Event calendar
  - Pre-news filtering
  - Post-news recovery

- `notification_manager.py` - Notification manager
  - Alert routing
  - Priority handling
  - Channel management

---

## Test Organization

### `tests/unit/` - Unit Tests (50 files)

**Coverage Areas**:
- Pattern detection algorithms
- ML pipeline components
- Risk management logic
- Data validation
- API clients
- Order execution
- Monitoring systems

**Example Tests**:
- `test_silver_bullet_detection.py` - Pattern detection tests
- `test_features.py` - Feature engineering tests
- `test_xgboost_trainer.py` - ML training tests
- `test_risk_orchestrator.py` - Risk management tests
- `test_trade_execution_pipeline.py` - Execution tests

**Running Tests**:
```bash
# All unit tests
pytest tests/unit/

# Specific test file
pytest tests/unit/test_silver_bullet_detection.py

# With coverage
pytest --cov=src tests/unit/
```

---

### `tests/integration/` - Integration Tests (23 files)

**Coverage Areas**:
- End-to-end workflows
- Multi-component interactions
- API integration
- Data pipeline
- Trading workflow

**Example Tests**:
- `test_silver_bullet_integration.py` - Full trading workflow
- `test_ml_pipeline_integration.py` - ML pipeline end-to-end
- `test_training_data_pipeline.py` - Training data generation
- `test_detection_pipeline.py` - Detection pipeline
- `test_streamlit_dashboard.py` - Dashboard functionality

**Running Tests**:
```bash
# All integration tests
pytest tests/integration/

# Specific test file
pytest tests/integration/test_silver_bullet_integration.py

# With detailed output
pytest -vv tests/integration/
```

---

## Standalone Scripts

### Data Collection Scripts

- `collect_historical_data.py` - Collect historical MNQ data from TradeStation
- `collect_realtime_data.py` - Stream real-time data
- `generate_test_data.py` - Generate synthetic test data
- `convert_csv_to_hdf5.py` - Convert CSV exports to HDF5
- `convert_to_time_bars.py` - Convert dollar bars to time bars
- `generate_ml_training_data.py` - Generate ML training datasets

### Backtest Scripts

- `simple_backtest.py` - Simple backtest for testing
- `run_optimized_silver_bullet.py` - Optimized baseline backtest
- `run_ml_backtest.py` - ML-enhanced backtest
- `run_ml_backtest_extended.py` - Extended ML backtest
- `run_ml_backtest_extended_fast.py` - Fast extended ML backtest
- `run_ml_backtest_fast.py` - Fast ML backtest
- `run_pattern_backtest.py` - Pattern-based backtest
- `run_silver_bullet_killzone_backtest.py` - Killzone backtest
- `run_dollar_bar_backtest.py` - Dollar bar backtest
- `run_enhanced_baseline.py` - Enhanced filters backtest
- `run_meta_labeling_backtest.py` - Meta-labeling A/B test

### Training Scripts

- `train_ml_model.py` - Train ML model
- `train_meta_model.py` - Train meta-model

### Testing Scripts

- `test_ml_components.py` - Test ML components
- `test_auth.py` - Test TradeStation authentication
- `test_current_contract.py` - Test current MNQ contract
- `test_lowercase.py` - Test lowercase handling
- `test_oauth_url.py` - Test OAuth URL generation
- `test_quotes_endpoint.py` - Test quotes endpoint
- `test_symbol_lookup.py` - Test symbol lookup

### Deployment Scripts

- `deploy_demo_mode.sh` - Deploy in demo mode (paper trading with simulated data)
- `deploy_paper_trading.sh` - Deploy paper trading (live data, paper execution)

---

## Configuration Files

### `config.yaml` - System Configuration
```yaml
system:
  environment: development
  log_level: INFO

data:
  dollar_bar_threshold: 50000000
  data_completeness_target: 0.9999

trading:
  symbol: MNQ
  exchange: CME
  timezone: America/New_York

risk:
  daily_loss_limit: 500
  max_drawdown_percent: 12
  max_position_size: 5

ml:
  probability_threshold: 0.65
  retraining_interval_days: 7

monitoring:
  health_check_interval_seconds: 5
  data_staleness_threshold_seconds: 30
```

### `.env` - Environment Variables (Template)
```
TRADESTATION_CLIENT_ID=your_client_id
TRADESTATION_CLIENT_SECRET=your_client_secret
TRADESTATION_REDIRECT_URI=http://localhost:8501/callback
```

### `pyproject.toml` - Python Dependencies
- Python 3.11+
- Poetry dependency management
- Dev tools (black, flake8, mypy, pytest)
- Pre-commit hooks

---

## Data Organization

### `data/raw/` - Raw Data
- CSV exports from TradeStation
- Unprocessed tick data
- Original format preserved

### `data/processed/` - Processed Data
- `dollar_bars/` - Volume-aggregated bars
- `time_bars/` - Time-based bars (1-min, 5-min, 15-min, 1-hour)
- HDF5 format for fast access

### `data/historical/` - Historical Data
- Long-term storage
- Compressed format
- Metadata indexed

### `data/ml_training/` - ML Training Data
- `silver_bullet_signals.parquet` - Signals with metadata
- `silver_bullet_trades.parquet` - Trade outcomes
- `metadata.json` - Training metadata

### `data/models/` - Trained Models
- `xgboost/30_minute/` - 30-min horizon models
  - `xgboost_model.pkl` - Trained model
  - `feature_pipeline.pkl` - Feature pipeline
  - `metadata.json` - Model metadata
  - `pipeline_metadata.json` - Pipeline metadata

### `data/features/` - Extracted Features
- Cached feature calculations
- Parquet format
- Organized by time horizon

### `data/reports/` - Backtest Reports
- CSV exports
- HTML reports
- Performance summaries
- Named by timestamp: `backtest_YYYY-MM-DD.csv`

### `data/audit/` - Audit Trail
- Immutable event log
- All system events
- Compliance records

### `data/state/` - System State
- Position state
- Risk limits
- Model versions
- Recovery data

### `data/logs/` - System Logs
- Application logs
- Error logs
- Performance logs

---

## Documentation Structure

### `docs/` - Main Documentation
- `ARCHITECTURE.md` - System architecture
- `SOURCE_TREE.md` - Source tree documentation (this file)
- `DEVELOPMENT.md` - Development guide
- `project-scan-report.json` - Project scan results

### `docs/research/` - Research Documentation
- `validation_and_ml_enhancement_report.md` - ML validation report
- `silver_bullet_optimization_recommendations.md` - Strategy optimization
- `alternative-data-sources.md` - Data source research
- `csv-format-example.csv` - CSV format example
- `tradingview-export-guide.md` - TradingView export guide

---

## BMAD Framework

### `_bmad/` - BMAD Framework
- `_config/` - Configuration
- `_memory/` - Memory files
- `bmm/` - Module configuration
  - `agents/` - Agent definitions
  - `workflows/` - Workflow definitions
  - `teams/` - Team configurations
- `cis/` - Collaborative innovation system
- `core/` - Core utilities

### `_bmad-output/` - BMAD Artifacts
- `implementation-artifacts/` - Implementation documents
- `planning-artifacts/` - Planning documents

---

## Summary Statistics

**Total Lines of Python Code**: 33,245
**Total Python Files**: 87
**Test Files**: 73 (50 unit + 23 integration)
**Documentation Files**: 15+
**Configuration Files**: 10+

**Largest Files**:
1. `src/dashboard/shared_state.py` - 1,294 lines
2. `src/dashboard/navigation.py` - 1,273 lines
3. `src/ml/walk_forward_optimizer.py` - 839 lines
4. `src/execution/immutable_audit_trail.py` - 798 lines
5. `src/research/silver_bullet_backtester.py` - 646 lines

**Technology Stack**:
- Language: Python 3.11+
- Package Manager: Poetry
- ML Framework: XGBoost, scikit-learn
- Data: pandas, numpy, scipy, h5py
- API: httpx, websockets
- Visualization: matplotlib, plotly, streamlit
- Testing: pytest, pytest-asyncio, pytest-mock

---

**Document Version**: 1.0.0
**Last Updated**: 2026-03-27
**Maintained By**: Development Team
