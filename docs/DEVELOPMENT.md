# Silver Bullet ML - Development Guide

**Last Updated**: 2026-03-27
**Version**: 0.1.0
**Status**: Active Development

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Code Standards](#code-standards)
4. [Testing](#testing)
5. [Debugging](#debugging)
6. [Adding Features](#adding-features)
7. [Common Tasks](#common-tasks)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)

---

## Getting Started

### Prerequisites

- **Python**: 3.11 or higher
- **Poetry**: Latest version (dependency manager)
- **Git**: For version control
- **TradeStation Account**: For real data (optional for development)

### Initial Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd silver-bullet-ml-bmad

# 2. Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# 3. Install dependencies
poetry install

# 4. Activate virtual environment
poetry shell

# 5. Create configuration file
cp .env.example .env

# 6. (Optional) Add TradeStation credentials to .env
# Edit .env with your Client ID and Secret

# 7. Run tests to verify setup
poetry run pytest

# 8. Generate test data
poetry run python generate_test_data.py

# 9. Run quick backtest
poetry run python simple_backtest.py
```

### Project Structure Overview

```
silver-bullet-ml-bmad/
├── src/                    # Source code (organized by capability)
│   ├── cli/               # Command-line interfaces
│   ├── dashboard/         # Streamlit dashboard
│   ├── data/              # Data collection and management
│   ├── detection/         # Pattern detection algorithms
│   ├── execution/         # Trade execution and order management
│   ├── ml/                # Machine learning pipeline
│   ├── monitoring/        # System monitoring and alerting
│   ├── research/          # Backtesting and analysis
│   └── risk/              # Risk management controls
├── tests/                 # Test suites
├── data/                  # Data storage
└── docs/                  # Documentation
```

---

## Development Workflow

### Branch Strategy

- **main**: Production-ready code
- **feature/***: Feature branches
- **bugfix/***: Bug fix branches
- **experiment/***: Experimental features

### Creating a Feature Branch

```bash
# Start from main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: Add your feature description"

# Push to remote
git push origin feature/your-feature-name
```

### Commit Message Format

Follow Conventional Commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

**Examples**:
```bash
git commit -m "feat(ml): Add meta-labeling model training"
git commit -m "fix(risk): Correct daily loss limit calculation"
git commit -m "docs: Update architecture documentation"
```

### Code Review Process

1. Create pull request
2. Request review from team members
3. Address feedback
4. Ensure CI checks pass
5. Merge with squash commit

---

## Code Standards

### Python Style Guide

We follow **PEP 8** with project-specific modifications:

**Line Length**: 88 characters (Black default)
**Indentation**: 4 spaces
**Imports**: Grouped and sorted (isort)
**Naming Conventions**:
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- `_leading_underscore` for protected members

### Code Formatting

```bash
# Format code with Black
make format
# OR
poetry run black src/ tests/

# Check formatting without making changes
poetry run black --check src/ tests/
```

### Linting

```bash
# Run flake8 and mypy
make lint
# OR

# Flake8 (style linting)
poetry run flake8 src/ tests/

# mypy (type checking)
poetry run mypy src/
```

### Pre-commit Hooks

Pre-commit hooks automatically run on git commit:

```bash
# Manual pre-commit run
poetry run pre-commit run --all-files

# Install pre-commit hooks (already done by poetry install)
poetry run pre-commit install
```

### Type Hints

All functions should have type hints:

```python
from typing import Optional

def calculate_signals(
    price_data: pd.DataFrame,
    lookback: int = 100,
    threshold: float = 0.5
) -> pd.DataFrame:
    """Calculate trading signals from price data.

    Args:
        price_data: Historical price data with OHLCV columns
        lookback: Number of bars to look back for calculations
        threshold: Minimum confidence threshold for signals

    Returns:
        DataFrame with signal timestamps, directions, and confidence scores
    """
    # Implementation...
```

### Docstrings

Use Google-style docstrings:

```python
def detect_silver_bullet(
    mss_event: MSSEvent,
    fvg_event: FVGEvent,
    sweep_event: Optional[LiquiditySweepEvent] = None
) -> Optional[SilverBulletSetup]:
    """Detect Silver Bullet setup from pattern events.

    A Silver Bullet setup occurs when MSS, FVG, and optionally sweep
    patterns align in confluence within a short time window.

    Args:
        mss_event: Market Structure Shift event
        fvg_event: Fair Value Gap event
        sweep_event: Optional liquidity sweep event for 3-pattern confluence

    Returns:
        SilverBulletSetup if valid setup detected, None otherwise

    Raises:
        ValueError: If event timestamps are invalid

    Examples:
        >>> mss = MSSEvent(...)
        >>> fvg = FVGEvent(...)
        >>> setup = detect_silver_bullet(mss, fvg)
        >>> if setup:
        ...     print(f"Setup detected at {setup.timestamp}")
    """
```

---

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests (50+ files)
│   ├── test_silver_bullet_detection.py
│   ├── test_features.py
│   ├── test_xgboost_trainer.py
│   └── ...
└── integration/       # Integration tests (23 files)
    ├── test_silver_bullet_integration.py
    ├── test_ml_pipeline_integration.py
    └── ...
```

### Writing Unit Tests

Use pytest with fixtures:

```python
import pytest
import pandas as pd
from src.detection.silver_bullet_detection import check_silver_bullet_setup
from src.data.models import MSSEvent, FVGEvent

@pytest.fixture
def sample_mss_event():
    """Create sample MSS event for testing."""
    return MSSEvent(
        timestamp=pd.Timestamp("2026-01-15 10:30:00"),
        direction="bullish",
        swing_point=SwingPoint(price=11850.0, side="high"),
        volume=1000000,
        bar_index=100
    )

@pytest.fixture
def sample_fvg_event():
    """Create sample FVG event for testing."""
    return FVGEvent(
        timestamp=pd.Timestamp("2026-01-15 10:35:00"),
        direction="bullish",
        gap_range=GapRange(top=11860.0, bottom=11880.0),
        size_bps=20.0,
        bar_index=105
    )

def test_silver_bullet_detection_bullish(sample_mss_event, sample_fvg_event):
    """Test bullish Silver Bullet setup detection."""
    setup = check_silver_bullet_setup(
        mss_event=sample_mss_event,
        fvg_event=sample_fvg_event,
        max_bar_distance=10
    )

    assert setup is not None
    assert setup.direction == "bullish"
    assert setup.confidence > 0.5

def test_silver_bullet_detection_too_far_apart(sample_mss_event, sample_fvg_event):
    """Test that setups too far apart are rejected."""
    setup = check_silver_bullet_setup(
        mss_event=sample_mss_event,
        fvg_event=sample_fvg_event,
        max_bar_distance=5  # Only 5 bars allowed
    )

    assert setup is None  # Should be rejected (10 bars apart)
```

### Writing Integration Tests

Test end-to-end workflows:

```python
import pytest
from src.research.silver_bullet_backtester import SilverBulletBacktester
from src.research.historical_data_loader import HistoricalDataLoader

@pytest.fixture
def backtester():
    """Create backtester instance for testing."""
    return SilverBulletBacktester(
        symbol="MNQ",
        start_date=pd.Timestamp("2025-12-01"),
        end_date=pd.Timestamp("2026-01-31")
    )

def test_full_backtest_workflow(backtester):
    """Test complete backtest workflow."""
    # Load data
    backtester.load_data()

    # Run detection
    signals = backtester.detect_patterns()
    assert len(signals) > 0

    # Simulate trades
    trades = backtester.simulate_trades(signals)
    assert len(trades) > 0

    # Calculate metrics
    metrics = backtester.calculate_metrics(trades)
    assert metrics['win_rate'] >= 0.0
    assert metrics['sharpe_ratio'] is not None
```

### Running Tests

```bash
# Run all tests
make test
# OR
poetry run pytest

# Run unit tests only
poetry run pytest tests/unit/

# Run integration tests only
poetry run pytest tests/integration/

# Run specific test file
poetry run pytest tests/unit/test_silver_bullet_detection.py

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run with verbose output
poetry run pytest -vv

# Run with specific marker
poetry run pytest -m "not slow"
```

### Test Coverage

```bash
# Generate coverage report
poetry run pytest --cov=src --cov-report=term-missing

# Generate HTML coverage report
poetry run pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Debugging

### Logging

Configure logging in your code:

```python
import logging

# Get logger for module
logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed debugging information")
logger.info("General information about execution")
logger.warning("Warning about potential issues")
logger.error("Error occurred but execution continues")
logger.critical("Critical error requiring immediate attention")
```

Set log level in `config.yaml`:

```yaml
system:
  log_level: DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Debugging Scripts

Use Python debugger:

```python
import pdb

def detect_patterns(data):
    # Set breakpoint
    pdb.set_trace()

    # Inspect variables
    print(data.head())

    # Continue execution
    patterns = process_data(data)
    return patterns
```

### Common Issues

**Issue**: Import errors
```bash
# Solution: Ensure you're using poetry shell
poetry shell
# OR use poetry run
poetry run python your_script.py
```

**Issue**: Module not found
```bash
# Solution: Install in development mode
poetry install
```

**Issue**: Tests failing
```bash
# Solution: Run with verbose output to see errors
poetry run pytest -vv tests/unit/test_failing_test.py
```

---

## Adding Features

### Adding a New Pattern Detector

1. **Create detector in `src/detection/`**:

```python
# src/detection/your_pattern_detector.py
from typing import Optional
from src.data.models import YourPatternEvent
import logging

logger = logging.getLogger(__name__)

def detect_your_pattern(
    price_data: pd.DataFrame,
    lookback: int = 100
) -> list[YourPatternEvent]:
    """Detect your custom pattern from price data.

    Args:
        price_data: OHLCV data
        lookback: Number of bars to analyze

    Returns:
        List of detected pattern events
    """
    events = []

    # Your detection logic here
    for i in range(lookback, len(price_data)):
        # Check if pattern exists
        if your_pattern_condition(price_data.iloc[i-lookback:i]):
            event = YourPatternEvent(
                timestamp=price_data.iloc[i].timestamp,
                # ... other fields
            )
            events.append(event)
            logger.info(f"Detected pattern at {event.timestamp}")

    return events
```

2. **Add event model in `src/data/models.py`**:

```python
from pydantic import BaseModel
from datetime import datetime

class YourPatternEvent(BaseModel):
    """Your custom pattern event."""

    timestamp: datetime
    direction: Literal["bullish", "bearish"]
    # ... other fields
```

3. **Add unit tests**:

```python
# tests/unit/test_your_pattern_detector.py
import pytest
from src.detection.your_pattern_detector import detect_your_pattern

def test_your_pattern_detection():
    """Test your pattern detection."""
    # Setup test data
    price_data = create_test_data()

    # Run detection
    events = detect_your_pattern(price_data)

    # Assert results
    assert len(events) > 0
    assert all(event.direction in ["bullish", "bearish"] for event in events)
```

4. **Integrate with pipeline**:

```python
# src/detection/pipeline.py
from src.detection.your_pattern_detector import detect_your_pattern

class DetectionPipeline:
    def run_detection(self, price_data: pd.DataFrame):
        # ... existing detectors
        your_patterns = detect_your_pattern(price_data)
        # ... use patterns
```

### Adding ML Features

1. **Add feature calculation in `src/ml/features.py`**:

```python
class FeatureEngineer:
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        # ... existing features

        # Add your feature
        df['your_feature'] = self.calculate_your_feature(data)

        return df

    def calculate_your_feature(self, data: pd.DataFrame) -> pd.Series:
        """Calculate your custom feature.

        Args:
            data: OHLCV data

        Returns:
            Series with feature values
        """
        # Your feature calculation logic
        return data['close'].pct_change()
```

2. **Add feature to training pipeline**:

```python
# src/ml/training_data.py
class TrainingDataPipeline:
    def build_features(self, signals: pd.DataFrame, price_data: pd.DataFrame):
        # ... existing features
        features['your_feature'] = self.feature_engineer.calculate_your_feature(price_data)
        return features
```

3. **Test feature importance**:

```python
# After training, check feature importance
import pandas as pd

feature_importance = pd.DataFrame({
    'feature': model.get_booster().feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
```

### Adding Risk Limits

1. **Create risk rule in `src/risk/`**:

```python
# src/risk/your_risk_limit.py
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class YourRiskLimit:
    """Your custom risk limit."""

    def __init__(self, limit: float):
        self.limit = limit
        self.current_value = 0.0

    def check_limit(self, value: float) -> bool:
        """Check if value is within limit.

        Args:
            value: Value to check

        Returns:
            True if within limit, False otherwise
        """
        if value > self.limit:
            logger.warning(f"Value {value} exceeds limit {self.limit}")
            return False
        return True
```

2. **Integrate with risk orchestrator**:

```python
# src/risk/risk_orchestrator.py
from src.risk.your_risk_limit import YourRiskLimit

class RiskOrchestrator:
    def __init__(self):
        # ... existing risk limits
        self.your_risk_limit = YourRiskLimit(limit=100.0)

    def check_trade(self, trade: Trade) -> bool:
        # ... existing checks
        if not self.your_risk_limit.check_limit(trade.value):
            return False
        return True
```

3. **Configure in `config.yaml`**:

```yaml
risk:
  # ... existing limits
  your_limit: 100.0
```

---

## Common Tasks

### Generating Test Data

```bash
# Generate 6 months of test data
poetry run python generate_test_data.py

# Check generated data
ls -la data/processed/dollar_bars/
```

### Running Backtests

```bash
# Simple backtest (for testing)
poetry run python simple_backtest.py

# Optimized baseline backtest
poetry run python run_optimized_silver_bullet.py \
    --start-date 2025-12-01 \
    --end-date 2026-03-06

# ML-enhanced backtest
poetry run python run_ml_backtest.py \
    --start-date 2025-12-01 \
    --end-date 2026-03-06 \
    --probability-threshold 0.65

# Meta-labeling A/B test
poetry run python run_meta_labeling_backtest.py \
    --start-date 2025-12-01 \
    --end-date 2026-03-06
```

### Training ML Models

```bash
# Train base ML model
poetry run python train_ml_model.py

# Train meta-model
poetry run python train_meta_model.py \
    --start-date 2024-01-01 \
    --end-date 2026-03-06
```

### Collecting Real Data

```bash
# Test authentication first
poetry run python test_auth.py

# Collect historical data
poetry run python collect_historical_data.py \
    --start-date 2024-01-01 \
    --end-date 2026-03-06

# Collect real-time data
poetry run python collect_realtime_data.py
```

### Running the Dashboard

```bash
# Start Streamlit dashboard
poetry run streamlit run src/dashboard/streamlit_app.py

# Access at http://localhost:8501
```

---

## Troubleshooting

### Common Errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Ensure you're in the project root
cd /path/to/silver-bullet-ml-bmad

# Use poetry run
poetry run python your_script.py
```

**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**:
```bash
# Check file permissions
ls -la data/

# Fix permissions if needed
chmod 755 data/
```

**Error**: `ConnectionError: Failed to connect to TradeStation API`

**Solution**:
```bash
# Check your credentials in .env
cat .env

# Test authentication
poetry run python test_auth.py

# Refresh token if needed
poetry run python exchange_auth_code.py
```

**Error**: `MemoryError: Unable to allocate array`

**Solution**:
```bash
# Reduce data size in config.yaml
# OR process data in smaller chunks
```

### Performance Issues

**Issue**: Slow backtest execution

**Solutions**:
1. Use faster data formats (HDF5 instead of CSV)
2. Reduce lookback period
3. Use `run_*_fast.py` scripts
4. Enable multiprocessing

**Issue**: High memory usage

**Solutions**:
1. Process data in chunks
2. Use `dtype` parameter to reduce memory footprint
3. Clear cache between operations
4. Use generators instead of lists

### Data Issues

**Issue**: Missing data or gaps

**Solutions**:
1. Check data completeness: `poetry run python -m src.data.cli validate`
2. Re-download missing data
3. Use forward fill for small gaps
4. Adjust data quality threshold in config

---

## Contributing

### Before Contributing

1. Read this development guide
2. Review existing code patterns
3. Understand the architecture (see `ARCHITECTURE.md`)
4. Set up your development environment

### Making Changes

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Make changes** following code standards
3. **Add tests** for your changes
4. **Update documentation** if needed
5. **Run tests**: `poetry run pytest`
6. **Format code**: `make format`
7. **Commit changes** with clear message
8. **Push and create pull request**

### Pull Request Checklist

- [ ] Code follows style guide (Black, flake8, mypy clean)
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Commit messages follow Conventional Commits
- [ ] No merge conflicts
- [ ] CI checks passing

### Getting Help

- **Documentation**: Check `docs/` directory
- **Existing Issues**: Review GitHub issues
- **Team Communication**: Contact team members
- **Code Review**: Request review early

---

## Best Practices

### Performance

- Use vectorized operations (pandas/numpy) instead of loops
- Cache expensive calculations
- Use appropriate data types (e.g., `category` for strings)
- Profile code before optimizing

### Security

- Never commit credentials (use `.env`)
- Validate all user inputs
- Use parameterized queries for database operations
- Sanitize data from external sources

### Testing

- Write tests before writing code (TDD)
- Test edge cases and error conditions
- Use fixtures for common test data
- Keep tests independent and fast

### Documentation

- Document public APIs with docstrings
- Keep README files up to date
- Document complex algorithms
- Add examples for non-trivial usage

---

## Resources

### Internal Documentation

- `ARCHITECTURE.md` - System architecture overview
- `SOURCE_TREE.md` - Source code organization
- `README.md` - Project overview
- `QUICK_START.md` - Quick start guide

### External Resources

- **ICT Concepts**: https://innercircletrader.net/
- **Python Docs**: https://docs.python.org/3.11/
- **pandas Docs**: https://pandas.pydata.org/docs/
- **XGBoost Docs**: https://xgboost.readthedocs.io/
- **Pytest Docs**: https://docs.pytest.org/

### Tools

- **Black**: Code formatter
- **flake8**: Style linting
- **mypy**: Type checking
- **pytest**: Testing framework
- **Poetry**: Dependency management

---

**Document Version**: 1.0.0
**Last Updated**: 2026-03-27
**Maintained By**: Development Team
