# Silver-Bullet-ML-BMAD

ICT Silver Bullet + ML Meta-Labeling Trading System for MNQ Futures.

## Overview

Hybrid trading system combining discretionary ICT concepts with machine learning meta-labeling for regime-adaptive edge in MNQ (Micro E-mini Nasdaq-100) futures trading.

## Project Structure

```
silver-bullet-ml-bmad/
├── src/               # Source code organized by capability
├── tests/             # Unit and integration tests
├── config/            # Configuration files
├── data/              # Data storage (raw, processed, features)
├── logs/              # Immutable audit trail
├── models/            # Trained ML models
└── notebooks/         # Research and analysis
```

## Installation

### Prerequisites

- Python 3.11 or higher
- Poetry (dependency manager)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd silver-bullet-ml-bmad
   ```

2. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Activate the virtual environment:
   ```bash
   poetry shell
   ```

5. Create configuration file:
   ```bash
   cp .env.example .env
   # Edit .env with your TradeStation API credentials
   ```

## Development

### Code Formatting

```bash
make format  # Format code with black
```

### Linting

```bash
make lint  # Run flake8 and mypy
```

### Testing

```bash
make test  # Run pytest (implemented in later stories)
```

### Pre-commit Hooks

Pre-commit hooks are automatically installed after `poetry install`. They will run:
- Black (code formatting)
- flake8 (linting)
- mypy (type checking)

## Configuration

Edit `config.yaml` to adjust system parameters:
- Risk limits
- ML thresholds
- Data completeness targets
- Trading hours

## License

Proprietary - All rights reserved
