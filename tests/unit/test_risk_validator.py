"""Unit tests for RiskValidator.

Tests the RiskValidator class that wraps RiskOrchestrator and converts
TradingSignal format between execution pipeline and risk management.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from src.execution.risk_integration import RiskValidator
from src.execution.trade_execution_pipeline import TradingSignal


@pytest.fixture
def mock_risk_orchestrator():
    """Create mock RiskOrchestrator."""
    mock_orchestrator = Mock()
    return mock_orchestrator


@pytest.fixture
def sample_trading_signal():
    """Create sample trading signal."""
    return TradingSignal(
        signal_id="SIG-123",
        symbol="MNQ 03-26",
        direction="bullish",
        confidence_score=0.85,
        timestamp=datetime.now(timezone.utc),
        entry_price=11800.00,
        patterns=[],
        prediction={}
    )


@pytest.fixture
def sample_trading_signal_short():
    """Create sample bearish trading signal."""
    return TradingSignal(
        signal_id="SIG-124",
        symbol="MNQ 03-26",
        direction="bearish",
        confidence_score=0.75,
        timestamp=datetime.now(timezone.utc),
        entry_price=11750.00,
        patterns=[],
        prediction={}
    )


class TestRiskValidator:
    """Test RiskValidator class."""

    def test_init(self, mock_risk_orchestrator):
        """Test RiskValidator initialization."""
        validator = RiskValidator(mock_risk_orchestrator)

        assert validator._risk_orchestrator == mock_risk_orchestrator

    def test_validate_trade_all_checks_pass(
        self,
        mock_risk_orchestrator,
        sample_trading_signal
    ):
        """Test validate_trade when all risk checks pass."""
        # Setup mock to return passing validation
        mock_risk_orchestrator.validate_trade.return_value = {
            'is_valid': True,
            'block_reason': None,
            'checks_passed': [
                'emergency_stop',
                'daily_loss',
                'drawdown',
                'position_size',
                'circuit_breaker',
                'news_events',
                'per_trade_risk'
            ],
            'checks_failed': [],
            'validation_details': {}
        }

        validator = RiskValidator(mock_risk_orchestrator)
        result = validator.validate_trade(sample_trading_signal)

        # Verify validation passed
        assert result['is_valid'] is True
        assert result['block_reason'] is None
        assert len(result['checks_passed']) == 7
        assert len(result['checks_failed']) == 0

        # Verify RiskOrchestrator.validate_trade was called
        mock_risk_orchestrator.validate_trade.assert_called_once()

        # Verify TradingSignal was converted properly
        call_args = mock_risk_orchestrator.validate_trade.call_args[0][0]
        assert hasattr(call_args, 'signal_id')
        assert hasattr(call_args, 'entry_price')
        assert hasattr(call_args, 'quantity')

    def test_validate_trade_daily_loss_limit_failed(
        self,
        mock_risk_orchestrator,
        sample_trading_signal
    ):
        """Test validate_trade when daily loss limit is breached."""
        # Setup mock to return failing validation
        mock_risk_orchestrator.validate_trade.return_value = {
            'is_valid': False,
            'block_reason': 'Daily loss limit breached',
            'checks_passed': ['emergency_stop', 'drawdown'],
            'checks_failed': ['daily_loss'],
            'validation_details': {
                'daily_loss': {
                    'passed': False,
                    'reason': 'Daily loss limit breached',
                    'status': 'BREACHED'
                }
            }
        }

        validator = RiskValidator(mock_risk_orchestrator)
        result = validator.validate_trade(sample_trading_signal)

        # Verify validation failed
        assert result['is_valid'] is False
        assert result['block_reason'] == 'Daily loss limit breached'
        assert 'daily_loss' in result['checks_failed']

    def test_validate_trade_drawdown_exceeded(
        self,
        mock_risk_orchestrator,
        sample_trading_signal
    ):
        """Test validate_trade when max drawdown is exceeded."""
        # Setup mock to return failing validation
        mock_risk_orchestrator.validate_trade.return_value = {
            'is_valid': False,
            'block_reason': 'Maximum drawdown exceeded',
            'checks_passed': ['emergency_stop', 'daily_loss'],
            'checks_failed': ['drawdown'],
            'validation_details': {
                'drawdown': {
                    'passed': False,
                    'reason': 'Maximum drawdown exceeded',
                    'status': 'EXCEEDED',
                    'current_drawdown': 15.0,
                    'max_drawdown': 12.0
                }
            }
        }

        validator = RiskValidator(mock_risk_orchestrator)
        result = validator.validate_trade(sample_trading_signal)

        # Verify validation failed
        assert result['is_valid'] is False
        assert result['block_reason'] == 'Maximum drawdown exceeded'
        assert 'drawdown' in result['checks_failed']

    def test_validate_trade_emergency_stop_active(
        self,
        mock_risk_orchestrator,
        sample_trading_signal
    ):
        """Test validate_trade when emergency stop is active."""
        # Setup mock to return failing validation
        mock_risk_orchestrator.validate_trade.return_value = {
            'is_valid': False,
            'block_reason': 'Emergency stop active: Manual stop activated',
            'checks_passed': [],
            'checks_failed': ['emergency_stop'],
            'validation_details': {
                'emergency_stop': {
                    'passed': False,
                    'reason': 'Emergency stop active: Manual stop activated',
                    'status': 'STOPPED'
                }
            }
        }

        validator = RiskValidator(mock_risk_orchestrator)
        result = validator.validate_trade(sample_trading_signal)

        # Verify validation failed
        assert result['is_valid'] is False
        assert 'Emergency stop active' in result['block_reason']
        assert 'emergency_stop' in result['checks_failed']
