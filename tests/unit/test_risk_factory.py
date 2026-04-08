"""Unit tests for RiskComponentFactory.

Tests the factory that initializes all risk management components
and creates a properly configured RiskOrchestrator.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.risk.factory import RiskComponentFactory
from src.risk.risk_orchestrator import RiskOrchestrator


class TestRiskComponentFactory:
    """Test RiskComponentFactory class."""

    def test_create_risk_orchestrator_success(self):
        """Test successful creation of RiskOrchestrator with all components."""
        orchestrator = RiskComponentFactory.create_risk_orchestrator()

        # Verify RiskOrchestrator was created
        assert orchestrator is not None
        assert isinstance(orchestrator, RiskOrchestrator)

        # Verify all risk components are initialized
        assert orchestrator._emergency_stop is not None
        assert orchestrator._daily_loss_tracker is not None
        assert orchestrator._drawdown_tracker is not None
        assert orchestrator._position_size_tracker is not None
        assert orchestrator._circuit_breaker_detector is not None
        assert orchestrator._news_event_filter is not None
        assert orchestrator._per_trade_risk_limit is not None
        assert orchestrator._notification_manager is not None

    def test_create_risk_orchestrator_with_custom_config(self, tmp_path):
        """Test creation with custom config file."""
        # Create custom config
        custom_config = tmp_path / "custom_config.yaml"
        custom_config.write_text("""
risk:
  daily_loss_limit: 1000
  max_drawdown_percent: 15
  max_position_size: 10
""")

        orchestrator = RiskComponentFactory.create_risk_orchestrator(
            config_path=str(custom_config)
        )

        # Verify RiskOrchestrator was created
        assert orchestrator is not None
        assert isinstance(orchestrator, RiskOrchestrator)

    def test_create_risk_orchestrator_default_config_values(self):
        """Test that default config values are used when no config provided."""
        orchestrator = RiskComponentFactory.create_risk_orchestrator()

        # Verify default risk limits are set
        # Daily loss limit: $500 USD (from config.yaml)
        assert orchestrator._daily_loss_tracker._daily_loss_limit == 500

        # Max drawdown: 12% as decimal (from config.yaml)
        assert orchestrator._drawdown_tracker._max_drawdown_percentage == 0.12

        # Max position size: 5 contracts (from config.yaml)
        assert orchestrator._position_size_tracker._max_position_size == 5
