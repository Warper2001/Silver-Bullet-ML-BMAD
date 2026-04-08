"""Factory for creating risk management components.

This module provides a factory for initializing all risk management
components and creating a properly configured RiskOrchestrator.

Features:
- One-line initialization of all 8 risk layers
- Load configuration from config.yaml
- Easy setup for paper trading and production
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

from src.risk.risk_orchestrator import RiskOrchestrator
from src.risk.emergency_stop import EmergencyStop
from src.risk.daily_loss_tracker import DailyLossTracker
from src.risk.drawdown_tracker import DrawdownTracker
from src.risk.position_size_tracker import PositionSizeTracker
from src.risk.circuit_breaker_detector import CircuitBreakerDetector
from src.risk.news_event_filter import NewsEventFilter
from src.risk.per_trade_risk_limit import PerTradeRiskLimit
from src.risk.notification_manager import NotificationManager

logger = logging.getLogger(__name__)


class RiskComponentFactory:
    """Factory for creating risk management components.

    Provides easy initialization of all 8 risk layers with proper
    configuration from config.yaml.

    Example:
        >>> orchestrator = RiskComponentFactory.create_risk_orchestrator()
        >>> result = orchestrator.validate_trade(signal)
    """

    DEFAULT_CONFIG_PATH = "config.yaml"
    DEFAULT_DAILY_LOSS_LIMIT = 500  # USD
    DEFAULT_MAX_DRAWDOWN = 0.12  # 12% as decimal
    DEFAULT_RECOVERY_THRESHOLD = 0.95  # 95% recovery threshold
    DEFAULT_MAX_POSITION_SIZE = 5  # contracts
    DEFAULT_ACCOUNT_BALANCE = 50000  # USD
    DEFAULT_PER_TRADE_RISK = 150  # USD

    @staticmethod
    def create_risk_orchestrator(
        config_path: Optional[str] = None
    ) -> RiskOrchestrator:
        """Create RiskOrchestrator with all 8 risk layers initialized.

        Initializes all risk management components with configuration
        loaded from config.yaml (or defaults if not provided).

        Risk Layers Created:
            1. EmergencyStop - Manual trading halt
            2. DailyLossTracker - Daily loss limit enforcement
            3. DrawdownTracker - Maximum drawdown enforcement
            4. PositionSizeTracker - Maximum position size enforcement
            5. CircuitBreakerDetector - Market circuit breaker detection
            6. NewsEventFilter - News event blackout periods
            7. PerTradeRiskLimit - Per-trade risk limit enforcement
            8. NotificationManager - Alert notifications

        Args:
            config_path: Path to config file (default: "config.yaml")

        Returns:
            Fully configured RiskOrchestrator instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config values are invalid

        Example:
            >>> orchestrator = RiskComponentFactory.create_risk_orchestrator()
            >>> result = orchestrator.validate_trade(signal)
            >>> if result['is_valid']:
            ...     submit_order(signal)
        """
        # Load configuration
        config = RiskComponentFactory._load_config(config_path)

        # Extract risk limits
        daily_loss_limit = config.get('risk', {}).get(
            'daily_loss_limit',
            RiskComponentFactory.DEFAULT_DAILY_LOSS_LIMIT
        )
        max_drawdown_percent = config.get('risk', {}).get(
            'max_drawdown_percent',
            RiskComponentFactory.DEFAULT_MAX_DRAWDOWN
        )
        # Convert percentage to decimal (e.g., 12 -> 0.12)
        if max_drawdown_percent >= 1.0:
            max_drawdown = max_drawdown_percent / 100.0
        else:
            max_drawdown = max_drawdown_percent

        max_position_size = config.get('risk', {}).get(
            'max_position_size',
            RiskComponentFactory.DEFAULT_MAX_POSITION_SIZE
        )

        # Create all risk components
        emergency_stop = RiskComponentFactory._create_emergency_stop(config)
        daily_loss_tracker = RiskComponentFactory._create_daily_loss_tracker(
            daily_loss_limit
        )
        drawdown_tracker = RiskComponentFactory._create_drawdown_tracker(
            max_drawdown
        )
        position_size_tracker = RiskComponentFactory._create_position_size_tracker(
            max_position_size
        )
        circuit_breaker_detector = (
            RiskComponentFactory._create_circuit_breaker_detector()
        )
        news_event_filter = RiskComponentFactory._create_news_event_filter()
        per_trade_risk_limit = RiskComponentFactory._create_per_trade_risk_limit()
        notification_manager = (
            RiskComponentFactory._create_notification_manager()
        )

        # Create RiskOrchestrator with all components
        orchestrator = RiskOrchestrator(
            emergency_stop=emergency_stop,
            daily_loss_tracker=daily_loss_tracker,
            drawdown_tracker=drawdown_tracker,
            position_size_tracker=position_size_tracker,
            circuit_breaker_detector=circuit_breaker_detector,
            news_event_filter=news_event_filter,
            per_trade_risk_limit=per_trade_risk_limit,
            notification_manager=notification_manager,
            audit_trail_path="logs/risk_validation.csv"
        )

        logger.info("RiskOrchestrator created with all 8 risk layers")

        return orchestrator

    @staticmethod
    def _load_config(config_path: Optional[str]) -> dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file (uses default if None)

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if config_path is None:
            config_path = RiskComponentFactory.DEFAULT_CONFIG_PATH

        config_file = Path(config_path)

        if not config_file.exists():
            logger.warning(
                "Config file not found: {}, using defaults".format(
                    config_path
                )
            )
            return {}

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        logger.info("Loaded configuration from {}".format(config_path))

        return config

    @staticmethod
    def _create_emergency_stop(config: dict) -> EmergencyStop:
        """Create emergency stop component.

        Args:
            config: Configuration dictionary

        Returns:
            EmergencyStop instance
        """
        return EmergencyStop(
            audit_trail_path="logs/emergency_stop.csv",
            state_path="logs/emergency_stop_state.json"
        )

    @staticmethod
    def _create_daily_loss_tracker(daily_loss_limit: float) -> DailyLossTracker:
        """Create daily loss tracker component.

        Args:
            daily_loss_limit: Maximum daily loss amount (USD)

        Returns:
            DailyLossTracker instance
        """
        return DailyLossTracker(
            daily_loss_limit=daily_loss_limit,
            account_balance=RiskComponentFactory.DEFAULT_ACCOUNT_BALANCE,
            reset_time_utc="13:00",  # 8:00 CT
            audit_trail_path="logs/daily_loss.csv"
        )

    @staticmethod
    def _create_drawdown_tracker(max_drawdown: float) -> DrawdownTracker:
        """Create drawdown tracker component.

        Args:
            max_drawdown: Maximum drawdown percentage (as decimal, e.g., 0.12)

        Returns:
            DrawdownTracker instance
        """
        return DrawdownTracker(
            max_drawdown_percentage=max_drawdown,
            recovery_threshold_percentage=RiskComponentFactory.DEFAULT_RECOVERY_THRESHOLD,
            initial_value=RiskComponentFactory.DEFAULT_ACCOUNT_BALANCE,
            audit_trail_path="logs/drawdown.csv"
        )

    @staticmethod
    def _create_position_size_tracker(max_position_size: int) -> PositionSizeTracker:
        """Create position size tracker component.

        Args:
            max_position_size: Maximum position size (contracts)

        Returns:
            PositionSizeTracker instance
        """
        return PositionSizeTracker(
            max_position_size=max_position_size,
            audit_trail_path="logs/position_size.csv"
        )

    @staticmethod
    def _create_circuit_breaker_detector() -> CircuitBreakerDetector:
        """Create circuit breaker detector component.

        Returns:
            CircuitBreakerDetector instance
        """
        return CircuitBreakerDetector(
            api_client=None,  # No API client for paper trading
            audit_trail_path="logs/circuit_breaker.csv"
        )

    @staticmethod
    def _create_news_event_filter() -> NewsEventFilter:
        """Create news event filter component.

        Returns:
            NewsEventFilter instance
        """
        return NewsEventFilter(
            audit_trail_path="logs/news_events.csv"
        )

    @staticmethod
    def _create_per_trade_risk_limit() -> PerTradeRiskLimit:
        """Create per-trade risk limit component.

        Returns:
            PerTradeRiskLimit instance
        """
        return PerTradeRiskLimit(
            max_risk_dollars=RiskComponentFactory.DEFAULT_PER_TRADE_RISK,
            account_balance=RiskComponentFactory.DEFAULT_ACCOUNT_BALANCE,
            audit_trail_path="logs/per_trade_risk.csv"
        )

    @staticmethod
    def _create_notification_manager() -> NotificationManager:
        """Create notification manager component.

        Returns:
            NotificationManager instance
        """
        return NotificationManager(
            enabled=True  # Enable notifications for paper trading
        )
