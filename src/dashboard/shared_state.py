"""Read shared system state for dashboard display."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Placeholder for future implementation
# Stories 8.2-8.11 will populate this with actual state reading logic


@dataclass
class AccountMetrics:
    """Account overview metrics."""

    equity: float
    daily_change_pct: float
    daily_change_usd: float
    daily_pnl: float
    open_positions_count: int
    open_contracts: int
    trade_count: int
    win_rate: float
    daily_drawdown: float
    daily_loss_limit: float
    system_uptime: str
    last_update: datetime


@dataclass
class DailyPerformance:
    """Daily trading performance."""

    trade_count: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float


def get_system_status() -> str:
    """Get current system status."""
    # TODO: Connect to actual system health check from Epic 6
    # For now, return placeholder
    return "RUNNING"


def get_last_update_time():
    """Get last system update timestamp."""
    return datetime.now()


def get_account_metrics() -> AccountMetrics:
    """Get current account metrics from shared state."""
    # TODO: Connect to actual shared state from Epic 4 (positions)
    # For now, return mock data
    logger.warning("Using mock account data - connect to actual shared state")

    return AccountMetrics(
        equity=100000.00,
        daily_change_pct=2.5,
        daily_change_usd=2500.00,
        daily_pnl=1500.00,
        open_positions_count=2,
        open_contracts=3,
        trade_count=5,
        win_rate=60.0,
        daily_drawdown=200.00,
        daily_loss_limit=500.00,
        system_uptime="4h 23m",
        last_update=datetime.now()
    )


def get_daily_pnl() -> float:
    """Get daily P&L from trade history."""
    # TODO: Read from Epic 4 audit trail
    # For now, return mock value
    return 1500.00


def get_open_positions_summary() -> dict:
    """Get summary of open positions."""
    # TODO: Read from Epic 4 position tracking
    # For now, return mock data
    return {
        "count": 2,
        "contracts": 3,
        "unrealized_pnl": 500.00
    }


def get_daily_performance() -> DailyPerformance:
    """Get daily trading performance."""
    # TODO: Read from Epic 4 trade execution logs
    # For now, return mock data
    return DailyPerformance(
        trade_count=5,
        winning_trades=3,
        losing_trades=2,
        win_rate=60.0,
        total_pnl=1500.00,
        max_drawdown=200.00
    )


def get_system_uptime() -> str:
    """Get system uptime from Epic 6 health checks."""
    # TODO: Read from Epic 6 monitoring
    # For now, calculate from process start time
    # Format: "Xh Ym" or "Xm" if < 1 hour
    uptime = timedelta(hours=4, minutes=23)
    hours = uptime.seconds // 3600
    minutes = (uptime.seconds % 3600) // 60
    return f"{hours}h {minutes}m"


def calculate_win_rate(winning_trades: int, total_trades: int) -> float:
    """Calculate win rate as percentage."""
    if total_trades == 0:
        return 0.0
    return (winning_trades / total_trades) * 100


def calculate_trend(current_value: float, previous_value: float) -> tuple[str, str]:
    """Calculate trend indicator and color.

    Returns: (indicator, color_hex)
    """
    if current_value > previous_value:
        return "↑", "#00FF00"  # Up, green
    elif current_value < previous_value:
        return "↓", "#FF0000"  # Down, red
    else:
        return "→", "#0080FF"  # Flat, blue


def get_est_time() -> datetime:
    """Get current time in EST."""
    utc_now = datetime.now(timezone.utc)
    est_offset = timedelta(hours=-5)  # EST is UTC-5
    est_now = utc_now + est_offset
    return est_now


# ============================================================================
# Story 8.3: Open Positions Data Models
# ============================================================================


class Direction(Enum):
    """Trade direction."""

    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class BarrierLevels:
    """Triple-barrier exit levels."""

    upper_barrier: float
    lower_barrier: float
    vertical_barrier: datetime
    entry_price: float


@dataclass
class PositionSignal:
    """Signal metadata for position."""

    signal_id: str
    direction: Direction
    confidence: int  # 1-5
    ml_probability: float  # 0.0-1.0
    mss_present: bool
    fvg_present: bool
    sweep_present: bool
    time_window: str


@dataclass
class OpenPosition:
    """Open position details."""

    signal_id: str
    direction: Direction
    entry_price: float
    current_price: float
    pnl_usd: float
    pnl_pct: float
    barriers: BarrierLevels
    confidence: int
    ml_probability: float
    entry_time: datetime


def get_open_positions() -> List[OpenPosition]:
    """Get open positions from shared state."""
    # TODO: Connect to actual shared state from Epic 4 (positions)
    # For now, return mock data
    logger.warning("Using mock position data - connect to actual shared state")

    return [
        OpenPosition(
            signal_id="SIGNAL-001",
            direction=Direction.LONG,
            entry_price=4500.00,
            current_price=4525.00,
            pnl_usd=125.00,
            pnl_pct=2.78,
            barriers=BarrierLevels(
                upper_barrier=4600.00,
                lower_barrier=4450.00,
                vertical_barrier=datetime.now() + timedelta(minutes=15),
                entry_price=4500.00
            ),
            confidence=4,
            ml_probability=0.75,
            entry_time=datetime.now() - timedelta(minutes=10)
        ),
        OpenPosition(
            signal_id="SIGNAL-002",
            direction=Direction.SHORT,
            entry_price=4480.00,
            current_price=4470.00,
            pnl_usd=40.00,
            pnl_pct=0.89,
            barriers=BarrierLevels(
                upper_barrier=4520.00,
                lower_barrier=4420.00,
                vertical_barrier=datetime.now() + timedelta(minutes=8),
                entry_price=4480.00
            ),
            confidence=3,
            ml_probability=0.65,
            entry_time=datetime.now() - timedelta(minutes=5)
        )
    ]


def calculate_barrier_progress(
    current_price: float, barrier: float, entry_price: float
) -> float:
    """Calculate percentage progress toward barrier (0-100%)."""
    total_range = abs(barrier - entry_price)
    current_distance = abs(current_price - entry_price)

    if total_range == 0:
        return 0.0

    progress = (current_distance / total_range) * 100
    return min(progress, 100.0)


def calculate_time_remaining(vertical_barrier: datetime) -> str:
    """Calculate time remaining to vertical barrier as MM:SS."""
    remaining = vertical_barrier - datetime.now()

    if remaining.total_seconds() <= 0:
        return "Expired"

    minutes = int(remaining.total_seconds() // 60)
    seconds = int(remaining.total_seconds() % 60)

    return f"{minutes:02d}:{seconds:02d}"


def format_position_pnl(pnl_usd: float) -> str:
    """Format P&L with color coding indicator."""
    color = "🟢" if pnl_usd >= 0 else "🔴"
    return f"{color} ${pnl_usd:+,.2f}"


def exit_position(signal_id: str, password: str) -> bool:
    """Execute manual exit for position.

    TODO: Call Epic 4 execution API to close position
    TODO: Verify password before executing
    """
    logger.info(f"Manual exit requested for {signal_id}")
    # Mock implementation - always succeeds for now
    return True


# ============================================================================
# Story 8.4: Live Silver Bullet Signals Data Models
# ============================================================================


class SignalStatus(Enum):
    """Signal execution status."""

    FILTERED = "Filtered"
    EXECUTED = "Executed"
    REJECTED = "Rejected"


@dataclass
class SilverBulletSignal:
    """Silver Bullet signal with ML metadata."""

    timestamp: datetime
    direction: Direction  # From Story 8.3
    confidence: int  # 1-5
    ml_probability: float  # 0.0-1.0
    mss_present: bool
    fvg_present: bool
    sweep_present: bool
    time_window: str
    status: SignalStatus
    signal_id: str


def get_silver_bullet_signals() -> List[SilverBulletSignal]:
    """Get Silver Bullet signals from shared state.

    Returns last 50 signals in reverse chronological order (most recent first).

    TODO: Connect to actual shared state from Epic 2 (signal detection)
    TODO: Connect to actual shared state from Epic 3 (ML prediction)
    """
    logger.warning("Using mock signal data - connect to actual shared state")

    now = datetime.now()
    return [
        SilverBulletSignal(
            timestamp=now - timedelta(minutes=5),
            direction=Direction.LONG,
            confidence=4,
            ml_probability=0.75,
            mss_present=True,
            fvg_present=True,
            sweep_present=False,
            time_window="09:30-16:00",
            status=SignalStatus.EXECUTED,
            signal_id="SIGNAL-001"
        ),
        SilverBulletSignal(
            timestamp=now - timedelta(minutes=10),
            direction=Direction.SHORT,
            confidence=3,
            ml_probability=0.65,
            mss_present=False,
            fvg_present=True,
            sweep_present=True,
            time_window="09:30-16:00",
            status=SignalStatus.FILTERED,
            signal_id="SIGNAL-002"
        ),
        SilverBulletSignal(
            timestamp=now - timedelta(minutes=15),
            direction=Direction.LONG,
            confidence=5,
            ml_probability=0.85,
            mss_present=True,
            fvg_present=True,
            sweep_present=True,
            time_window="09:30-16:00",
            status=SignalStatus.EXECUTED,
            signal_id="SIGNAL-003"
        ),
        SilverBulletSignal(
            timestamp=now - timedelta(minutes=20),
            direction=Direction.SHORT,
            confidence=2,
            ml_probability=0.45,
            mss_present=False,
            fvg_present=False,
            sweep_present=True,
            time_window="09:30-16:00",
            status=SignalStatus.REJECTED,
            signal_id="SIGNAL-004"
        ),
        SilverBulletSignal(
            timestamp=now - timedelta(minutes=25),
            direction=Direction.LONG,
            confidence=4,
            ml_probability=0.70,
            mss_present=True,
            fvg_present=False,
            sweep_present=False,
            time_window="09:30-16:00",
            status=SignalStatus.FILTERED,
            signal_id="SIGNAL-005"
        )
    ]


def filter_signals(
    signals: List[SilverBulletSignal],
    status: str = "All",
    direction: str = "All",
    confidence: str = "All"
) -> List[SilverBulletSignal]:
    """Filter signals by status, direction, and confidence.

    Args:
        signals: List of signals to filter
        status: Status filter ("All", "Filtered", "Executed", "Rejected")
        direction: Direction filter ("All", "LONG", "SHORT")
        confidence: Confidence filter ("All", "5★", "4★+", "3★+")

    Returns:
        Filtered list of signals
    """
    filtered = signals.copy()

    if status != "All":
        filtered = [s for s in filtered if s.status.value == status]

    if direction != "All":
        filtered = [s for s in filtered if s.direction.value == direction]

    if confidence != "All":
        # Parse confidence filter: "5★" -> 5, "4★+" -> 4, "3★+" -> 3
        min_conf = {"5★": 5, "4★+": 4, "3★+": 3}.get(confidence, 0)
        filtered = [s for s in filtered if s.confidence >= min_conf]

    return filtered


def format_signal_status(status: SignalStatus) -> str:
    """Format signal status with color coding indicator.

    Args:
        status: Signal status enum

    Returns:
        Formatted status with emoji indicator
    """
    color_map = {
        SignalStatus.EXECUTED: "🟢",
        SignalStatus.FILTERED: "🟡",
        SignalStatus.REJECTED: "⚪"
    }
    return f"{color_map.get(status, '?')} {status.value}"


def format_confidence_stars(confidence: int) -> str:
    """Format confidence as star rating.

    Args:
        confidence: Confidence level (1-5)

    Returns:
        Star rating string (e.g., "★★★★★" for 5)
    """
    if confidence < 1 or confidence > 5:
        return "?"
    return "★" * confidence + "☆" * (5 - confidence)


def format_ml_probability_bar(probability: float) -> str:
    """Format ML probability as visual progress bar.

    Args:
        probability: Probability value (0.0-1.0)

    Returns:
        Visual progress bar string with percentage (e.g., "██████░░░░ 75.0%")
    """
    if probability < 0.0 or probability > 1.0:
        return "??.?%"

    # Create a 10-segment bar
    filled_segments = int(probability * 10)
    bar = "█" * filled_segments + "░" * (10 - filled_segments)
    return f"{bar} {probability:.1%}"


# ============================================================================
# Story 8.5: Dollar Bar Charts with Patterns
# ============================================================================


class MarkerType(Enum):
    """Type of chart marker for pattern overlays."""

    MSS_BULLISH = "MSS_BULLISH"
    MSS_BEARISH = "MSS_BEARISH"
    FVG_BULLISH = "FVG_BULLISH"
    FVG_BEARISH = "FVG_BEARISH"
    SWEEP = "SWEEP"
    ENTRY = "ENTRY"
    EXIT_PROFIT = "EXIT_PROFIT"
    EXIT_LOSS = "EXIT_LOSS"


@dataclass
class ChartMarker:
    """Generic marker for chart overlay.

    Attributes:
        timestamp: When the marker occurred
        price: Price level of the marker
        marker_type: Type of marker (MSS, FVG, sweep, entry, exit)
        signal_id: Associated signal ID (optional)
    """

    timestamp: datetime
    price: float
    marker_type: MarkerType
    signal_id: str = ""


@dataclass
class FVGZone:
    """Fair Value Gap zone for shaded rectangles.

    Attributes:
        start_time: Zone start time
        end_time: Zone end time
        top_price: Upper price boundary
        bottom_price: Lower price boundary
        direction: "bullish" or "bearish"
    """

    start_time: datetime
    end_time: datetime
    top_price: float
    bottom_price: float
    direction: str  # "bullish" or "bearish"


@dataclass
class TradeMarker:
    """Trade entry/exit marker.

    Attributes:
        timestamp: When the trade occurred
        price: Entry or exit price
        trade_type: "entry" or "exit"
        pnl_usd: P&L in USD (None for entries, required for exits)
        signal_id: Associated signal ID
    """

    timestamp: datetime
    price: float
    trade_type: str  # "entry" or "exit"
    pnl_usd: float | None = None
    signal_id: str = ""


def get_dollar_bars(time_range: str = "today") -> "pd.DataFrame":
    """Get Dollar Bar data for charting.

    Args:
        time_range: "hour", "today", or "week"

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume

    TODO: Connect to Epic 1 Dollar Bar data from HDF5
    """
    import pandas as pd
    import numpy as np

    logger.warning("Using mock dollar bar data - connect to Epic 1")

    # Generate mock candlestick data
    now = datetime.now()
    timestamps = pd.date_range(
        end=now,
        periods=100,
        freq="1min"
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [4500.0 + i * 0.5 + np.random.randn() * 2 for i in range(100)],
        "high": [4502.0 + i * 0.5 + np.random.randn() * 2 for i in range(100)],
        "low": [4498.0 + i * 0.5 + np.random.randn() * 2 for i in range(100)],
        "close": [4500.0 + i * 0.5 + np.random.randn() * 2 for i in range(100)],
        "volume": [1000 + np.random.randint(-100, 100) for _ in range(100)]
    })


def get_pattern_overlays(time_range: str = "today") -> tuple[List[ChartMarker], List[FVGZone]]:
    """Get pattern overlay markers (MSS, FVG, sweeps) and FVG zones.

    Args:
        time_range: "hour", "today", or "week"

    Returns:
        Tuple of (List of ChartMarker objects, List of FVGZone objects)

    TODO: Connect to Epic 2 pattern detection results
    """
    logger.warning("Using mock pattern data - connect to Epic 2")

    now = datetime.now()

    # Return point markers (MSS, sweeps)
    markers = [
        ChartMarker(
            timestamp=now - timedelta(minutes=30),
            price=4510.0,
            marker_type=MarkerType.MSS_BULLISH,
            signal_id="MSS-001"
        ),
        ChartMarker(
            timestamp=now - timedelta(minutes=10),
            price=4502.0,
            marker_type=MarkerType.SWEEP,
            signal_id="SWEEP-001"
        )
    ]

    # Return FVG zones as shaded rectangles
    zones = [
        FVGZone(
            start_time=now - timedelta(minutes=25),
            end_time=now - timedelta(minutes=15),
            top_price=4515.0,
            bottom_price=4500.0,
            direction="bullish"
        ),
        FVGZone(
            start_time=now - timedelta(minutes=35),
            end_time=now - timedelta(minutes=30),
            top_price=4495.0,
            bottom_price=4485.0,
            direction="bearish"
        )
    ]

    return markers, zones


def get_trade_markers(time_range: str = "today") -> List[TradeMarker]:
    """Get trade entry/exit markers.

    Args:
        time_range: "hour", "today", or "week"

    Returns:
        List of TradeMarker objects

    TODO: Connect to Epic 4 trade execution data
    """
    logger.warning("Using mock trade data - connect to Epic 4")

    now = datetime.now()
    return [
        TradeMarker(
            timestamp=now - timedelta(minutes=25),
            price=4510.0,
            trade_type="entry",
            signal_id="SIGNAL-001"
        ),
        TradeMarker(
            timestamp=now - timedelta(minutes=15),
            price=4525.0,
            trade_type="exit",
            pnl_usd=125.00,
            signal_id="SIGNAL-001"
        )
    ]


# ============================================================================
# Story 8.6: System Configuration Panel
# ============================================================================


@dataclass
class RiskLimits:
    """Risk limit configuration."""

    daily_loss_limit: float  # USD
    max_drawdown_pct: float  # Percentage
    per_trade_risk_pct: float  # Percentage
    max_position_contracts: int  # Number of contracts


@dataclass
class TimeWindow:
    """Time window configuration."""

    enabled: bool
    start_time: str  # HH:MM format
    end_time: str  # HH:MM format


@dataclass
class MLConfig:
    """ML configuration."""

    min_probability: float  # 0.0-1.0


@dataclass
class SystemConfig:
    """Complete system configuration."""

    risk_limits: RiskLimits
    london_am: TimeWindow
    ny_am: TimeWindow
    ny_pm: TimeWindow
    ml_config: MLConfig


def get_system_config() -> SystemConfig:
    """Get current system configuration from shared state.

    Returns:
        Current system configuration

    TODO: Connect to actual shared state from Epic 6 (configuration)
    """
    logger.warning("Using mock configuration data - connect to actual shared state")

    return SystemConfig(
        risk_limits=RiskLimits(
            daily_loss_limit=500.0,
            max_drawdown_pct=12.0,
            per_trade_risk_pct=2.0,
            max_position_contracts=5
        ),
        london_am=TimeWindow(
            enabled=True,
            start_time="02:00",
            end_time="05:00"
        ),
        ny_am=TimeWindow(
            enabled=True,
            start_time="09:30",
            end_time="11:00"
        ),
        ny_pm=TimeWindow(
            enabled=True,
            start_time="13:30",
            end_time="15:30"
        ),
        ml_config=MLConfig(
            min_probability=0.65
        )
    )


def get_default_config() -> SystemConfig:
    """Get default system configuration.

    Returns:
        Default system configuration

    TODO: Read from config/default_config.yaml
    """
    logger.warning("Using mock default configuration - read from config/default_config.yaml")

    return SystemConfig(
        risk_limits=RiskLimits(
            daily_loss_limit=500.0,
            max_drawdown_pct=12.0,
            per_trade_risk_pct=2.0,
            max_position_contracts=5
        ),
        london_am=TimeWindow(
            enabled=True,
            start_time="02:00",
            end_time="05:00"
        ),
        ny_am=TimeWindow(
            enabled=True,
            start_time="09:30",
            end_time="11:00"
        ),
        ny_pm=TimeWindow(
            enabled=True,
            start_time="13:30",
            end_time="15:30"
        ),
        ml_config=MLConfig(
            min_probability=0.65
        )
    )


def validate_config(config: SystemConfig) -> tuple[bool, str]:
    """Validate configuration values.

    Args:
        config: Configuration to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate risk limits are positive
    if config.risk_limits.daily_loss_limit <= 0:
        return False, "Daily loss limit must be positive"

    if config.risk_limits.max_drawdown_pct <= 0 or config.risk_limits.max_drawdown_pct > 100:
        return False, "Max drawdown must be between 0-100%"

    if config.risk_limits.per_trade_risk_pct <= 0 or config.risk_limits.per_trade_risk_pct > 100:
        return False, "Per-trade risk must be between 0-100%"

    if config.risk_limits.max_position_contracts <= 0:
        return False, "Max position contracts must be positive"

    # Validate time windows
    def validate_time_window(tw: TimeWindow, name: str) -> tuple[bool, str]:
        if not tw.enabled:
            return True, ""
        try:
            start = datetime.strptime(tw.start_time, "%H:%M")
            end = datetime.strptime(tw.end_time, "%H:%M")
            if start >= end:
                return False, f"{name}: Start time must be before end time"
        except ValueError:
            return False, f"{name}: Invalid time format (use HH:MM)"
        return True, ""

    valid, msg = validate_time_window(config.london_am, "London AM")
    if not valid:
        return False, msg

    valid, msg = validate_time_window(config.ny_am, "NY AM")
    if not valid:
        return False, msg

    valid, msg = validate_time_window(config.ny_pm, "NY PM")
    if not valid:
        return False, msg

    # Validate ML threshold
    if config.ml_config.min_probability < 0.0 or config.ml_config.min_probability > 1.0:
        return False, "ML threshold must be between 0.0-1.0"

    return True, ""


def save_system_config(
    new_config: SystemConfig,
    password: str
) -> bool:
    """Save system configuration to shared state.

    Args:
        new_config: New configuration to save
        password: Password for confirmation

    Returns:
        True if save successful, False otherwise

    TODO: Implement actual password verification
    TODO: Write to data/state/config.json
    TODO: Send notification via Epic 6
    """
    # Mock password verification
    if password != "test_password":
        logger.warning("Password verification failed - mock only")
        return False

    # Validate configuration
    is_valid, error_msg = validate_config(new_config)
    if not is_valid:
        logger.error(f"Configuration validation failed: {error_msg}")
        return False

    # Get old configuration for logging
    old_config = get_system_config()

    # Log changes
    logger.info(
        "Configuration change",
        extra={
            "daily_loss_limit": (old_config.risk_limits.daily_loss_limit, new_config.risk_limits.daily_loss_limit),
            "max_drawdown_pct": (old_config.risk_limits.max_drawdown_pct, new_config.risk_limits.max_drawdown_pct),
            "per_trade_risk_pct": (old_config.risk_limits.per_trade_risk_pct, new_config.risk_limits.per_trade_risk_pct),
            "max_position_contracts": (old_config.risk_limits.max_position_contracts, new_config.risk_limits.max_position_contracts),
            "ml_min_probability": (old_config.ml_config.min_probability, new_config.ml_config.min_probability),
        }
    )

    # TODO: Save to shared state (data/state/config.json)
    # TODO: Send notification "Configuration updated: {changes}"
    logger.info("Configuration saved successfully (mock)")

    return True


# =============================================================================
# Story 8.7: Health Indicator Data Models and Readers
# =============================================================================

@dataclass
class APIConnectionStatus:
    """API connection status."""

    connected: bool
    last_ping_time: datetime  # When last successful ping occurred
    ping_latency_ms: float  # Latency in milliseconds


@dataclass
class ResourceUsage:
    """System resource usage metrics."""

    cpu_percent: float  # 0.0-100.0
    memory_percent: float  # 0.0-100.0
    disk_percent: float  # 0.0-100.0


@dataclass
class PipelineComponentStatus:
    """Pipeline component status."""

    component_name: str  # "Data Flow", "Signal Detection", etc.
    is_healthy: bool
    last_execution_time: datetime
    error_count: int  # Number of errors since last restart


@dataclass
class SystemHealth:
    """Complete system health status."""

    api_status: APIConnectionStatus
    resources: ResourceUsage
    data_flow_status: PipelineComponentStatus
    signal_detection_status: PipelineComponentStatus
    ml_prediction_status: PipelineComponentStatus
    execution_status: PipelineComponentStatus
    active_alerts_count: int
    system_uptime: str  # "Xh Ym" or "Xm"
    last_restart_time: datetime


def get_system_health() -> SystemHealth:
    """Get current system health from shared state.

    Returns:
        Current system health status

    TODO: Connect to actual health monitoring from Epic 6
    """
    logger.warning("Using mock health data - connect to Epic 6 monitoring")

    now = datetime.now()

    return SystemHealth(
        api_status=APIConnectionStatus(
            connected=True,
            last_ping_time=now - timedelta(seconds=5),
            ping_latency_ms=45.0
        ),
        resources=ResourceUsage(
            cpu_percent=35.2,
            memory_percent=62.8,
            disk_percent=45.1
        ),
        data_flow_status=PipelineComponentStatus(
            component_name="Data Flow",
            is_healthy=True,
            last_execution_time=now - timedelta(seconds=2),
            error_count=0
        ),
        signal_detection_status=PipelineComponentStatus(
            component_name="Signal Detection",
            is_healthy=True,
            last_execution_time=now - timedelta(seconds=2),
            error_count=0
        ),
        ml_prediction_status=PipelineComponentStatus(
            component_name="ML Prediction",
            is_healthy=True,
            last_execution_time=now - timedelta(seconds=2),
            error_count=0
        ),
        execution_status=PipelineComponentStatus(
            component_name="Execution",
            is_healthy=True,
            last_execution_time=now - timedelta(minutes=5),
            error_count=0
        ),
        active_alerts_count=0,
        system_uptime="4h 23m",
        last_restart_time=now - timedelta(hours=4, minutes=23)
    )


def get_alerts_summary() -> dict:
    """Get summary of active alerts.

    Returns:
        Dictionary with 'count' key containing number of active alerts

    TODO: Connect to actual alert tracking from Epic 6
    """
    logger.warning("Using mock alert data - connect to Epic 6 alert tracking")

    return {
        "count": 0,
        "summary": "No active alerts"
    }


def get_resource_color(usage_percent: float) -> str:
    """Get color code based on usage percentage.

    Args:
        usage_percent: Usage percentage (0-100)

    Returns:
        Hex color code: #00FF00 (green), #FFCC00 (yellow), #FF0000 (red)
    """
    if usage_percent < 80:
        return "#00FF00"  # Green
    elif usage_percent < 90:
        return "#FFCC00"  # Yellow
    else:
        return "#FF0000"  # Red


def format_resource_usage(usage_percent: float) -> tuple[str, str]:
    """Format resource usage with color.

    Args:
        usage_percent: Usage percentage (0-100)

    Returns:
        Tuple of (color_hex, emoji_indicator)
    """
    color = get_resource_color(usage_percent)
    emoji = "✅" if usage_percent < 80 else "⚠️" if usage_percent < 90 else "🔴"
    return color, emoji


def is_data_stale(last_ping_time: datetime) -> bool:
    """Check if data is stale (> 30 seconds old).

    Args:
        last_ping_time: Last successful data timestamp

    Returns:
        True if data is stale (> 30 seconds old), False otherwise
    """
    age_seconds = calculate_data_age(last_ping_time)
    return age_seconds > 30


def calculate_data_age(last_ping_time: datetime) -> int:
    """Calculate data age in seconds.

    Args:
        last_ping_time: Last successful data timestamp

    Returns:
        Age in seconds
    """
    return int((datetime.now() - last_ping_time).total_seconds())


# ============================================================================
# Story 8.8: Manual Trade Submission Form
# ============================================================================


@dataclass
class ManualTradeRequest:
    """Manual trade submission request."""

    direction: str  # "Buy" or "Sell"
    quantity: int  # Number of contracts (1-5)
    order_type: str  # "Market" or "Limit"
    limit_price: Optional[float]  # Required if order_type is "Limit"
    submit_time: datetime  # When trade was submitted
    submitted_by: str  # User identifier


@dataclass
class TradePreview:
    """Trade risk and barrier preview."""

    dollar_risk: float  # Risk amount in dollars
    stop_loss_price: float  # Stop loss price (1.2x ATR)
    upper_barrier_price: float  # Upper barrier (2.5x ATR)
    lower_barrier_price: float  # Lower barrier (1.2x ATR)
    vertical_barrier_time: datetime  # 45 minutes from entry
    margin_required: float  # Margin requirement for trade
    margin_sufficient: bool  # Whether account has sufficient margin
    position_size_valid: bool  # Whether quantity < 5 contracts
    per_trade_risk_valid: bool  # Whether risk < 2% equity
    validation_errors: List[str]  # List of validation error messages


@dataclass
class OrderSubmissionResult:
    """Result of order submission."""

    success: bool
    order_id: Optional[str] = None
    error: Optional[str] = None


def validate_position_size(quantity: int, max_position: int = 5) -> tuple[bool, str]:
    """Validate position size limit.

    Args:
        quantity: Number of contracts
        max_position: Maximum allowed position (default 5)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if quantity <= 0:
        return False, "Quantity must be greater than 0"
    if quantity > max_position:
        return False, f"Quantity exceeds maximum position size ({max_position} contracts)"
    return True, ""


def validate_per_trade_risk(
    dollar_risk: float,
    account_equity: float,
    max_risk_percent: float = 0.02
) -> tuple[bool, str]:
    """Validate per-trade risk limit.

    Args:
        dollar_risk: Risk amount in dollars
        account_equity: Total account equity
        max_risk_percent: Maximum risk as fraction of equity (default 2%)

    Returns:
        Tuple of (is_valid, error_message)
    """
    max_risk_dollars = account_equity * max_risk_percent
    if dollar_risk > max_risk_dollars:
        return False, f"Risk ${dollar_risk:.2f} exceeds 2% equity limit (${max_risk_dollars:.2f})"
    return True, ""


def validate_margin_requirement(
    quantity: int,
    account_equity: float,
    margin_per_contract: float = 500.0  # MNQ margin approx $500/contract
) -> tuple[bool, str, float]:
    """Validate margin requirement.

    Args:
        quantity: Number of contracts
        account_equity: Total account equity
        margin_per_contract: Margin required per contract

    Returns:
        Tuple of (is_valid, error_message, margin_required)
    """
    margin_required = quantity * margin_per_contract
    if margin_required > account_equity:
        return False, f"Insufficient margin: need ${margin_required:.2f}, have ${account_equity:.2f}", margin_required
    return True, "", margin_required


def calculate_trade_preview(
    request: ManualTradeRequest,
    current_price: float,
    atr: float,
    account_equity: float
) -> TradePreview:
    """Calculate trade preview with risk and barriers.

    Args:
        request: Manual trade request
        current_price: Current market price
        atr: Average True Range for barrier calculations
        account_equity: Total account equity

    Returns:
        Trade preview with risk and barrier levels
    """
    # Calculate stop loss (1.2x ATR)
    if request.direction == "Buy":
        stop_loss_price = current_price - (1.2 * atr)
        upper_barrier_price = current_price + (2.5 * atr)
        lower_barrier_price = stop_loss_price
        dollar_risk = abs(current_price - stop_loss_price) * request.quantity * 20  # MNQ $20/tick
    else:  # Sell
        stop_loss_price = current_price + (1.2 * atr)
        upper_barrier_price = stop_loss_price
        lower_barrier_price = current_price - (2.5 * atr)
        dollar_risk = abs(stop_loss_price - current_price) * request.quantity * 20

    # Vertical barrier (45 minutes from now)
    vertical_barrier_time = datetime.now() + timedelta(minutes=45)

    # Validate position size
    position_size_valid, position_error = validate_position_size(request.quantity)

    # Validate per-trade risk
    per_trade_risk_valid, risk_error = validate_per_trade_risk(dollar_risk, account_equity)

    # Validate margin
    margin_valid, margin_error, margin_required = validate_margin_requirement(request.quantity, account_equity)

    # Validate limit price for limit orders
    limit_price_valid = True
    limit_price_error = None
    if request.order_type == "Limit" and request.limit_price is None:
        limit_price_valid = False
        limit_price_error = "Limit price required for limit orders"

    # Collect validation errors
    validation_errors = []
    if not position_size_valid:
        validation_errors.append(position_error)
    if not per_trade_risk_valid:
        validation_errors.append(risk_error)
    if not margin_valid:
        validation_errors.append(margin_error)
    if not limit_price_valid:
        validation_errors.append(limit_price_error)

    return TradePreview(
        dollar_risk=dollar_risk,
        stop_loss_price=stop_loss_price,
        upper_barrier_price=upper_barrier_price,
        lower_barrier_price=lower_barrier_price,
        vertical_barrier_time=vertical_barrier_time,
        margin_required=margin_required,
        margin_sufficient=margin_valid,
        position_size_valid=position_size_valid,
        per_trade_risk_valid=per_trade_risk_valid,
        validation_errors=validation_errors
    )


def submit_manual_trade(request: ManualTradeRequest) -> OrderSubmissionResult:
    """Submit manual trade to execution system.

    Args:
        request: Validated manual trade request

    Returns:
        Order submission result

    TODO: Connect to Epic 4 TradeExecutor for actual order submission
    """
    logger.warning("Using mock order submission - connect to Epic 4")

    # Mock successful submission
    return OrderSubmissionResult(
        success=True,
        order_id=f"MANUAL-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )


def get_current_price() -> float:
    """Get current MNQ price.

    Returns:
        Current market price

    TODO: Connect to actual data pipeline
    """
    logger.warning("Using mock current price - connect to data pipeline")
    return 11500.0  # Typical MNQ price


def get_current_atr() -> float:
    """Get current Average True Range.

    Returns:
        Current ATR value

    TODO: Connect to actual ML pipeline for ATR calculation
    """
    logger.warning("Using mock ATR - connect to ML pipeline")
    return 50.0  # Typical MNQ ATR


def validate_password(password: str) -> bool:
    """Validate system password.

    Args:
        password: Password to validate

    Returns:
        True if password is correct

    TODO: Connect to actual password from Story 8.6
    """
    logger.warning("Using mock password validation - connect to Story 8.6")
    return password == "admin123"  # Mock password

