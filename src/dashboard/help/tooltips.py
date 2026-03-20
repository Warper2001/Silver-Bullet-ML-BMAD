"""Tooltip definitions for dashboard metrics."""

TOOLTIPS = {
    # Overview page tooltips
    "equity": "Total account value including open positions and cash",
    "daily_pnl": "Today's profit or loss in USD (green=profit, red=loss)",
    "win_rate": "Percentage of winning trades: (Winning / Total) × 100",
    "drawdown": "Current loss vs daily loss limit - how close to hitting limit",
    "open_positions": "Number of active positions and total contracts",
    "system_uptime": "Time since last system restart",

    # Positions page tooltips
    "barrier_levels": "Upper (take-profit), Lower (stop-loss), Vertical (timeout) barriers",
    "vertical_barrier": "45-minute time limit from entry - position closes automatically",
    "manual_exit": "Click Exit to manually close position (requires password)",

    # Signals page tooltips
    "confidence": "1-5 stars based on pattern confluence (MSS + FVG + Sweep)",
    "ml_probability": "XGBoost prediction score (0-100%) - higher is better",

    # Charts page tooltips
    "chart_markers": "▲/▼ = MSS, 🟣 = Sweep, 🔵 = Entry, ✅/❌ = Exit",
    "color_coding": "Green=profit, Red=loss, Purple=sweep",

    # Settings page tooltips
    "daily_loss_limit": "Maximum daily loss - trading stops when reached",
    "max_drawdown": "Maximum % drop from account peak allowed",
    "per_trade_risk": "Maximum % of equity risked per trade (1-2% recommended)",
    "max_position": "Maximum number of contracts per position (3-5 recommended)",
    "ml_threshold": "Signals below this ML score are filtered (0.60-0.75 recommended)",
}


def get_tooltip(metric: str) -> str:
    """Get tooltip text for a metric.

    Args:
        metric: Metric name

    Returns:
        Tooltip text, or empty string if not found
    """
    return TOOLTIPS.get(metric, "")


def add_tooltip(label: str, tooltip_text: str) -> str:
    """Add tooltip to a label (for display in UI).

    Args:
        label: The label text
        tooltip_text: The tooltip content

    Returns:
        Label with tooltip indicator
    """
    # In Streamlit, we'll use a pattern like: "Label ℹ️"
    # where ℹ️ indicates help is available
    return f"{label} ℹ️"
