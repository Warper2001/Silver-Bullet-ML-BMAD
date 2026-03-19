"""Color scheme and styling utilities for Streamlit dashboard."""

# Trading colors
COLOR_PROFIT = "#00FF00"  # Green
COLOR_LOSS = "#FF0000"  # Red
COLOR_NEUTRAL = "#0080FF"  # Blue


def get_profit_color() -> str:
    """Return green color for profit/bullish indicators."""
    return COLOR_PROFIT


def get_loss_color() -> str:
    """Return red color for loss/bearish indicators."""
    return COLOR_LOSS


def get_neutral_color() -> str:
    """Return blue color for neutral indicators."""
    return COLOR_NEUTRAL


def format_pnl(value: float) -> str:
    """Format P&L value with color and sign."""
    color = get_profit_color() if value >= 0 else get_loss_color()
    sign = "+" if value > 0 else ""
    return f":{color}[{sign}${value:,.2f}]"
