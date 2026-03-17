"""Risk management components for trading.

This package contains components for:
- Position sizing based on risk parameters
- Risk limit enforcement
- Circuit breakers
"""

from src.risk.position_sizer import PositionSizer, PositionSizeResult

__all__ = [
    "PositionSizer",
    "PositionSizeResult",
]
