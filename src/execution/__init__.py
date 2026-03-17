"""Trade execution components for automated order submission.

This package contains components for:
- Order type selection based on position size
- Market order submission
- Limit order submission
- Order tracking and management
"""

from src.execution.order_type_selector import (
    OrderTypeDecision,
    OrderTypeSelector,
)
from src.execution.market_order_submitter import (
    MarketOrderSubmitter,
    OrderSubmissionError,
    OrderSubmissionResult,
)

__all__ = [
    "OrderTypeSelector",
    "OrderTypeDecision",
    "MarketOrderSubmitter",
    "OrderSubmissionError",
    "OrderSubmissionResult",
]
