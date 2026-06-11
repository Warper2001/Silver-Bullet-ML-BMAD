
from src.data.models import DollarBar
from datetime import datetime, timezone

try:
    bar = DollarBar(
        timestamp=datetime.now(timezone.utc),
        open=27000.0,
        high=27010.0,
        low=26990.0,
        close=27005.0,
        volume=100,
        notional_value=0  # This should fail
    )
    print("Success")
except Exception as e:
    print(f"Caught: {type(e).__name__}: {e}")
