"""Generated prices and explicit schedules without input files."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .engine import MINUTE, Session


def bundled_fixture() -> tuple[list[dict[str, Any]], list[Session]]:
    # Explicit synthetic dates straddle DST; the sixth session is short.
    sessions = [
        Session(
            pd.Timestamp(f"{date} 09:30", tz="America/New_York"),
            pd.Timestamp(f"{date} {end}", tz="America/New_York"),
        )
        for date, end in (
            ("2026-03-02", "16:00"),
            ("2026-03-03", "16:00"),
            ("2026-03-04", "16:00"),
            ("2026-03-05", "16:00"),
            ("2026-03-06", "16:00"),
            ("2026-03-09", "11:00"),
        )
    ]
    rows: list[dict[str, Any]] = []
    for session in sessions:
        for stamp in pd.date_range(
            session.opening, session.close - MINUTE, freq="min"
        ):
            n = len(rows)
            price = 20000 + (n // 60) * 0.25
            close = price + (n % 3 - 1) * 0.25
            rows.append(
                dict(
                    timestamp=stamp,
                    contract="SYNTHETIC_MNQ",
                    open=price,
                    high=max(price, close) + 0.25,
                    low=min(price, close) - 0.25,
                    close=close,
                    volume=10 + n % 5,
                )
            )
    return rows, sessions
