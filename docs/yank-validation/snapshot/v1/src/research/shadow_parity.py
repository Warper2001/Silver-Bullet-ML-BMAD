"""Hash-chained shadow-parity logger for the ProjectX data-feed migration.

Stage-1 (shadow) deliverable: while a bot keeps TradeStation as its SIGNAL source,
it also fetches ProjectX bars in parallel and logs a per-minute TS-vs-PX comparison
here. This accumulates live parity evidence (the cutover gate) without changing any
trade behaviour. Self-contained (no import side effects); mirrors the SHA-256
hash-chain style of mim_nb_live.ChainedCsv so the artifact is tamper-evident.

Logging discipline:
  - one row per SETTLED minute (>= SETTLE_LAG_MIN old, so both feeds have delivered it)
  - only minutes newer than the last logged minute (idempotent across repeated polls)
so the file grows ~1 row/min during market hours, never re-logging the same minute.
"""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

TICK = 0.25                # MNQ tick — OHLC parity tolerance flag
SETTLE_LAG_MIN = 2         # only log minutes this many minutes old (both feeds settled)

FIELDS = [
    "logged_at", "minute", "coverage",
    "ts_open", "ts_high", "ts_low", "ts_close", "ts_vol",
    "px_open", "px_high", "px_low", "px_close", "px_vol",
    "d_open", "d_high", "d_low", "d_close", "d_vol",
    "ohlc_max_abs", "within_tick", "px_fetch_ms", "px_error", "chain",
]


def bars_by_minute(rows) -> dict:
    """Index TradeStation-shaped dicts ({TimeStamp,Open,High,Low,Close,TotalVolume})
    by their minute string."""
    out = {}
    for r in rows:
        out[r["TimeStamp"]] = (float(r["Open"]), float(r["High"]), float(r["Low"]),
                               float(r["Close"]), float(r.get("TotalVolume", 0) or 0))
    return out


class ShadowParityLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._chain = "GENESIS"
        self._last_minute = ""
        if self.path.exists():
            try:
                with self.path.open() as f:
                    last = None
                    for last in csv.DictReader(f):
                        pass
                    if last:
                        self._chain = last.get("chain", "GENESIS")
                        self._last_minute = last.get("minute", "")
            except Exception:
                pass

    def _append(self, row: dict) -> None:
        payload = "|".join(str(row.get(k, "")) for k in FIELDS if k != "chain")
        self._chain = hashlib.sha256((self._chain + "|" + payload).encode()).hexdigest()[:16]
        row["chain"] = self._chain
        write_header = not self.path.exists()
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)

    def log_poll(self, ts_by_min: dict, px_by_min: dict, now_utc: datetime,
                 fetch_ms: float, error: str = "") -> int:
        """Log settled, not-yet-logged minutes. Returns number of rows written."""
        cutoff = (now_utc - timedelta(minutes=SETTLE_LAG_MIN)).strftime("%Y-%m-%dT%H:%M:%SZ")
        minutes = sorted(set(ts_by_min) | set(px_by_min))
        n = 0
        for m in minutes:
            if m > cutoff or m <= self._last_minute:
                continue
            t = ts_by_min.get(m)
            p = px_by_min.get(m)
            coverage = "both" if (t and p) else ("ts_only" if t else "px_only")
            row = {"logged_at": datetime.now(timezone.utc).isoformat(), "minute": m,
                   "coverage": coverage, "px_fetch_ms": round(fetch_ms, 1), "px_error": error}
            if t:
                row.update(ts_open=t[0], ts_high=t[1], ts_low=t[2], ts_close=t[3], ts_vol=t[4])
            if p:
                row.update(px_open=p[0], px_high=p[1], px_low=p[2], px_close=p[3], px_vol=p[4])
            if t and p:
                d = [abs(t[i] - p[i]) for i in range(4)]
                row.update(d_open=round(t[0]-p[0], 4), d_high=round(t[1]-p[1], 4),
                           d_low=round(t[2]-p[2], 4), d_close=round(t[3]-p[3], 4),
                           d_vol=round(t[4]-p[4], 1), ohlc_max_abs=round(max(d), 4),
                           within_tick=int(max(d) <= TICK))
            self._append(row)
            self._last_minute = m
            n += 1
        return n
