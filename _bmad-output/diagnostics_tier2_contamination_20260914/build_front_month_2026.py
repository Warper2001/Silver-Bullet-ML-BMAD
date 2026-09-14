"""Build a front-month replacement for mnq_1min_2026_ytd.csv, Jan-Feb 2026 only (replay_plan.md).

Source: raw TradeStation 1-min records in /root/mnq_historical.json stamped in
[2026-01-01, 2026-03-01) UTC; all are MNQH26 (asserted). Output matches the CSV schema the
Tier2 loader reads: timestamp, open, high, low, close, volume, notional with
notional = close * volume * 20, the convention of both existing CSVs.

Run: .venv/bin/python _bmad-output/diagnostics_tier2_contamination_20260914/build_front_month_2026.py <out.csv>
"""
from __future__ import annotations

import re
import sys

import pandas as pd

RAW = "/root/mnq_historical.json"
LO, HI = "2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z"
FIELD = re.compile(r'^\s*"(High|Low|Open|Close|TimeStamp|TotalVolume|Contract)":\s*"?([^",]*)"?,?\s*$')


def main(out: str) -> int:
    rows, cur = [], {}
    with open(RAW) as f:
        for line in f:
            m = FIELD.match(line)
            if not m:
                continue
            k, v = m.groups()
            cur[k] = v
            if k == "Contract":
                ts = cur["TimeStamp"]
                if LO <= ts < HI:
                    rows.append((ts, float(cur["Open"]), float(cur["High"]), float(cur["Low"]),
                                 float(cur["Close"]), int(float(cur["TotalVolume"])), v))
                cur = {}
    d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "contract"])
    assert set(d["contract"]) == {"MNQH26"}, set(d["contract"])
    assert d["ts"].is_monotonic_increasing and not d["ts"].duplicated().any()
    d["timestamp"] = pd.to_datetime(d["ts"], utc=True).map(lambda t: t.isoformat())
    d["notional"] = d["close"] * d["volume"] * 20
    d[["timestamp", "open", "high", "low", "close", "volume", "notional"]].to_csv(out, index=False)
    print(f"{len(d)} rows {d['timestamp'].iloc[0]} -> {d['timestamp'].iloc[-1]}  median vol/min {d['volume'].median()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
