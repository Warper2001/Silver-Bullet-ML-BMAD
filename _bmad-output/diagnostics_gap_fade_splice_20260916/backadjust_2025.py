"""Back-adjust the front-month 2025 bars so no gap is ever measured across two contracts.

The problem (measured in results.md): on the first session of a new contract the "overnight
gap" is prior-contract close -> new-contract open, so it folds in the calendar spread. In
2025 that happens 4 times and distorts one GAP-1 trade worth 9.4% of the sealed net.

The fix is Panama back-adjustment of the front-month rebuild:
  * segment the bars by contract (the emitting minute's label, from the pinned raw extract);
  * estimate each roll's spread from ADJACENT raw minutes where the label changes
    (delta = close(new at t+1) - close(old at t)), median over the roll window — the
    underlying moves little in a minute, and there are thousands of such pairs per roll
    because roll weeks interleave minute by minute;
  * shift every earlier segment by the cumulative spread, leaving the last segment as-is.

Because each shift is a constant within a segment, every intra-segment gap, range and
stop distance is unchanged: ONLY the four boundary sessions can change. The script
asserts that property against the unadjusted rebuild.

Volume and notional are copied unchanged; notional is no longer price-consistent after a
shift, and GAP-1 does not read it.

Inputs: mnq_1min_2025_frontmonth.csv (from rebuild_2025_frontmonth.py) + the pinned extract.
Output: mnq_1min_2025_frontmonth_adjusted.csv in this folder.

Run: .venv/bin/python _bmad-output/diagnostics_gap_fade_splice_20260916/backadjust_2025.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = Path("/root/Silver-Bullet-ML-BMAD")
EXTRACT = (MAIN / "_bmad-output/planning-artifacts/research"
           / "technical-yank-bar-provenance-and-pilot-evidence-g-2026-09-07/imports/provenance-raw-2025.jsonl")
SRC = HERE / "mnq_1min_2025_frontmonth.csv"
OUT = HERE / "mnq_1min_2025_frontmonth_adjusted.csv"


def raw_minutes() -> pd.DataFrame:
    rows = []
    for line in EXTRACT.open():
        b = json.loads(line)["bar"]
        rows.append((b["TimeStamp"], float(b["Close"]), b["Contract"]))
    d = pd.DataFrame(rows, columns=["ts", "close", "contract"])
    d["ts"] = pd.to_datetime(d["ts"], format="%Y-%m-%dT%H:%M:%SZ", utc=True)
    return d.sort_values("ts").reset_index(drop=True)


def roll_spreads(raw: pd.DataFrame) -> dict:
    """median(new - old) over adjacent minutes whose contract label changes."""
    prev_c = raw["contract"].shift()
    prev_close = raw["close"].shift()
    changed = prev_c.notna() & (raw["contract"] != prev_c)
    diffs: dict = defaultdict(list)
    for c_new, c_old, close_new, close_old in zip(raw.loc[changed, "contract"], prev_c[changed],
                                                  raw.loc[changed, "close"], prev_close[changed]):
        diffs[(c_old, c_new)].append(close_new - close_old)
    out = {}
    for (c_old, c_new), vals in diffs.items():
        out[f"{c_old}->{c_new}"] = {"n_pairs": len(vals), "median": float(np.median(vals)),
                                    "iqr": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))]}
    return out


def main() -> int:
    raw = raw_minutes()
    spreads = roll_spreads(raw)
    # contract of each emitted bar = label of the minute it is stamped at
    label = dict(zip(raw["ts"], raw["contract"]))
    bars = pd.read_csv(SRC)
    ts = pd.to_datetime(bars["timestamp"], utc=True, format="ISO8601")
    bars["contract"] = [label.get(t) for t in ts]
    missing = int(bars["contract"].isna().sum())
    bars["contract"] = bars["contract"].ffill().bfill()

    order = list(dict.fromkeys(bars["contract"]))          # chronological segments
    # cumulative offset: last segment unshifted; earlier ones shifted by the spreads after them
    pair_med = {k: v["median"] for k, v in spreads.items()}
    offsets, running = {order[-1]: 0.0}, 0.0
    for i in range(len(order) - 1, 0, -1):
        key = f"{order[i - 1]}->{order[i]}"
        if key not in pair_med:
            raise SystemExit(f"no adjacent-minute spread estimate for {key}")
        running += pair_med[key]
        offsets[order[i - 1]] = running

    adj = bars.copy()
    shift = adj["contract"].map(offsets)
    for col in ("open", "high", "low", "close"):
        adj[col] = (adj[col] + shift).round(2)
    adj[["timestamp", "open", "high", "low", "close", "volume", "notional"]].to_csv(OUT, index=False)

    # property check: within a segment nothing changed; every change is a constant per segment
    deltas = {c: sorted(set((adj.loc[bars["contract"] == c, "close"]
                             - bars.loc[bars["contract"] == c, "close"]).round(2)))
              for c in order}
    res = {"segments": order, "offsets_pts": offsets, "roll_spreads": spreads,
           "bars": len(bars), "bars_without_raw_label": missing,
           "per_segment_shift_is_constant": all(len(v) == 1 for v in deltas.values()),
           "observed_shifts": {c: v for c, v in deltas.items()},
           "out_csv": str(OUT)}
    (HERE / "backadjust_meta.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
