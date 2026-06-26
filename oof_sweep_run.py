"""Out-of-fold ceiling probe — SL x TP grid sweep on ONE instrument, in-sample only.

Runs the frozen YANK structural engine (ML-off) over an SL x TP grid on the
in-sample window (2025-05-19..2026-02-28). NEVER touches the sealed holdout.
Saves per-config trade lists (entry_time, exit_type, pnl) for walk-forward
analysis by oof_analyze.py. Bars are loaded ONCE and reused across configs.

Usage: PYTHONPATH=. .venv/bin/python oof_sweep.py <hg|pl>
"""
import sys, csv, time, asyncio
from datetime import datetime, timezone
from pathlib import Path
import backtest_tier2_1year_validation as bt

INST = sys.argv[1]
SL_GRID = [2.0, 3.0, 5.0]
TP_GRID = [6.0, 8.0, 10.0, 12.0]
START = datetime(2025, 5, 19, tzinfo=timezone.utc)
END = datetime(2026, 2, 28, 23, 59, 59, tzinfo=timezone.utc)
HOLDOUT = datetime(2026, 3, 1, tzinfo=timezone.utc)
OUTDIR = Path(f"data/reports/oof_sweep/{INST}")
OUTDIR.mkdir(parents=True, exist_ok=True)

inst = bt.INSTRUMENTS[INST]
symbol = inst["symbol"]

bars = []
for csv_path, _fs, _fe in inst["files"]:
    bars += bt.load_bars(Path(csv_path), start=START, end=END)
bars.sort(key=lambda b: b.timestamp)
assert all(b.timestamp < HOLDOUT for b in bars), "HOLDOUT LEAK — aborting"
print(f"{INST}: {len(bars):,} bars  {bars[0].timestamp.date()}..{bars[-1].timestamp.date()}  symbol={symbol}", flush=True)

for sl in SL_GRID:
    for tp in TP_GRID:
        t0 = time.time()
        overrides = {**bt.STRUCTURAL_OVERRIDES, "sl_multiplier": sl, "tp_multiplier": tp}
        trades = asyncio.run(bt.run_backtest(bars, ml_threshold=0.0, symbol=symbol, config_overrides=overrides))
        out = OUTDIR / f"SL{sl}_TP{tp}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["entry_time", "exit_type", "pnl"])
            for t in trades:
                w.writerow([t.entry_time.isoformat(), t.exit_type, round(t.pnl, 2)])
        net = sum(t.pnl for t in trades)
        print(f"  SL{sl}/TP{tp}: {len(trades):>3} trades  net=${net:>9,.0f}  ({time.time()-t0:.0f}s)", flush=True)

print("SWEEP DONE", flush=True)
