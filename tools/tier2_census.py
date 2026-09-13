#!/usr/bin/env python3
"""Signal census for the Tier2/YANK engine: an instrumented, faithful replay that records
every pending order the engine creates (filled or later expired).

How: wraps ``Tier2StreamingTrader._enter_trade`` and ``MetaLabelingFilter.predict_proba``
IN-PROCESS, then runs ``backtest_tier2_1year_validation.run_backtest``. No repository file
is edited. Replay persistence stays suppressed by run_backtest (TradeLogger(persist=False),
StatePersistence mocked, ML decision log disabled); the tool snapshots data/trades.db,
logs/tier2_trade_log.csv and logs/yank_ml_canary.csv before/after and reports whether they
changed. Live bots also write trades.db, so a change there must be inspected before it is
blamed on the replay.

Holdout: refuses any --end on/after backtest_tier2_1year_validation.HOLDOUT_CUTOFF.

Config pins: --pin KEY=VALUE applies a StrategyConfig override in memory only. To reproduce
the 2026-06-15 seal-138cab1 replay CSVs, pin max_daily_loss=-750 (the YAML now says -300).

Outputs (in --out-dir): signals_<tag>.csv, census_trades_<tag>.csv, census_meta_<tag>.json

Example (from the repo root; about 22 min per arm for 2025-05-19..2026-02-28):
  .venv/bin/python tools/tier2_census.py --ml-threshold 0.50 --start 2025-05-19 \
      --end 2026-02-28 --tag ml050 --pin max_daily_loss=-750 --out-dir <dir>

Time a two-week slice first (AGENTS.md): replay wall-clock is not reproducible.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GUARDED = ("data/trades.db", "logs/tier2_trade_log.csv", "logs/yank_ml_canary.csv")


def parse_pins(pins: list[str]) -> dict:
    """Parse KEY=VALUE pins; values become int, float, bool, or str."""
    out = {}
    for p in pins or []:
        if "=" not in p:
            raise ValueError(f"pin must be KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k:
            raise ValueError(f"empty pin key in {p!r}")
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def check_window(start: str, end: str, cutoff: datetime) -> tuple[datetime, datetime]:
    """Return the UTC window; raise if it reaches the sealed-holdout cutoff."""
    g_start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    g_end = datetime.fromisoformat(end).replace(tzinfo=timezone.utc, hour=23, minute=59, second=59)
    if g_end >= cutoff:
        raise SystemExit(f"refusing: --end {end} reaches the sealed-holdout cutoff {cutoff.date()}")
    if g_start > g_end:
        raise SystemExit("refusing: --start is after --end")
    return g_start, g_end


def guard_snapshot(root: Path) -> dict:
    snap = {}
    for p in GUARDED:
        q = root / p
        snap[p] = (q.stat().st_size, q.stat().st_mtime) if q.exists() else None
    return snap


def install_hooks(tier2_mod, signals: list, last_proba: dict) -> None:
    """Wrap the trader in-process so every newly created pending order is recorded."""
    orig_proba = tier2_mod.MetaLabelingFilter.predict_proba
    orig_enter = tier2_mod.Tier2StreamingTrader._enter_trade

    def proba_wrapper(self, features):
        p = orig_proba(self, features)
        last_proba["p"] = p
        return p

    async def enter_wrapper(self, fvg, bar, idx, is_backfill):
        before = self.active_trade
        await orig_enter(self, fvg, bar, idx, is_backfill)
        after = self.active_trade
        if after is not None and after is not before:
            signals.append({
                "signal_ts": after.entry_time.isoformat(),
                "direction": after.direction,
                "entry": after.entry_price,
                "sl": after.sl_price,
                "tp": after.tp_price,
                "gap": after.gap_size,
                "fvg_top": getattr(fvg, "high", None),
                "fvg_bot": getattr(fvg, "low", None),
                "contracts": after.contracts,
                "ml_proba": last_proba["p"],
            })
        last_proba["p"] = None

    tier2_mod.MetaLabelingFilter.predict_proba = proba_wrapper
    tier2_mod.Tier2StreamingTrader._enter_trade = enter_wrapper


async def run(args) -> dict:
    root = Path(args.repo_root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    import backtest_tier2_1year_validation as btv  # noqa: E402
    import src.research.tier2_streaming_working as tier2_mod  # noqa: E402

    g_start, g_end = check_window(args.start, args.end, btv.HOLDOUT_CUTOFF)
    pins = parse_pins(args.pin)
    signals: list = []
    install_hooks(tier2_mod, signals, {"p": None})

    guard0 = guard_snapshot(root)
    bars = []
    for csv_path, _fs, _fe in btv.INSTRUMENTS["mnq"]["files"]:
        bars += btv.load_bars(Path(csv_path), start=g_start, end=g_end)
    bars.sort(key=lambda b: b.timestamp)
    print(f"[{args.tag}] {len(bars):,} bars {bars[0].timestamp} -> {bars[-1].timestamp}", flush=True)

    t0 = time.time()
    trades = await btv.run_backtest(bars, ml_threshold=args.ml_threshold, symbol="MNQM26",
                                    config_overrides=pins or None)
    wall = time.time() - t0

    filled_ts = {t.entry_time.isoformat() for t in trades}
    for s in signals:
        s["filled"] = s["signal_ts"] in filled_ts

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"signals_{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(signals[0].keys()) if signals else ["signal_ts"])
        w.writeheader()
        w.writerows(signals)
    with open(out / f"census_trades_{args.tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entry_time", "exit_time", "direction", "entry_price", "exit_price", "exit_type", "bars_held", "pnl"])
        for t in trades:
            w.writerow([t.entry_time.isoformat(), t.exit_time.isoformat(), t.direction, t.entry_price,
                        t.exit_price, t.exit_type, t.bars_held, round(t.pnl, 2)])
    guard1 = guard_snapshot(root)
    meta = {"tag": args.tag, "ml_threshold": args.ml_threshold, "start": args.start, "end": args.end,
            "pins": pins, "bars": len(bars), "signals": len(signals), "filled": sum(s["filled"] for s in signals),
            "trades": len(trades), "wall_seconds": round(wall, 1),
            "live_files_unchanged": guard0 == guard1,
            "changed_files": [p for p in GUARDED if guard0[p] != guard1[p]]}
    (out / f"census_meta_{args.tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    return meta


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ml-threshold", type=float, required=True, help="0.50 = ML arm, 0.0 = no-ML arm")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC), must be before the holdout cutoff")
    ap.add_argument("--tag", required=True, help="suffix for output files, e.g. ml050")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pin", action="append", default=[], help="StrategyConfig override KEY=VALUE (repeatable)")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]),
                    help="repository whose engine, models and bars are replayed (default: this checkout)")
    return ap


if __name__ == "__main__":
    asyncio.run(run(build_parser().parse_args()))
