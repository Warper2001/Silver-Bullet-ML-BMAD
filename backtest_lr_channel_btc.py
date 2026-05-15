#!/usr/bin/env python3
"""LR Channel BTC Backtest — parametric entry/exit exploration.

Uses three OLS linear-regression channels (300/100/30 bars) over Kraken
1-minute BTC/USD data.  Entry trigger, MTF slope filter, and channel lengths
are all configurable via BacktestConfig.

Exit: fixed — close crosses above 300-bar midline (midline return).
Fill: next-bar open (bar-close confirmed signal, no intrabar fills).

Output:
  data/reports/backtest_lr_channel_btc_{ts}.csv   — per-trade detail
  stdout — summary metrics

Usage:
  .venv/bin/python backtest_lr_channel_btc.py
  .venv/bin/python backtest_lr_channel_btc.py --entry upper --no-mtf-filter
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.models import DollarBar
from src.research.lr_channel import LRChannel, compute_lr_channel, detect_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

BTC_TICK = 0.5
COMMISSION = 2.0          # $2 per side (round-trip $4)
CSV_PATH = "data/kraken/PF_XBTUSD_1min.csv"
REPORTS_DIR = Path("data/reports")


@dataclass
class BacktestConfig:
    entry_line: Literal["lower", "mid", "upper"] = "lower"
    lengths: tuple = (300, 100, 30)
    mtf_slope_filter: bool = True
    max_hold_bars: int = 480     # 8 hours at 1-min bars; forced exit if midline never hit
    contract_value: float = 1.0  # $1/point per contract (PF_XBTUSD USD-linear)
    position_size: int = 1


def round_tick(price: float) -> float:
    return round(round(price / BTC_TICK) * BTC_TICK, 10)


def load_csv_as_bars(csv_path: str) -> list[DollarBar]:
    logger.info(f"Loading 1-minute BTC data from {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    bars: list[DollarBar] = []
    skipped = 0
    for row in df.itertuples(index=False):
        try:
            if row.open == 0 or row.high == 0 or row.low == 0 or row.close == 0:
                skipped += 1
                continue
            ts = row.timestamp.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            bars.append(DollarBar(
                timestamp=ts,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                notional_value=float(row.close),
                is_forward_filled=False,
            ))
        except Exception as exc:
            skipped += 1
            logger.debug(f"Skipped row: {exc}")

    logger.info(f"Loaded {len(bars):,} bars ({skipped} skipped)")
    return bars



def _run_sequential(
    bars: list[DollarBar],
    closes: np.ndarray,
    timestamps,
    ch300: LRChannel,
    entries: list[dict],
    exits: list[dict],
    cfg: BacktestConfig,
) -> list[dict]:
    """Sequential, non-overlapping trade simulation over all entry signals.

    All fills at next-bar open: entry fills at bars[entry_bar+1].open,
    exit fills at bars[signal_bar+1].open (signal confirmed at signal_bar close).
    """
    trades: list[dict] = []
    last_exit_bar = -1

    for signal in entries:
        entry_bar = signal["bar_idx"]
        if entry_bar <= last_exit_bar:
            continue   # skip entries while a previous trade is still open

        fill_bar = entry_bar + 1
        if fill_bar >= len(bars) - 1:   # need fill_bar+1 to exist for exit fill
            break

        entry_price = round_tick(bars[fill_bar].open)
        exit_price = None
        exit_bar = None
        exit_reason = "timeout"

        # Scan for midline return; signal at bar j → fill at bars[j+1].open
        scan_end = min(fill_bar + cfg.max_hold_bars, len(bars) - 2)  # need j+1 valid
        for j in range(fill_bar + 1, scan_end + 1):
            if (
                not np.isnan(ch300.mid[j])
                and not np.isnan(ch300.mid[j - 1])
                and closes[j - 1] <= ch300.mid[j - 1]
                and closes[j] > ch300.mid[j]
            ):
                exit_price = round_tick(bars[j + 1].open)
                exit_bar = j + 1
                exit_reason = "midline_return"
                break

        if exit_price is None:
            timeout_bar = min(fill_bar + cfg.max_hold_bars + 1, len(bars) - 1)
            exit_price = round_tick(bars[timeout_bar].open)
            exit_bar = timeout_bar

        pnl = (exit_price - entry_price) * cfg.contract_value * cfg.position_size
        pnl -= COMMISSION * 2

        last_exit_bar = exit_bar
        trades.append({
            "entry_ts": bars[fill_bar].timestamp,
            "exit_ts": bars[exit_bar].timestamp,
            "entry_bar": fill_bar,
            "exit_bar": exit_bar,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "trigger": signal["trigger"],
            "exit_reason": exit_reason,
            "hold_bars": exit_bar - fill_bar,
            "pnl": round(pnl, 2),
            "winner": pnl > 0,
        })

    return trades


def analyze_and_print(trades: list[dict], cfg: BacktestConfig) -> None:
    if not trades:
        print("\n--- LR Channel Backtest Results ---")
        print("No trades generated.")
        return

    df = pd.DataFrame(trades)
    n = len(df)
    winners = df["winner"].sum()
    win_rate = winners / n * 100
    gross_profit = df[df["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(df[df["pnl"] <= 0]["pnl"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_pnl = df["pnl"].sum()
    avg_hold = df["hold_bars"].mean()
    avg_win = df[df["pnl"] > 0]["pnl"].mean() if winners > 0 else 0.0
    avg_loss = df[df["pnl"] <= 0]["pnl"].mean() if (n - winners) > 0 else 0.0

    daily = df.groupby(df["entry_ts"].apply(lambda t: t.date()))["pnl"].sum()
    sharpe = (daily.mean() / daily.std() * (252 ** 0.5)) if daily.std() > 0 else 0.0

    print("\n--- LR Channel Backtest Results ---")
    print(f"Config : entry_line={cfg.entry_line!r}  "
          f"lengths={cfg.lengths}  "
          f"mtf_slope_filter={cfg.mtf_slope_filter}")
    print(f"Trades     : {n}")
    print(f"Win rate   : {win_rate:.1f}%")
    print(f"Profit factor: {pf:.3f}")
    print(f"Total PnL  : ${total_pnl:,.2f}")
    print(f"Avg win    : ${avg_win:,.2f}  |  Avg loss: ${avg_loss:,.2f}")
    print(f"Avg hold   : {avg_hold:.1f} bars ({avg_hold:.0f} min)")
    print(f"Sharpe (daily, ann.) : {sharpe:.2f}")


def save_report(trades: list[dict], cfg: BacktestConfig) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"backtest_lr_channel_btc_{ts}.csv"
    pd.DataFrame(trades).to_csv(out, index=False)
    logger.info(f"Report saved → {out}")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="LR Channel BTC backtest")
    p.add_argument("--entry", choices=["lower", "mid", "upper"], default="lower")
    p.add_argument("--lengths", nargs=3, type=int, default=[300, 100, 30],
                   metavar=("L300", "L100", "L30"))
    p.add_argument("--no-mtf-filter", action="store_true")
    p.add_argument("--max-hold", type=int, default=480)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = BacktestConfig(
        entry_line=args.entry,
        lengths=tuple(args.lengths),
        mtf_slope_filter=not args.no_mtf_filter,
        max_hold_bars=args.max_hold,
    )

    bars = load_csv_as_bars(CSV_PATH)
    if not bars:
        logger.error("No bars loaded — check CSV path.")
        sys.exit(1)

    closes = np.array([b.close for b in bars], dtype=np.float64)
    timestamps = [b.timestamp for b in bars]
    l300, l100, l30 = cfg.lengths

    logger.info(f"Computing LR channels ({l300}/{l100}/{l30}) ...")
    ch300 = compute_lr_channel(closes, l300)
    ch100 = compute_lr_channel(closes, l100)
    ch30 = compute_lr_channel(closes, l30)

    logger.info("Detecting signals ...")
    entries, exits = detect_signals(
        closes, timestamps, ch300, ch100, ch30,
        entry_line=cfg.entry_line,
        mtf_slope_filter=cfg.mtf_slope_filter,
    )
    logger.info(f"Signals: {len(entries)} entries, {len(exits)} exits")

    trades = _run_sequential(bars, closes, timestamps, ch300, entries, exits, cfg)

    analyze_and_print(trades, cfg)
    if trades:
        save_report(trades, cfg)


if __name__ == "__main__":
    main()
