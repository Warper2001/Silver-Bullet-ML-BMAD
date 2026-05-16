#!/usr/bin/env python3
"""LR Channel BTC Grid Search — find optimal entry/exit conditions.

Sweeps:
  entry_line    : lower | mid | upper            (3 values)
  exit_type     : midline | upper_rail | timeout  (3 values)
  mtf_filter    : True | False                   (2 values)
  slow_length   : 200 | 300 | 400 | 500          (4 values)
Total            : 72 combinations

Medium (100) and fast (30) channel lengths are fixed.
All fills at next-bar open. Minimum 10 trades to appear in ranked output.

Output:
  data/reports/grid_search_lr_channel_{ts}.csv  — all 72 rows, sorted by PF
  stdout                                         — top-10 table
"""

import sys
import logging
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.research.lr_channel import LRChannel, compute_lr_channel, detect_signals
from backtest_lr_channel_btc import BacktestConfig, load_csv_as_bars, round_tick

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

CSV_PATH   = "data/kraken/PF_XBTUSD_1min.csv"
REPORTS_DIR = Path("data/reports")
COMMISSION  = 2.0   # per side; round-trip = 4.0
CONTRACT_VALUE = 1.0
POSITION_SIZE  = 1
MAX_HOLD_BARS  = 480
MEDIUM_LENGTH  = 100
FAST_LENGTH    = 30
MIN_TRADES     = 10

PARAM_GRID = {
    "entry_line"      : ["lower", "mid", "upper"],
    "exit_type"       : ["midline", "upper_rail", "timeout"],
    "mtf_slope_filter": [True, False],
    "slow_length"     : [200, 300, 400, 500],
}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _simulate_combo(
    bars,
    closes: np.ndarray,
    ch_slow: LRChannel,
    ch100: LRChannel,
    ch30: LRChannel,
    entry_line: str,
    exit_type: str,
    mtf_slope_filter: bool,
) -> list[dict]:
    """Non-overlapping simulation for one parameter combination."""
    timestamps = [b.timestamp for b in bars]
    entries, _ = detect_signals(
        closes, timestamps, ch_slow, ch100, ch30,
        entry_line=entry_line,
        mtf_slope_filter=mtf_slope_filter,
    )

    trades: list[dict] = []
    last_exit_bar = -1
    n = len(bars)

    for signal in entries:
        entry_bar = signal["bar_idx"]
        if entry_bar <= last_exit_bar:
            continue

        fill_bar = entry_bar + 1
        if fill_bar >= n - 1:   # need fill_bar+1 for exit fill
            break

        entry_price = round_tick(bars[fill_bar].open)
        exit_price  = None
        exit_bar    = None
        exit_reason = "timeout"

        if exit_type != "timeout":
            scan_end = min(fill_bar + MAX_HOLD_BARS, n - 2)
            for j in range(fill_bar + 1, scan_end + 1):
                if exit_type == "midline":
                    hit = (
                        not np.isnan(ch_slow.mid[j])
                        and not np.isnan(ch_slow.mid[j - 1])
                        and closes[j - 1] <= ch_slow.mid[j - 1]
                        and closes[j] > ch_slow.mid[j]
                    )
                else:  # upper_rail
                    hit = (
                        not np.isnan(ch_slow.upper[j])
                        and not np.isnan(ch_slow.upper[j - 1])
                        and closes[j - 1] <= ch_slow.upper[j - 1]
                        and closes[j] > ch_slow.upper[j]
                    )
                if hit:
                    exit_price  = round_tick(bars[j + 1].open)
                    exit_bar    = j + 1
                    exit_reason = exit_type
                    break

        if exit_price is None:
            timeout_bar = min(fill_bar + MAX_HOLD_BARS + 1, n - 1)
            exit_price  = round_tick(bars[timeout_bar].open)
            exit_bar    = timeout_bar

        pnl = (exit_price - entry_price) * CONTRACT_VALUE * POSITION_SIZE
        pnl -= COMMISSION * 2
        last_exit_bar = exit_bar

        trades.append({
            "entry_bar"  : fill_bar,
            "exit_bar"   : exit_bar,
            "entry_price": entry_price,
            "exit_price" : exit_price,
            "exit_reason": exit_reason,
            "hold_bars"  : exit_bar - fill_bar,
            "pnl"        : round(pnl, 2),
            "winner"     : pnl > 0,
        })

    return trades


def _summarise(
    trades: list[dict],
    slow_length: int,
    entry_line: str,
    exit_type: str,
    mtf_slope_filter: bool,
) -> dict:
    row: dict = {
        "slow_length"      : slow_length,
        "entry_line"       : entry_line,
        "exit_type"        : exit_type,
        "mtf_slope_filter" : mtf_slope_filter,
        "trade_count"      : len(trades),
        "win_rate_pct"     : 0.0,
        "profit_factor"    : 0.0,
        "avg_hold_bars"    : 0.0,
        "total_pnl"        : 0.0,
        "sharpe"           : 0.0,
    }
    if not trades:
        return row

    df = pd.DataFrame(trades)
    n  = len(df)
    winners     = df["winner"].sum()
    gross_win   = df[df["pnl"] > 0]["pnl"].sum()
    gross_loss  = abs(df[df["pnl"] <= 0]["pnl"].sum())

    row["trade_count"]   = n
    row["win_rate_pct"]  = round(winners / n * 100, 1)
    row["profit_factor"] = round(gross_win / gross_loss, 4) if gross_loss > 0 else float("inf")
    row["avg_hold_bars"] = round(df["hold_bars"].mean(), 1)
    row["total_pnl"]     = round(df["pnl"].sum(), 2)

    daily = df.groupby(
        df["entry_bar"].apply(lambda i: i // 1440)  # approx daily buckets
    )["pnl"].sum()
    if daily.std() > 0:
        row["sharpe"] = round(daily.mean() / daily.std() * (252 ** 0.5), 2)

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    bars = load_csv_as_bars(CSV_PATH)
    if not bars:
        logger.error("No bars loaded.")
        sys.exit(1)

    closes = np.array([b.close for b in bars], dtype=np.float64)

    # Pre-compute fixed channels once
    logger.info(f"Pre-computing ch{MEDIUM_LENGTH} and ch{FAST_LENGTH} ...")
    ch100 = compute_lr_channel(closes, MEDIUM_LENGTH)
    ch30  = compute_lr_channel(closes, FAST_LENGTH)

    # Pre-compute one slow channel per unique slow_length
    slow_lengths = PARAM_GRID["slow_length"]
    logger.info(f"Pre-computing slow channels {slow_lengths} ...")
    slow_channels = {L: compute_lr_channel(closes, L) for L in slow_lengths}

    # Run grid
    keys   = ["entry_line", "exit_type", "mtf_slope_filter", "slow_length"]
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    logger.info(f"Running {len(combos)} combinations ...")

    results = []
    for entry_line, exit_type, mtf_filter, slow_len in combos:
        trades = _simulate_combo(
            bars, closes,
            slow_channels[slow_len], ch100, ch30,
            entry_line, exit_type, mtf_filter,
        )
        results.append(_summarise(trades, slow_len, entry_line, exit_type, mtf_filter))

    # Save full results
    df_all = pd.DataFrame(results).sort_values("profit_factor", ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"grid_search_lr_channel_{ts}.csv"
    df_all.to_csv(out, index=False)
    logger.info(f"Full results saved → {out}")

    # Print top-10 (≥ MIN_TRADES)
    df_ranked = df_all[df_all["trade_count"] >= MIN_TRADES].head(10)

    if df_ranked.empty:
        print(f"\n⚠  No config met the {MIN_TRADES}-trade minimum. Full results saved to CSV.")
        return

    print(f"\n{'─'*88}")
    print(f" Top-{len(df_ranked)} configs  (min {MIN_TRADES} trades, ranked by PF)  —  {len(combos)} combos swept")
    print(f"{'─'*88}")
    header = (
        f"{'slow':>4}  {'entry':<6}  {'exit':<11}  {'mtf':<5}  "
        f"{'trades':>6}  {'WR%':>5}  {'PF':>6}  {'avg_hold':>8}  {'PnL':>9}"
    )
    print(header)
    print("─" * 88)
    for _, r in df_ranked.iterrows():
        pf_str = f"{r.profit_factor:.3f}" if r.profit_factor != float("inf") else "  inf"
        print(
            f"{int(r.slow_length):>4}  {r.entry_line:<6}  {r.exit_type:<11}  "
            f"{'Y' if r.mtf_slope_filter else 'N':<5}  "
            f"{int(r.trade_count):>6}  {r.win_rate_pct:>5.1f}  {pf_str:>6}  "
            f"{r.avg_hold_bars:>8.1f}  {r.total_pnl:>9.2f}"
        )
    print(f"{'─'*88}\n")


if __name__ == "__main__":
    main()
