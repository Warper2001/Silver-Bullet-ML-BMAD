#!/usr/bin/env python3
"""LR Channel BTC — Stop-Loss / Take-Profit Grid Search.

Extends the LR channel grid with SL% and TP% sweep dimensions.
All entry/exit/MTF/slow combinations from the first grid are retained,
and every combo is now run across a 3×3 SL/TP matrix.

Total: 3 entry × 3 exit × 2 mtf × 4 slow × 3 SL × 3 TP = 648 combinations

SL/TP fill logic (long-only, per bar):
  Gap-down  : bar.open <= sl_price  → fill at open  (worst-case gap protection)
  Gap-up    : bar.open >= tp_price  → fill at open  (gap-up fill at open)
  Intrabar SL : bar.low  <= sl_price → fill at sl_price  (SL checked before TP)
  Intrabar TP : bar.high >= tp_price → fill at tp_price
  Signal exit : midline / upper_rail close condition → fill at bar+1 open
  Timeout     : absolute fallback at max_hold_bars

Output:
  data/reports/sl_tp_grid_lr_channel_{ts}.csv   — all rows sorted by PF desc
  stdout                                          — top-20 table
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
from backtest_lr_channel_btc import load_csv_as_bars, round_tick

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

CSV_PATH       = "data/kraken/PF_XBTUSD_1min.csv"
REPORTS_DIR    = Path("data/reports")
COMMISSION     = 2.0   # per side; round-trip = 4.0
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
    "sl_pct"          : [0.005, 0.010, 0.020],   # 0.5 % | 1.0 % | 2.0 %
    "tp_pct"          : [0.010, 0.020, 0.030],   # 1.0 % | 2.0 % | 3.0 %
}
# 3 × 3 × 2 × 4 × 3 × 3 = 648 combos


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
    sl_pct: float,
    tp_pct: float,
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
        if fill_bar >= n - 2:
            break

        entry_price = round_tick(bars[fill_bar].open)
        sl_price    = entry_price * (1.0 - sl_pct)
        tp_price    = entry_price * (1.0 + tp_pct)

        exit_price  = None
        exit_bar    = None
        exit_reason = "timeout"

        scan_end = min(fill_bar + MAX_HOLD_BARS, n - 2)
        for j in range(fill_bar + 1, scan_end + 1):
            bar = bars[j]

            # --- Gap protection (open already beyond bracket) ---
            if bar.open <= sl_price:
                exit_price  = round_tick(bar.open)
                exit_bar    = j
                exit_reason = "sl"
                break
            if bar.open >= tp_price:
                exit_price  = round_tick(bar.open)
                exit_bar    = j
                exit_reason = "tp"
                break

            # --- Intrabar SL (checked before TP — conservative) ---
            if bar.low <= sl_price:
                exit_price  = round_tick(sl_price)
                exit_bar    = j
                exit_reason = "sl"
                break

            # --- Intrabar TP ---
            if bar.high >= tp_price:
                exit_price  = round_tick(tp_price)
                exit_bar    = j
                exit_reason = "tp"
                break

            # --- Signal-based exit (close confirmed, fill at j+1 open) ---
            if exit_type != "timeout":
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
            exit_reason = "timeout"

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
    sl_pct: float,
    tp_pct: float,
) -> dict:
    row: dict = {
        "slow_length"      : slow_length,
        "entry_line"       : entry_line,
        "exit_type"        : exit_type,
        "mtf_slope_filter" : mtf_slope_filter,
        "sl_pct"           : sl_pct,
        "tp_pct"           : tp_pct,
        "trade_count"      : len(trades),
        "win_rate_pct"     : 0.0,
        "profit_factor"    : 0.0,
        "avg_hold_bars"    : 0.0,
        "total_pnl"        : 0.0,
        "sharpe"           : 0.0,
        "pct_sl"           : 0.0,
        "pct_tp"           : 0.0,
        "pct_timeout"      : 0.0,
    }
    if not trades:
        return row

    df = pd.DataFrame(trades)
    n  = len(df)
    winners    = df["winner"].sum()
    gross_win  = df[df["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(df[df["pnl"] <= 0]["pnl"].sum())

    row["trade_count"]   = n
    row["win_rate_pct"]  = round(winners / n * 100, 1)
    row["profit_factor"] = round(gross_win / gross_loss, 4) if gross_loss > 0 else float("inf")
    row["avg_hold_bars"] = round(df["hold_bars"].mean(), 1)
    row["total_pnl"]     = round(df["pnl"].sum(), 2)

    for reason in ("sl", "tp", "timeout"):
        row[f"pct_{reason}"] = round((df["exit_reason"] == reason).sum() / n * 100, 1)

    daily = df.groupby(
        df["entry_bar"].apply(lambda i: i // 1440)
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

    logger.info(f"Pre-computing ch{MEDIUM_LENGTH} and ch{FAST_LENGTH} ...")
    ch100 = compute_lr_channel(closes, MEDIUM_LENGTH)
    ch30  = compute_lr_channel(closes, FAST_LENGTH)

    slow_lengths = PARAM_GRID["slow_length"]
    logger.info(f"Pre-computing slow channels {slow_lengths} ...")
    slow_channels = {L: compute_lr_channel(closes, L) for L in slow_lengths}

    keys   = ["entry_line", "exit_type", "mtf_slope_filter", "slow_length", "sl_pct", "tp_pct"]
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    logger.info(f"Running {len(combos)} combinations ...")

    results = []
    for entry_line, exit_type, mtf_filter, slow_len, sl_pct, tp_pct in combos:
        trades = _simulate_combo(
            bars, closes,
            slow_channels[slow_len], ch100, ch30,
            entry_line, exit_type, mtf_filter, sl_pct, tp_pct,
        )
        results.append(_summarise(
            trades, slow_len, entry_line, exit_type, mtf_filter, sl_pct, tp_pct
        ))

    df_all = pd.DataFrame(results).sort_values("profit_factor", ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"sl_tp_grid_lr_channel_{ts}.csv"
    df_all.to_csv(out, index=False)
    logger.info(f"Full results saved → {out}")

    df_ranked = df_all[df_all["trade_count"] >= MIN_TRADES].head(20)

    if df_ranked.empty:
        print(f"\n  No config met the {MIN_TRADES}-trade minimum. Full results saved.")
        return

    print(f"\n{'─'*116}")
    print(f" Top-{len(df_ranked)} configs  (min {MIN_TRADES} trades, ranked by PF)  —  {len(combos)} combos swept")
    print(f"{'─'*116}")
    header = (
        f"{'slow':>4}  {'entry':<6}  {'exit':<11}  {'m':<1}  "
        f"{'SL%':>4}  {'TP%':>4}  "
        f"{'trades':>6}  {'WR%':>5}  {'PF':>6}  {'hold':>6}  "
        f"{'%SL':>5}  {'%TP':>5}  {'%TO':>5}  {'Sharpe':>6}  {'PnL':>9}"
    )
    print(header)
    print("─" * 116)
    for _, r in df_ranked.iterrows():
        pf_str = f"{r.profit_factor:.3f}" if r.profit_factor != float("inf") else "   inf"
        print(
            f"{int(r.slow_length):>4}  {r.entry_line:<6}  {r.exit_type:<11}  "
            f"{'Y' if r.mtf_slope_filter else 'N':<1}  "
            f"{r.sl_pct*100:>4.1f}  {r.tp_pct*100:>4.1f}  "
            f"{int(r.trade_count):>6}  {r.win_rate_pct:>5.1f}  {pf_str:>6}  "
            f"{r.avg_hold_bars:>6.1f}  "
            f"{r.pct_sl:>5.1f}  {r.pct_tp:>5.1f}  {r.pct_timeout:>5.1f}  "
            f"{r.sharpe:>6.2f}  {r.total_pnl:>9.2f}"
        )
    print(f"{'─'*116}\n")


if __name__ == "__main__":
    main()
