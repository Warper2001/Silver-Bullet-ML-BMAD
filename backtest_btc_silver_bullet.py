#!/usr/bin/env python3
"""BTC Silver Bullet Base Backtest (No ML).

Ports the validated MNQ backtest logic (backtest_paper_trade_6months.py) to
PF_XBTUSD using 1-minute Kraken candle data.

Key differences from MNQ version:
- Data: CSV (not JSON)
- Kill zones: America/Chicago (CDT=UTC-5 summer, CST=UTC-6 winter), London AM / NY AM / Asia
- Contract: BTC_CONTRACT_VALUE=1.0, BTC_TICK=0.5, COMMISSION=$2.00
- No ML filter

Output: data/reports/backtest_btc_silver_bullet_{ts}.txt + .csv
"""

import sys
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import yaml
from tqdm import tqdm

_CHICAGO = pytz.timezone("America/Chicago")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.models import DollarBar
from src.detection.fvg_detection import detect_bullish_fvg, detect_bearish_fvg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

BTC_TICK = 0.5
BTC_CONTRACT_VALUE = 1.0  # $1 per point per contract (PF_XBTUSD USD-linear)
COMMISSION = 2.0
POSITION_SIZE = 1
LIMIT_CANCEL_BARS = 15
MAX_HOLD_BARS = 120
MIN_RR = 2.0
SWING_LOOKBACK = 200


def round_tick(price: float) -> float:
    return round(round(price / BTC_TICK) * BTC_TICK, 10)


def _load_kill_zones(config_path: str = "config_kraken.yaml"):
    cfg = yaml.safe_load(open(config_path))
    result = []
    for kz in cfg["kill_zones"]:
        sh, sm = map(int, kz["start"].split(":"))
        eh, em = map(int, kz["end"].split(":"))
        result.append((kz["name"], sh, sm, eh, em))
    return result


def _in_kill_zone(ts_utc: datetime, kill_zones: list) -> tuple[bool, Optional[str]]:
    local = ts_utc.astimezone(_CHICAGO)
    local_min = local.hour * 60 + local.minute
    for name, sh, sm, eh, em in kill_zones:
        if sh * 60 + sm <= local_min < eh * 60 + em:
            return True, name
    return False, None


def _cdt_date(ts_utc: datetime) -> str:
    return ts_utc.astimezone(_CHICAGO).date().isoformat()


def load_csv_as_bars(csv_path: str) -> list[DollarBar]:
    logger.info(f"Loading 1-minute BTC data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    bars = []
    skipped = 0
    for row in df.itertuples(index=False):
        try:
            if row.high == 0 or row.low == 0 or row.close == 0:
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
                volume=0,
                notional_value=float(row.close),
                is_forward_filled=False,
            ))
        except Exception as exc:
            skipped += 1
            logger.debug(f"Skipped row: {exc}")

    logger.info(f"Loaded {len(bars):,} bars ({skipped} skipped)")
    return bars


def detect_swing_points(bars: list[DollarBar], lookback: int = 3) -> tuple[list, list]:
    swing_highs, swing_lows = [], []
    for i in range(lookback, len(bars) - lookback):
        try:
            is_high = all(bars[i].high >= bars[i + k].high for k in range(-lookback, lookback + 1) if k != 0)
            if is_high:
                swing_highs.append({"index": i, "price": bars[i].high, "timestamp": bars[i].timestamp})
            is_low = all(bars[i].low <= bars[i + k].low for k in range(-lookback, lookback + 1) if k != 0)
            if is_low:
                swing_lows.append({"index": i, "price": bars[i].low, "timestamp": bars[i].timestamp})
        except Exception:
            continue
    logger.info(f"Swing points: {len(swing_highs)} highs, {len(swing_lows)} lows")
    return swing_highs, swing_lows


def detect_mss_events(bars: list[DollarBar], swing_highs: list, swing_lows: list) -> list:
    mss_events = []
    for swing in tqdm(swing_highs, desc="Bullish MSS"):
        try:
            for j in range(swing["index"] + 1, min(swing["index"] + 60, len(bars))):
                if bars[j].high > swing["price"]:
                    mss_events.append({
                        "index": j,
                        "timestamp": bars[j].timestamp,
                        "direction": "bullish",
                        "breakout_price": bars[j].high,
                        "swing_point": swing,
                    })
                    break
        except Exception:
            continue
    for swing in tqdm(swing_lows, desc="Bearish MSS"):
        try:
            for j in range(swing["index"] + 1, min(swing["index"] + 60, len(bars))):
                if bars[j].low < swing["price"]:
                    mss_events.append({
                        "index": j,
                        "timestamp": bars[j].timestamp,
                        "direction": "bearish",
                        "breakout_price": bars[j].low,
                        "swing_point": swing,
                    })
                    break
        except Exception:
            continue
    logger.info(f"MSS events: {len(mss_events)}")
    return mss_events


def detect_fvg_setups(bars: list[DollarBar]) -> list:
    # Pass full bars list + index directly — avoid O(n²) slice creation
    fvg_setups = []
    for i in tqdm(range(2, len(bars)), desc="FVG detection"):
        try:
            bullish = detect_bullish_fvg(bars, i, atr_filter=None, volume_confirmer=None)
            if bullish:
                fvg_setups.append({
                    "index": i,
                    "timestamp": bars[i].timestamp,
                    "direction": "bullish",
                    "entry_top": bullish.gap_range.top,
                    "entry_bottom": bullish.gap_range.bottom,
                    "gap_size": bullish.gap_size_dollars,
                })
            bearish = detect_bearish_fvg(bars, i, atr_filter=None, volume_confirmer=None)
            if bearish:
                fvg_setups.append({
                    "index": i,
                    "timestamp": bars[i].timestamp,
                    "direction": "bearish",
                    "entry_top": bearish.gap_range.top,
                    "entry_bottom": bearish.gap_range.bottom,
                    "gap_size": bearish.gap_size_dollars,
                })
        except Exception:
            continue
    logger.info(f"FVG setups: {len(fvg_setups)}")
    return fvg_setups


def detect_confluence(mss_events: list, fvg_setups: list) -> list:
    # Index FVGs by bar index for O(1) lookup; match window = [mss_idx, mss_idx+10]
    from collections import defaultdict
    fvg_by_idx: dict[int, list] = defaultdict(list)
    for fvg in fvg_setups:
        fvg_by_idx[fvg["index"]].append(fvg)

    setups = []
    for mss in tqdm(mss_events, desc="Confluence"):
        try:
            for offset in range(0, 11):  # FVG at or after MSS, within 10 bars
                for fvg in fvg_by_idx.get(mss["index"] + offset, []):
                    if mss["direction"] != fvg["direction"]:
                        continue
                    fvg_midpoint = (fvg["entry_top"] + fvg["entry_bottom"]) / 2
                    swing_price = mss["swing_point"]["price"]
                    breakout_price = mss["breakout_price"]
                    equilibrium = (swing_price + breakout_price) / 2
                    if mss["direction"] == "bullish" and fvg_midpoint >= equilibrium:
                        continue
                    if mss["direction"] == "bearish" and fvg_midpoint <= equilibrium:
                        continue
                    setups.append({
                        "index": fvg["index"],
                        "timestamp": fvg["timestamp"],
                        "direction": mss["direction"],
                        "entry_zone_top": fvg["entry_top"],
                        "entry_zone_bottom": fvg["entry_bottom"],
                        "gap_size": fvg["gap_size"],
                        "mss_event": mss,
                        "fvg_event": fvg,
                    })
        except Exception:
            continue
    logger.info(f"Confluence setups: {len(setups)}")
    return setups


def filter_by_kill_zone(setups: list, kill_zones: list) -> list:
    result = []
    for setup in setups:
        in_kz, name = _in_kill_zone(setup["timestamp"], kill_zones)
        if in_kz:
            setup["killzone_window"] = name
            result.append(setup)
    logger.info(f"Kill zone filtered: {len(result)} setups")
    return result


def _find_next_liquidity_pool(
    bars: list[DollarBar],
    setup_idx: int,
    direction: str,
    entry_price: float,
    lookback: int = SWING_LOOKBACK,
) -> Optional[float]:
    start = max(0, setup_idx - lookback)
    candidates = []
    for i in range(setup_idx - 1, start - 1, -1):
        bar = bars[i]
        if direction == "bullish":
            left = bars[i - 1].high if i > 0 else 0
            right = bars[i + 1].high if i < len(bars) - 1 else 0
            if bar.high > left and bar.high > right and bar.high > entry_price:
                swept = any(bars[j].close > bar.high for j in range(i + 1, setup_idx))
                if not swept:
                    candidates.append(bar.high)
        else:
            left = bars[i - 1].low if i > 0 else float("inf")
            right = bars[i + 1].low if i < len(bars) - 1 else float("inf")
            if bar.low < left and bar.low < right and bar.low < entry_price:
                swept = any(bars[j].close < bar.low for j in range(i + 1, setup_idx))
                if not swept:
                    candidates.append(bar.low)
    if not candidates:
        return None
    return min(candidates, key=lambda p: abs(p - entry_price))


def execute_backtest(bars: list[DollarBar], setups: list) -> list:
    trades = []
    window_trades: dict[str, set] = {}

    for setup in tqdm(setups, desc="Executing trades"):
        try:
            setup_idx = setup["index"]
            if setup_idx >= len(bars) - LIMIT_CANCEL_BARS - 1:
                continue

            direction = setup["direction"]
            window = setup.get("killzone_window", "unknown")
            fvg_midpoint = round_tick((setup["entry_zone_top"] + setup["entry_zone_bottom"]) / 2)

            setup_date = _cdt_date(bars[setup_idx].timestamp)
            fired_dates = window_trades.setdefault(window, set())
            if setup_date in fired_dates:
                continue

            # Wait for FVG touch (limit fill simulation)
            fill_idx = None
            for i in range(setup_idx + 1, min(setup_idx + LIMIT_CANCEL_BARS + 1, len(bars))):
                bar = bars[i]
                if direction == "bullish" and bar.low <= fvg_midpoint:
                    fill_idx = i
                    break
                elif direction == "bearish" and bar.high >= fvg_midpoint:
                    fill_idx = i
                    break

            if fill_idx is None:
                continue

            entry_price = fvg_midpoint
            fill_bar = bars[fill_idx]

            # Stop loss
            gap_size = setup["entry_zone_top"] - setup["entry_zone_bottom"]
            if direction == "bullish":
                stop_loss = round_tick(setup["entry_zone_bottom"] - gap_size * 0.5)
            else:
                stop_loss = round_tick(setup["entry_zone_top"] + gap_size * 0.5)

            stop_distance = abs(entry_price - stop_loss)
            if stop_distance == 0:
                continue

            # Target: nearest unswept swing pool, min 2R
            target_price = _find_next_liquidity_pool(bars, fill_idx, direction, entry_price)
            if target_price is None:
                logger.debug("No liquidity pool — skipped")
                continue

            target_price = round_tick(target_price)
            target_distance = abs(target_price - entry_price)
            if target_distance < MIN_RR * stop_distance:
                logger.debug(f"R:R {target_distance/stop_distance:.2f} < {MIN_RR} — skipped")
                continue

            # Guard: fill bar breaches stop
            if direction == "bullish" and fill_bar.low <= stop_loss:
                continue
            if direction == "bearish" and fill_bar.high >= stop_loss:
                continue

            # Bar-by-bar simulation
            exit_idx = fill_idx
            exit_price = 0.0
            pnl = 0.0
            exit_reason = None

            for j in range(fill_idx + 1, min(fill_idx + MAX_HOLD_BARS + 1, len(bars))):
                exit_bar = bars[j]
                if direction == "bullish":
                    if exit_bar.low <= stop_loss:
                        pnl = (stop_loss - entry_price) * POSITION_SIZE * BTC_CONTRACT_VALUE
                        exit_price = stop_loss
                        exit_reason = "stop_loss"
                        exit_idx = j
                        break
                    elif exit_bar.high >= target_price:
                        pnl = (target_price - entry_price) * POSITION_SIZE * BTC_CONTRACT_VALUE
                        exit_price = target_price
                        exit_reason = "target"
                        exit_idx = j
                        break
                else:
                    if exit_bar.high >= stop_loss:
                        pnl = (entry_price - stop_loss) * POSITION_SIZE * BTC_CONTRACT_VALUE
                        exit_price = stop_loss
                        exit_reason = "stop_loss"
                        exit_idx = j
                        break
                    elif exit_bar.low <= target_price:
                        pnl = (entry_price - target_price) * POSITION_SIZE * BTC_CONTRACT_VALUE
                        exit_price = target_price
                        exit_reason = "target"
                        exit_idx = j
                        break

                if j - fill_idx >= MAX_HOLD_BARS:
                    if direction == "bullish":
                        pnl = (exit_bar.close - entry_price) * POSITION_SIZE * BTC_CONTRACT_VALUE
                    else:
                        pnl = (entry_price - exit_bar.close) * POSITION_SIZE * BTC_CONTRACT_VALUE
                    exit_price = exit_bar.close
                    exit_reason = "time_exit"
                    exit_idx = j
                    break

            if exit_reason is None:
                continue

            pnl -= COMMISSION
            fired_dates.add(setup_date)

            trades.append({
                "entry_time": fill_bar.timestamp.isoformat(),
                "exit_time": bars[exit_idx].timestamp.isoformat(),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "exit_price": exit_price,
                "direction": direction,
                "pnl": pnl,
                "exit_reason": exit_reason,
                "killzone_window": window,
                "bars_held": exit_idx - fill_idx,
                "rr_target": round(target_distance / stop_distance, 2),
            })

        except Exception as exc:
            logger.debug(f"Trade error: {exc}")
            continue

    logger.info(f"Trades executed: {len(trades)}")
    return trades


def analyze_performance(trades: list, bars: list[DollarBar]) -> dict:
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    total = len(df)
    wins = (df["pnl"] > 0).sum()
    losses = (df["pnl"] < 0).sum()
    win_rate = wins / total
    total_pnl = df["pnl"].sum()
    gross_profit = df[df["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(df[df["pnl"] < 0]["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["running_max"] = df["cumulative_pnl"].cummax()
    df["drawdown"] = df["cumulative_pnl"] - df["running_max"]
    max_drawdown = df["drawdown"].min()

    date_range_days = (bars[-1].timestamp - bars[0].timestamp).days or 1
    trading_days = df["entry_time"].apply(lambda t: t[:10]).nunique()
    trades_per_day = total / trading_days if trading_days > 0 else 0

    # Sharpe
    df["entry_date"] = df["entry_time"].apply(lambda t: t[:10])
    daily_pnl = df.groupby("entry_date")["pnl"].sum()
    sharpe = float(np.sqrt(252) * daily_pnl.mean() / daily_pnl.std()) if daily_pnl.std() > 0 else 0.0

    # Per-kill-zone breakdown
    zone_stats = {}
    for zone, grp in df.groupby("killzone_window"):
        z_wins = (grp["pnl"] > 0).sum()
        z_losses = (grp["pnl"] < 0).sum()
        z_gp = grp[grp["pnl"] > 0]["pnl"].sum()
        z_gl = abs(grp[grp["pnl"] < 0]["pnl"].sum())
        zone_stats[zone] = {
            "trades": len(grp),
            "win_rate": z_wins / len(grp),
            "profit_factor": z_gp / z_gl if z_gl > 0 else float("inf"),
            "avg_pnl": grp["pnl"].mean(),
            "total_pnl": grp["pnl"].sum(),
        }

    exit_counts = df["exit_reason"].value_counts().to_dict()

    return {
        "total_trades": total,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": df["pnl"].mean(),
        "avg_win": df[df["pnl"] > 0]["pnl"].mean() if wins > 0 else 0,
        "avg_loss": df[df["pnl"] < 0]["pnl"].mean() if losses > 0 else 0,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "trades_per_day": trades_per_day,
        "avg_hold_bars": df["bars_held"].mean(),
        "date_range_days": date_range_days,
        "trading_days": trading_days,
        "exit_counts": exit_counts,
        "zone_stats": zone_stats,
        "cumulative_pnl": df["cumulative_pnl"].tolist(),
    }


def generate_report(perf: dict, bars: list[DollarBar]) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("BTC (PF_XBTUSD) SILVER BULLET BASE BACKTEST — NO ML FILTER")
    lines.append("=" * 72)
    lines.append(f"Data:    {bars[0].timestamp.date()} → {bars[-1].timestamp.date()}")
    lines.append(f"Config:  BTC_TICK={BTC_TICK}  CONTRACT_VALUE=${BTC_CONTRACT_VALUE}  COMMISSION=${COMMISSION}  MIN_RR={MIN_RR}")
    lines.append(f"Windows: London AM 03-04 CDT | NY AM 09-10 CDT | Asia 19-20 CDT")
    lines.append("")

    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 72)
    lines.append(f"  Total trades:      {perf['total_trades']}")
    lines.append(f"  Win rate:          {perf['win_rate']:.1%}")
    lines.append(f"  Profit factor:     {perf['profit_factor']:.3f}")
    lines.append(f"  Sharpe ratio:      {perf['sharpe_ratio']:.3f}")
    lines.append(f"  Total P&L:         ${perf['total_pnl']:,.2f}")
    lines.append(f"  Avg P&L/trade:     ${perf['avg_pnl']:,.2f}")
    lines.append(f"  Avg win:           ${perf['avg_win']:,.2f}")
    lines.append(f"  Avg loss:          ${perf['avg_loss']:,.2f}")
    lines.append(f"  Max drawdown:      ${perf['max_drawdown']:,.2f}")
    lines.append(f"  Trades/day:        {perf['trades_per_day']:.2f}")
    lines.append(f"  Avg hold (bars):   {perf['avg_hold_bars']:.1f}  (~{perf['avg_hold_bars']:.0f} min)")
    lines.append(f"  Trading days:      {perf['trading_days']}")
    lines.append(f"  Date range:        {perf['date_range_days']} days")
    lines.append("")

    lines.append("EXIT BREAKDOWN")
    lines.append("-" * 72)
    for reason, count in perf["exit_counts"].items():
        lines.append(f"  {reason:<20} {count:>6}  ({count/perf['total_trades']:.1%})")
    lines.append("")

    lines.append("KILL ZONE BREAKDOWN")
    lines.append("-" * 72)
    for zone, stats in perf["zone_stats"].items():
        pf_str = f"{stats['profit_factor']:.3f}" if stats["profit_factor"] != float("inf") else "inf"
        lines.append(
            f"  {zone:<12} | trades={stats['trades']:>4}  win={stats['win_rate']:.1%}"
            f"  PF={pf_str}  avg=${stats['avg_pnl']:>8.2f}  total=${stats['total_pnl']:>10,.2f}"
        )
    lines.append("")

    lines.append("INSTITUTIONAL GRADE ASSESSMENT")
    lines.append("-" * 72)
    checks = [
        ("Profit Factor ≥ 1.5", perf["profit_factor"] >= 1.5),
        ("Win Rate ≥ 40%",      perf["win_rate"] >= 0.40),
        ("Sharpe ≥ 1.0",        perf["sharpe_ratio"] >= 1.0),
        ("Max DD ≤ $5,000",     abs(perf["max_drawdown"]) <= 5000),
    ]
    passed = sum(1 for _, ok in checks if ok)
    for label, ok in checks:
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {label}")
    lines.append(f"\n  {'PASS' if passed >= 3 else 'FAIL'} — {passed}/4 thresholds met")
    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    logger.info("BTC Silver Bullet Base Backtest starting...")

    kill_zones = _load_kill_zones("config_kraken.yaml")
    bars = load_csv_as_bars("data/kraken/PF_XBTUSD_1min.csv")

    if len(bars) < 1000:
        logger.error("Insufficient data — aborting")
        return

    swing_highs, swing_lows = detect_swing_points(bars)
    mss_events = detect_mss_events(bars, swing_highs, swing_lows)

    if not mss_events:
        logger.error("No MSS events detected — aborting")
        return

    fvg_setups = detect_fvg_setups(bars)
    if not fvg_setups:
        logger.error("No FVG setups detected — aborting")
        return

    setups = detect_confluence(mss_events, fvg_setups)
    if not setups:
        logger.error("No confluence setups — aborting")
        return

    kz_setups = filter_by_kill_zone(setups, kill_zones)
    if not kz_setups:
        logger.warning("No setups inside kill zones — using all setups for diagnostics")
        kz_setups = setups

    trades = execute_backtest(bars, kz_setups)
    if not trades:
        logger.error("No trades executed — aborting")
        return

    perf = analyze_performance(trades, bars)
    report = generate_report(perf, bars)
    print("\n" + report)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / f"backtest_btc_silver_bullet_{ts}.txt"
    txt_path.write_text(report)
    logger.info(f"Report: {txt_path}")

    csv_path = out_dir / f"backtest_btc_silver_bullet_{ts}.csv"
    pd.DataFrame(trades).to_csv(csv_path, index=False)
    logger.info(f"Trades: {csv_path}")


if __name__ == "__main__":
    main()
