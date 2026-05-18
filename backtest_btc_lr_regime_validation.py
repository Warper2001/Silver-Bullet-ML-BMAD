#!/usr/bin/env python3
"""BTC LR Regime Walk-Forward Validation.

Compares two tracks over the full available history:
  Baseline : Silver Bullet signals filtered only by kill zones + trade_days
  Deployed : Same signals + LR counter-trend regime gate (fast=390, slow=1950)

Data:   data/kraken/PF_XBTUSD_1min.csv  (Nov 2024 – May 2026)
Output: data/reports/backtest_btc_lr_regime_{date}.txt
        data/reports/backtest_btc_lr_regime_{date}.csv   (deployed equity curve)
        models/xgboost/lr_regime_config_btc.json         (if all gates pass)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backtest_btc_silver_bullet import (
    BTC_CONTRACT_VALUE,
    BTC_TICK,
    COMMISSION,
    LIMIT_CANCEL_BARS,
    MAX_HOLD_BARS,
    MIN_RR,
    POSITION_SIZE,
    SWING_LOOKBACK,
    _find_next_liquidity_pool,
    _load_kill_zones,
    detect_confluence,
    detect_fvg_setups,
    detect_mss_events,
    detect_swing_points,
    filter_by_kill_zone,
    load_csv_as_bars,
    round_tick,
)
from src.ml.regime_detection.lr_channel_detector import LRChannelRegimeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

_CHICAGO = pytz.timezone("America/Chicago")

# LR regime parameters (clock-time equivalents match MNQ tier2 deployment)
LR_FAST = 390       # ~6.5 hours of 1-min bars (1 MNQ session equivalent)
LR_SLOW = 1950      # ~32.5 hours

MIN_TRAIN_MONTHS = 3   # seed before first test window

# Promotion gate thresholds
GATE_PF_MIN = 1.10
GATE_MIN_FILTER_RATIO = 0.25   # regime must reject ≥25% of test-period setups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def month_key(ts_utc: datetime) -> str:
    return ts_utc.strftime("%Y-%m")


def _cdt_date(ts_utc: datetime) -> str:
    return ts_utc.astimezone(_CHICAGO).date().isoformat()


def regime_counter(lr_regime: str, direction: str) -> bool:
    """Counter-trend gate: pass when regime opposes signal direction."""
    if lr_regime == "UP":
        return direction == "bearish"
    if lr_regime == "DOWN":
        return direction == "bullish"
    return True   # SIDEWAYS always passes


def regime_trend(lr_regime: str, direction: str) -> bool:
    """Trend-following gate: pass when regime aligns with signal direction."""
    if lr_regime == "UP":
        return direction == "bullish"
    if lr_regime == "DOWN":
        return direction == "bearish"
    return True   # SIDEWAYS always passes


def filter_by_trade_days(setups: list, trade_days: list | None) -> list:
    """Restrict setups to allowed weekdays (America/Chicago local DOW)."""
    if trade_days is None:
        return setups
    return [
        s for s in setups
        if s["timestamp"].astimezone(_CHICAGO).weekday() in trade_days
    ]


# ---------------------------------------------------------------------------
# Per-month execution (fresh window-dedup per call, matching live behaviour)
# ---------------------------------------------------------------------------

def execute_month(bars: list, setups: list) -> list:
    """Execute trades for a subset of setups using the full bars array.

    Window dedup resets each call — one trade per kill-zone window per CDT day.
    """
    trades = []
    window_trades: dict[str, set] = {}

    for setup in setups:
        try:
            setup_idx = setup["index"]
            if setup_idx >= len(bars) - LIMIT_CANCEL_BARS - 1:
                continue

            direction = setup["direction"]
            window = setup.get("killzone_window", "unknown")
            fvg_midpoint = round_tick(
                (setup["entry_zone_top"] + setup["entry_zone_bottom"]) / 2
            )

            setup_date = _cdt_date(bars[setup_idx].timestamp)
            fired_dates = window_trades.setdefault(window, set())
            if setup_date in fired_dates:
                continue

            # Wait for FVG touch
            fill_idx = None
            for i in range(
                setup_idx + 1,
                min(setup_idx + LIMIT_CANCEL_BARS + 1, len(bars)),
            ):
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

            gap_size = setup["entry_zone_top"] - setup["entry_zone_bottom"]
            if direction == "bullish":
                stop_loss = round_tick(setup["entry_zone_bottom"] - gap_size * 0.5)
            else:
                stop_loss = round_tick(setup["entry_zone_top"] + gap_size * 0.5)

            stop_distance = abs(entry_price - stop_loss)
            if stop_distance == 0:
                continue

            target_price = _find_next_liquidity_pool(
                bars, fill_idx, direction, entry_price
            )
            if target_price is None:
                continue
            target_price = round_tick(target_price)
            target_distance = abs(target_price - entry_price)
            if target_distance < MIN_RR * stop_distance:
                continue

            if direction == "bullish" and fill_bar.low <= stop_loss:
                continue
            if direction == "bearish" and fill_bar.high >= stop_loss:
                continue

            exit_idx = fill_idx
            exit_price = 0.0
            pnl = 0.0
            exit_reason = None

            for j in range(
                fill_idx + 1, min(fill_idx + MAX_HOLD_BARS + 1, len(bars))
            ):
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
                "month": month_key(fill_bar.timestamp),
                "entry_time": fill_bar.timestamp.isoformat(),
                "exit_time": bars[exit_idx].timestamp.isoformat(),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "direction": direction,
                "pnl": pnl,
                "exit_reason": exit_reason,
                "killzone_window": window,
                "lr_regime": setup.get("lr_regime", "SIDEWAYS"),
            })

        except Exception as exc:
            logger.debug(f"Trade error: {exc}")
            continue

    return trades


def compute_metrics(trades: list) -> dict:
    if not trades:
        return {
            "trades": 0, "wins": 0, "pf": 0.0,
            "wr": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0, "max_dd": 0.0,
        }
    df = pd.DataFrame(trades)
    wins = int((df["pnl"] > 0).sum())
    gp = df[df["pnl"] > 0]["pnl"].sum()
    gl = abs(df[df["pnl"] < 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else float("inf")
    cumul = df["pnl"].cumsum()
    max_dd = float((cumul - cumul.cummax()).min())
    return {
        "trades": len(df),
        "wins": wins,
        "pf": pf,
        "wr": wins / len(df),
        "total_pnl": float(df["pnl"].sum()),
        "avg_pnl": float(df["pnl"].mean()),
        "max_dd": max_dd,
    }


def pf_str(v: float) -> str:
    return f"{v:.3f}" if v != float("inf") else "  inf"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("BTC LR Regime Walk-Forward Validation starting...")

    # --- Config & bars ---
    config = yaml.safe_load(open("config_kraken.yaml"))
    trade_days: list | None = config.get("trade_days")
    kill_zones = _load_kill_zones("config_kraken.yaml")
    bars = load_csv_as_bars("data/kraken/PF_XBTUSD_1min.csv")

    if len(bars) < LR_SLOW + 100:
        logger.error("Insufficient bars — aborting")
        return

    # --- Signal detection (full history, computed once) ---
    swing_highs, swing_lows = detect_swing_points(bars)
    mss_events = detect_mss_events(bars, swing_highs, swing_lows)
    fvg_setups = detect_fvg_setups(bars)
    all_setups = detect_confluence(mss_events, fvg_setups)
    kz_setups = filter_by_kill_zone(all_setups, kill_zones)
    kz_setups = filter_by_trade_days(kz_setups, trade_days)
    logger.info(f"Kill-zone + trade-day setups: {len(kz_setups)}")

    # --- LR regime labels (full history, computed once) ---
    logger.info(f"Computing LR regimes (fast={LR_FAST}, slow={LR_SLOW})...")
    closes = np.array([b.close for b in bars], dtype=np.float64)
    detector = LRChannelRegimeDetector(LR_FAST, LR_SLOW)
    all_regimes = detector.fit_predict(closes)
    logger.info("LR regime computation complete.")

    # Attach regime label to each setup
    for setup in kz_setups:
        idx = setup["index"]
        setup["lr_regime"] = str(all_regimes[idx]) if idx < len(all_regimes) else "SIDEWAYS"

    # --- Walk-forward month splits ---
    first_bar_dt = bars[0].timestamp
    test_start_month = first_bar_dt.month + MIN_TRAIN_MONTHS
    test_start_year = first_bar_dt.year
    if test_start_month > 12:
        test_start_year += (test_start_month - 1) // 12
        test_start_month = (test_start_month - 1) % 12 + 1
    test_start_key = f"{test_start_year:04d}-{test_start_month:02d}"

    all_months = sorted(set(month_key(s["timestamp"]) for s in kz_setups))
    test_months = [m for m in all_months if m >= test_start_key]

    if not test_months:
        logger.error("No test months available — aborting")
        return

    logger.info(
        f"Walk-forward: {test_months[0]} → {test_months[-1]}  ({len(test_months)} months)"
    )

    # --- Per-month walk-forward (both polarities in one pass) ---
    monthly_rows_ct: list[dict] = []   # counter-trend
    monthly_rows_tf: list[dict] = []   # trend-following
    baseline_all: list[dict] = []
    ct_all: list[dict] = []
    tf_all: list[dict] = []

    for month in tqdm(test_months, desc="Walk-forward"):
        month_setups = [s for s in kz_setups if month_key(s["timestamp"]) == month]
        if not month_setups:
            continue

        ct_setups = [s for s in month_setups if regime_counter(s["lr_regime"], s["direction"])]
        tf_setups = [s for s in month_setups if regime_trend(s["lr_regime"], s["direction"])]

        base_trades = execute_month(bars, month_setups)
        ct_trades   = execute_month(bars, ct_setups)
        tf_trades   = execute_month(bars, tf_setups)

        base_m = compute_metrics(base_trades)
        ct_m   = compute_metrics(ct_trades)
        tf_m   = compute_metrics(tf_trades)

        ct_filter = 1.0 - len(ct_setups) / len(month_setups)
        tf_filter = 1.0 - len(tf_setups) / len(month_setups)

        def _delta(dep_pf, base_pf):
            return dep_pf - base_pf if base_pf not in (float("inf"), 0.0) else 0.0

        monthly_rows_ct.append({
            "month": month,
            "base_trades": base_m["trades"], "dep_trades": ct_m["trades"],
            "base_pf": base_m["pf"],         "dep_pf":   ct_m["pf"],
            "delta_pf": _delta(ct_m["pf"], base_m["pf"]),
            "base_pnl": base_m["total_pnl"], "dep_pnl":  ct_m["total_pnl"],
            "filter_ratio": ct_filter,
        })
        monthly_rows_tf.append({
            "month": month,
            "base_trades": base_m["trades"], "dep_trades": tf_m["trades"],
            "base_pf": base_m["pf"],         "dep_pf":   tf_m["pf"],
            "delta_pf": _delta(tf_m["pf"], base_m["pf"]),
            "base_pnl": base_m["total_pnl"], "dep_pnl":  tf_m["total_pnl"],
            "filter_ratio": tf_filter,
        })

        baseline_all.extend(base_trades)
        ct_all.extend(ct_trades)
        tf_all.extend(tf_trades)

    # --- Aggregates ---
    base_total = compute_metrics(baseline_all)
    ct_total   = compute_metrics(ct_all)
    tf_total   = compute_metrics(tf_all)

    test_setups = [s for s in kz_setups if month_key(s["timestamp"]) >= test_start_key]
    ct_test = [s for s in test_setups if regime_counter(s["lr_regime"], s["direction"])]
    tf_test = [s for s in test_setups if regime_trend(s["lr_regime"], s["direction"])]
    ct_filter_ratio = 1.0 - len(ct_test) / len(test_setups) if test_setups else 0.0
    tf_filter_ratio = 1.0 - len(tf_test) / len(test_setups) if test_setups else 0.0

    ct_pos_delta = sum(1 for r in monthly_rows_ct if r["delta_pf"] > 0)
    tf_pos_delta = sum(1 for r in monthly_rows_tf if r["delta_pf"] > 0)

    # Regime distribution over test setups
    reg_dist: dict[str, int] = {"UP": 0, "DOWN": 0, "SIDEWAYS": 0}
    for s in test_setups:
        reg_dist[s["lr_regime"]] = reg_dist.get(s["lr_regime"], 0) + 1

    # --- Promotion gate (trend-following polarity) ---
    tf_gate_pf     = tf_total["pf"] >= GATE_PF_MIN
    tf_gate_delta  = tf_total["pf"] > base_total["pf"]
    tf_gate_filter = tf_filter_ratio >= GATE_MIN_FILTER_RATIO
    tf_gate_passed = tf_gate_pf and tf_gate_delta and tf_gate_filter

    ct_gate_pf     = ct_total["pf"] >= GATE_PF_MIN
    ct_gate_delta  = ct_total["pf"] > base_total["pf"]
    ct_gate_filter = ct_filter_ratio >= GATE_MIN_FILTER_RATIO
    ct_gate_passed = ct_gate_pf and ct_gate_delta and ct_gate_filter

    # --- Report ---
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    today_iso = now.strftime("%Y-%m-%d")

    lines: list[str] = []
    W = 90
    lines.append("=" * W)
    lines.append("BTC (PF_XBTUSD) LR REGIME WALK-FORWARD VALIDATION — DUAL POLARITY")
    lines.append("=" * W)
    lines.append(f"Data   : {bars[0].timestamp.date()} → {bars[-1].timestamp.date()}  ({len(bars):,} 1-min bars)")
    lines.append(f"LR     : fast={LR_FAST} bars ({LR_FAST/60:.1f}h)  slow={LR_SLOW} bars ({LR_SLOW/60:.1f}h)")
    lines.append(f"Test   : {test_months[0]} → {test_months[-1]}  ({len(test_months)} months)")
    lines.append(f"Config : trade_days={trade_days}")
    lines.append("")

    lines.append("REGIME DISTRIBUTION — test-period setups")
    lines.append("-" * W)
    n_test = len(test_setups) or 1
    for regime, cnt in reg_dist.items():
        lines.append(f"  {regime:<10} {cnt:>5}  ({cnt/n_test:.1%})")
    lines.append(f"  Counter-trend filter ratio: {ct_filter_ratio:.1%}  |  Trend-following filter ratio: {tf_filter_ratio:.1%}")
    lines.append("")

    def _monthly_table(rows: list, label: str) -> list[str]:
        out = []
        out.append(f"WALK-FORWARD MONTHLY RESULTS — {label}")
        out.append("-" * W)
        hdr = f"  {'Month':<9} {'B.Tr':>5} {'D.Tr':>5}  {'B.PF':>6}  {'D.PF':>6}  {'ΔPF':>7}  {'B.P&L':>9}  {'D.P&L':>9}  {'Filt%':>6}"
        out.append(hdr)
        out.append("  " + "-" * (len(hdr) - 2))
        for r in rows:
            ds = f"{r['delta_pf']:+.3f}" if r["base_pf"] not in (float("inf"), 0.0) else "   n/a"
            out.append(
                f"  {r['month']:<9} {r['base_trades']:>5} {r['dep_trades']:>5}"
                f"  {pf_str(r['base_pf']):>6}  {pf_str(r['dep_pf']):>6}"
                f"  {ds:>7}"
                f"  ${r['base_pnl']:>8.2f}  ${r['dep_pnl']:>8.2f}"
                f"  {r['filter_ratio']:>5.1%}"
            )
        return out

    lines.extend(_monthly_table(monthly_rows_ct, "COUNTER-TREND (UP→bearish, DOWN→bullish)"))
    lines.append("")
    lines.extend(_monthly_table(monthly_rows_tf, "TREND-FOLLOWING (UP→bullish, DOWN→bearish)"))
    lines.append("")

    lines.append("AGGREGATE COMPARISON")
    lines.append("-" * W)
    lines.append(f"  {'Metric':<28} {'Baseline':>12} {'CounterTrend':>14} {'TrendFollow':>13}")
    lines.append(f"  {'-'*67}")
    lines.append(f"  {'Total trades':<28} {base_total['trades']:>12} {ct_total['trades']:>14} {tf_total['trades']:>13}")
    lines.append(f"  {'Win rate':<28} {base_total['wr']:>12.1%} {ct_total['wr']:>14.1%} {tf_total['wr']:>13.1%}")
    lines.append(f"  {'Profit factor':<28} {pf_str(base_total['pf']):>12} {pf_str(ct_total['pf']):>14} {pf_str(tf_total['pf']):>13}")
    lines.append(f"  {'Total P&L':<28} ${base_total['total_pnl']:>11.2f} ${ct_total['total_pnl']:>13.2f} ${tf_total['total_pnl']:>12.2f}")
    lines.append(f"  {'Avg P&L/trade':<28} ${base_total['avg_pnl']:>11.2f} ${ct_total['avg_pnl']:>13.2f} ${tf_total['avg_pnl']:>12.2f}")
    lines.append(f"  {'Max drawdown':<28} ${base_total['max_dd']:>11.2f} ${ct_total['max_dd']:>13.2f} ${tf_total['max_dd']:>12.2f}")
    lines.append(f"  {'Months pos ΔPF':<28} {'—':>12} {ct_pos_delta:>11}/{len(monthly_rows_ct)} {tf_pos_delta:>9}/{len(monthly_rows_tf)}")
    lines.append(f"  {'Filter ratio':<28} {'0.0%':>12} {ct_filter_ratio:>13.1%} {tf_filter_ratio:>12.1%}")
    lines.append("")

    def _gate_block(label: str, dep_total: dict, gate_pf: bool, gate_delta: bool,
                    gate_filter: bool, gate_passed: bool, filter_ratio: float) -> list[str]:
        out = [f"PROMOTION GATE — {label}", "-" * W]
        out.append(
            f"  {'PASS' if gate_pf else 'FAIL'}  Deployed PF ≥ {GATE_PF_MIN}         "
            f"(actual: {pf_str(dep_total['pf'])})"
        )
        dmb = (
            dep_total["pf"] - base_total["pf"]
            if dep_total["pf"] != float("inf") and base_total["pf"] != float("inf")
            else float("nan")
        )
        out.append(
            f"  {'PASS' if gate_delta else 'FAIL'}  Deployed PF > Baseline PF  "
            f"(Δ = {dmb:+.3f})"
        )
        out.append(
            f"  {'PASS' if gate_filter else 'FAIL'}  Filter ratio ≥ {GATE_MIN_FILTER_RATIO:.0%}         "
            f"(actual: {filter_ratio:.1%})"
        )
        out.append("")
        if gate_passed:
            out.append("  ✅ ALL GATES PASSED")
        else:
            out.append("  ❌ GATES FAILED")
        return out

    lines.extend(_gate_block(
        "COUNTER-TREND", ct_total,
        ct_gate_pf, ct_gate_delta, ct_gate_filter, ct_gate_passed, ct_filter_ratio
    ))
    lines.append("")
    lines.extend(_gate_block(
        "TREND-FOLLOWING", tf_total,
        tf_gate_pf, tf_gate_delta, tf_gate_filter, tf_gate_passed, tf_filter_ratio
    ))
    lines.append("=" * W)

    report = "\n".join(lines)
    print("\n" + report)

    # --- Write report ---
    out_dir = Path("data/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / f"backtest_btc_lr_regime_{date_str}.txt"
    txt_path.write_text(report)
    logger.info(f"Report written: {txt_path}")

    # Equity CSVs for both filtered tracks
    for label, trades in [("ct", ct_all), ("tf", tf_all)]:
        if trades:
            df_t = pd.DataFrame(trades)
            df_t["cumulative_pnl"] = df_t["pnl"].cumsum()
            csv_path = out_dir / f"backtest_btc_lr_regime_{date_str}_{label}.csv"
            df_t.to_csv(csv_path, index=False)
            logger.info(f"{label.upper()} equity CSV: {csv_path}")

    # --- Write config for whichever polarity passed (prefer trend-following) ---
    winning_polarity = None
    if tf_gate_passed:
        winning_polarity = "trend_following"
    elif ct_gate_passed:
        winning_polarity = "counter_trend"

    if winning_polarity:
        btc_cfg = {
            "fast_len": LR_FAST,
            "slow_len": LR_SLOW,
            "polarity": winning_polarity,
            "enabled": True,
            "validated_date": today_iso,
            "gate_criteria": {
                "pf_min": GATE_PF_MIN,
                "min_filter_ratio": GATE_MIN_FILTER_RATIO,
            },
            "note": "Written by backtest_btc_lr_regime_validation.py — all gate criteria passed",
        }
        cfg_path = Path("models/xgboost/lr_regime_config_btc.json")
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(btc_cfg, indent=2))
        logger.info(f"Config written: {cfg_path} (polarity={winning_polarity})")
    else:
        logger.warning(
            "Both polarities failed promotion gate — lr_regime_config_btc.json NOT written."
        )


if __name__ == "__main__":
    main()
