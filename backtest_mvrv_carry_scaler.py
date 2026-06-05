#!/usr/bin/env python3
"""Backtest: MVRV On-Chain Position Scaler for BTC Carry (MVRV-SCALER).

Pre-registration: _bmad-output/preregistration_mvrv_carry_scaler.md
Sealed: 2026-06-05

Applies MVRV-based position sizing to the existing carry backtest.
Compares MVRV-scaled Sharpe / MaxDD against the base carry backtest.

Requires:
  data/kraken/PF_XBTUSD_funding_rate.csv
  data/macro/MVRV_BTC.csv  (from download_mvrv_data.py)

Decision rule (from preregistration):
  PASS if MVRV-scaled Sharpe > baseline + 0.10
        OR  MVRV-scaled MaxDD < baseline MaxDD × 0.85
  FAIL if MVRV-scaled total return < baseline × 0.80
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# --- Frozen carry parameters (from preregistration_btc_carry_backtest.md) ---
HURDLE_ANNUAL_PCT = 10.0
COST_BPS_ENTRY = 15
COST_BPS_EXIT = 15
NEG_STOP_THRESHOLD = -0.0001
NEG_STOP_PERIODS = 3
BACKTEST_START = "2024-11-01"
BASE_NOTIONAL = 10_000.0  # USD

# --- Frozen MVRV scaler parameters (from preregistration_mvrv_carry_scaler.md) ---
MVRV_THRESHOLDS = [1.0, 2.0, 3.0]
MVRV_SCALES = {
    "deep_value": 2.0,   # MVRV < 1.0
    "fair_value": 1.0,   # 1.0 <= MVRV < 2.0
    "overvalued": 0.5,   # 2.0 <= MVRV < 3.0
    "distribution": 0.0, # MVRV >= 3.0
}
MVRV_LAG_DAYS = 1

# --- Paths ---
FUNDING_PATH = Path("data/kraken/PF_XBTUSD_funding_rate.csv")
MVRV_PATH = Path("data/macro/MVRV_BTC.csv")


def load_funding() -> pd.DataFrame:
    if not FUNDING_PATH.exists():
        logger.error(f"Funding rate data not found at {FUNDING_PATH}")
        sys.exit(1)
    df = pd.read_csv(FUNDING_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"] >= BACKTEST_START].reset_index(drop=True)
    df["annualized"] = df["funding_rate"] * 3 * 365
    logger.info(f"Funding data: {len(df)} periods | "
                f"{df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def load_mvrv() -> pd.DataFrame:
    if not MVRV_PATH.exists():
        logger.error(f"MVRV data not found at {MVRV_PATH} — run download_mvrv_data.py first")
        sys.exit(1)
    df = pd.read_csv(MVRV_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"MVRV data: {len(df)} rows | "
                f"{df['date'].min().date()} → {df['date'].max().date()}")
    logger.info(f"Current MVRV: {df['mvrv'].iloc[-1]:.3f}")
    return df


def mvrv_to_scale(mvrv: float) -> float:
    if mvrv < MVRV_THRESHOLDS[0]:
        return MVRV_SCALES["deep_value"]
    elif mvrv < MVRV_THRESHOLDS[1]:
        return MVRV_SCALES["fair_value"]
    elif mvrv < MVRV_THRESHOLDS[2]:
        return MVRV_SCALES["overvalued"]
    else:
        return MVRV_SCALES["distribution"]


def run_carry(df: pd.DataFrame, mvrv_daily: pd.DataFrame | None) -> tuple[pd.Series, pd.Series]:
    """Run carry simulation with and without MVRV scaling.

    Returns (base_pnl_series, scaled_pnl_series) indexed by funding timestamp.
    """
    # Build MVRV lookup (date → lagged scale)
    mvrv_lookup: dict = {}
    if mvrv_daily is not None:
        for _, row in mvrv_daily.iterrows():
            lag_date = row["date"] + pd.Timedelta(days=MVRV_LAG_DAYS)
            mvrv_lookup[lag_date.date()] = mvrv_to_scale(row["mvrv"])

    hurdle = HURDLE_ANNUAL_PCT / 100
    neg_consecutive = 0
    in_carry = False

    base_pnl = []
    scaled_pnl = []
    transitions = 0

    for i, row in df.iterrows():
        ts = row["timestamp"]
        rate = row["funding_rate"]
        ann = row["annualized"]

        # Entry/exit logic (identical to backtest_btc_carry.py)
        was_in = in_carry
        if not in_carry:
            if ann > hurdle:
                in_carry = True
                transitions += 1
        else:
            if rate < NEG_STOP_THRESHOLD:
                neg_consecutive += 1
                if neg_consecutive >= NEG_STOP_PERIODS:
                    in_carry = False
                    neg_consecutive = 0
                    transitions += 1
            else:
                neg_consecutive = 0

        # MVRV scale for this period
        trade_date = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        scale = mvrv_lookup.get(trade_date, 1.0) if mvrv_lookup else 1.0

        # P&L
        cost = 0.0
        if was_in != in_carry:
            cost = COST_BPS_ENTRY / 10_000 * BASE_NOTIONAL

        period_base = (in_carry * rate * BASE_NOTIONAL) - cost
        period_scaled = (in_carry * rate * BASE_NOTIONAL * scale) - cost

        base_pnl.append((ts, period_base))
        scaled_pnl.append((ts, period_scaled))

    base = pd.Series(
        [x[1] for x in base_pnl],
        index=pd.DatetimeIndex([x[0] for x in base_pnl]),
        name="base_pnl",
    )
    scaled = pd.Series(
        [x[1] for x in scaled_pnl],
        index=pd.DatetimeIndex([x[0] for x in scaled_pnl]),
        name="scaled_pnl",
    )
    logger.info(f"Carry transitions: {transitions} (entry/exit events)")
    return base, scaled


def compute_metrics(pnl: pd.Series, label: str) -> dict:
    n_periods = len(pnl)
    periods_per_year = 3 * 365  # three 8h periods per day
    total_return = pnl.sum() / BASE_NOTIONAL * 100
    ann_return = total_return / (n_periods / periods_per_year)

    # Sharpe on 8h P&L series (annualized)
    std = pnl.std()
    sharpe = (pnl.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

    # Max drawdown
    cum = pnl.cumsum()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / BASE_NOTIONAL
    max_dd = dd.min()

    metrics = {
        "label": label,
        "n_periods": n_periods,
        "total_return_pct": total_return,
        "ann_return_pct": ann_return,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
    }

    logger.info(f"  [{label}] Total return: {total_return:.2f}% | "
                f"Ann: {ann_return:.2f}% | Sharpe: {sharpe:.3f} | MaxDD: {max_dd*100:.2f}%")
    return metrics


def main() -> None:
    funding = load_funding()

    mvrv = None
    if MVRV_PATH.exists():
        mvrv = load_mvrv()
    else:
        logger.warning("MVRV data not found — scaled backtest will use scale=1.0 everywhere")

    logger.info("Running carry simulation...")
    base_pnl, scaled_pnl = run_carry(funding, mvrv)

    logger.info("=" * 60)
    logger.info("MVRV-SCALER RESULTS")
    logger.info("=" * 60)
    base_m = compute_metrics(base_pnl, "BASE (unscaled)")
    scaled_m = compute_metrics(scaled_pnl, "MVRV-SCALED")

    sharpe_delta = scaled_m["sharpe"] - base_m["sharpe"]
    dd_ratio = scaled_m["max_dd_pct"] / base_m["max_dd_pct"] if base_m["max_dd_pct"] != 0 else 1.0
    return_ratio = (scaled_m["total_return_pct"] / base_m["total_return_pct"]
                    if base_m["total_return_pct"] != 0 else 1.0)

    logger.info("-" * 60)
    logger.info(f"Sharpe delta:    {sharpe_delta:+.3f}")
    logger.info(f"MaxDD ratio:     {dd_ratio:.3f}  (lower = better protection)")
    logger.info(f"Return ratio:    {return_ratio:.3f}")

    if mvrv is not None:
        cur = mvrv["mvrv"].iloc[-1]
        cur_scale = mvrv_to_scale(cur)
        logger.info(f"Current MVRV: {cur:.3f} → position scale = {cur_scale:.1f}×")

    # --- Decision rule ---
    passes_sharpe = sharpe_delta >= 0.10
    passes_dd = dd_ratio <= 0.85
    fails_return = return_ratio < 0.80

    if fails_return:
        verdict = "FAIL"
        detail = "MVRV scaler hurts total return too much; discard"
    elif passes_sharpe or passes_dd:
        verdict = "PASS"
        detail = "Integrate MVRV scaler into live carry executor"
    else:
        verdict = "AMBIGUOUS"
        detail = "Test alternative thresholds (e.g. 2.5/3.5 instead of 2.0/3.0)"

    logger.info(f"Passes Sharpe threshold (Δ≥0.10): {passes_sharpe}")
    logger.info(f"Passes DD threshold (ratio≤0.85):  {passes_dd}")
    logger.info(f"Fails return floor (ratio<0.80):   {fails_return}")
    logger.info(f"VERDICT: {verdict} — {detail}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
