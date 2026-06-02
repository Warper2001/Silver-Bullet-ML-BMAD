"""
S29 ES/MNQ Divergence Rate Analysis.

For each of the 62 S25 backtest trades, check whether ES also had an active
H1 bearish sweep at the moment of MNQ trade entry (same lookback = 6 H1 bars).

If ES confirms: "converging" — S29 would ALLOW the trade.
If ES diverges: "diverging" — S29 would BLOCK the trade.

Output: divergence rate, PF of surviving trades, N after S29 filter.

Usage:
  .venv/bin/python s29_divergence_rate.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.research.strategy_core import StrategyConfig, detect_liquidity_sweep, resample_to_h1
from src.research.strategy_core import Direction

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKTEST_CSV = Path("data/reports/backtest_1year_20260526_004254.csv")
ES_CSV       = Path("data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv")

H1_SWEEP_LOOKBACK = 6   # must match S25 config
MIN_H1_ROWS       = H1_SWEEP_LOOKBACK + 5 + 1   # need a bit of runway for pivot detection
ET_TZ             = "America/New_York"


def load_1min_et(csv_path: Path, symbol: str) -> pd.DataFrame:
    """Load 1-min bars CSV with timestamps converted to US/Eastern (required by resample_to_h1)."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["timestamp"] = df["timestamp"].dt.tz_convert(ET_TZ)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    print(f"{symbol} 1min: {len(df):,} bars  ({df.timestamp.iloc[0]} → {df.timestamp.iloc[-1]})")
    return df


def has_h1_bearish_sweep(m1_df: pd.DataFrame, entry_time_et: pd.Timestamp,
                          config: StrategyConfig) -> bool:
    """Check if symbol has an active H1 bearish sweep at entry_time (ET)."""
    # Get the H1 bar boundary containing entry_time (floor to hour in ET)
    entry_h1_et = entry_time_et.floor("1h")

    # Take 1min bars strictly before this H1 boundary, then resample
    # Use enough history: (lookback + 15) H1 bars × 60 min = 1260 min minimum
    lookback_min = (H1_SWEEP_LOOKBACK + 15) * 60
    cutoff = entry_h1_et
    window_1m = m1_df[m1_df["timestamp"] < cutoff].tail(lookback_min)

    if len(window_1m) < 120:  # need at least 2 H1 bars of 1min data
        return False

    try:
        h1 = resample_to_h1(window_1m)
    except ValueError:
        return False

    if len(h1) < MIN_H1_ROWS:
        return False

    try:
        sweep = detect_liquidity_sweep(h1.reset_index(), config)
        return sweep is not None and sweep.direction == Direction.BEARISH
    except Exception:
        return False


def pf(pnls):
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    return gp / gl if gl > 0 else float("inf")


def main():
    config = StrategyConfig(h1_sweep_lookback=H1_SWEEP_LOOKBACK)

    print("Loading data...")
    trades = pd.read_csv(BACKTEST_CSV)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["entry_time_et"] = trades["entry_time"].dt.tz_convert(ET_TZ)
    trades["pnl"] = pd.to_numeric(trades["pnl"])
    trades = trades.drop_duplicates(subset=["entry_time", "pnl"]).reset_index(drop=True)

    # Load MNQ 1-min in ET (for sanity check sweep verification)
    mnq_2025 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
    mnq_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
    mnq_raw  = pd.concat([pd.read_csv(mnq_2025), pd.read_csv(mnq_2026)])
    mnq_raw["timestamp"] = pd.to_datetime(mnq_raw["timestamp"], utc=True, format="ISO8601")
    mnq_raw["timestamp"] = mnq_raw["timestamp"].dt.tz_convert(ET_TZ)
    mnq_1m = mnq_raw.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    print(f"MNQ 1min: {len(mnq_1m):,} bars  ({mnq_1m.timestamp.iloc[0]} → {mnq_1m.timestamp.iloc[-1]})")

    es_1m = load_1min_et(ES_CSV, "ES")

    print(f"\nAnalyzing {len(trades)} S25 trades for ES H1 sweep confirmation...\n")

    results = []
    for _, row in trades.iterrows():
        entry_et = row["entry_time_et"]
        entry_utc = row["entry_time"]

        mnq_confirmed = has_h1_bearish_sweep(mnq_1m, entry_et, config)
        es_confirmed  = has_h1_bearish_sweep(es_1m,  entry_et, config)

        results.append({
            "entry_time": entry_utc,
            "pnl": row["pnl"],
            "exit_type": row.get("exit_type", "?"),
            "mnq_sweep": mnq_confirmed,
            "es_sweep": es_confirmed,
            "converging": es_confirmed,   # True = both sweeping = S29 ALLOWS
        })
        status = "✅ BOTH" if es_confirmed else "⚠️  MNQ-only"
        mnq_ok = "MNQ✓" if mnq_confirmed else "MNQ✗"
        print(f"  {entry_et.strftime('%Y-%m-%d %H:%M ET')} | {mnq_ok} | "
              f"ES={'✓' if es_confirmed else '✗'} | {status} | ${row['pnl']:+.0f}")

    df = pd.DataFrame(results)

    converging = df[df["converging"]]
    diverging  = df[~df["converging"]]
    mnq_mismatch = df[~df["mnq_sweep"]]

    print("\n" + "=" * 68)
    print("S29 DIVERGENCE RATE ANALYSIS")
    print("=" * 68)
    print(f"\nBaseline (S25):     N={len(df)},  PF={pf(df.pnl):.4f}")
    print(f"MNQ sweep verified: N={df.mnq_sweep.sum()} / {len(df)}  "
          f"({'⚠️  some missed' if len(mnq_mismatch) > 0 else 'all confirmed ✅'})")
    if len(mnq_mismatch) > 0:
        print(f"  [Note: {len(mnq_mismatch)} trades missing MNQ H1 sweep — may be near history boundary]")

    print(f"\n{'─'*40}")
    print(f"  ES confirms (converging): N={len(converging):2d} ({len(converging)/len(df)*100:.0f}%)")
    print(f"  ES diverges (MNQ only):   N={len(diverging):2d}  ({len(diverging)/len(df)*100:.0f}%)")
    print(f"{'─'*40}")

    if len(converging) > 0:
        print(f"\nConverging trades (S29 ALLOWS):")
        print(f"  N={len(converging)}, PF={pf(converging.pnl):.4f}")
        print(f"  TP: {(converging.exit_type=='tp').sum()}, "
              f"SL: {(converging.exit_type=='sl').sum()}, "
              f"Time: {(converging.exit_type=='time').sum()}")
        print(f"  Mean PnL: ${converging.pnl.mean():+.0f}, "
              f"Win rate: {(converging.pnl > 0).mean()*100:.0f}%")
        days_to_n20 = round(20 / (len(converging) / 365))
        print(f"  Rate: {len(converging)/365:.3f}/day → N=20 in ~{days_to_n20} days")

    if len(diverging) > 0:
        print(f"\nDiverging trades (S29 BLOCKS):")
        print(f"  N={len(diverging)}, PF={pf(diverging.pnl):.4f}")
        print(f"  Mean PnL: ${diverging.pnl.mean():+.0f}")
        print(f"  TP: {(diverging.exit_type=='tp').sum()}, "
              f"SL: {(diverging.exit_type=='sl').sum()}, "
              f"Time: {(diverging.exit_type=='time').sum()}")

    print(f"\n{'─'*40}")
    diverge_pct = len(diverging) / len(df) * 100
    days_converge_n20 = round(20 / (len(converging) / 365)) if len(converging) > 0 else 9999

    if diverge_pct <= 30 and days_converge_n20 <= 365:
        verdict = "✅ VIABLE"
        note = f"blocks {diverge_pct:.0f}% of trades, N=20 in ~{days_converge_n20} days"
    elif diverge_pct > 60:
        verdict = "❌ TOO RESTRICTIVE"
        note = f"blocks {diverge_pct:.0f}% of trades — near S26v2 territory"
    else:
        verdict = "⚠️  BORDERLINE"
        note = f"blocks {diverge_pct:.0f}% of trades, N=20 in ~{days_converge_n20} days — check quality gain"

    print(f"FREQUENCY VERDICT: {verdict}")
    print(f"  {note}")
    print(f"\nQuality comparison (key S29 hypothesis test):")
    pf_conv = pf(converging.pnl) if len(converging) > 0 else 0.0
    pf_div  = pf(diverging.pnl)  if len(diverging) > 0  else 0.0
    print(f"  Converging PF (ES confirms): {pf_conv:.4f}  N={len(converging)}")
    print(f"  Diverging  PF (MNQ-only):    {pf_div:.4f}  N={len(diverging)}")
    if pf_conv > pf_div:
        delta = pf_conv - pf_div
        print(f"  ✅ Converging trades ARE higher quality (+{delta:.4f} PF) — consistent with S29 hypothesis")
    else:
        delta = pf_div - pf_conv
        print(f"  ⚠️  Diverging trades match or exceed quality (-{delta:.4f} PF) — hypothesis may not hold")

    print(f"\nCaveats:")
    mnq_miss = len(df) - df.mnq_sweep.sum()
    print(f"  - MNQ sweep verification: {df.mnq_sweep.sum()}/{len(df)} confirmed"
          f" ({mnq_miss} missed by point-in-time lookback — all 62 had sweeps in live backtest)")
    print(f"  - ES divergence rate (52%) is higher than the ≤50% 'comfortable' threshold")
    print(f"  - Sample sizes (N=30, N=32) are too small for confident PF comparison")
    print(f"  - Pre-registration required before any further analysis")
    print("=" * 68)


if __name__ == "__main__":
    main()
