#!/usr/bin/env python3
"""weekly_backtest.py — Rolling post-holdout backtest for weekly config review.

Runs the current StrategyConfig against the most recent N weeks of 1-min data
that falls AFTER the sealed holdout cutoff (2026-05-19). Use this each week to
evaluate the deployed config on fresh, non-holdout data before deciding on a
config change.

Usage:
    PYTHONPATH=. .venv/bin/python tools/weekly_backtest.py [options]

Examples:
    # Default: last 4 weeks of post-holdout MNQ data
    python tools/weekly_backtest.py

    # Smoke test against 2025 data (treat 2024-12-15 as the cutoff)
    python tools/weekly_backtest.py \\
        --holdout-cutoff 2024-12-15 \\
        --weeks 2 \\
        --data-file data/processed/dollar_bars/1_minute/mnq_1min_2025.csv

    # Multi-symbol (when MES/M2K data files are available)
    python tools/weekly_backtest.py --symbols MNQM26,MESM26,M2KM26
"""

import argparse
import math
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "dollar_bars" / "1_minute"
HOLDOUT_CUTOFF_DEFAULT = "2026-05-19"

# Maps symbol → file prefix used in the CSV naming convention
SYMBOL_DATA_PREFIXES: dict[str, str] = {
    "MNQM26": "mnq",
    "MESM26": "mes",
    "M2KM26": "m2k",
}


def _find_data_file(symbol: str) -> Path | None:
    """Return the most recent 1-min CSV for *symbol*, or None if not found."""
    prefix = SYMBOL_DATA_PREFIXES.get(symbol)
    if prefix is None:
        return None
    import datetime
    current_year = datetime.date.today().year
    for year in (current_year, current_year - 1):
        for suffix in ("_ytd", ""):
            p = DATA_DIR / f"{prefix}_1min_{year}{suffix}.csv"
            if p.exists():
                return p
    return None


def _load_and_filter(
    csv_path: Path,
    holdout_cutoff: str,
    weeks: int,
) -> pd.DataFrame | None:
    """Load 1-min bars from *csv_path*, keep only rows after *holdout_cutoff* and within last *weeks* calendar weeks."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    df = df.set_index("timestamp").sort_index()

    cutoff = pd.Timestamp(holdout_cutoff, tz="America/New_York")
    df = df[df.index > cutoff]
    if len(df) == 0:
        return None

    end_ts = df.index[-1]
    start_ts = end_ts - pd.Timedelta(weeks=weeks)
    df = df[df.index >= start_ts]
    return df if len(df) > 0 else None


def _write_temp_csv(df: pd.DataFrame) -> Path:
    """Write a filtered DataFrame to a temp CSV for BacktestEngine."""
    df_out = df.reset_index()
    df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    # Ensure required columns exist; drop notional if present (BacktestEngine ignores it)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df_out.to_csv(tmp.name, index=False)
    tmp.close()
    return Path(tmp.name)




def _print_table(rows: list[dict]) -> None:
    """Print a formatted results table."""
    header = f"{'Symbol':<8} | {'N':>4} | {'PF':>6} | {'WR':>6} | {'TIME_STOP%':>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    def fmt_row(r: dict) -> str:
        sym = r["symbol"]
        n = r["N"]
        note = r.get("note", "")
        if n == 0:
            return f"{sym:<8} | {n:>4} | {'—':>6} | {'—':>6} | {'—':>10}  ({note})"
        pf = r["pf"]
        pf_str = f"{pf:.3f}" if not math.isinf(pf) else "inf"
        wr_str = f"{r['wr']:.3f}"
        ts_str = f"{r['tstop_pct']:.0f}%"
        return f"{sym:<8} | {n:>4} | {pf_str:>6} | {wr_str:>6} | {ts_str:>10}"

    for r in rows:
        print(fmt_row(r))

    # POOLED row (only when multiple symbols with data)
    data_rows = [r for r in rows if r["N"] > 0]
    if len(rows) > 1 and data_rows:
        # Re-create pooled metric from aggregated values
        total_n = sum(r["N"] for r in data_rows)
        gross_win = sum(r.get("_gross_win", 0.0) for r in data_rows)
        gross_loss = sum(r.get("_gross_loss", 0.0) for r in data_rows)
        total_wins = sum(r.get("_wins", 0) for r in data_rows)
        total_tstop = sum(r.get("_tstops", 0) for r in data_rows)
        if total_n > 0:
            pf = gross_win / gross_loss if gross_loss > 0 else math.inf
            wr = total_wins / total_n
            tstop = total_tstop / total_n * 100
            pf_str = f"{pf:.3f}" if not math.isinf(pf) else "inf"
            print(sep)
            ts_str = f"{tstop:.0f}%"
            print(f"{'POOLED':<8} | {total_n:>4} | {pf_str:>6} | {wr:>6.3f} | {ts_str:>10}")

    print(sep)


def _run_all(
    symbols: list[str],
    config,
    holdout_cutoff: str,
    weeks: int,
    data_file_override: Path | None,
) -> list[dict]:
    """Run backtest for each symbol and collect aggregated metrics."""
    from src.research.backtest_engine import BacktestEngine

    all_rows = []
    for i, symbol in enumerate(symbols):
        override = data_file_override if i == 0 and data_file_override is not None else None
        csv_path = override if override is not None else _find_data_file(symbol)
        if csv_path is None:
            all_rows.append({
                "symbol": symbol, "N": 0, "pf": math.nan, "wr": math.nan,
                "tstop_pct": math.nan, "note": "no data file",
                "_gross_win": 0.0, "_gross_loss": 0.0, "_wins": 0, "_tstops": 0,
            })
            continue

        filtered = _load_and_filter(csv_path, holdout_cutoff, weeks)
        if filtered is None:
            all_rows.append({
                "symbol": symbol, "N": 0, "pf": math.nan, "wr": math.nan,
                "tstop_pct": math.nan, "note": "no post-holdout data in window",
                "_gross_win": 0.0, "_gross_loss": 0.0, "_wins": 0, "_tstops": 0,
            })
            continue

        tmp_path = _write_temp_csv(filtered)
        try:
            engine = BacktestEngine(str(tmp_path), config)
            trades = engine.run()
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

        n = len(trades)
        if n == 0:
            all_rows.append({
                "symbol": symbol, "N": 0, "pf": math.nan, "wr": math.nan,
                "tstop_pct": math.nan, "note": "0 trades in window",
                "_gross_win": 0.0, "_gross_loss": 0.0, "_wins": 0, "_tstops": 0,
            })
            continue

        pnls = [t.pnl_usd for t in trades]
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        wins = sum(1 for p in pnls if p > 0)
        tstops = sum(1 for t in trades if t.exit_reason == "TIME_STOP")
        pf = gross_win / gross_loss if gross_loss > 0 else math.inf
        wr = wins / n
        tstop_pct = tstops / n * 100

        all_rows.append({
            "symbol": symbol, "N": n, "pf": pf, "wr": wr, "tstop_pct": tstop_pct, "note": "",
            "_gross_win": gross_win, "_gross_loss": gross_loss, "_wins": wins, "_tstops": tstops,
        })

    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rolling post-holdout backtest for weekly config evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\nUsage:")[1] if "\n\nUsage:" in __doc__ else "",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("strategy_config.yaml"),
        help="Path to YAML config file (default: strategy_config.yaml)",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=4,
        help="Number of calendar weeks of post-holdout data to backtest (default: 4)",
    )
    parser.add_argument(
        "--symbols",
        default="MNQM26",
        help="Comma-separated symbol list (default: MNQM26)",
    )
    parser.add_argument(
        "--holdout-cutoff",
        default=HOLDOUT_CUTOFF_DEFAULT,
        help=f"Exclude data on or before this date (default: {HOLDOUT_CUTOFF_DEFAULT})",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Override CSV data file path (single symbol; applies to first symbol when multiple given)",
    )
    args = parser.parse_args()

    # Load config
    if args.config.exists():
        from src.research.config_loader import load_strategy_config
        config = load_strategy_config(args.config)
        print(f"Config: {args.config} (min_gap_atr_ratio={config.min_gap_atr_ratio})")
    else:
        from src.research.strategy_core import StrategyConfig
        config = StrategyConfig()
        print(f"Config: StrategyConfig() defaults (min_gap_atr_ratio={config.min_gap_atr_ratio})")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"Symbols: {', '.join(symbols)} | Weeks: {args.weeks} | Holdout cutoff: {args.holdout_cutoff}")
    print()

    rows = _run_all(symbols, config, args.holdout_cutoff, args.weeks, args.data_file)
    _print_table(rows)


if __name__ == "__main__":
    main()
