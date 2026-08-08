#!/usr/bin/env python3
"""FVG feasibility diagnostic — can the gap filters be satisfied at all?

Built after the 2026-08-07 party-mode sidebar on YANK's 17-day signal drought.

``detect_fvg`` (src/research/strategy_core.py:387) applies two gap gates that are
denominated differently:

    gap_pts * POINT_VALUE_USD <= config.max_gap_dollars    # ABSOLUTE currency ceiling
    gap_pts                   >= config.min_gap_atr_ratio * h1_atr   # SCALE-FREE floor

The floor rescales with the market; the ceiling does not. So the set of gap sizes
that satisfies BOTH is

    [ min_gap_atr_ratio * h1_atr ,  max_gap_dollars / POINT_VALUE_USD ]

which is EMPTY whenever

    h1_atr > max_gap_dollars / (POINT_VALUE_USD * min_gap_atr_ratio)

At the live config (60.0 / 2.0 / 0.25) that bound is 120 index points. Above it no
FVG of any size can pass, and the bot goes silent while every upstream stage keeps
logging healthy structure. In 2026-06 and 2026-07 the MEDIAN H1 hour was above the
bound — the window was not merely tight, it was empty.

This tool measures that window against recorded bars. It imports the live
``StrategyConfig`` loader and the live ``calc_atr``/``resample_to_h1`` rather than
reimplementing them, so the diagnostic cannot drift from the engine it describes.

It is READ-ONLY. It never trades, never writes to logs/, never mutates state.

Usage:
    .venv/bin/python tools/fvg_feasibility.py
    .venv/bin/python tools/fvg_feasibility.py --days 30
    .venv/bin/python tools/fvg_feasibility.py --csv data/processed/dollar_bars/1_minute/mnq_1min_2025.csv
    .venv/bin/python tools/fvg_feasibility.py --json
    .venv/bin/python tools/fvg_feasibility.py --plot _bmad-output/fvg_feasibility.png

Exit codes (for cron / alerting):
    0 = window open in the recent period
    1 = window empty for >= --empty-warn of recent H1 bars   (default 25%)
    2 = window empty for >= --empty-crit of recent H1 bars   (default 50%)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.research.strategy_core import (  # noqa: E402
    POINT_VALUE_USD,
    calc_atr,
    resample_to_h1,
)
from src.research.config_loader import load_strategy_config  # noqa: E402
from src.research.strategy_core import StrategyConfig  # noqa: E402

# Live bar record written by the YANK shadow-parity logger. Preferred source because
# these are the bars the bot actually saw, not what the venue serves for that session
# today — TradeStation silently revises 1-minute history (see the 2026-08-06 sigma
# corroboration: 145 of 390 bars for 2026-08-04 differ on re-fetch).
DEFAULT_BARS = BASE / "logs" / "yank_shadow_parity.csv"
DEFAULT_CONFIG = BASE / "strategy_config.yaml"

OK, WARN, CRIT = 0, 1, 2


def load_config(path: Path) -> StrategyConfig:
    if path.exists():
        return load_strategy_config(path)
    return StrategyConfig()


def _read_shadow_parity(path: Path) -> pd.DataFrame:
    """Load the shadow-parity log into canonical 1-min bars.

    Uses the ``ts_*`` (TradeStation) columns because those are the bars the live
    trader fed to the strategy; ``px_*`` is the ProjectX shadow.
    """
    df = pd.read_csv(
        path,
        usecols=["minute", "ts_open", "ts_high", "ts_low", "ts_close", "ts_vol"],
    )
    df = df.dropna(subset=["ts_open", "ts_high", "ts_low", "ts_close"])
    df["timestamp"] = pd.to_datetime(df["minute"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.rename(
        columns={
            "ts_open": "open",
            "ts_high": "high",
            "ts_low": "low",
            "ts_close": "close",
            "ts_vol": "volume",
        }
    )
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def _read_ohlcv(path: Path) -> pd.DataFrame:
    """Load a generic 1-min OHLCV csv (the processed/ and sealed_holdout/ layout)."""
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    tcol = next(
        (lower[k] for k in ("timestamp", "datetime", "date", "time") if k in lower),
        None,
    )
    if tcol is None:
        raise SystemExit(f"{path}: no timestamp-like column in {list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    out = pd.DataFrame({"timestamp": df["timestamp"]})
    for c in ("open", "high", "low", "close", "volume"):
        if c not in lower:
            raise SystemExit(f"{path}: missing column {c!r}")
        out[c] = df[lower[c]]
    return out


def load_bars(path: Path) -> pd.DataFrame:
    """Return canonical 1-min bars indexed by tz-aware timestamp, sorted, deduped."""
    if not path.exists():
        raise SystemExit(f"bar source not found: {path}")
    with path.open() as fh:
        header = fh.readline()
    df = _read_shadow_parity(path) if "ts_close" in header else _read_ohlcv(path)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    df.index.name = "timestamp"
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype("float64")
    df["volume"] = df["volume"].fillna(0).astype("int64")
    return df


def h1_atr_series(bars_1m: pd.DataFrame) -> pd.Series:
    """Rolling H1 ATR in index points, using the live ``calc_atr`` verbatim.

    ``calc_atr`` is a 20-bar mean True Range over the LAST 20 rows handed to it, so
    the rolling series is built by replaying the same trailing window the live trader
    holds — not by a pandas rolling mean, which would use a different TR convention.
    """
    h1 = resample_to_h1(bars_1m)
    vals, idx = [], []
    for i in range(20, len(h1) + 1):
        vals.append(calc_atr(h1.iloc[i - 20:i]))
        idx.append(h1.index[i - 1])
    return pd.Series(vals, index=pd.DatetimeIndex(idx, name="timestamp"), name="h1_atr")


def feasibility(atr: pd.Series, cfg: StrategyConfig) -> pd.DataFrame:
    """Per-H1-bar feasible gap window, in index points."""
    cap_pts = cfg.max_gap_dollars / POINT_VALUE_USD
    floor_pts = cfg.min_gap_atr_ratio * atr
    width = (cap_pts - floor_pts).clip(lower=0.0)
    return pd.DataFrame(
        {
            "h1_atr": atr,
            "floor_pts": floor_pts,
            "cap_pts": cap_pts,
            "width_pts": width,
            "empty": floor_pts > cap_pts,
        }
    )


def atr_bound(cfg: StrategyConfig) -> float:
    """H1 ATR above which the two gates are mutually unsatisfiable."""
    if cfg.min_gap_atr_ratio <= 0:
        return float("inf")
    return cfg.max_gap_dollars / (POINT_VALUE_USD * cfg.min_gap_atr_ratio)


def summarize(f: pd.DataFrame, by: str) -> pd.DataFrame:
    # tz_localize(None) first: to_period() drops the tz anyway and warns while doing it.
    key = f.index.tz_localize(None).to_period("M") if by == "month" else f.index.date
    g = f.groupby(key)
    return pd.DataFrame(
        {
            "h1_bars": g.size(),
            "med_atr": g["h1_atr"].median(),
            "pct_empty": g["empty"].mean() * 100.0,
            "med_width": g["width_pts"].median(),
            "med_floor": g["floor_pts"].median(),
        }
    )


def render_plot(f: pd.DataFrame, cfg: StrategyConfig, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap_pts = cfg.max_gap_dollars / POINT_VALUE_USD
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(f.index, f["floor_pts"], lw=1.0, color="#c1121f",
            label=f"required min gap = {cfg.min_gap_atr_ratio} x H1 ATR")
    ax.axhline(cap_pts, color="#003049", lw=1.4, ls="--",
               label=f"max gap = ${cfg.max_gap_dollars:.0f} / ${POINT_VALUE_USD:.0f} = {cap_pts:.0f} pts")
    ax.fill_between(f.index, f["floor_pts"], cap_pts,
                    where=f["floor_pts"] <= cap_pts, color="#2a9d8f", alpha=0.28,
                    label="feasible gap sizes")
    ax.fill_between(f.index, cap_pts, f["floor_pts"],
                    where=f["floor_pts"] > cap_pts, color="#c1121f", alpha=0.30,
                    label="NO satisfiable gap (window empty)")
    ax.set_ylabel("gap size (MNQ index points)")
    ax.set_title(
        f"FVG feasibility — window is empty whenever H1 ATR > {atr_bound(cfg):.0f} pts "
        f"({f['empty'].mean() * 100:.1f}% of the shown period)"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_BARS,
                    help=f"1-min bar source (default: {DEFAULT_BARS.relative_to(BASE)})")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="strategy_config.yaml to read the gap gates from")
    ap.add_argument("--days", type=int, default=14,
                    help="days of per-day detail and the window judged for the exit code")
    ap.add_argument("--empty-warn", type=float, default=25.0,
                    help="%% of recent H1 bars with an empty window that exits 1")
    ap.add_argument("--empty-crit", type=float, default=50.0,
                    help="%% of recent H1 bars with an empty window that exits 2")
    ap.add_argument("--plot", type=Path, help="write a PNG of the window over time")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bars = load_bars(args.csv)
    atr = h1_atr_series(bars)
    if atr.empty:
        print("not enough H1 bars to compute a 20-bar ATR", file=sys.stderr)
        return WARN
    f = feasibility(atr, cfg)

    cap_pts = cfg.max_gap_dollars / POINT_VALUE_USD
    bound = atr_bound(cfg)
    cutoff = f.index.max() - pd.Timedelta(days=args.days)
    recent = f[f.index >= cutoff]
    pct_empty = float(recent["empty"].mean() * 100.0) if len(recent) else 0.0

    level = CRIT if pct_empty >= args.empty_crit else (WARN if pct_empty >= args.empty_warn else OK)

    if args.json:
        print(json.dumps({
            "bar_source": str(args.csv),
            "config": str(args.config),
            "min_gap_atr_ratio": cfg.min_gap_atr_ratio,
            "max_gap_dollars": cfg.max_gap_dollars,
            "point_value_usd": POINT_VALUE_USD,
            "cap_pts": cap_pts,
            "atr_bound_pts": bound,
            "range": [str(f.index.min()), str(f.index.max())],
            "recent_days": args.days,
            "recent_h1_bars": int(len(recent)),
            "recent_pct_empty": round(pct_empty, 2),
            "recent_med_atr": round(float(recent["h1_atr"].median()), 1) if len(recent) else None,
            "recent_med_width_pts": round(float(recent["width_pts"].median()), 1) if len(recent) else None,
            "level": level,
        }, indent=2))
    else:
        print("=== FVG feasibility ===")
        print(f"bars   : {args.csv.relative_to(BASE) if args.csv.is_relative_to(BASE) else args.csv}")
        print(f"config : {args.config.name}  "
              f"min_gap_atr_ratio={cfg.min_gap_atr_ratio}  max_gap_dollars=${cfg.max_gap_dollars:.0f}")
        print(f"gates  : {cfg.min_gap_atr_ratio} x H1_ATR  <=  gap_pts  <=  "
              f"{cap_pts:.0f} pts (${cfg.max_gap_dollars:.0f} / ${POINT_VALUE_USD:.0f})")
        print(f"bound  : window is EMPTY when H1 ATR > {bound:.0f} pts")
        print(f"period : {f.index.min():%Y-%m-%d} .. {f.index.max():%Y-%m-%d}  ({len(f)} H1 bars)")
        print()
        print("by month:")
        m = summarize(f, "month")
        print(f"  {'period':<10} {'H1 bars':>8} {'med ATR':>8} {'% EMPTY':>8} "
              f"{'med width':>10} {'med min-gap':>12}")
        for k, r in m.iterrows():
            print(f"  {str(k):<10} {int(r.h1_bars):>8} {r.med_atr:>8.1f} {r.pct_empty:>7.1f}% "
                  f"{r.med_width:>9.1f}p {r.med_floor:>11.1f}p")
        if len(recent):
            print()
            print(f"last {args.days} days:")
            d = summarize(recent, "day")
            print(f"  {'date':<12} {'H1 bars':>8} {'med ATR':>8} {'% EMPTY':>8} {'med width':>10}")
            for k, r in d.iterrows():
                print(f"  {str(k):<12} {int(r.h1_bars):>8} {r.med_atr:>8.1f} "
                      f"{r.pct_empty:>7.1f}% {r.med_width:>9.1f}p")
            print()
            tag = {OK: "OK", WARN: "WARN", CRIT: "CRITICAL"}[level]
            print(f"[{tag}] {pct_empty:.1f}% of the last {args.days} days' H1 bars had NO "
                  f"satisfiable gap size (median feasible width "
                  f"{recent['width_pts'].median():.1f} pts).")
            if level:
                print("       A silent bot in this regime is arithmetic, not market conditions.")
                print("       Changing either gate is a SEALED-CONFIG change — pre-register first.")

    if args.plot:
        render_plot(f, cfg, args.plot)
        print(f"\nwrote {args.plot}")

    return level


if __name__ == "__main__":
    sys.exit(main())
