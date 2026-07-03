"""Option C retrospective (H-C1): did policy-shock-flagged days have worse
realized performance for our trading systems than unflagged days?

Flag calendar was compiled from the public news record only (no contact with
P&L) and frozen in data/macro/policy_shock_calendar_2025_2026.csv before this
script was written. Two flag variants:

  DAY    — flagged calendar dates, mapped to their trading session (weekends/
           holidays roll to next session; known evening events also flag the
           following session).
  WINDOW — all trading days inside the multi-day shock windows.

Samples:
  yank_bt   — YANK sealed 1-year backtest, run 20260615_181838 (ml0.50, 5ct)
  mim_live  — MIM-NB live combine trades (hash-chained CSV, 1ct)
  gap_live  — GAP-1 TS SIM paper trades
  yank_live — YANK live combine closes from trades.db (symbol IS NOT NULL
              filters out the pre-idempotency replay contamination)
  s26_paper / s27_paper — shadow data collectors (modeled PnL), secondary

Usage:  .venv/bin/python tools/option_c_retrospective.py [--repo /path/to/main/checkout]
"""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

HOLIDAYS = {
    date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25), date(2026, 1, 1),
    date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3),
}

# Calendar dates whose event broke after the cash close (per calendar
# reconciliation notes): flag the next session as well.
EVENING_EVENTS = {date(2025, 7, 10), date(2025, 7, 22), date(2025, 8, 25),
                  date(2026, 4, 7)}

ANALYSIS_START = date(2025, 6, 1)
ANALYSIS_END = date(2026, 7, 2)

N_PERMUTATIONS = 10_000
RNG_SEED = 20260703


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in HOLIDAYS


def next_trading_day(d: date) -> date:
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def load_flags(macro_dir: Path) -> tuple[set[date], set[date]]:
    day_flags: set[date] = set()
    with open(macro_dir / "policy_shock_calendar_2025_2026.csv") as f:
        for row in csv.DictReader(f):
            d = date.fromisoformat(row["date"])
            day_flags.add(next_trading_day(d))
            if d in EVENING_EVENTS:
                day_flags.add(next_trading_day(d + timedelta(days=1)))
    window_flags: set[date] = set()
    with open(macro_dir / "policy_shock_windows_2025_2026.csv") as f:
        for row in csv.DictReader(f):
            d = date.fromisoformat(row["start"])
            end = date.fromisoformat(row["end"])
            while d <= end:
                if is_trading_day(d):
                    window_flags.add(d)
                d += timedelta(days=1)
    return day_flags, window_flags | day_flags


def session_date(ts: datetime) -> date:
    """CME session date: timestamps >= 18:00 ET belong to the next day."""
    ts_et = ts.astimezone(ET)
    d = ts_et.date()
    if ts_et.hour >= 18:
        d += timedelta(days=1)
    return d


def in_analysis_window(d: date) -> bool:
    return ANALYSIS_START <= d <= ANALYSIS_END


def load_yank_backtest(repo: Path) -> list[tuple[date, float]]:
    trades = []
    path = repo / "data/reports/backtest_1year_20260615_181838.csv"
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["entry_time"])
            d = session_date(ts)
            if in_analysis_window(d):
                trades.append((d, float(row["pnl"])))
    return trades


def load_mim_live(repo: Path) -> list[tuple[date, float]]:
    trades = []
    with open(repo / "data/mim_nb/trades.csv") as f:
        for row in csv.DictReader(f):
            trades.append((date.fromisoformat(row["day"]),
                           float(row["pnl_usd"].replace("+", ""))))
    return trades


def load_gap_live(repo: Path) -> list[tuple[date, float]]:
    trades = []
    with open(repo / "data/gap_fade/trades.csv") as f:
        for row in csv.DictReader(f):
            trades.append((date.fromisoformat(row["date_et"]),
                           float(row["pnl_usd"])))
    return trades


def load_db(repo: Path, trader: str, real_only: bool = False) -> list[tuple[date, float]]:
    con = sqlite3.connect(repo / "data/trades.db")
    q = "SELECT timestamp, pnl FROM trades WHERE trader_id=?"
    if real_only:
        q += " AND symbol IS NOT NULL"
    trades = []
    for ts_s, pnl in con.execute(q, (trader,)).fetchall():
        d = session_date(datetime.fromisoformat(ts_s))
        if in_analysis_window(d):
            trades.append((d, float(pnl)))
    con.close()
    return trades


def profit_factor(pnls: list[float]) -> float | None:
    gp = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    if gl == 0:
        return None
    return gp / gl


def fmt_pf(pf: float | None) -> str:
    return "inf" if pf is None else f"{pf:.3f}"


def split_stats(trades: list[tuple[date, float]], flags: set[date]) -> dict:
    fl = [p for d, p in trades if d in flags]
    un = [p for d, p in trades if d not in flags]
    return {
        "flagged": fl,
        "unflagged": un,
        "n_days_fl": len({d for d, _ in trades if d in flags}),
        "n_days_un": len({d for d, _ in trades if d not in flags}),
    }


def permutation_test(trades: list[tuple[date, float]], flags: set[date]) -> float | None:
    """Day-level label permutation: p-value for flagged mean-daily-PnL being
    LOWER than unflagged (one-sided, H-C1 direction)."""
    daily: dict[date, float] = {}
    for d, p in trades:
        daily[d] = daily.get(d, 0.0) + p
    days = sorted(daily)
    fl_days = [d for d in days if d in flags]
    un_days = [d for d in days if d not in flags]
    if len(fl_days) < 3 or len(un_days) < 3:
        return None
    obs = (sum(daily[d] for d in fl_days) / len(fl_days)
           - sum(daily[d] for d in un_days) / len(un_days))
    vals = [daily[d] for d in days]
    k = len(fl_days)
    rng = random.Random(RNG_SEED)
    hits = 0
    for _ in range(N_PERMUTATIONS):
        rng.shuffle(vals)
        perm = sum(vals[:k]) / k - sum(vals[k:]) / (len(vals) - k)
        if perm <= obs:
            hits += 1
    return hits / N_PERMUTATIONS


def report_sample(name: str, trades: list[tuple[date, float]],
                  day_flags: set[date], window_flags: set[date],
                  lines: list[str]) -> None:
    if not trades:
        lines.append(f"### {name}\n\nNo trades in analysis window.\n")
        return
    total = sum(p for _, p in trades)
    lines.append(f"### {name}\n")
    lines.append(f"N={len(trades)} trades over {len({d for d, _ in trades})} "
                 f"active days, total PnL ${total:,.0f}, "
                 f"PF {fmt_pf(profit_factor([p for _, p in trades]))}\n")
    for label, flags in (("DAY", day_flags), ("WINDOW", window_flags)):
        lines.append(f"**{label} variant**\n")
        lines.append("| Side | N trades | Active days | Total PnL | "
                     "Expectancy/trade | Win% | PF |")
        lines.append("|---|---|---|---|---|---|---|")
        s = split_stats(trades, flags)
        for side, pnls, nd in (("flagged", s["flagged"], s["n_days_fl"]),
                               ("unflagged", s["unflagged"], s["n_days_un"])):
            if pnls:
                wr = 100 * sum(1 for p in pnls if p > 0) / len(pnls)
                lines.append(
                    f"| {side} | {len(pnls)} | {nd} | "
                    f"${sum(pnls):,.0f} | ${sum(pnls)/len(pnls):,.0f} | "
                    f"{wr:.0f}% | {fmt_pf(profit_factor(pnls))} |")
            else:
                lines.append(f"| {side} | 0 | 0 | – | – | – | – |")
        p = permutation_test(trades, flags)
        if p is not None:
            lines.append(f"\n{label} permutation test (one-sided, flagged "
                         f"worse): p = {p:.3f} ({N_PERMUTATIONS} shuffles, "
                         f"day-level)\n")
        else:
            lines.append(f"\n{label}: too few days on one side for a "
                         f"permutation test.\n")
    lines.append("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/root/Silver-Bullet-ML-BMAD",
                    help="main checkout holding live data (read-only)")
    ap.add_argument("--out", default=None, help="markdown report path")
    args = ap.parse_args()
    repo = Path(args.repo)
    here = Path(__file__).resolve().parent.parent

    day_flags, window_flags = load_flags(here / "data/macro")
    all_days = [ANALYSIS_START + timedelta(days=i)
                for i in range((ANALYSIS_END - ANALYSIS_START).days + 1)]
    trading_days = [d for d in all_days if is_trading_day(d)]

    lines = [
        "# Option C Retrospective — H-C1 verdict input",
        "",
        f"Generated by tools/option_c_retrospective.py; analysis window "
        f"{ANALYSIS_START} .. {ANALYSIS_END}.",
        "",
        f"Flag coverage: DAY = {len([d for d in trading_days if d in day_flags])}"
        f"/{len(trading_days)} trading days flagged "
        f"({100*len([d for d in trading_days if d in day_flags])/len(trading_days):.0f}%), "
        f"WINDOW = {len([d for d in trading_days if d in window_flags])}"
        f"/{len(trading_days)} "
        f"({100*len([d for d in trading_days if d in window_flags])/len(trading_days):.0f}%).",
        "",
    ]

    samples = [
        ("YANK sealed backtest 181838 (ml0.50, 5ct)", load_yank_backtest(repo)),
        ("MIM-NB live (combine, 1ct)", load_mim_live(repo)),
        ("GAP-1 live (TS SIM paper)", load_gap_live(repo)),
        ("YANK live (combine, real rows only)", load_db(repo, "trader-yank", real_only=True)),
        ("s26 paper collector (modeled)", load_db(repo, "trader-s26")),
        ("s27 paper collector (modeled)", load_db(repo, "trader-s27")),
    ]
    live_combined = samples[1][1] + samples[2][1] + samples[3][1]
    samples.insert(4, ("Live book combined (MIM + GAP-1 + YANK live)", live_combined))

    for name, trades in samples:
        report_sample(name, trades, day_flags, window_flags, lines)

    out = Path(args.out) if args.out else (
        here / "_bmad-output/option_c_retrospective_20260703.md")
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[written to {out}]")


if __name__ == "__main__":
    main()
