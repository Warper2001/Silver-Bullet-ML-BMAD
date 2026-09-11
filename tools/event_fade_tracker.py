#!/usr/bin/env python3
"""
MNQ econ-event fade/trend -- prospective tracker (observation only).

Seal: _bmad-output/preregistration_event_fade_prospective.md (2026-09-02)

Scores tier-1 economic events per the FROZEN rule and appends them to a paper
ledger. Executes nothing, risks nothing, authorizes nothing. Its only job is to
accrue evidence on a clock that is measured in years, so that the clock starts
tonight instead of the next time someone remembers.

FROZEN MECHANICS (may not change without a new pre-registration)
    pre-event reference : close of the event minute
    impulse window K    : 3 minutes
    hold M              : 30 minutes
    FOMC                : FADE the impulse
    NFP                 : FOLLOW the impulse
    CPI                 : logged, excluded from the decision rule
    cost                : $2.24 per round trip, 1 contract

ONLY events on/after SEAL_DATE count. The historical events that generated the
hypothesis are the discovery sample and are excluded by construction.

LOUD-WHEN-EMPTY: if the calendar has no coverage after today, this exits non-zero
and says so. A tracker that runs clean while measuring nothing is the failure
mode this shop named on 2026-08-08 (a bot that looks healthy is not a bot that
is working) and hit again with BTC-CARRY on 2026-09-02.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SEAL_DATE = date(2026, 9, 2)
SEAL_DOC = "_bmad-output/preregistration_event_fade_prospective.md"

CALENDAR = REPO / "data/macro/econ_calendar_2025_2026.csv"
LEDGER = REPO / "data/event_fade/prospective_trades.csv"
BAR_SOURCES = [
    REPO / "data/processed/mnq_1min_2026_ytd.csv",
    REPO / "data/processed/mnq_1min_2025.csv",
]

K_IMPULSE_MIN = 3
M_HOLD_MIN = 30
COST_PER_RT = 2.24
POINT_VALUE = 2.0  # MNQ: $2 per index point
N_TARGET = 30

# FADE = trade against the K-minute impulse; FOLLOW = trade with it.
DIRECTION_RULE = {"FOMC": "FADE", "NFP": "FOLLOW"}
DECISION_TYPES = ("FOMC", "NFP")  # CPI logged only

LEDGER_COLS = [
    "event_date", "event_time_et", "event_type", "rule",
    "ref_price", "impulse_price", "exit_price", "impulse_pts",
    "direction", "gross_pnl", "net_pnl", "scored_at", "seal_doc",
]


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th <weekday> of a month (weekday: Mon=0 .. Sun=6)."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def generated_nfp(start: date, end: date) -> list[dict]:
    """NFP is the first Friday of each month, 08:30 ET -- deterministic, so it
    can be generated rather than transcribed. FOMC and CPI cannot: their dates
    are set by committee/BLS schedule and MUST come from the calendar file."""
    out = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        d = nth_weekday(y, m, 4, 1)  # Friday
        if start <= d <= end:
            out.append({"date": d.isoformat(), "time_et": "08:30", "event": "NFP",
                        "tier": 1, "source": "generated"})
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def load_calendar() -> pd.DataFrame:
    rows: list[dict] = []
    if CALENDAR.exists():
        df = pd.read_csv(CALENDAR)
        for _, r in df.iterrows():
            if int(r["tier"]) == 1:
                rows.append({"date": str(r["date"]), "time_et": str(r["time_et"]),
                             "event": str(r["event"]), "tier": 1, "source": "calendar"})
    known = {(r["date"], r["event"]) for r in rows}
    horizon = date.today() + timedelta(days=400)
    for r in generated_nfp(SEAL_DATE, horizon):
        if (r["date"], r["event"]) not in known:
            rows.append(r)
    cal = pd.DataFrame(rows)
    return cal.sort_values("date").reset_index(drop=True)


def load_bars() -> pd.DataFrame | None:
    frames = []
    for p in BAR_SOURCES:
        if p.exists():
            df = pd.read_csv(p)
            tcol = next((c for c in df.columns if c.lower() in
                         ("timestamp", "datetime", "time", "date")), None)
            if tcol is None:
                continue
            df["ts"] = pd.to_datetime(df[tcol], format="mixed", errors="coerce", utc=True)
            df = df.dropna(subset=["ts"])
            df.columns = [c.lower() for c in df.columns]
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames).drop_duplicates("ts").sort_values("ts").set_index("ts")


def load_ledger() -> set[tuple[str, str]]:
    if not LEDGER.exists():
        return set()
    with open(LEDGER) as fh:
        return {(r["event_date"], r["event_type"]) for r in csv.DictReader(fh)}


def append_ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def score_event(bars: pd.DataFrame, ev: dict) -> dict | None:
    """Apply the frozen rule to one event. None if bars don't cover it."""
    et = pd.Timestamp(f"{ev['date']} {ev['time_et']}", tz="America/New_York").tz_convert("UTC")
    window = bars.loc[et: et + timedelta(minutes=M_HOLD_MIN + K_IMPULSE_MIN + 5)]
    if len(window) < K_IMPULSE_MIN + M_HOLD_MIN:
        return None
    close = window["close"]
    ref = float(close.iloc[0])
    impulse_px = float(close.iloc[K_IMPULSE_MIN])
    exit_px = float(close.iloc[min(K_IMPULSE_MIN + M_HOLD_MIN, len(close) - 1)])
    impulse = impulse_px - ref
    rule = DIRECTION_RULE.get(ev["event"], "LOG_ONLY")
    if rule == "LOG_ONLY" or impulse == 0:
        direction = "FLAT"
        gross = 0.0
    else:
        # FADE: trade against the impulse. FOLLOW: trade with it.
        sign = -1 if rule == "FADE" else 1
        direction = "SHORT" if sign * impulse < 0 else "LONG"
        gross = sign * (exit_px - impulse_px) * (1 if impulse > 0 else -1) * POINT_VALUE
    return {
        "event_date": ev["date"], "event_time_et": ev["time_et"], "event_type": ev["event"],
        "rule": rule, "ref_price": round(ref, 2), "impulse_price": round(impulse_px, 2),
        "exit_price": round(exit_px, 2), "impulse_pts": round(impulse, 2),
        "direction": direction, "gross_pnl": round(gross, 2),
        "net_pnl": round(gross - COST_PER_RT, 2) if rule != "LOG_ONLY" else 0.0,
        "scored_at": datetime.utcnow().isoformat(), "seal_doc": SEAL_DOC,
    }


def status() -> None:
    print(f"seal: {SEAL_DOC}  (sealed {SEAL_DATE})")
    if not LEDGER.exists():
        print(f"ledger: {LEDGER} -- not created yet, N=0 for every event type")
    else:
        df = pd.read_csv(LEDGER)
        print(f"ledger: {LEDGER}  ({len(df)} rows)")
        for t in DECISION_TYPES:
            g = df[df.event_type == t]
            print(f"  {t:<5} N={len(g):>3}/{N_TARGET}", end="")
            if len(g):
                pf_num = g[g.net_pnl > 0].net_pnl.sum()
                pf_den = -g[g.net_pnl < 0].net_pnl.sum()
                pf = (pf_num / pf_den) if pf_den else float("inf")
                print(f"   mean ${g.net_pnl.mean():+.2f}/ct   PF {pf:.2f}"
                      f"   (no decision until N>={N_TARGET})")
            else:
                print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="show accrual and exit")
    args = ap.parse_args()

    if args.status:
        status()
        return 0

    cal = load_calendar()
    today = date.today()
    future = cal[cal.date > today.isoformat()]
    scored_window = cal[(cal.date >= SEAL_DATE.isoformat()) & (cal.date <= today.isoformat())]

    print(f"calendar: {len(cal)} tier-1 events; {len(future)} still ahead")
    print(f"eligible to score (on/after seal, already past): {len(scored_window)}")

    already = load_ledger()
    bars = load_bars()
    scored = skipped = 0
    if len(scored_window):
        if bars is None:
            print("WARNING: no MNQ 1-min bar source found; cannot score", file=sys.stderr)
        else:
            for _, ev in scored_window.iterrows():
                if (ev["date"], ev["event"]) in already:
                    continue
                row = score_event(bars, ev.to_dict())
                if row is None:
                    skipped += 1
                    continue
                append_ledger(row)
                scored += 1
                print(f"  scored {ev['date']} {ev['event']:<5} "
                      f"{row['rule']:<7} net ${row['net_pnl']:+.2f}")
    print(f"newly scored: {scored}   skipped (no bars): {skipped}")

    status()

    # --- loud-when-empty ---------------------------------------------------
    horizon_days = 45
    soon = cal[(cal.date > today.isoformat())
               & (cal.date <= (today + timedelta(days=horizon_days)).isoformat())]
    fomc_cpi_ahead = future[future.event.isin(["FOMC", "CPI"])]
    if fomc_cpi_ahead.empty:
        print(
            f"\n*** CALENDAR EXHAUSTED ***\n"
            f"No FOMC or CPI events after {today}. NFP dates are generated "
            f"deterministically (first Friday), but FOMC and CPI dates are set by "
            f"committee/BLS schedule and MUST be transcribed into\n"
            f"  {CALENDAR}\n"
            f"Until that happens this tracker measures NFP only, and the FOMC leg "
            f"of the hypothesis -- the stronger one in the scout -- accrues NOTHING "
            f"while appearing to run normally. That is the exact failure this seal "
            f"was written to prevent.",
            file=sys.stderr,
        )
        return 1
    if soon.empty:
        print(f"\nWARNING: no tier-1 events in the next {horizon_days} days.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
