"""Feasibility check for the two proposed prospective accruals.

Uses only the VOLATILITY of MNQ day-session P&L (never a Monday mean — the observed
effect size is taken from the sealed Option-6 verdict, not re-derived here) to ask:
how many observations, and therefore how many YEARS, would each accrual need?
"""
import csv
from collections import defaultdict
from datetime import datetime, timezone
from math import sqrt
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
POINT = 2.0  # MNQ $/point
FILES = [
    "/root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute/mnq_1min_2025.csv",
    "/root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv",
]

ZA, ZP = 1.959964, 0.841621

sessions = defaultdict(list)  # date_et -> [(ts, close)]
for path in FILES:
    with open(path) as f:
        for r in csv.DictReader(f):
            ts = datetime.fromisoformat(r["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            et = ts.astimezone(ET)
            # RTH day session 09:30–16:00 ET
            mins = et.hour * 60 + et.minute
            if 570 <= mins <= 960:
                sessions[et.date()].append((mins, float(r["close"]), float(r["open"])))

daily = {}
for d, rows in sessions.items():
    if len(rows) < 100:
        continue
    rows.sort()
    open_px = rows[0][2]
    close_px = rows[-1][1]
    daily[d] = (close_px - open_px) * POINT

alldays = list(daily.values())
mondays = [v for d, v in daily.items() if d.weekday() == 0]

def sd(xs):
    m = sum(xs) / len(xs)
    return sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

sd_all, sd_mon = sd(alldays), sd(mondays)
print(f"MNQ day-session P&L per 1 contract, {min(daily)} → {max(daily)}")
print(f"  all days : N={len(alldays):4d}  sigma=${sd_all:,.2f}")
print(f"  Mondays  : N={len(mondays):4d}  sigma=${sd_mon:,.2f}   (sigma only — no mean printed)")

print("\n=== MON-1: how long to confirm the Option-6 Monday cell? ===")
print("  (effect sizes taken from the SEALED Option-6 verdict, not re-derived)")
for label, eff in [("Monday vs zero        (+$56.49/day)", 56.49),
                   ("Monday minus baseline (+$52.31/day)", 56.49 - 4.18)]:
    n = ((ZA + ZP) * sd_mon / eff) ** 2
    print(f"  {label}: n={n:7.0f} Mondays  ≈ {n/52:6.1f} years")

print("\n=== EVENT-FADE: how long at ~34 scheduled events/yr? ===")
for n_target in (30, 50):
    print(f"  N={n_target} aggregate : {n_target/34:5.1f} years")
    print(f"  N={n_target} per type (3 types, ~11/yr each) : {n_target/11:5.1f} years")
