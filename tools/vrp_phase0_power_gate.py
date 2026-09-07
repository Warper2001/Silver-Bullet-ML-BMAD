#!/usr/bin/env python3
"""VRP-1 Phase 0 — POWER GATE (firewalled).

Pre-registration: _bmad-output/preregistration_vrp_gate0.md (seal d1afdf5), §5 Phase 0.

FIREWALL — this script deliberately computes and prints ONLY:
  * N (daily observations) and calendar span of the development-window front-contract series
  * annualized realized volatility of that series
  * the resulting minimum detectable annualized Sharpe (MDE)

It NEVER computes, prints, stores or returns the MEAN, the sign, a Sharpe estimate, or any P&L.
The mean is explicitly deleted rather than merely unused, so a later reader can verify no effect
size was observed at this stage. Holdout rows (>= 2019-01-01) are dropped before any statistic.

Usage:  PYTHONPATH=. python tools/vrp_phase0_power_gate.py
"""
from __future__ import annotations

import asyncio
import statistics
import sys
from datetime import date, datetime

import requests

sys.path.insert(0, "/root/Silver-Bullet-ML-BMAD")
from src.data.auth_v3 import TradeStationAuthV3  # noqa: E402

MD = "https://api.tradestation.com/v3/marketdata"

# Sealed parameters — do not edit without a new seal.
DEV_START = date(2007, 6, 21)
DEV_END = date(2018, 12, 31)          # holdout begins 2019-01-01; never read here
ROLL_DAYS_BEFORE_SETTLE = 3           # sealed roll rule
SHARPE_BAR = 0.50                     # inherited from TSC-1
ALPHA_Z = 1.959964                    # two-sided alpha = 0.05
POWER_Z = 0.841621                    # 80% power
TRADING_DAYS = 252

MONTH_CODES = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]


def fetch(tok: str, sym: str) -> list[dict]:
    try:
        r = requests.get(f"{MD}/barcharts/{sym}",
                         headers={"Authorization": f"Bearer {tok}"},
                         params={"unit": "Daily", "barsback": 400}, timeout=60)
    except Exception:  # noqa: BLE001
        return []
    if r.status_code != 200:
        return []
    return r.json().get("Bars", [])


async def main() -> int:
    auth = TradeStationAuthV3.from_file("/root/Silver-Bullet-ML-BMAD/.access_token")
    tok = await auth.authenticate()
    print("VRP-1 Phase 0 — POWER GATE (no P&L, no mean, dev window only)\n")

    # ---- build the contract universe covering the dev window -------------------
    contracts: dict[str, list[dict]] = {}
    for yy in range(7, 20):                      # 2007..2019 (2019 only to close 2018's front)
        for mc in MONTH_CODES:
            sym = f"VX{mc}{yy:02d}"
            bars = fetch(tok, sym)
            if bars:
                contracts[sym] = bars
    print(f"contracts fetched with data : {len(contracts)}")

    # settlement proxy = last bar date of each contract
    settle = {s: datetime.fromisoformat(b[-1]["TimeStamp"].replace("Z", "+00:00")).date()
              for s, b in contracts.items()}

    # per-contract date -> close
    closes: dict[str, dict[date, float]] = {}
    all_dates: set[date] = set()
    for s, bars in contracts.items():
        d = {}
        for b in bars:
            dt = datetime.fromisoformat(b["TimeStamp"].replace("Z", "+00:00")).date()
            d[dt] = float(b["Close"])
        closes[s] = d
        all_dates |= set(d)

    # dev-window trading dates only — holdout rows dropped here, before any statistic
    dates = sorted(d for d in all_dates if DEV_START <= d <= DEV_END)
    print(f"dev-window trading dates    : {len(dates)}  "
          f"({dates[0]} → {dates[-1]})")

    # ---- roll rule: front = nearest settlement more than N trading days ahead ---
    idx = {d: i for i, d in enumerate(dates)}

    def front_for(d: date) -> str | None:
        best, best_s = None, None
        for s, sd in settle.items():
            if sd <= d or d not in closes[s]:
                continue
            # require at least ROLL_DAYS_BEFORE_SETTLE trading days of life left
            if sd in idx and d in idx and idx[sd] - idx[d] < ROLL_DAYS_BEFORE_SETTLE:
                continue
            if best is None or sd < best:
                best, best_s = sd, s
        return best_s

    # ---- daily returns of the held front contract (within-contract only) -------
    rets: list[float] = []
    held_prev: tuple[str, date] | None = None
    for d in dates:
        s = front_for(d)
        if s is None:
            held_prev = None
            continue
        if held_prev and held_prev[0] == s:
            p0 = closes[s].get(held_prev[1])
            p1 = closes[s].get(d)
            if p0 and p1 and p0 > 0:
                rets.append(p1 / p0 - 1.0)
        held_prev = (s, d)

    n = len(rets)
    if n < 30:
        print(f"\nInsufficient series ({n} returns) — cannot evaluate. ABORT.")
        return 1

    # ---- FIREWALL: volatility only. The mean is computed nowhere and the series
    # is discarded immediately after sigma so no effect size can leak into Phase 0.
    sigma_d = statistics.pstdev(rets)
    del rets                                   # explicit: nothing else may be derived
    sigma_ann = sigma_d * (TRADING_DAYS ** 0.5)

    span_years = n / TRADING_DAYS
    z = ALPHA_Z + POWER_Z
    mde_ann = z / (span_years ** 0.5)
    years_needed = (z / SHARPE_BAR) ** 2

    print(f"\n{'='*64}")
    print("  INPUTS (volatility and N only — no mean, no Sharpe, no P&L)")
    print(f"{'='*64}")
    print(f"  N daily observations      : {n}")
    print(f"  effective span            : {span_years:.2f} years")
    print(f"  annualized realized vol   : {sigma_ann*100:.1f}%   (context only)")
    print(f"\n{'='*64}")
    print("  POWER (alpha=0.05 two-sided, 80% power)")
    print(f"{'='*64}")
    print(f"  minimum detectable Sharpe : {mde_ann:.3f}")
    print(f"  sealed bar to detect      : {SHARPE_BAR:.3f}")
    print(f"  span needed for that bar  : {years_needed:.1f} years "
          f"(have {span_years:.2f})")
    print(f"  achieved power at SR={SHARPE_BAR:.2f}   : "
          f"{_power_at(SHARPE_BAR, span_years):.1%}")

    print(f"\n{'='*64}")
    if mde_ann > SHARPE_BAR:
        print("  VERDICT: ⛔ UNDERPOWERED — sealed rule §5 Phase 0 closes the study.")
        print(f"  MDE {mde_ann:.3f} > bar {SHARPE_BAR:.2f}. No further phase runs;")
        print("  no holdout is read. This is a statement about the DESIGN's")
        print("  resolving power, NOT evidence that VRP lacks an edge.")
    else:
        print("  VERDICT: ✅ POWERED — Phase 1 (dev-window Gate 0) is authorized.")
    print(f"{'='*64}\n")
    return 0


def _power_at(sr: float, span_years: float) -> float:
    """Achieved power to detect annualized Sharpe `sr` over `span_years`."""
    from math import erf, sqrt
    ncp = sr * (span_years ** 0.5)
    # P(Z > z_alpha - ncp) for a two-sided test, upper tail
    x = ALPHA_Z - ncp
    return 0.5 * (1.0 - erf(x / sqrt(2.0)))


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
