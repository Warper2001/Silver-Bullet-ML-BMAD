"""YANK-FLOOR out-of-sample test. Seal ed8b226.

Both floors (live 0.25 and candidate 0.10) run on the SAME 2021-2024 bars, which
YANK's derivation has never seen. One knob differs. Verdict per seal §6 is both
statistical AND economic.
"""
from __future__ import annotations
import math, statistics, sys
from dataclasses import replace
import pandas as pd

sys.path.insert(0, "/root/Silver-Bullet-ML-BMAD")
from yank_compressed_cascade_phase1 import _load_bars, _precompute_gates, _run_cascade
from yank_lrc_grid_search import BASE_CONFIG

OOS = "data/mim_x/mnq_1min_2021_2024_frontmonth.csv"
LIVE_FLOOR, CAND_FLOOR = 0.25, 0.10
NOTIONAL, FEE_BPS, BAR_3X = 59000.0, 0.51, 1.53
SHAPE = [0.25, 0.20, 0.15, 0.125, 0.10, 0.075, 0.05]


def stats(pnls, ct):
    n = len(pnls)
    if n < 2: return None
    m, sd = statistics.mean(pnls), statistics.stdev(pnls)
    gp = sum(p for p in pnls if p > 0); gl = -sum(p for p in pnls if p < 0)
    bps = 10000 * (m / ct) / NOTIONAL
    return dict(n=n, mean=m, sd=sd, t=m/(sd/math.sqrt(n)), net=sum(pnls),
                pf=gp/gl if gl else float("inf"),
                wr=100*sum(1 for p in pnls if p > 0)/n, bps=bps, ratio=bps/FEE_BPS)


def main():
    cfg = replace(BASE_CONFIG, sl_multiplier=2.0, tp_multiplier=8.0)
    ct = cfg.contracts_per_trade
    bars = _load_bars(OOS)
    days = (bars.index[-1] - bars.index[0]).days or 1
    print("=" * 92)
    print("YANK-FLOOR — out-of-sample test   |   seal ed8b226")
    print("=" * 92)
    print(f"  bars {len(bars):,}   {bars.index[0].date()} -> {bars.index[-1].date()}   ({days} days)")
    print(f"  NEVER SEEN by YANK's derivation (its seals used 2025 + the 2026 holdout)")
    print(f"  SEALED POWER: detectable @t=2 $27.72 | @80% $38.81 | 2025 observed $42.56")
    print(f"  one knob: min_gap_atr_ratio  {LIVE_FLOOR} (live) vs {CAND_FLOOR} (candidate)\n", flush=True)

    gates = _precompute_gates(bars, cfg, "1h", "15min")
    print("  gates precomputed\n", flush=True)

    out = {}
    for fl in (LIVE_FLOOR, CAND_FLOOR):
        r = _run_cascade(bars, replace(cfg, min_gap_atr_ratio=fl), gates)
        out[fl] = stats(r.pnls, ct)
        out[fl]["trades"] = r.trades or []

    print(f"  {'floor':>7}{'N':>7}{'/day':>8}{'PF':>8}{'WR':>7}{'net$':>12}"
          f"{'mean$':>9}{'t':>8}{'bps':>8}{'ratio':>8}")
    print("  " + "-" * 90)
    for fl in (LIVE_FLOOR, CAND_FLOOR):
        s = out[fl]
        tag = "  <- LIVE" if fl == LIVE_FLOOR else "  <- CANDIDATE"
        print(f"  {fl:>7.3f}{s['n']:>7}{s['n']/days:>8.3f}{s['pf']:>8.3f}{s['wr']:>6.1f}%"
              f"{s['net']:>12,.0f}{s['mean']:>9.1f}{s['t']:>8.3f}{s['bps']:>8.2f}"
              f"{s['ratio']:>7.1f}x{tag}")

    c, l = out[CAND_FLOOR], out[LIVE_FLOOR]
    print("\n" + "=" * 92)
    print("VERDICT (seal §6)")
    print("=" * 92)
    print(f"  candidate 0.10 : mean ${c['mean']:+.2f}  t={c['t']:.3f}  {c['bps']:.2f} bps"
          f"  ({c['ratio']:.1f}x friction)   PF {c['pf']:.3f}")
    print(f"  live      0.25 : mean ${l['mean']:+.2f}  t={l['t']:.3f}  {l['bps']:.2f} bps"
          f"  ({l['ratio']:.1f}x friction)   PF {l['pf']:.3f}")
    print(f"  bars: t>=2.0 | >={BAR_3X} bps | PF(0.10) > PF(0.25)")
    if c["mean"] <= 0:
        v = "FAILS"
    elif c["t"] < 2.0:
        v = "UNPROVEN"
    elif c["bps"] < BAR_3X:
        v = "MARGINAL"
    elif c["pf"] <= l["pf"]:
        v = "MARGINAL"
    else:
        v = "PASS"
    print(f"\n  >>> {v} <<<")
    if v == "FAILS":
        print("  The loosening does not survive OOS. No second floor value (§7.1).")
    elif v == "UNPROVEN":
        print("  Positive but not significant. No deployment (§6).")
    elif v == "MARGINAL":
        print("  Real but sub-scale, or not better than live. No deployment (§6).")
    else:
        print("  Triggers §8 — authorises a DEPLOYMENT pre-registration, not a deployment.")

    print("\n" + "-" * 92)
    print("SECONDARY — reported, never decision-bearing (§6)")
    print("  per-year, candidate floor 0.10:")
    tr = pd.DataFrame(c["trades"])
    if len(tr):
        tr["yr"] = pd.to_datetime(tr["entry_ts"], utc=True, format="ISO8601").dt.year
        for yr, g in tr.groupby("yr"):
            print(f"    {yr}: N={len(g):>4}  mean=${g['pnl'].mean():>8.2f}  net=${g['pnl'].sum():>10,.0f}")
    print("\n  OOS floor sweep (SHAPE ONLY — §7.2 forbids acting on it):")
    print(f"    {'floor':>7}{'N':>7}{'PF':>8}{'mean$':>9}{'bps':>8}")
    for fl in SHAPE:
        r = _run_cascade(bars, replace(cfg, min_gap_atr_ratio=fl), gates)
        s = stats(r.pnls, ct)
        if s: print(f"    {fl:>7.3f}{s['n']:>7}{s['pf']:>8.3f}{s['mean']:>9.1f}{s['bps']:>8.2f}", flush=True)
    print(f"\n  backtest runs {ct} contracts / -750 daily; live YANK is 2ct / -300 (§9).")
    print("  Live bot unmodified. This document authorises no deployment.")


if __name__ == "__main__":
    main()
