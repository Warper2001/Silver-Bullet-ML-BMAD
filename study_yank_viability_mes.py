"""YANK-VIA — does YANK's sealed config generalise to MES? Seal 3f9167e.

Identical config, one instrument change. bps is point-value invariant, so the
engine's internal $2/pt does not distort the metric and the daily breaker binds at
the same POINT threshold. Dollar columns are meaningless for MES and not reported.
"""
from __future__ import annotations
import math, statistics, sys
from dataclasses import replace
import pandas as pd

sys.path.insert(0, "/root/Silver-Bullet-ML-BMAD")
from yank_compressed_cascade_phase1 import _load_bars, _precompute_gates, _run_cascade
from yank_lrc_grid_search import BASE_CONFIG

MES = "data/mim_x/mes_1min_2021_2024_frontmonth.csv"
MES_PV, MES_FRIC_USD = 5.0, 3.25
ENGINE_PV = 2.0          # what the cascade uses internally
BAR_MULT = 3.0


def main():
    cfg = replace(BASE_CONFIG, sl_multiplier=2.0, tp_multiplier=8.0, min_gap_atr_ratio=0.25)
    ct = cfg.contracts_per_trade
    bars = _load_bars(MES)
    lvl = bars["close"].mean()
    notional = MES_PV * lvl
    fric_bps = 10000 * MES_FRIC_USD / notional
    bar_bps = BAR_MULT * fric_bps
    days = (bars.index[-1] - bars.index[0]).days or 1

    print("=" * 92)
    print("YANK-VIA — YANK's sealed config on MES   |   seal 3f9167e")
    print("=" * 92)
    print(f"  bars {len(bars):,}   {bars.index[0].date()} -> {bars.index[-1].date()}  ({days} days)")
    print(f"  NEVER touched by YANK — no seal, sweep or backtest in this project has used MES")
    print(f"  config: live YANK unchanged, min_gap_atr_ratio={cfg.min_gap_atr_ratio}")
    print(f"  MES: level {lvl:,.0f}  notional/ct ${notional:,.0f}  friction {fric_bps:.3f} bps"
          f"  3x bar {bar_bps:.3f} bps")
    print(f"  SEALED POWER: detectable @t=2 (N~500) = 2.47 bps, below the {bar_bps:.2f} bps bar\n",
          flush=True)

    gates = _precompute_gates(bars, cfg, "1h", "15min")
    print("  gates precomputed\n", flush=True)
    res = _run_cascade(bars, cfg, gates)
    pnls = res.pnls
    n = len(pnls)
    if n < 2:
        print(f"  N={n} — insufficient. Reported as such.")
        return

    # engine dollars -> index points -> bps (point-value invariant)
    pts = [p / (ENGINE_PV * ct) for p in pnls]          # index points per contract
    bps = [10000 * x / lvl for x in pts]
    m, sd = statistics.mean(bps), statistics.stdev(bps)
    t = m / (sd / math.sqrt(n))
    gp = sum(x for x in bps if x > 0); gl = -sum(x for x in bps if x < 0)
    pf = gp / gl if gl else float("inf")
    wr = 100 * sum(1 for x in bps if x > 0) / n

    print(f"  N = {n}   ({n/days:.3f}/day)")
    print(f"  mean = {m:+.3f} bps/trade   sd = {sd:.2f}   t = {t:.3f}")
    print(f"  PF = {pf:.3f}   WR = {wr:.1f}%")
    print(f"  net = {sum(bps):+,.1f} bps  ({sum(pts):+,.0f} index points/contract)")

    # achieved power, per §5
    det = 2 * sd / math.sqrt(n)
    print(f"\n  ACHIEVED POWER (§5 requires recomputing if N or sd differ):")
    print(f"    realised N={n}, sd={sd:.2f} -> detectable @t=2.0 = {det:.3f} bps")
    print(f"    economic bar = {bar_bps:.3f} bps  ->  "
          f"{'ADEQUATE (detectable < bar)' if det < bar_bps else 'UNDERPOWERED (detectable > bar)'}")

    print("\n" + "=" * 92)
    print("VERDICT (seal §6)")
    print("=" * 92)
    if m <= 0:
        v = "DOES NOT GENERALISE"
    elif t < 2.0:
        v = "UNPROVEN"
    elif m < bar_bps:
        v = "REAL BUT SUB-SCALE"
    else:
        v = "STRUCTURAL"
    print(f"  mean {m:+.3f} bps | t {t:.3f} | bar {bar_bps:.3f} bps")
    print(f"\n  >>> {v} <<<")
    msg = {
     "DOES NOT GENERALISE": "  The edge is not present in MES. Record; close (§6).",
     "UNPROVEN":            "  No claim either way. No deployment (§6).",
     "REAL BUT SUB-SCALE":  "  Present, but cannot pay MES friction. No deployment (§6).",
     "STRUCTURAL":          "  Generalises to a sibling index future. Triggers §8. No deployment.",
    }[v]
    print(msg)
    print("  §7.4: this says nothing about YANK on MNQ in either direction.")

    print("\n" + "-" * 92)
    print("SECONDARY — reported, never decision-bearing (§6)")
    tr = res.trades or []
    if tr:
        d = pd.DataFrame(tr)
        d["yr"] = pd.to_datetime(d["entry_ts"], utc=True, format="ISO8601").dt.year
        d["bps"] = d["pnl"] / (ENGINE_PV * ct) / lvl * 10000
        for yr, g in d.groupby("yr"):
            print(f"    {yr}: N={len(g):>4}  mean={g['bps'].mean():+8.3f} bps  net={g['bps'].sum():+9.1f}")
        print(f"\n  exit mix: {d['exit_reason'].value_counts().to_dict()}")
    print(f"\n  At MNQ's friction (0.823 bps) this would be {m/0.823:.2f}x — shown for comparability only.")
    print("  Live bot unmodified. No deployment authorised.")


if __name__ == "__main__":
    main()
