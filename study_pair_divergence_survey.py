"""
Cross-Pair Divergence-Fade Survey — Gate 0 (2026-06-12)

Generalization of study_stat_arb_short_only.py (the validated MNQ/MES template,
live on combine acct 23884932) to a frozen universe of pairs defined in
pair_survey_config.yaml. Protocol, gates, and decision rule are pre-committed in
_bmad-output/precommit_pair_divergence_survey_2026-06.md — both files must be
committed to git BEFORE this script runs.

Per pair: rolling 60-bar OLS beta on 1-min price changes; divergence = 5-bar
cum change of leg A − β × 5-bar cum change of leg B; fade leg A when |div|
exceeds a dollar-normalized threshold. BOTH directions simulated and gated
independently (MNQ/MES lesson: decisive directional asymmetry).

The MNQ_ES control pair must reproduce the template's numbers exactly —
that is the refactor-correctness test.

Usage:
  .venv/bin/python study_pair_divergence_survey.py            # all pairs
  .venv/bin/python study_pair_divergence_survey.py --pair SI_GC
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path("pair_survey_config.yaml")
GRID_CSV_OUT = Path("_bmad-output/pair_survey_gate0_grid.csv")

# Hard-cut dev window — module constants, not CLI-overridable (holdout discipline).
DEV_START = "2025-05-01"
DEV_END = "2026-02-28"
HOLDOUT_CUTOFF = date(2026, 3, 1)

DIRECTIONS = {-1: "SHORT_A", +1: "LONG_A"}


def load_et(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()


def load_leg(csvs):
    df = pd.concat([load_et(p) for p in csvs])
    return df[~df.index.duplicated(keep="first")].sort_index()


def prepare_pair(pair, g, session=None):
    """Join legs, slice dev window + session, compute beta and divergence."""
    rth_start, sess_close = session or (g["rth_start"], g["session_close"])
    a = load_leg(pair["leg_a"]["csvs"])
    b = load_leg(pair["leg_b"]["csvs"])
    both = (a[["close"]].rename(columns={"close": "a"})
            .join(b[["close"]].rename(columns={"close": "b"}), how="inner"))
    both = both[DEV_START:DEV_END]
    rth = both.between_time(rth_start, sess_close).copy()

    rth["a_chg"] = rth["a"].diff()
    rth["b_chg"] = rth["b"].diff()
    roll_cov = rth["a_chg"].rolling(g["beta_window"]).cov(rth["b_chg"])
    roll_var = rth["b_chg"].rolling(g["beta_window"]).var()
    # clip is a numerical-stability bound, sized to the pair's price-change
    # scale (template default [0,10] fits index/metals; BTC/ETH needs wider)
    clip_lo, clip_hi = pair.get("beta_clip", [0.0, 10.0])
    rth["beta"] = (roll_cov / roll_var.replace(0, np.nan)).ffill().clip(clip_lo, clip_hi)
    rth["div"] = (rth["a_chg"].rolling(g["spread_window"]).sum()
                  - rth["beta"] * rth["b_chg"].rolling(g["spread_window"]).sum())
    rth = rth.dropna(subset=["div", "beta"])

    # Holdout-discipline assertion: no bar may postdate the dev window.
    last = rth.index[-1].date()
    assert last < HOLDOUT_CUTOFF, f"{pair['name']}: dev data leaks past cutoff ({last})"
    return rth, sess_close


def run_simulation(rth, sess_close, pv, comm, thresh_pts, stop_mult, direction,
                   roll_dates, stop_cap, hold_max):
    """direction = -1: short leg A on div > +thresh; +1: long leg A on div < -thresh."""
    trades = []
    active = None
    hold_count = 0
    px = rth["a"].values
    dv = rth["div"].values
    ts_arr = rth.index

    for k in range(len(rth)):
        ts, p, d = ts_arr[k], px[k], dv[k]

        if active is not None:
            hold_count += 1
            if direction == -1:
                hit_tp, hit_stop = p <= active["tp"], p >= active["stop"]
            else:
                hit_tp, hit_stop = p >= active["tp"], p <= active["stop"]
            at_close = ts.strftime("%H:%M") >= sess_close

            if hit_tp:
                pnl = (active["tp"] - active["entry"]) * direction * pv - comm
                trades.append({**active, "exit": active["tp"], "pnl": pnl,
                               "win": True, "reason": "TP"})
                active = None; hold_count = 0
            elif hit_stop:
                pnl = (active["stop"] - active["entry"]) * direction * pv - comm
                trades.append({**active, "exit": active["stop"], "pnl": pnl,
                               "win": False, "reason": "STOP"})
                active = None; hold_count = 0
            elif at_close or hold_count >= hold_max:
                pnl = (p - active["entry"]) * direction * pv - comm
                trades.append({**active, "exit": p, "pnl": pnl, "win": pnl > 0,
                               "reason": "CLOSE" if at_close else "TIME"})
                active = None; hold_count = 0
            continue

        # entry: leg A overshot (short) or undershot (long) by > thresh
        if direction == -1 and d <= thresh_pts:
            continue
        if direction == +1 and d >= -thresh_pts:
            continue
        if ts.date() in roll_dates:
            continue  # stitched-contract roll day on either leg — fake divergence risk

        div_abs = abs(d)
        stop_usd = div_abs * stop_mult * pv
        if stop_usd > stop_cap:
            continue

        entry = p
        tp_price = entry + direction * div_abs
        sl_price = entry - direction * div_abs * stop_mult
        active = {"entry": entry, "div": div_abs, "tp": tp_price, "stop": sl_price,
                  "stop_usd": stop_usd, "date": ts.date(),
                  "month": ts.to_period("M"), "hour": ts.hour}
        hold_count = 0

    if active:
        pnl = (px[-1] - active["entry"]) * direction * pv - comm
        trades.append({**active, "exit": px[-1], "pnl": pnl,
                       "win": pnl > 0, "reason": "END"})
    return trades


def summarise(trades, n_days):
    if not trades:
        return dict(n=0, wr=0.0, freq=0.0, avg_pnl=0.0, pf=0.0, worst_mo=0.0,
                    stop_med=0.0, edge_density=0.0, total=0.0,
                    pnls=np.array([]), mo={}, exit_tp=0, exit_stop=0, exit_time=0)
    n = len(trades)
    wins = sum(t["win"] for t in trades)
    pnls = np.array([t["pnl"] for t in trades])
    stops = np.array([t["stop_usd"] for t in trades])
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    mo: dict = {}
    for t in trades:
        m = t["month"]
        mo.setdefault(m, [0, 0])
        mo[m][0 if t["win"] else 1] += 1
    worst_mo = min((w / (w + l) if w + l else 0) for w, l in mo.values()) if mo else 0.0
    freq = n / n_days
    avg = pnls.mean()
    return dict(n=n, wr=wins / n, freq=freq, avg_pnl=avg,
                pf=gross_w / max(1e-9, gross_l), worst_mo=worst_mo,
                stop_med=float(np.median(stops)), edge_density=avg * freq,
                total=float(pnls.sum()), pnls=pnls, mo=mo,
                exit_tp=sum(1 for t in trades if t["reason"] == "TP"),
                exit_stop=sum(1 for t in trades if t["reason"] == "STOP"),
                exit_time=sum(1 for t in trades if t["reason"] in ("TIME", "CLOSE", "END")))


def month_table(trades, summary, rth, worst_mo_min):
    mo_pnl: dict = {}
    for t in trades:
        mo_pnl.setdefault(t["month"], []).append(t["pnl"])
    print(f"    {'Month':<9}  {'N':>4}  {'WR':>7}  {'AvgP&L':>9}  {'TotalP&L':>10}")
    for m in sorted(summary["mo"]):
        w, l = summary["mo"][m]
        n_mo = w + l
        mwr = w / n_mo if n_mo else 0
        flag = "⚠️ N<5" if n_mo < 5 else ("❌" if mwr < worst_mo_min else "✅")
        print(f"    {str(m):<9}  {n_mo:>4}  {mwr:>7.1%}  ${np.mean(mo_pnl[m]):>7.2f}  "
              f"${np.sum(mo_pnl[m]):>8.0f}  {flag}")


def gate0_verdict(s, be_wr, be_wr_stress, g0, robust_pos, informational):
    checks = [
        (s["wr"] >= be_wr, f"WR ≥ {be_wr:.1%} (per-pair breakeven)", f"{s['wr']:.1%}"),
        (s["avg_pnl"] > g0["ev_min"], "Avg net P&L > $0/trade", f"${s['avg_pnl']:.2f}"),
        (s["freq"] >= g0["freq_min"], f"Frequency ≥ {g0['freq_min']}/day", f"{s['freq']:.2f}/d"),
        (s["stop_med"] <= 150.0, "Median stop ≤ $150/contract", f"${s['stop_med']:.0f}"),
        (s["worst_mo"] >= g0["worst_month_wr_min"],
         f"Worst-month WR ≥ {g0['worst_month_wr_min']:.0%}", f"{s['worst_mo']:.1%}"),
    ]
    for ok, label, meas in checks:
        print(f"    {'✅' if ok else '❌'}  {label:<44} [measured: {meas}]")
    g0_pass = all(ok for ok, _, _ in checks)
    q_checks = [
        (g0_pass, "Gate 0 primary spec (all five)"),
        (robust_pos >= 4, f"Robustness: ≥4/8 grid cells positive EV (got {robust_pos})"),
        (s["wr"] >= be_wr_stress, f"WR clears slippage-stressed BE {be_wr_stress:.1%}"),
        (not informational, "Not informational-only"),
    ]
    qualified = all(ok for ok, _ in q_checks)
    print(f"    {'─' * 60}")
    for ok, label in q_checks:
        print(f"    {'✅' if ok else '❌'}  {label}")
    print(f"    → {'🟢 QUALIFIES for Gate 1 prereg consideration' if qualified else '🔴 does not qualify'}")
    return g0_pass, qualified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", help="run a single pair by name (e.g. SI_GC)")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    g = cfg["global"]
    assert g["dev_start"] == DEV_START and g["dev_end"] == DEV_END, \
        "YAML dev window differs from hard-coded constants"
    g0 = g["gate0"]
    primary_usd = g["primary_thresh_usd"]
    primary_sm = g["primary_stop_mult"]

    pairs = sorted(cfg["pairs"], key=lambda p: p["order"])
    if args.pair:
        pairs = [p for p in pairs if p["name"] == args.pair]
        if not pairs:
            sys.exit(f"Unknown pair {args.pair}")

    grid_rows = []
    final_rows = []

    for pair in pairs:
        name, pv, comm = pair["name"], pair["point_value"], pair["commission_rt"]
        slip = pair["slippage_stress_rt"]
        informational = pair.get("informational_only", False)
        roll_dates = {date.fromisoformat(d) for d in pair["roll_dates"]}
        missing = [p for p in pair["leg_a"]["csvs"] + pair["leg_b"]["csvs"]
                   if not Path(p).exists()]
        if missing:
            print(f"\n{'=' * 88}\n{name}: SKIPPED — missing data: {missing}")
            continue

        print(f"\n{'=' * 88}")
        print(f"PAIR {pair['order']}: {name} — {pair['description']}")
        print(f"  traded leg econ: {pair['traded_micro']}  PV=${pv:g}/pt  "
              f"comm=${comm}/RT  slip-stress=${slip}/RT")
        rth, sess_close = prepare_pair(pair, g)
        n_days = rth.index.normalize().nunique()
        print(f"  Dev window: {rth.index[0].date()} → {rth.index[-1].date()}  "
              f"({len(rth):,} RTH bars, {n_days} days)  [cutoff check OK]")

        thresh_pts = {usd: usd / pv for usd in g["thresh_usd_grid"]}
        results = {}
        print(f"\n  {'Dir':>8} {'Thr$':>5} {'Stop×':>6} {'BE WR':>7} {'N':>6} "
              f"{'Freq/d':>7} {'WR':>7} {'PF':>6} {'AvgP&L':>9} {'WorstMo':>8} {'$/day':>8}")
        for direction in (-1, +1):
            for usd in g["thresh_usd_grid"]:
                for sm in g["stop_mults"]:
                    t = run_simulation(rth, sess_close, pv, comm, thresh_pts[usd], sm,
                                       direction, roll_dates, g["stop_cap_usd"],
                                       g["hold_max"])
                    s = summarise(t, n_days)
                    results[(direction, usd, sm)] = (t, s)
                    stop_usd = usd * sm
                    be = (stop_usd + comm) / (2 * stop_usd)
                    prim = " ◀ PRIMARY" if usd == primary_usd and sm == primary_sm else ""
                    print(f"  {DIRECTIONS[direction]:>8} {usd:>5} {sm:>5.1f}× {be:>7.1%} "
                          f"{s['n']:>6} {s['freq']:>6.2f}/d {s['wr']:>7.1%} {s['pf']:>6.2f} "
                          f"${s['avg_pnl']:>7.2f} {s['worst_mo']:>8.1%} "
                          f"${s['edge_density']:>6.2f}{prim}")
                    grid_rows.append(dict(pair=name, direction=DIRECTIONS[direction],
                                          thresh_usd=usd, stop_mult=sm, n=s["n"],
                                          freq=s["freq"], wr=s["wr"], pf=s["pf"],
                                          avg_pnl=s["avg_pnl"], worst_mo=s["worst_mo"],
                                          stop_med=s["stop_med"],
                                          edge_density=s["edge_density"], total=s["total"]))

        be_wr = (primary_usd + comm) / (2 * primary_usd)
        be_wr_stress = (primary_usd + comm + slip) / (2 * primary_usd)

        for direction in (-1, +1):
            t, s = results[(direction, primary_usd, primary_sm)]
            robust_pos = sum(1 for usd in g["thresh_usd_grid"] for sm in g["stop_mults"]
                             if results[(direction, usd, sm)][1]["avg_pnl"] > 0)
            print(f"\n  ── {name} {DIRECTIONS[direction]} — primary spec "
                  f"(${primary_usd} thr, {primary_sm}× stop) "
                  f"N={s['n']} WR={s['wr']:.1%} PF={s['pf']:.2f} "
                  f"avg=${s['avg_pnl']:.2f} total=${s['total']:.0f}")
            if s["n"]:
                print(f"    Exits: TP={s['exit_tp']} STOP={s['exit_stop']} "
                      f"TIME/CLOSE={s['exit_time']}")
                month_table(t, s, rth, g0["worst_month_wr_min"])
            g0_pass, qualified = gate0_verdict(s, be_wr, be_wr_stress, g0,
                                               robust_pos, informational)
            final_rows.append(dict(pair=name, direction=DIRECTIONS[direction],
                                   informational=informational, n=s["n"], wr=s["wr"],
                                   pf=s["pf"], freq=s["freq"], avg_pnl=s["avg_pnl"],
                                   worst_mo=s["worst_mo"], edge_density=s["edge_density"],
                                   total=s["total"], be_wr=be_wr,
                                   be_wr_stress=be_wr_stress, robust_pos=robust_pos,
                                   gate0_pass=g0_pass, qualified=qualified))

        # Labeled-exploratory appendix: metals pit-aligned session, primary spec only.
        if pair.get("metals") or name.endswith("_GC"):
            alt = tuple(g["metals_alt_session"])
            rth_alt, close_alt = prepare_pair(pair, g, session=alt)
            nd_alt = rth_alt.index.normalize().nunique()
            print(f"\n  [EXPLORATORY APPENDIX — pit session {alt[0]}–{alt[1]} ET; "
                  f"cannot promote the pair]")
            for direction in (-1, +1):
                t = run_simulation(rth_alt, close_alt, pv, comm,
                                   thresh_pts[primary_usd], primary_sm, direction,
                                   roll_dates, g["stop_cap_usd"], g["hold_max"])
                s = summarise(t, nd_alt)
                print(f"    {DIRECTIONS[direction]:>8}: N={s['n']} freq={s['freq']:.2f}/d "
                      f"WR={s['wr']:.1%} PF={s['pf']:.2f} avg=${s['avg_pnl']:.2f}")

    # ── cross-pair summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 88}")
    print("CROSS-PAIR SUMMARY — primary spec, ranked by net edge density ($/day/contract)")
    print(f"{'=' * 88}")
    print(f"  {'Pair':<8} {'Dir':>8} {'N':>5} {'WR':>7} {'BE':>6} {'BEstr':>6} "
          f"{'PF':>6} {'Avg$':>8} {'$/day':>8} {'WorstMo':>8}  Verdict")
    for r in sorted(final_rows, key=lambda r: -r["edge_density"]):
        verdict = ("🟢 QUALIFIED" if r["qualified"]
                   else ("🟡 G0 pass (info-only)" if r["gate0_pass"] and r["informational"]
                         else ("🟡 G0 pass, robustness/stress fail" if r["gate0_pass"]
                               else "🔴 FAIL")))
        print(f"  {r['pair']:<8} {r['direction']:>8} {r['n']:>5} {r['wr']:>7.1%} "
              f"{r['be_wr']:>6.1%} {r['be_wr_stress']:>6.1%} {r['pf']:>6.2f} "
              f"${r['avg_pnl']:>6.2f} ${r['edge_density']:>6.2f} "
              f"{r['worst_mo']:>8.1%}  {verdict}")

    if grid_rows:
        GRID_CSV_OUT.parent.mkdir(exist_ok=True)
        pd.DataFrame(grid_rows).to_csv(GRID_CSV_OUT, index=False)
        print(f"\nGrid results → {GRID_CSV_OUT}")
    print(f"\nDev window used: {DEV_START} → {DEV_END} (hard-coded; holdout untouched)")


if __name__ == "__main__":
    main()
