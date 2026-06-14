#!/usr/bin/env python3
"""
rerun_sigc_combined_mc.py
=========================
Scheduled follow-up to the 2026-06-14 SI-GC second-edge scoping (see memory
project_pair_survey_20260612.md / project_mim_nb_expectations_reconciled.md).

Fired by sigc-combined-rerun.timer daily Jun 19-23 (after the 20:20 UTC SIL
analysis). Idempotent via logs/sigc_combined_mc.done — acts at most once.

Logic:
  - No definitive SIL verdict in logs/sil_quote_analysis.log yet  -> exit quietly (retry next day)
  - Verdict FAIL                                                  -> record "family closed final", done
  - Verdict PASS                                                  -> extract measured median SIL spread,
        re-run the combined-equity combine MC (MIM-NB + SI-GC) with the MEASURED cost plugged in,
        compare vs MIM-alone against breakeven ~$7.3/RT (additive) and ~$5/RT (clearly worthwhile),
        write a results report, and if it clears, DRAFT a Gate-1 prereg (does NOT consume the
        sealed SI/GC 2026-03+ holdout; does NOT touch any live config).
"""
import os, re, sys, importlib.util
from datetime import date, datetime, timezone
import numpy as np, pandas as pd, yaml

ROOT = "/root/Silver-Bullet-ML-BMAD"
VERDICT_LOG = f"{ROOT}/logs/sil_quote_analysis.log"
DONE = f"{ROOT}/logs/sigc_combined_mc.done"
REPORT = f"{ROOT}/logs/sigc_combined_mc_result.log"
PREREG = f"{ROOT}/_bmad-output/preregistration_sigc_combined_edge_gate1.md"
OV0, OV1 = date(2025, 5, 1), date(2026, 2, 28)        # in-sample overlap; holdout (2026-03+) stays sealed
BREAKEVEN, WORTHWHILE = 7.3, 5.0

def log(msg):
    line = f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {msg}"
    print(line, flush=True)
    with open(REPORT, "a") as f:
        f.write(line + "\n")

def latest_verdict():
    """Return ('PASS'|'FAIL'|None, median_spread_usd_or_None) from the last run block."""
    if not os.path.exists(VERDICT_LOG):
        return None, None
    txt = open(VERDICT_LOG, encoding="utf-8", errors="replace").read()
    blocks = re.split(r"=== run .*? ===", txt)
    if not blocks:
        return None, None
    last = blocks[-1]
    med = None
    m = re.search(r"median spread:\s*\$([0-9.]+)", last)
    if m:
        med = float(m.group(1))
    if "PASS" in last and "median spread" in last:
        return "PASS", med
    if "FAIL" in last:
        return "FAIL", med
    return None, med   # INSUFFICIENT DATA / error / not run yet

# ---------------- combined-MC (mirrors the 2026-06-14 scoping test) ----------------
def run_combined_mc(measured_spread):
    spec = importlib.util.spec_from_file_location("pds", f"{ROOT}/study_pair_divergence_survey.py")
    pds = importlib.util.module_from_spec(spec); spec.loader.exec_module(pds)
    g = yaml.safe_load(open(f"{ROOT}/pair_survey_5m_config.yaml")); glob = g["global"]
    sigc = next(p for p in g["pairs"] if p["name"] == "SI_GC")
    for leg in ("leg_a", "leg_b"):
        sigc[leg]["csvs"] = [f"{ROOT}/{c}" for c in sigc[leg]["csvs"]]
    pv, comm = sigc["point_value"], sigc["commission_rt"]
    roll = set(pd.to_datetime(d).date() for d in sigc["roll_dates"])
    rth, sc = pds.prepare_pair(sigc, glob)
    tdf = pd.DataFrame(pds.run_simulation(rth, sc, pv, comm, glob["primary_thresh_usd"]/pv,
                       glob["primary_stop_mult"], +1, roll, glob["stop_cap_usd"], glob["hold_max"]))
    mim = pd.read_csv(f"{ROOT}/data/reports/mim_nb_catstop_s500_pooled.csv")
    mim["net_usd"] = (mim["pnl_pts"] - 1.12) * 2.0
    mim["day"] = pd.to_datetime(mim["day"]).dt.date
    mim["exit_t"] = mim["exit_t"].astype(str)
    mim_ov = mim[(mim["day"] >= OV0) & (mim["day"] <= OV1)].sort_values("exit_t")
    mim_only = [list(gg["net_usd"].values) for _, gg in mim_ov.groupby("day")]

    def mc(byday, n=5000, md=90, seed=42):
        rng = np.random.default_rng(seed); nd = len(byday); pn = bn = 0
        for _ in range(n):
            bal, fl, bd, oc = 50000., 48000., 0., None
            for di in rng.integers(0, nd, size=md):
                dp = 0.
                for p1 in byday[di]:
                    bal += p1; dp += p1
                    if bal <= fl: oc = "blow"; break
                    if dp <= -1000.: break
                if oc: break
                bd = max(bd, dp); pr = bal - 50000.
                if pr >= 3000. and bd < 0.5*pr: oc = "pass"; break
                fl = min(50000., max(fl, bal - 2000.))
            if oc == "pass": pn += 1
            elif oc == "blow": bn += 1
        return pn/n, bn/n

    def combined(slip):
        sg = tdf.assign(net=tdf["pnl"]-slip).groupby("date")["net"].sum()
        sg = sg[(sg.index >= OV0) & (sg.index <= OV1)]
        days = {}
        for d, gg in mim_ov.groupby("day"): days.setdefault(d, []).extend(list(gg["net_usd"].values))
        for d, v in sg.items(): days.setdefault(d, []).append(float(v))
        return [v for _, v in sorted(days.items())], sg.mean()

    p0, b0 = mc(mim_only)
    # full spread (conservative) and half spread (optimistic) as the realized-cost bracket
    rows = []
    for label, slip in [("full-spread", measured_spread), ("half-spread", measured_spread/2)]:
        byday, sg_ev = combined(slip)
        p, b = mc(byday)
        rows.append((label, slip, sg_ev, p, b, p - p0))
    return p0, b0, rows, len(tdf), float(tdf["win"].mean())

def draft_prereg(measured_spread, p0, rows):
    best = max(r[5] for r in rows)
    ts = datetime.now(timezone.utc)
    body = f"""# Pre-Registration (DRAFT): SI-GC Combined-Edge Gate 1 (holdout OOS)

**Generated:** {ts:%Y-%m-%d} (auto-drafted by rerun_sigc_combined_mc.py)
**Status:** DRAFT — requires Alex review before sealing. No holdout consumed, no live config changed.
**Trigger:** SIL slippage verdict = PASS; measured median SILN26 RTH spread = ${measured_spread:.2f}/RT.

## Rationale
The 2026-06-14 in-sample scoping showed SI-GC LONG 5m is uncorrelated with MIM-NB
(daily-P&L corr -0.06) and, in the combined-equity 50K combine MC, lifts pass% from
{p0*100:.1f}% (MIM-alone) materially once transaction cost is ≲ $7.3/RT. With measured
spread ${measured_spread:.2f}/RT, the re-run shows combined Δpass up to {best*100:+.1f}pp
(see logs/sigc_combined_mc_result.log).

## Pre-registered one-shot OOS test (frozen BEFORE touching the holdout)
- Strategy: SI-GC LONG 5m primary (thresh $80, stop 1.0×, hold 30 bars), traded via SIL,
  cost basis = measured ${measured_spread:.2f}/RT spread (full-spread, conservative).
- Data: sealed SI/GC holdout `data/sealed_holdout/{{si,gc}}_1min_holdout_20260301_plus.csv` (2026-03-01+),
  consumed ONCE. MIM-NB daily P&L over the same dates from the live ledger / sealed engine.
- Gates (all must pass):
  1. SI-GC standalone net PF ≥ 1.05 and net expectancy > 0 at the measured cost.
  2. corr(SI-GC, MIM-NB daily P&L) ≤ 0 on the holdout.
  3. Combined-equity combine MC pass% > MIM-alone pass% (same engine, holdout day-bootstrap).
- Decision: all three PASS → authorize a SEPARATE live-deployment pre-registration for the
  combined sleeve. Any FAIL → SI-GC family closure is final.

## Guardrails
- Holdout is consumed exactly once; no parameter sweeps; no re-runs on FAIL.
- This draft changes nothing live. MIM-NB continues unchanged at 1 MNQ.
"""
    os.makedirs(os.path.dirname(PREREG), exist_ok=True)
    with open(PREREG, "w") as f:
        f.write(body)
    log(f"DRAFTED Gate-1 prereg -> {PREREG} (review required; holdout NOT consumed)")

def main():
    if os.path.exists(DONE):
        log("already acted (sentinel present) — no-op."); return
    verdict, med = latest_verdict()
    if verdict is None:
        log("no definitive SIL verdict yet — will retry next scheduled run."); return
    if verdict == "FAIL":
        log("SIL verdict = FAIL — SI-GC divergence family closure is CONFIRMED FINAL. No action.")
        open(DONE, "w").write(f"FAIL {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}\n"); return
    # PASS
    if med is None:
        log("verdict=PASS but could not parse 'median spread' — aborting, retry next run."); return
    log(f"SIL verdict = PASS; measured median SIL spread = ${med:.2f}/RT. Re-running combined MC...")
    try:
        p0, b0, rows, n_sig, wr = run_combined_mc(med)
    except Exception as e:
        log(f"ERROR during combined MC: {e!r} — will retry next run."); return
    log(f"SI-GC engine reproduced: N={n_sig} trades, WR={wr*100:.1f}%")
    log(f"MIM-alone combine MC: pass={p0*100:.1f}% blow={b0*100:.1f}%")
    for label, slip, ev, p, b, d in rows:
        verdict_word = "ADDS VALUE" if d > 0 else "no help"
        log(f"  {label:11} (cost ${slip:.2f}/RT): SIG EV=${ev:+.1f}/day | "
            f"combined pass={p*100:.1f}% blow={b*100:.1f}% | Δpass={d*100:+.1f}pp [{verdict_word}]")
    best = max(r[5] for r in rows)
    if best > 0 and med <= BREAKEVEN:
        tier = "CLEARLY WORTHWHILE" if med <= WORTHWHILE else "ADDITIVE"
        log(f"VERDICT: measured ${med:.2f}/RT ≤ breakeven ${BREAKEVEN} → SI-GC {tier} as 2nd edge.")
        draft_prereg(med, p0, rows)
    else:
        log(f"VERDICT: measured ${med:.2f}/RT > breakeven ${BREAKEVEN} (or no Δpass>0) → "
            f"SI-GC remains a net drag on the combine; second-edge thread stays closed.")
    open(DONE, "w").write(f"PASS med=${med:.2f} {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}\n")
    log("Done. Sentinel written; timer will no-op on remaining fires.")

if __name__ == "__main__":
    main()
