"""Re-run of the H2/L2 gate's EXPLORATORY variants on corrected bars (2026-09-16).

The originals (diagnostics_h2l2_power_gate_20260913/exploratory_frequency.py) ran on the
contaminated CSVs and were explicitly left out of the C1 correction (correction_plan.md
section 4). They are re-run here on the C1b front-month rebuild.

Still exploratory: not pre-registered, and it cannot change any verdict. Constructs,
filters, geometry, costs, effect sizes, placebo and power formula are the committed
gates'; only the bars differ.

FIREWALL unchanged: dispersion comes only from placebo pairings shifted 5..ND-5 sessions;
no statistic of price after a real event; nothing under data/sealed_holdout/.

MNQ: C1b bars (front-month rebuild, 27 interleaved sessions dropped, EMA restarted per
contract segment), identical to the corrected gate's input.

ES: there is NO raw ES source in this repo, so the front-month rebuild is impossible.
The ES rows are therefore only APPROXIMATELY cleaned:
  - 2026 is dropped entirely (the back-month defect cannot be checked without raw data)
  - the same 27 roll-week sessions are dropped (ES and MNQ roll on the same schedule)
They are reported as indicative, not corrected.

Run:  .venv/bin/python _bmad-output/diagnostics_h2l2_exploratory_clean_20260916/exploratory_clean.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
H2 = REPO / "_bmad-output/diagnostics_h2l2_power_gate_20260913"
CORR = REPO / "_bmad-output/diagnostics_h2l2_contamination_20260914"

sys.path.insert(0, str(CORR))
sys.path.insert(0, str(H2))
import corrected_rerun as cr  # noqa: E402  (module import verifies CSV hashes, parses the raw feed)
import exploratory_frequency as ex  # noqa: E402  (the original variants' detect() and size())

pg = ex.pg
VARIANTS = {
    "A_plan_primary (prox 1.00pt)": dict(prox=1.00, side=True, colour=True, doji=True),
    "B_prox_price_scaled (4.00pt)": dict(prox=4.00, side=True, colour=True, doji=True),
    "C_no_proximity": dict(prox=None, side=True, colour=True, doji=True),
    "D_raw_H2L2_no_filters": dict(prox=None, side=False, colour=False, doji=False),
}
CORRECTED_A_N = 47          # C1b h2l2 results.json, the guard below


def load_es_approx() -> pd.DataFrame:
    """ES 5-min bars, 2025 only, roll-week sessions dropped. Approximate: no raw ES exists."""
    old_in, old_cut = pg.INPUTS, pg.CUTOFF
    pg.INPUTS = [REPO / "data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv"]
    pg.CUTOFF = pd.Timestamp("2026-01-01", tz=cr.NY)
    try:
        b = pg.load_5min()
    finally:
        pg.INPUTS, pg.CUTOFF = old_in, old_cut
    keep = ~pd.Index(b["session"]).isin(list(cr.INTERLEAVED))
    return b[keep].copy()


def es_range_diagnostic(clean: pd.DataFrame) -> dict:
    """Data-only: does ES show the same roll-week fake-bar signature as MNQ?"""
    old_in, old_cut = pg.INPUTS, pg.CUTOFF
    pg.INPUTS = [REPO / "data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv"]
    pg.CUTOFF = pd.Timestamp("2026-01-01", tz=cr.NY)
    try:
        allb = pg.load_5min()
    finally:
        pg.INPUTS, pg.CUTOFF = old_in, old_cut
    rng = (allb["high"] - allb["low"]).groupby(allb["session"]).max()
    roll = pd.Index(rng.index).isin(list(cr.INTERLEAVED))
    return {"sessions_all": int(len(rng)), "sessions_roll_week": int(roll.sum()),
            "max_5min_range_pts_median_rollweek": float(rng[roll].median()) if roll.any() else None,
            "max_5min_range_pts_median_clean": float(rng[~roll].median()),
            "max_5min_range_pts_p100_clean": float(rng[~roll].max())}


def main() -> None:
    mnq = cr.load_c1b()
    res = {"mnq_bars": {"sessions": int(mnq["session"].nunique()), "bars": int(len(mnq)),
                        "first": str(mnq.index.min()), "last": str(mnq.index.max())},
           "raw_sha256": cr.atr.sha(cr.atr.RAW), "script_sha256": cr.atr.sha(Path(__file__)),
           "MNQ_5min_C1b": {}, "ES_5min_as_MES_approx": {}}

    for k, v in VARIANTS.items():
        ev = ex.detect(mnq, tick=pg.TICK, **v)
        if k.startswith("A_") and len(ev) != CORRECTED_A_N:
            sys.exit(f"GUARD FAILED: variant A gives N={len(ev)}, corrected gate has {CORRECTED_A_N}")
        res["MNQ_5min_C1b"][k] = ex.size(ev, mnq, pg.PV, pg.COSTS["primary"])
    print("guard: variant A reproduces the corrected gate's N =", CORRECTED_A_N, flush=True)

    es = load_es_approx()
    res["es_bars"] = {"sessions": int(es["session"].nunique()), "bars": int(len(es)),
                      "first": str(es.index.min()), "last": str(es.index.max())}
    res["es_range_diagnostic"] = es_range_diagnostic(es)
    for k in ("A_plan_primary (prox 1.00pt)", "D_raw_H2L2_no_filters"):
        res["ES_5min_as_MES_approx"][k] = ex.size(ex.detect(es, tick=0.25, **VARIANTS[k]), es, 5.0,
                                                  pg.COSTS["primary"])

    (HERE / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("mnq_bars", "es_bars", "es_range_diagnostic")}, indent=1, default=float))
    for inst in ("MNQ_5min_C1b", "ES_5min_as_MES_approx"):
        for k, v in res[inst].items():
            print(inst, "|", k, {kk: (round(x, 3) if isinstance(x, float) else x) for kk, x in v.items()})


if __name__ == "__main__":
    main()
