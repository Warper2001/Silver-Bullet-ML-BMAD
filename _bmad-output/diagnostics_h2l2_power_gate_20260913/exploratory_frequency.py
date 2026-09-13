"""EXPLORATORY, not pre-registered. Written after power_gate.py ran.

Question: is the UNDERPOWERED verdict an artefact of the plan's literal 4-tick
(1.00 pt) EMA-proximity translation, or of the filters generally? Re-counts
events with filters relaxed and re-sizes power under the same firewall (placebo
pairing only; no statistic of price after a real fill). Effect sizes, costs and
the power formula are the plan's. It cannot change the pre-registered verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import power_gate as pg  # noqa: E402


def detect(b: pd.DataFrame, prox: float | None, side: bool, colour: bool, doji: bool, tick: float) -> pd.DataFrame:
    ev = []
    sidx = {s: i for i, s in enumerate(sorted(b["session"].unique()))}
    for s, g in b.groupby("session", sort=True):
        o, h, l, c, e, sl = (g[k].to_numpy() for k in ("open", "high", "low", "close", "ema", "slot"))
        for d in (1, -1):
            ext = h if d == 1 else -l
            leg, count, armed = ext[0], 0, False
            for t in range(1, len(g)):
                if ext[t] > leg:
                    leg, count, armed = ext[t], 0, False
                    continue
                if ext[t] > ext[t - 1] and armed:
                    count, armed = count + 1, False
                    if count == 2 and sl[t] <= pg.LAST_TRIGGER_SLOT:
                        q = t - 1
                        rng, body = h[q] - l[q], abs(c[q] - o[q])
                        ok = rng > 0
                        if prox is not None:
                            ok &= (l[q] <= e[q] + prox) if d == 1 else (h[q] >= e[q] - prox)
                        if side:
                            ok &= (c[q] > e[q]) if d == 1 else (c[q] < e[q])
                        if colour:
                            ok &= (c[q] > o[q]) if d == 1 else (c[q] < o[q])
                        if doji:
                            ok &= body > pg.DOJI_BODY * rng
                        if ok:
                            ev.append({"session": sidx[s], "slot": int(sl[t]), "dir": d, "r_pts": rng + 2 * tick})
                elif ext[t] < ext[t - 1]:
                    armed = True
    return pd.DataFrame(ev).sort_values(["session", "slot"]).reset_index(drop=True)


def size(ev: pd.DataFrame, b: pd.DataFrame, pv: float, cost: float) -> dict:
    nd = b["session"].nunique()
    H, L, C = pg.grids(b, nd)
    old = pg.PV
    pg.PV = pv
    try:
        st = pg.shift_stats(ev, H, L, C, 1.0)
    finally:
        pg.PV = old
    r_usd = pv * ev["r_pts"].to_numpy()
    rbar, sig = float(r_usd.mean()), st["sigma_cluster_usd"]
    out = {"N": len(ev), "per_session": len(ev) / nd, "R_usd_median": float(np.median(r_usd)),
           "sigma_cluster_usd_1R": sig}
    for th_n, th in pg.THETAS.items():
        mu = th * rbar - cost
        out[f"power_{th_n}"] = pg.power(mu, sig, len(ev))
        out[f"years80_{th_n}"] = (((pg.Z_A + pg.Z_B) * sig / mu) ** 2 / (len(ev) / nd) / 252) if mu > 0 else None
    return out


def load_es() -> pd.DataFrame:
    old = pg.INPUTS
    pg.INPUTS = [pg.REPO / "data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv"]
    try:
        return pg.load_5min()
    finally:
        pg.INPUTS = old


def main() -> None:
    b = pg.load_5min()
    variants = {
        "A_plan_primary (prox 1.00pt)": dict(prox=1.00, side=True, colour=True, doji=True),
        "B_prox_price_scaled (4.00pt ~ 1 ES pt x NQ/ES ratio ~3.7-4)": dict(prox=4.00, side=True, colour=True, doji=True),
        "C_no_proximity": dict(prox=None, side=True, colour=True, doji=True),
        "D_raw_H2L2_no_filters": dict(prox=None, side=False, colour=False, doji=False),
    }
    res = {"MNQ_5min": {k: size(detect(b, tick=pg.TICK, **v), b, pg.PV, pg.COSTS["primary"])
                        for k, v in variants.items()}}
    es = load_es()
    # Wade's own instrument, literal 4-tick proximity; sized as MES ($5/pt), same $5.80 cost
    res["ES_5min_as_MES"] = {"A_literal_4tick": size(detect(es, tick=0.25, **variants["A_plan_primary (prox 1.00pt)"]),
                                                     es, 5.0, pg.COSTS["primary"]),
                             "D_raw_H2L2_no_filters": size(detect(es, tick=0.25, **variants["D_raw_H2L2_no_filters"]),
                                                           es, 5.0, pg.COSTS["primary"])}
    (HERE / "exploratory_frequency.json").write_text(json.dumps(res, indent=2, default=float))
    for inst, rows in res.items():
        for k, v in rows.items():
            print(inst, "|", k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
