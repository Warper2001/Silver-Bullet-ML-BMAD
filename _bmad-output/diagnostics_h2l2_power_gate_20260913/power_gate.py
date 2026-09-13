"""H2/L2 second-entry power gate (MNQ 5-min RTH).

Plan: analysis_plan.md (SHA-256 pinned below; the script refuses to run on a mismatch).

FIREWALL (plan section 3). Real H2/L2 events are used only for their count, timing,
direction and risk. Every price-path statistic is computed under a mismatched
pairing: the event's geometry is laid on a *different* session's path, shifted by
k whole sessions with 5 <= k <= ND-5. The identity pairing is refused. Nothing
under data/sealed_holdout/ is opened, and no bar on/after 2026-03-01 survives.

Run:  .venv/bin/python _bmad-output/diagnostics_h2l2_power_gate_20260913/power_gate.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PLAN = HERE / "analysis_plan.md"
PLAN_SHA = "1e08bb4e27999907542bf03cd23d563d5eb7c66ca0498e590978cd2cbc692076"
INPUTS = [REPO / "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv",
          REPO / "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"]
START, CUTOFF = "2025-01-01", pd.Timestamp("2026-03-01", tz="America/New_York")

# --- frozen by the plan ------------------------------------------------------
TICK, PV = 0.25, 2.0
EMA_SPAN, PROX, DOJI_BODY = 21, 1.00, 0.30
SLOTS = 78                      # 5-min bars closing 09:35 .. 16:00
LAST_TRIGGER_SLOT = 75          # bar closing 15:50
FLATTEN_SLOT = 76               # bar closing 15:55
HOLD = 24
COSTS = {"primary": 5.80, "double": 11.60}
THETAS = {"pessimistic": 0.05, "central": 0.10, "optimistic": 0.20}
TARGETS = {"1R": 1.0, "2R": 2.0}
MIN_SHIFT = 5
HOLD_SESSIONS = 55              # 2026-03-01 .. 2026-05-19, per ACCESS_LOG; file not opened
Z_A, Z_B = stats.norm.ppf(0.95), stats.norm.ppf(0.80)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_5min() -> pd.DataFrame:
    frames = []
    for p in INPUTS:
        d = pd.read_csv(p)
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, format="ISO8601").dt.tz_convert("America/New_York")
        frames.append(d)
    m = pd.concat(frames).drop_duplicates("timestamp").set_index("timestamp").sort_index()
    m = m[(m.index >= pd.Timestamp(START, tz="America/New_York")) & (m.index < CUTOFF)]
    t = m.index.hour * 60 + m.index.minute
    m = m[(t > 9 * 60 + 30) & (t <= 16 * 60)]            # close-stamped: (09:30, 16:00]
    b = m.resample("5min", closed="right", label="right").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")).dropna()
    tb = b.index.hour * 60 + b.index.minute
    b = b[(tb >= 9 * 60 + 35) & (tb <= 16 * 60)].copy()
    assert b.index.max() < CUTOFF, "post-cutoff bar survived"
    b["ema"] = b["close"].ewm(span=EMA_SPAN, adjust=False).mean()
    b["session"] = b.index.date
    b["slot"] = ((b.index.hour * 60 + b.index.minute) - (9 * 60 + 35)) // 5
    return b


def detect(b: pd.DataFrame) -> pd.DataFrame:
    """H2/L2 events. Uses bars up to and including the trigger bar's high/low test only."""
    ev = []
    sessions = sorted(b["session"].unique())
    sidx = {s: i for i, s in enumerate(sessions)}
    for s, g in b.groupby("session", sort=True):
        o, h, l, c, e, sl = (g[k].to_numpy() for k in ("open", "high", "low", "close", "ema", "slot"))
        for d in (1, -1):
            ext = h if d == 1 else -l                      # mirror: short side on negated lows
            leg, count, armed = ext[0], 0, False
            for t in range(1, len(g)):
                if ext[t] > leg:
                    leg, count, armed = ext[t], 0, False
                    continue
                if ext[t] > ext[t - 1] and armed:
                    count, armed = count + 1, False
                    if count == 2 and sl[t] <= LAST_TRIGGER_SLOT:
                        q = t - 1
                        rng = h[q] - l[q]
                        body = abs(c[q] - o[q])
                        if d == 1:
                            ok = (l[q] <= e[q] + PROX) and (c[q] > e[q]) and (c[q] > o[q])
                        else:
                            ok = (h[q] >= e[q] - PROX) and (c[q] < e[q]) and (c[q] < o[q])
                        ok = ok and rng > 0 and body > DOJI_BODY * rng
                        if ok:
                            ev.append({"session": sidx[s], "slot": int(sl[t]), "dir": d, "r_pts": rng + 2 * TICK})
                elif ext[t] < ext[t - 1]:
                    armed = True
    return pd.DataFrame(ev).sort_values(["session", "slot"]).reset_index(drop=True)


def thin(ev: pd.DataFrame) -> pd.DataFrame:
    keep, last = [], {}
    for i, r in ev.iterrows():
        if r.session not in last or r.slot - last[r.session] >= HOLD:
            keep.append(i)
            last[r.session] = r.slot
    return ev.loc[keep]


def grids(b: pd.DataFrame, nd: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, L, C = (np.full((nd, SLOTS), np.nan) for _ in range(3))
    sess = {s: i for i, s in enumerate(sorted(b["session"].unique()))}
    si = b["session"].map(sess).to_numpy()
    sl = b["slot"].to_numpy()
    H[si, sl], L[si, sl], C[si, sl] = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
    return H, L, C


def placebo(ev: pd.DataFrame, H, L, C, k: int, target: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gross $ P&L of each event's geometry laid on session (j+k) mod ND. Identity refused."""
    nd = H.shape[0]
    if k % nd == 0:
        raise AssertionError("FIREWALL VIOLATION: identity pairing requested (plan section 3).")
    j = (ev["session"].to_numpy() + k) % nd
    slot = ev["slot"].to_numpy()
    d = ev["dir"].to_numpy().astype(float)
    R = ev["r_pts"].to_numpy()
    entry = C[j, slot]
    ok = ~np.isnan(entry)
    offs = np.arange(1, HOLD + 1)
    ps = slot[:, None] + offs[None, :]
    valid = ps <= FLATTEN_SLOT
    psc = np.minimum(ps, SLOTS - 1)
    ph, pl, pc = H[j[:, None], psc], L[j[:, None], psc], C[j[:, None], psc]
    ph[~valid] = pl[~valid] = pc[~valid] = np.nan
    stop = entry - d * R
    tgt = entry + d * target * R
    fav = np.where(d[:, None] > 0, ph >= tgt[:, None], pl <= tgt[:, None])
    adv = np.where(d[:, None] > 0, pl <= stop[:, None], ph >= stop[:, None])
    big = HOLD + 1
    i_adv = np.where(adv.any(1), adv.argmax(1), big)
    i_fav = np.where(fav.any(1), fav.argmax(1), big)
    # time exit: last non-NaN close in the path
    has = ~np.isnan(pc)
    last_i = np.where(has.any(1), HOLD - 1 - has[:, ::-1].argmax(1), -1)
    last_close = np.where(last_i >= 0, pc[np.arange(len(pc)), np.maximum(last_i, 0)], entry)
    exitp = np.where(i_adv <= i_fav, np.where(i_adv < big, stop, last_close), tgt)   # stop wins ties
    x = d * (exitp - entry) * PV
    return x[ok], (2 * R)[ok], ev["session"].to_numpy()[ok]


def shift_stats(ev, H, L, C, target: float) -> dict:
    nd = H.shape[0]
    sd, sig_cl, sd_r, mu = [], [], [], []
    for k in range(MIN_SHIFT, nd - MIN_SHIFT + 1):
        x, rd, g = placebo(ev, H, L, C, k, target)
        n = len(x)
        xc = x - x.mean()
        cl = pd.Series(xc).groupby(g).sum().to_numpy()
        se_cl = np.sqrt((cl ** 2).sum()) / n
        sd.append(x.std(ddof=1)); sig_cl.append(se_cl * np.sqrt(n)); sd_r.append((x / rd).std(ddof=1)); mu.append(x.mean())
    return {"n_shifts": len(sd), "sd_iid_usd": float(np.median(sd)), "sigma_cluster_usd": float(np.median(sig_cl)),
            "sd_R": float(np.median(sd_r)), "placebo_mean_usd": float(np.median(mu)),
            "sigma_cluster_p10_p90": [float(np.percentile(sig_cl, 10)), float(np.percentile(sig_cl, 90))]}


def power(mu: float, sigma: float, n: float) -> float:
    return float(stats.norm.cdf(mu * np.sqrt(n) / sigma - Z_A)) if n > 0 else 0.05


def main() -> int:
    if sha(PLAN) != PLAN_SHA:
        sys.exit(f"PLAN HASH MISMATCH: {sha(PLAN)} != pinned {PLAN_SHA}")
    b = load_5min()
    nd = b["session"].nunique()
    ev = detect(b)
    evl = thin(ev)
    H, L, C = grids(b, nd)
    r_usd = 2 * ev["r_pts"].to_numpy()
    rbar = float(r_usd.mean())

    res: dict = {"plan_sha256": PLAN_SHA, "script_sha256": sha(Path(__file__)),
                 "inputs": {str(p.relative_to(REPO)): sha(p) for p in INPUTS},
                 "window": {"first": str(b.index.min()), "last": str(b.index.max()), "sessions": nd,
                            "bars": len(b)},
                 "events": {"N_upper": len(ev), "N_lower": len(evl),
                            "per_session_upper": len(ev) / nd, "per_session_lower": len(evl) / nd,
                            "long": int((ev["dir"] == 1).sum()), "short": int((ev["dir"] == -1).sum()),
                            "sessions_with_event": int(ev["session"].nunique())},
                 "risk": {"R_usd_mean": rbar, "R_usd_median": float(np.median(r_usd)),
                          "R_usd_p10": float(np.percentile(r_usd, 10)), "R_usd_p90": float(np.percentile(r_usd, 90)),
                          "share_R_over_150": float((r_usd > 150).mean())}}
    res["cost"] = {name: {"usd": c, "mean_c_over_R": float((c / r_usd).mean()), "breakeven_theta_R": c / rbar}
                   for name, c in COSTS.items()}

    res["placebo"] = {tn: shift_stats(ev, H, L, C, tv) for tn, tv in TARGETS.items()}

    grid = {}
    for tn in TARGETS:
        sig = res["placebo"][tn]["sigma_cluster_usd"]
        sig_iid = res["placebo"][tn]["sd_iid_usd"]
        for cn, c in COSTS.items():
            for th_n, th in THETAS.items():
                mu = th * rbar - c
                key = f"{tn}|{cn}|{th_n}"
                row = {"mu_net_usd": mu,
                       "power_IS_upper": power(mu, sig, len(ev)), "power_IS_lower": power(mu, sig, len(evl)),
                       "power_IS_upper_iid": power(mu, sig_iid, len(ev)),
                       "power_HOLD_upper": power(mu, sig, len(ev) / nd * HOLD_SESSIONS),
                       "power_HOLD_lower": power(mu, sig, len(evl) / nd * HOLD_SESSIONS)}
                if mu > 0:
                    n80 = ((Z_A + Z_B) * sig / mu) ** 2
                    row.update({"N_for_80": n80,
                                "years_for_80_upper_rate": n80 / (len(ev) / nd) / 252,
                                "years_for_80_lower_rate": n80 / (len(evl) / nd) / 252})
                else:
                    row.update({"N_for_80": None, "years_for_80_upper_rate": None, "years_for_80_lower_rate": None})
                grid[key] = row
    res["grid"] = grid

    def verdict(tn: str, cn: str) -> str:
        r = grid[f"{tn}|{cn}|central"]
        if r["mu_net_usd"] <= 0:
            return "COST-BOUND"
        if r["power_IS_lower"] >= 0.80:
            return "POWERED"
        if r["power_IS_upper"] >= 0.80:
            return "POWERED-IF-DENSE"
        if r["power_IS_upper"] >= 0.50:
            return "MARGINAL"
        return "UNDERPOWERED"

    res["verdict"] = {"PRIMARY_1R_primary_cost": verdict("1R", "primary"),
                      "sens_2R_primary_cost": verdict("2R", "primary"),
                      "sens_1R_double_cost": verdict("1R", "double"),
                      "sens_2R_double_cost": verdict("2R", "double")}
    (HERE / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("window", "events", "risk", "cost", "placebo", "verdict")}, indent=1, default=float))
    for k, v in grid.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
