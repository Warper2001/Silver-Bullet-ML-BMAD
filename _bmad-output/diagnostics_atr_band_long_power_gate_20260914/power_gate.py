"""ATR-band long power gate (MNQ 5-min; RTH primary, Globex sensitivity).

Plan: analysis_plan.md (SHA-256 pinned below; the script refuses to run on a mismatch).

FIREWALL (plan section 4). Real fills are used only for their count, timing and
entry-known geometry (ATR, stop and target distances). Every price-path statistic
is computed under a mismatched pairing: the event's geometry is laid on a
*different* eligible session's path, shifted by k whole sessions with
5 <= k <= ND-5. The identity pairing is refused. Nothing under
data/sealed_holdout/ is opened; raw records stamped on/after 2026-03-01 are
skipped during parsing and never stored.

Run:  .venv/bin/python _bmad-output/diagnostics_atr_band_long_power_gate_20260914/power_gate.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PLAN = HERE / "analysis_plan.md"
PLAN_SHA = "a900bd8808dc84841efda17fe9147677ec498a0d90876fd756125ed79ba55968"
RAW = Path("/root/mnq_historical.json")
CUT_UTC = "2026-03-01T00:00:00Z"            # raw records at/after this are skipped unparsed-into-memory
CUTOFF_ET = pd.Timestamp("2026-03-01", tz="America/New_York")
NY = "America/New_York"

# --- frozen by the plan ------------------------------------------------------
TICK, PV = 0.25, 2.0
ATR_N = 14
BAND, TP_M, SL_M = 3.1, 2.0, 1.5
WARMUP_SESSIONS = 3
COSTS = {"low": 2.22, "primary": 5.80, "double": 11.60}
THETAS = {"pessimistic": 0.05, "central": 0.10, "optimistic": 0.20}
MIN_SHIFT = 5
HOLD_SESSIONS = 55              # 2026-03-01 .. 2026-05-19, per ACCESS_LOG; file not opened
Z_A, Z_B = stats.norm.ppf(0.95), stats.norm.ppf(0.80)

VARIANTS = {
    # first_close: minute-of-day (ET) of slot 0's close; slots: bars per full session
    "RTH": {"globex": False, "first_close": 9 * 60 + 35, "slots": 78, "min_minutes": 380,
            "last_fill_slot": 75, "flatten_slot": 76},
    "GLOBEX": {"globex": True, "first_close": 18 * 60 + 5, "slots": 264, "min_minutes": 1300,
               "last_fill_slot": 261, "flatten_slot": 262},
}
FILLS = ("through", "touch")
PRIMARY = ("RTH", "through")

_FIELD = re.compile(r'^\s*"(High|Low|Open|Close|TimeStamp|Contract)":\s*"?([^",]*)"?,?\s*$')


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_raw(path: Path = RAW) -> pd.DataFrame:
    """Stream the pretty-printed TradeStation JSON; keep only records stamped before CUT_UTC."""
    rows, cur, skipped = [], {}, 0
    with open(path) as f:
        for line in f:
            m = _FIELD.match(line)
            if not m:
                continue
            k, v = m.groups()
            cur[k] = v
            if k == "Contract":
                ts = cur["TimeStamp"]
                if ts < CUT_UTC:
                    rows.append((ts, float(cur["Open"]), float(cur["High"]), float(cur["Low"]),
                                 float(cur["Close"]), v))
                else:
                    skipped += 1
                cur = {}
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "contract"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
    df.attrs["skipped_post_cutoff"] = skipped
    assert not df["ts"].duplicated().any(), "duplicate raw timestamps"
    return df.set_index("ts").sort_index()


def session_minutes(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Kept 1-min records with their session key (plan 1.3)."""
    mc = raw.index.hour * 60 + raw.index.minute
    if cfg["globex"]:
        keep = (mc > 18 * 60) | (mc <= 16 * 60)
        r = raw[keep].copy()
        r["session"] = (r.index + pd.Timedelta(hours=6)).date
    else:
        keep = (mc > 9 * 60 + 30) & (mc <= 16 * 60)
        r = raw[keep].copy()
        r["session"] = r.index.date
    return r


def eligible(r: pd.DataFrame, cfg: dict) -> pd.Series:
    g = r.groupby("session")
    return (g["contract"].nunique() == 1) & (g.size() >= cfg["min_minutes"])


def to_bars(r: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    b = r.resample("5min", closed="right", label="right").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        contract=("contract", "first"), session=("session", "first")).dropna(subset=["close"])
    mc = b.index.hour * 60 + b.index.minute
    since = (mc - cfg["first_close"]) % 1440 if cfg["globex"] else mc - cfg["first_close"]
    b["slot"] = since // 5
    assert b["slot"].between(0, cfg["slots"] - 1).all(), "bar outside session slots"
    return b


def add_atr(b: pd.DataFrame) -> pd.DataFrame:
    """Wilder ATR(14), SMA-seeded, continuous; TR = high-low when the previous bar is another contract."""
    h, l, c = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
    con = b["contract"].to_numpy()
    pc = np.r_[np.nan, c[:-1]]
    same = np.r_[False, con[1:] == con[:-1]]
    tr = np.where(same, np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))), h - l)
    atr = np.full(len(tr), np.nan)
    if len(tr) >= ATR_N:
        atr[ATR_N - 1] = tr[:ATR_N].mean()
        for i in range(ATR_N, len(tr)):
            atr[i] = atr[i - 1] + (tr[i] - atr[i - 1]) / ATR_N
    b = b.copy()
    b["tr"], b["atr"] = tr, atr
    return b


def build(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    r = session_minutes(raw, cfg)
    ok = eligible(r, cfg)
    r = r[r["session"].map(ok)]
    b = add_atr(to_bars(r, cfg))
    assert b.index.max() < CUTOFF_ET, "post-cutoff bar survived"
    sessions = sorted(b["session"].unique())
    b["sidx"] = b["session"].map({s: i for i, s in enumerate(sessions)})
    info = {"sessions_all": int(ok.size), "sessions_eligible": int(ok.sum()), "bars": int(len(b)),
            "first_bar": str(b.index.min()), "last_bar": str(b.index.max())}
    return b, info


def detect(b: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Fill events of the resting buy limit. Reads only bar t (order) and bar t+1's open/low (the fill test)."""
    sidx, sl = b["sidx"].to_numpy(), b["slot"].to_numpy()
    c, a = b["close"].to_numpy(), b["atr"].to_numpy()
    lo = b["low"].to_numpy()
    nxt = np.r_[(sidx[1:] == sidx[:-1]) & (sl[1:] == sl[:-1] + 1), False]
    t = np.flatnonzero(nxt & ~np.isnan(a) & (a > 0) & (sidx >= WARMUP_SESSIONS))
    t = t[sl[t + 1] <= cfg["last_fill_slot"]]
    lim = np.floor((c[t] - BAND * a[t]) / TICK) * TICK
    low1 = lo[t + 1]
    touch = low1 <= lim
    ev = pd.DataFrame({"session": sidx[t + 1], "slot": sl[t + 1], "atr": a[t], "limit": lim,
                       "r_pts": np.round(SL_M * a[t] / TICK) * TICK,
                       "tp_pts": np.round(TP_M * a[t] / TICK) * TICK,
                       "through": low1 <= lim - TICK, "touch": touch})
    ev = ev[ev["touch"]].reset_index(drop=True)
    assert (ev["r_pts"] > 0).all()
    return ev


def first_per_session(ev: pd.DataFrame) -> pd.DataFrame:
    return ev.sort_values(["session", "slot"]).groupby("session", sort=True).head(1)


def grids(b: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nd = int(b["sidx"].max()) + 1
    H, L, C = (np.full((nd, cfg["slots"]), np.nan) for _ in range(3))
    si, sl = b["sidx"].to_numpy(), b["slot"].to_numpy()
    H[si, sl], L[si, sl], C[si, sl] = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
    return H, L, C


def placebo(ev: pd.DataFrame, H, L, C, k: int, flatten_slot: int) -> dict:
    """Gross $ P&L of each long event's geometry laid on session (j+k) mod ND. Identity refused."""
    nd = H.shape[0]
    if k % nd == 0:
        raise AssertionError("FIREWALL VIOLATION: identity pairing requested (plan section 4.1).")
    j = (ev["session"].to_numpy() + k) % nd
    slot = ev["slot"].to_numpy()
    R, TP = ev["r_pts"].to_numpy(), ev["tp_pts"].to_numpy()
    entry = C[j, slot]
    ok = ~np.isnan(entry)
    span = flatten_slot - int(slot.min())
    offs = np.arange(1, span + 1)
    ps = slot[:, None] + offs[None, :]
    valid = ps <= flatten_slot
    psc = np.minimum(ps, H.shape[1] - 1)
    ph, pl, pc = H[j[:, None], psc], L[j[:, None], psc], C[j[:, None], psc]
    ph[~valid] = pl[~valid] = pc[~valid] = np.nan
    stop, tgt = entry - R, entry + TP
    fav = ph >= tgt[:, None]
    adv = pl <= stop[:, None]
    big = span + 1
    i_adv = np.where(adv.any(1), adv.argmax(1), big)
    i_fav = np.where(fav.any(1), fav.argmax(1), big)
    has = ~np.isnan(pc)
    last_i = np.where(has.any(1), span - 1 - has[:, ::-1].argmax(1), -1)
    last_close = np.where(last_i >= 0, pc[np.arange(len(pc)), np.maximum(last_i, 0)], entry)
    hit_stop = (i_adv <= i_fav) & (i_adv < big)
    hit_tgt = (i_fav < i_adv)
    exitp = np.where(hit_stop, stop, np.where(hit_tgt, tgt, last_close))   # stop wins ties
    x = (exitp - entry) * PV
    return {"x": x[ok], "r_usd": (PV * R)[ok], "g": ev["session"].to_numpy()[ok],
            "flat_share": float((~hit_stop & ~hit_tgt)[ok].mean()) if ok.any() else float("nan")}


def cluster_sigma(x: np.ndarray, g: np.ndarray) -> float:
    n = len(x)
    cl = pd.Series(x - x.mean()).groupby(g).sum().to_numpy()
    return float(np.sqrt((cl ** 2).sum()) / n * np.sqrt(n))


def shift_stats(ev: pd.DataFrame, H, L, C, flatten_slot: int) -> dict:
    nd = H.shape[0]
    out = {k: [] for k in ("sd", "sig_cl", "sd_R", "sig_cl_R", "mu", "flat")}
    for k in range(MIN_SHIFT, nd - MIN_SHIFT + 1):
        p = placebo(ev, H, L, C, k, flatten_slot)
        x, xr = p["x"], p["x"] / p["r_usd"]
        for key, val in (("sd", x.std(ddof=1)), ("sig_cl", cluster_sigma(x, p["g"])),
                         ("sd_R", xr.std(ddof=1)), ("sig_cl_R", cluster_sigma(xr, p["g"])),
                         ("mu", x.mean()), ("flat", p["flat_share"])):
            out[key].append(val)
    med = {k: float(np.median(v)) for k, v in out.items()}
    return {"n_shifts": len(out["sd"]), "sd_iid_usd": med["sd"], "sigma_cluster_usd": med["sig_cl"],
            "sd_iid_R": med["sd_R"], "sigma_cluster_R": med["sig_cl_R"], "placebo_mean_usd": med["mu"],
            "placebo_flatten_share": med["flat"],
            "sigma_cluster_p10_p90": [float(np.percentile(out["sig_cl"], 10)),
                                      float(np.percentile(out["sig_cl"], 90))]}


def bracket_bound(ev: pd.DataFrame) -> dict:
    """sigma if every trade ends at a bracket with p = 0.5 (plan 4.2). Uses entry-known geometry only."""
    R, TP = PV * ev["r_pts"].to_numpy(), PV * ev["tp_pts"].to_numpy()
    usd = np.sqrt(np.mean(((TP + R) / 2) ** 2) + np.var((TP - R) / 2))
    tr = TP / R
    rr = np.sqrt(np.mean(((tr + 1) / 2) ** 2) + np.var((tr - 1) / 2))
    return {"sigma_bb_usd": float(usd), "sigma_bb_R": float(rr)}


def power(mu: float, sigma: float, n: float) -> float:
    return float(stats.norm.cdf(mu * np.sqrt(n) / sigma - Z_A)) if n > 0 and sigma > 0 else 0.05


def verdict(row: dict) -> str:
    if row["mu_net"] <= 0:
        return "COST-BOUND"
    if row["power_IS_lower"] >= 0.80:
        return "POWERED"
    if row["power_IS_upper"] >= 0.80:
        return "POWERED-IF-DENSE"
    if row["power_IS_upper"] >= 0.50:
        return "MARGINAL"
    return "UNDERPOWERED"


def evaluate(ev: pd.DataFrame, nd: int, H, L, C, cfg: dict) -> dict:
    evl = first_per_session(ev)
    r_usd = PV * ev["r_pts"].to_numpy()
    rbar = float(r_usd.mean())
    pl = shift_stats(ev, H, L, C, cfg["flatten_slot"])
    bb = bracket_bound(ev)
    de = max(1.0, pl["sigma_cluster_usd"] / pl["sd_iid_usd"])
    de_r = max(1.0, pl["sigma_cluster_R"] / pl["sd_iid_R"])
    sig = {"contract": {"placebo": pl["sigma_cluster_usd"], "v": max(pl["sigma_cluster_usd"], bb["sigma_bb_usd"] * de)},
           "risk": {"placebo": pl["sigma_cluster_R"], "v": max(pl["sigma_cluster_R"], bb["sigma_bb_R"] * de_r)}}
    n_up, n_lo = len(ev), len(evl)
    rate_up, rate_lo = n_up / nd, n_lo / nd
    grid = {}
    for sizing in ("contract", "risk"):
        for cn, c in COSTS.items():
            mean_c_R = float((c / r_usd).mean())
            for th_n, th in THETAS.items():
                mu = th * rbar - c if sizing == "contract" else th - mean_c_R
                for sn, s in sig[sizing].items():
                    row = {"mu_net": mu, "sigma": s,
                           "power_IS_upper": power(mu, s, n_up), "power_IS_lower": power(mu, s, n_lo),
                           "power_HOLD_upper": power(mu, s, rate_up * HOLD_SESSIONS),
                           "power_HOLD_lower": power(mu, s, rate_lo * HOLD_SESSIONS)}
                    if mu > 0:
                        n80 = ((Z_A + Z_B) * s / mu) ** 2
                        row.update({"N_for_80": n80, "years_for_80_upper_rate": n80 / rate_up / 252,
                                    "years_for_80_lower_rate": n80 / rate_lo / 252})
                    else:
                        row.update({"N_for_80": None, "years_for_80_upper_rate": None, "years_for_80_lower_rate": None})
                    row["verdict"] = verdict(row) if th_n == "central" else None
                    grid[f"{sizing}|{cn}|{th_n}|{sn}"] = row
    hours = (ev["slot"] * 5 + cfg["first_close"]) // 60 % 24
    return {
        "events": {"N_upper": n_up, "N_lower": n_lo, "per_session_upper": rate_up, "per_session_lower": rate_lo,
                   "sessions_with_event": int(ev["session"].nunique()),
                   "max_events_one_session": int(ev.groupby("session").size().max()) if n_up else 0,
                   "by_fill_hour_ET": {int(k): int(v) for k, v in hours.value_counts().sort_index().items()}},
        "geometry": {"atr_pts_median": float(ev["atr"].median()), "R_usd_mean": rbar,
                     "R_usd_median": float(np.median(r_usd)), "R_usd_p10": float(np.percentile(r_usd, 10)),
                     "R_usd_p90": float(np.percentile(r_usd, 90))},
        "cost": {cn: {"usd": c, "mean_c_over_R": float((c / r_usd).mean()), "breakeven_theta_R": c / rbar}
                 for cn, c in COSTS.items()},
        "placebo": pl, "bracket_bound": bb, "cluster_ratio_usd": de, "cluster_ratio_R": de_r,
        "sigma_used": sig, "grid": grid,
    }


def main() -> int:
    if sha(PLAN) != PLAN_SHA:
        sys.exit(f"PLAN HASH MISMATCH: {sha(PLAN)} != pinned {PLAN_SHA}")
    raw = parse_raw()
    res: dict = {"plan_sha256": PLAN_SHA, "script_sha256": sha(Path(__file__)),
                 "inputs": {str(RAW): sha(RAW)},
                 "raw": {"records_kept": int(len(raw)), "records_skipped_post_cutoff": raw.attrs["skipped_post_cutoff"],
                         "first": str(raw.index.min()), "last": str(raw.index.max())},
                 "variants": {}}
    for vn, cfg in VARIANTS.items():
        b, info = build(raw, cfg)
        ev_all = detect(b, cfg)
        H, L, C = grids(b, cfg)
        nd = H.shape[0]
        res["variants"][vn] = {"window": info}
        for fill in FILLS:
            ev = ev_all[ev_all[fill]].reset_index(drop=True)
            res["variants"][vn][fill] = evaluate(ev, nd, H, L, C, cfg)
            print(vn, fill, json.dumps(res["variants"][vn][fill]["events"], default=float), flush=True)
    p = res["variants"][PRIMARY[0]][PRIMARY[1]]["grid"]["contract|primary|central|v"]
    res["verdict_primary"] = p["verdict"]
    res["verdict_table"] = {f"{vn}|{fill}|{key}": r["verdict"]
                            for vn in VARIANTS for fill in FILLS
                            for key, r in res["variants"][vn][fill]["grid"].items() if r["verdict"]}
    (HERE / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print("PRIMARY VERDICT:", res["verdict_primary"])
    for k, v in res["verdict_table"].items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
