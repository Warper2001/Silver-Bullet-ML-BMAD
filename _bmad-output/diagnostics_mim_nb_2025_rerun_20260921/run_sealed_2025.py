"""Sealed MIM-NB engine re-run on the 2025 bars — IN-SAMPLE (2025 is the sealed DEV window).

What this is: the sealed `run_catstop()` from study_mim_nb_catstop.py (prereg 6957daa; file sha256 pinned below), lifted
VERBATIM by AST (importing the script would execute its module-level code: it loads the spent 2026 OOS file and writes
into data/reports/). Run at S=250 (live since 2026-06-25) and S=500 (the sealed original), on:
  A. the frozen 2025 CSV (md5 3ba83a32… = the sealed dev file)      -> PARITY check vs the sealed 2025-06-11 pooled trades
  B. the 2025 CSV rebuilt one-contract-per-session (roll-week splices removed, sha pinned below)
It does NOT read data/sealed_holdout/, mnq_1min_2026_ytd.csv, or anything from 2026. It applies no gate and tests no threshold:
2025 is the window the spec was developed and gated on, so nothing here is out-of-sample. Descriptive statistics only, in the
same units as the third-party report (bp/trade, win rate, payoff), plus splits fixed in advance: calendar halves, and the
defect-day exclusion (the 40 interleaved sessions + the 4 contract-boundary sessions from rebuild_meta.json).

Run: .venv/bin/python _bmad-output/diagnostics_mim_nb_2025_rerun_20260921/run_sealed_2025.py
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from collections import defaultdict, deque

REPO = Path("/root/Silver-Bullet-ML-BMAD")
HERE = Path(__file__).parent
SEALED_SCRIPT = REPO / "study_mim_nb_catstop.py"
SEALED_SHA_PREFIX = "210518d63a76374f"
FROZEN = REPO / "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
FROZEN_MD5 = "3ba83a32cac3fa1284e09277259887c9"
FRONT = HERE / "mnq_1min_2025_frontmonth.csv"
FRONT_SHA = "f1fe5b36abba90681d8b1439a3975f94e4b4d1040093c7c0368629a807d219d4"
META = HERE / "rebuild_meta.json"

assert "sealed_holdout" not in str(FROZEN) and "2026" not in FROZEN.name
sha = hashlib.sha256(SEALED_SCRIPT.read_bytes()).hexdigest()
assert sha.startswith(SEALED_SHA_PREFIX), f"sealed script changed: {sha}"
assert hashlib.md5(FROZEN.read_bytes()).hexdigest() == FROZEN_MD5, "frozen 2025 CSV changed"
assert hashlib.sha256(FRONT.read_bytes()).hexdigest() == FRONT_SHA, "front-month rebuild does not match pinned sha"

# ---- lift constants + the three functions verbatim, nothing else ---------------------------------
src = SEALED_SCRIPT.read_text()
tree = ast.parse(src)
keep = []
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        keep.append(node)
    elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("BASE", "COST_PTS", "PT_VAL", "ET", "LOOKBACK"):
        keep.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in ("load", "run_catstop", "stats"):
        keep.append(node)
ns: dict = {}
exec(compile(ast.Module(body=keep, type_ignores=[]), str(SEALED_SCRIPT), "exec"), ns)
load, run_catstop, sealed_stats = ns["load"], ns["run_catstop"], ns["stats"]
COST_PTS, PT_VAL = ns["COST_PTS"], ns["PT_VAL"]

meta = json.loads(META.read_text())
DEFECT_DAYS = set(meta["mixed_session_detail"]) | set(meta["contract_switches_between_sessions"])
rng = np.random.default_rng(20260921)


def describe(t: pd.DataFrame, opens: pd.Series, label: str) -> dict:
    """Same units as the third-party report. bp = pnl_pts / that day's 09:31 open (entry price is not stored by the sealed fn)."""
    t = t.copy()
    t["day"] = pd.to_datetime(t["day"])
    net = t["pnl_pts"] - COST_PTS
    t["net_usd"] = net * PT_VAL
    t["bp_gross"] = t["pnl_pts"] / t["day"].map(opens) * 1e4
    t["bp_net"] = net / t["day"].map(opens) * 1e4
    w, l = t[t.net_usd > 0], t[t.net_usd < 0]
    n = len(t)
    boot = rng.choice(t["bp_net"].to_numpy(), size=(20000, n)).mean(axis=1)
    return {"label": label, "N": n, "net_usd": float(t.net_usd.sum()),
            "PF_net": float(w.net_usd.sum() / -l.net_usd.sum()), "win_rate_net": len(w) / n,
            "payoff_net": float(w.net_usd.mean() / -l.net_usd.mean()),
            "mean_bp_gross": float(t.bp_gross.mean()), "mean_bp_net": float(t.bp_net.mean()),
            "sd_bp_net": float(t.bp_net.std(ddof=1)),
            "mean_bp_net_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "exp_usd_per_trade": float(t.net_usd.mean())}


def daily_sharpe(t: pd.DataFrame, days: list, opens: pd.Series) -> dict:
    net = (t["pnl_pts"] - COST_PTS) * PT_VAL
    d = net.groupby(pd.to_datetime(t["day"])).sum().reindex(pd.to_datetime(days), fill_value=0.0)
    r = d / (opens.mean() * PT_VAL)
    sh = r.mean() / r.std(ddof=1) * np.sqrt(252)
    bs = [(lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252))(rng.choice(r.to_numpy(), size=len(r))) for _ in range(20000)]
    return {"sessions": len(days), "trade_days": int((d != 0).sum()), "sharpe": float(sh),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


out: dict = {"sealed_script_sha256": sha, "frozen_md5": FROZEN_MD5, "front_sha256": FRONT_SHA, "runs": {}}
for tag, path in (("A_frozen", FROZEN), ("B_frontmonth", FRONT)):
    df = load(str(path))
    sess = df.groupby("day").first()
    opens = pd.Series({pd.Timestamp(d): g for d, g in sess["open"].items()})
    valid_days = [pd.Timestamp(d) for d, g in df.groupby("day") if g["hm"].iloc[0] == "09:31" and "16:00" in set(g["hm"])]
    for S in (250, 500):
        t = run_catstop(df, S)
        t["day_ts"] = pd.to_datetime(t["day"])
        r: dict = {"S": S, "sealed_stats(N,PF,exp$,worst)": [float(x) for x in sealed_stats(t)]}
        r["all"] = describe(t, opens, "all 2025 trades")
        r["ex_defect_days"] = describe(t[~t["day"].astype(str).isin(DEFECT_DAYS)], opens, "excluding defect sessions")
        for lab, lo, hi in (("H1_2025", "2025-01-01", "2025-06-30"), ("H2_2025", "2025-07-01", "2025-12-31")):
            m = (t.day_ts >= lo) & (t.day_ts <= hi)
            r[lab] = describe(t[m], opens, lab)
        # concentration: top 5 winning days removed (pre-specified, same slice used on the live ledger)
        pd_ = ((t["pnl_pts"] - COST_PTS) * PT_VAL).groupby(t["day"].astype(str)).sum()
        top5 = set(pd_.sort_values(ascending=False).index[:5])
        r["ex_top5_days"] = describe(t[~t["day"].astype(str).isin(top5)], opens, "excluding 5 best days")
        r["top5_days"] = sorted(top5)
        r["reasons"] = t.groupby("reason")["pnl_pts"].agg(["size", "mean"]).round(2).reset_index().to_dict("records")
        r["daily_sharpe"] = daily_sharpe(t, [d for d in valid_days if d.year == 2025], opens)
        monthly = t.assign(m=t.day_ts.dt.strftime("%Y-%m"), net=(t["pnl_pts"] - COST_PTS) * PT_VAL) \
                   .groupby("m").agg(N=("net", "size"), net_usd=("net", "sum")).round(1).reset_index()
        r["monthly"] = monthly.to_dict("records")
        out["runs"][f"{tag}_S{S}"] = r
        t.to_csv(HERE / f"trades_{tag}_S{S}.csv", index=False)

# ---- PARITY vs the sealed 2025-06-11 pooled trades (dev part = 2025 dates) ------------------------
par = {}
for S in (250, 500):
    sealed = pd.read_csv(REPO / f"data/reports/mim_nb_catstop_s{S}_pooled.csv")
    dev = sealed[sealed["day"].astype(str) < "2026-01-01"]
    mine = pd.read_csv(HERE / f"trades_A_frozen_S{S}.csv")
    par[f"S{S}"] = {"sealed_N": len(dev), "mine_N": len(mine),
                    "sealed_sum_pts": float(dev.pnl_pts.sum()), "mine_sum_pts": float(mine.pnl_pts.sum()),
                    "identical_rows": bool(len(dev) == len(mine) and np.allclose(dev.pnl_pts.to_numpy(), mine.pnl_pts.to_numpy())
                                           and (dev.day.astype(str).to_numpy() == mine.day.astype(str).to_numpy()).all())}
out["parity_vs_sealed_pooled"] = par
(HERE / "results.json").write_text(json.dumps(out, indent=2, default=float))

for k, r in out["runs"].items():
    a = r["all"]
    print(f"[{k}] N={a['N']} PF_net={a['PF_net']:.3f} exp=${a['exp_usd_per_trade']:+.2f} net=${a['net_usd']:+,.0f} "
          f"WR={a['win_rate_net']:.1%} payoff={a['payoff_net']:.2f} bp_net={a['mean_bp_net']:+.2f} "
          f"CI[{a['mean_bp_net_ci95'][0]:+.1f},{a['mean_bp_net_ci95'][1]:+.1f}] | H1 N={r['H1_2025']['N']} PF={r['H1_2025']['PF_net']:.2f} "
          f"H2 N={r['H2_2025']['N']} PF={r['H2_2025']['PF_net']:.2f} | ex-defect N={r['ex_defect_days']['N']} PF={r['ex_defect_days']['PF_net']:.3f} "
          f"| ex-top5 N={r['ex_top5_days']['N']} PF={r['ex_top5_days']['PF_net']:.3f} | dailySharpe={r['daily_sharpe']['sharpe']:+.2f}")
print("PARITY:", json.dumps(par))
