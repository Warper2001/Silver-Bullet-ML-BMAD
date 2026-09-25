"""GAP-1-C21 — the one-shot confirmatory run sealed in preregistration_gap1_confirm_2021_2024.md (9351402).

Order, as sealed:
  1. REPRODUCTION GATE: the unchanged rescore replay() must reproduce the corrected Gate-0 on its dev bars
     (N=115, PF 1.646, trade-for-trade equal to corrected_gate0_trades.csv). If not, stop; the target is not replayed.
  2. SETUP GATE: replay the target, then select the primary window. The selected dates must equal the
     outcome-blind power-gate list (307 unseen minus the 29 roll-week dates = 278). If not, stop.
  3. The single test: one-sided one-sample t-test, mean net $/trade > 0, alpha 0.05, cost $5.45/trade.
  4. Descriptives: never a verdict.
Writes only into this folder. Does not touch data/sealed_holdout/ or any live file.

Run: .venv/bin/python _bmad-output/gap1_c21_confirm_20260925/run_c21.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/root/Silver-Bullet-ML-BMAD")
OUT = Path(__file__).resolve().parent
RESCORE = REPO / "_bmad-output/diagnostics_gap_fade_gate0_rescore_20260916/rescore_gate0.py"
DEV_TRADES = RESCORE.parent / "corrected_gate0_trades.csv"
SPLICE_WT = REPO / ".claude/worktrees/gapfade-gate0-rescore/_bmad-output/diagnostics_gap_fade_splice_20260916"
GATE = REPO / "_bmad-output/diagnostics_gap_fade_2021_2024_power_gate_20260925"
TARGET = REPO / "data/mim_x/mnq_1min_2021_2024_frontmonth.csv"
BY_CONTRACT = REPO / "data/mim_x/mnq_1min_by_contract.csv"
COST = 5.45
ALPHA = 0.05
SEED = 20260925
SEEN = {(2023, m) for m in (9, 10, 11)} | {(2024, m) for m in (9, 10, 11)}
assert "sealed_holdout" not in str(TARGET) + str(BY_CONTRACT)

spec = importlib.util.spec_from_file_location("rs", RESCORE)
rs = importlib.util.module_from_spec(spec)
sys.modules["rs"] = rs
spec.loader.exec_module(rs)
gfl = rs.gfl


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pf(x: np.ndarray) -> float | None:
    loss = -x[x < 0].sum()
    return round(float(x[x > 0].sum() / loss), 3) if loss else None


def reproduction_gate() -> dict:
    rs.SPLICE = SPLICE_WT
    b25 = rs.load_csv(SPLICE_WT / "mnq_1min_2025_frontmonth.csv")
    closes, dominant = rs.extract_2025_closes()
    raw26, closes26, _ = rs.raw_2026_frontmonth()
    closes.update(closes26)
    csv26 = rs.load_csv(rs.CSV_2026)
    post = csv26[(csv26.index.date >= rs.ROLL_2026) & (csv26.index <= rs.END)].copy()
    post_rth = post[post.index.map(lambda t: gfl._is_rth(t))]
    for day, g in post_rth.groupby(post_rth.index.date):
        closes[("MNQM26", day)] = float(g["close"].iloc[-1])
    cod: dict = {d: dominant.get(d) for d in sorted({t.date() for t in b25.index})}
    cod.update({d: "MNQH26" for d in {t.date() for t in raw26.index}})
    cod.update({d: "MNQM26" for d in {t.date() for t in post.index}})
    bars = pd.concat([b25[["open", "high", "low", "close"]], raw26[["open", "high", "low", "close"]],
                      post[["open", "high", "low", "close"]]]).sort_index()
    trades, _ = rs.replay(bars, cod, closes)
    got = pd.DataFrame(trades)
    ref = pd.read_csv(DEV_TRADES)
    same = (len(got) == len(ref) and (got["date"].astype(str).values == ref["date"].astype(str).values).all()
            and np.allclose(got["pnl_usd"].values, ref["pnl_usd"].values))
    s = rs.summary(trades)
    return {"N": s["N"], "PF": s["PF"], "net_usd": s["net_usd"], "trade_for_trade_equal": bool(same),
            "passed": bool(same and s["N"] == 115 and s["PF"] == 1.646)}


def target_inputs():
    bars = rs.load_csv(TARGET)
    byc = pd.read_csv(BY_CONTRACT, usecols=["contract", "timestamp", "open", "high", "low", "close"])
    byc["ts"] = pd.to_datetime(byc["timestamp"], utc=True).dt.tz_convert(gfl.ET)
    h, mi = byc["ts"].dt.hour, byc["ts"].dt.minute
    rth = byc[((((h == 9) & (mi >= 30)) | (h > 9)) & (h < 16))]
    sample = byc["ts"].sample(5000, random_state=0)
    hh, mm = sample.dt.hour, sample.dt.minute
    assert all(gfl._is_rth(t) == bool((((a == 9) and (b >= 30)) or a > 9) and a < 16)
               for t, a, b in zip(sample, hh, mm))
    last = rth.sort_values("ts").groupby(["contract", rth["ts"].dt.date])["close"].last()
    closes = {(c, d): float(v) for (c, d), v in last.items()}
    # contract of each session = the contract of its RTH bars (verified single-contract in the power gate)
    fb = bars.reset_index().rename(columns={"timestamp": "ts"})
    m = fb.merge(byc[["ts", "open", "close", "contract"]], on=["ts", "open", "close"], how="left")
    assert m["contract"].notna().all() and len(m) == len(fb)
    mr = m[[gfl._is_rth(t) for t in m["ts"]]]
    per = mr.groupby(mr["ts"].dt.date)["contract"].agg(["nunique", "first"])
    assert (per["nunique"] == 1).all()
    return bars, per["first"].to_dict(), closes


def ambiguous_exit(day_rth: pd.DataFrame, t: dict, pc: float) -> bool:
    """Report-only: did the exit bar touch both target and stop? Mirrors replay()'s scan."""
    entry = day_rth["open"].iloc[0]
    gap_abs = abs(entry - pc)
    direction = -1 if t["dir"] == "short" else 1
    stop = entry + gfl.STOP_MULT * gap_abs if direction == -1 else entry - gfl.STOP_MULT * gap_abs
    for ts_b, bar in day_rth.iloc[1:].iterrows():
        if ts_b.hour >= gfl.TIME_STOP_HOUR:
            return False
        hit_t = bar["low"] <= pc if direction == -1 else bar["high"] >= pc
        hit_s = bar["high"] >= stop if direction == -1 else bar["low"] <= stop
        if hit_t or hit_s:
            return bool(hit_t and hit_s)
    return False


def clustered_t(x: np.ndarray, groups: np.ndarray) -> dict:
    n, mu = len(x), x.mean()
    g = pd.Series(x - mu).groupby(groups).sum().to_numpy()
    G = len(g)
    se = np.sqrt((g ** 2).sum() * G / (G - 1)) / n
    t = mu / se
    return {"clusters": G, "t": round(float(t), 3), "p_one_sided": float(stats.t.sf(t, G - 1))}


def describe(df: pd.DataFrame, rng) -> dict:
    g, net = df["pnl_usd"].to_numpy(float), df["net"].to_numpy(float)
    tt = stats.ttest_1samp(net, 0.0, alternative="greater")
    boot = rng.choice(net, size=(20_000, len(net)), replace=True).mean(axis=1)
    top3 = np.sort(g)[-3:].sum()
    return {"N": len(df), "mean_net": round(float(net.mean()), 2), "sd_net": round(float(net.std(ddof=1)), 2),
            "t": round(float(tt.statistic), 3), "p_one_sided": float(tt.pvalue),
            "boot95_mean_net": [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)],
            "gross_PF": pf(g), "net_PF": pf(net), "WR": round(float((g > 0).mean()) * 100, 1),
            "gross_total": round(float(g.sum()), 2), "net_total": round(float(net.sum()), 2),
            "top3_share_of_gross": round(float(top3 / g.sum()), 3) if g.sum() > 0 else None,
            "month_clustered": clustered_t(net, df["date"].str[:7].to_numpy())}


def main() -> int:
    res: dict = {"prereg": "preregistration_gap1_confirm_2021_2024.md @ 9351402",
                 "script_sha256": sha(Path(__file__)), "target_sha256": sha(TARGET),
                 "by_contract_sha256": sha(BY_CONTRACT), "cost_per_trade": COST}
    res["reproduction_gate"] = rg = reproduction_gate()
    print("reproduction gate:", rg, flush=True)
    if not rg["passed"]:
        (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
        raise SystemExit("REPRODUCTION GATE FAILED — target not replayed (prereg §3)")

    bars, cod, closes = target_inputs()
    trades, skipped = rs.replay(bars, cod, closes)
    df = pd.DataFrame(trades)
    df["dt"] = pd.to_datetime(df["date"])
    df = df[(df["dt"] >= "2021-01-04") & (df["dt"] <= "2024-12-31")]
    df = df[[(d.year, d.month) not in SEEN for d in df["dt"]]]
    roll = set(json.loads((GATE / "addendum_rollweek.json").read_text())["roll_week_dates"])
    blind = pd.read_csv(GATE / "unseen_setups_outcome_blind.csv")
    all307 = df.copy()
    prim = df[~df["date"].isin(roll)].copy()
    expect = sorted(set(blind["date"]) - roll)
    res["setup_gate"] = {"n_307_replayed": len(all307), "n_primary": len(prim),
                         "dates_equal_307": sorted(all307["date"]) == sorted(blind["date"]),
                         "dates_equal_278": sorted(prim["date"]) == expect,
                         "skipped_no_same_contract_prior": skipped}
    print("setup gate:", {k: v for k, v in res["setup_gate"].items() if k != "skipped_no_same_contract_prior"}, flush=True)
    if not (res["setup_gate"]["dates_equal_307"] and res["setup_gate"]["dates_equal_278"]):
        (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
        raise SystemExit("SETUP GATE FAILED — replay setups differ from the sealed outcome-blind list")

    rng = np.random.default_rng(SEED)
    for d in (prim, all307):
        d["net"] = d["pnl_usd"] - COST
    p = describe(prim, rng)
    if p["p_one_sided"] < ALPHA:
        verdict = "CONFIRMED"
    elif p["mean_net"] > 0:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "REFUTED"
    res["primary"] = p
    res["VERDICT"] = verdict
    res["sensitivity_all_307"] = describe(all307, rng)
    res["by_year"] = {str(y): {"N": len(g), "net": round(float(g["net"].sum()), 2),
                               "mean_net": round(float(g["net"].mean()), 2), "gross_PF": pf(g["pnl_usd"].to_numpy(float))}
                      for y, g in prim.groupby(prim["dt"].dt.year)}
    res["by_side"] = {s: {"N": len(g), "net": round(float(g["net"].sum()), 2),
                          "gross_PF": pf(g["pnl_usd"].to_numpy(float))} for s, g in prim.groupby("dir")}
    res["by_outcome"] = prim["outcome"].value_counts().to_dict()
    rthb = bars[bars.index.map(lambda ts: gfl._is_rth(ts))]
    by_day = {d: g for d, g in rthb.groupby(rthb.index.date)}
    days = sorted(by_day)
    prev_of = dict(zip(days[1:], days[:-1]))
    amb = 0
    for t in prim.to_dict("records"):
        if t["outcome"] in ("fill", "stop"):
            d = pd.Timestamp(t["date"]).date()
            amb += ambiguous_exit(by_day[d], t, closes[(cod[d], prev_of[d])])
    res["same_bar_target_and_stop"] = int(amb)
    prim.drop(columns=["dt"]).to_csv(OUT / "trades_primary.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("setup_gate",)}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
