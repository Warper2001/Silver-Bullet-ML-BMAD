#!/usr/bin/env python3
"""ETFTM-1 A0.4-A0.6 — build the total-return panel, split the holdout, write manifest + audit.

Inputs: data/etf_daily/raw/{ts,yahoo}/ from tools/fetch_etf_daily.py.
Total return is rebuilt from Yahoo 'close' (split-adjusted, price-only) plus Yahoo ex-date dividends,
one internally consistent source:
    r_t = (close_t + div_t) / close_{t-1} - 1
Why not TradeStation closes (decided 2026-09-25 from data-quality checks only, before any signal or
strategy return existed): TS closes for QQQ, XLE and XLV carry undocumented historical adjustments
(e.g. XLE TS/Yahoo price ratio drifts 0.91 -> 1.00 from 1999 to 2019), so TS close + dividends would
double-count distributions. TradeStation is kept as the INDEPENDENT check: its price returns are
compared with Yahoo price returns on days without a dividend. Yahoo 'adjclose' is a second check.
Outputs:
  data/etf_daily/panel_dev.csv                         dates < CUTOFF (development)
  data/etf_daily/rf_dev.csv                            ^IRX daily risk-free, dates < CUTOFF
  data/sealed_holdout/etf_daily_holdout_20211001_plus.csv  dates >= CUTOFF (chmod 444)
  data/etf_daily/manifest.json                         SHA-256 of every input and output + script
  _bmad-output/etftm1_data_audit_<date>.md              audit (no strategy statistic is computed)
The builder reports only data-quality statistics. It computes no signal, no strategy return.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.etftm1_universe import ALL_SYMBOLS, ASSET_CLASS  # noqa: E402

ROOT = Path("/root/Silver-Bullet-ML-BMAD")
RAW = ROOT / "data/etf_daily/raw"
OUT = ROOT / "data/etf_daily"
HOLDOUT = ROOT / "data/sealed_holdout/etf_daily_holdout_20211001_plus.csv"
CUTOFF = "2021-10-01"
MIN_YEARS_GATE = 15.0


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_symbol(sym: str) -> tuple[pd.DataFrame, dict]:
    raw_ts = pd.read_csv(RAW / "ts" / f"{sym}.csv", parse_dates=["date"])[["date", "close", "volume"]]
    raw_ts = raw_ts.drop_duplicates("date").rename(columns={"close": "close_tsx"})
    yf = pd.read_csv(RAW / "yahoo" / f"{sym}.csv", parse_dates=["date"]).drop_duplicates("date")
    dv = pd.read_csv(RAW / "yahoo" / f"{sym}_div.csv", parse_dates=["date"])
    ts = yf.sort_values("date").reset_index(drop=True)            # primary: Yahoo close + dividends
    ts["div"] = ts["date"].map(dv.groupby("date")["amount"].sum()).fillna(0.0)
    unmatched_div = int((~dv["date"].isin(ts["date"])).sum())
    ts["tr_ret"] = (ts["close"] + ts["div"]) / ts["close"].shift(1) - 1
    ts = ts.merge(raw_ts, on="date", how="left")
    m = ts.copy()
    m["yf_adj_ret"] = m["adjclose"] / m["adjclose"].shift(1) - 1
    both = m.dropna(subset=["tr_ret", "yf_adj_ret"])
    diff = (both["tr_ret"] - both["yf_adj_ret"]).abs()
    mon = both.set_index("date")[["tr_ret", "yf_adj_ret"]].add(1).resample("ME").prod().sub(1)
    te_m = (mon["tr_ret"] - mon["yf_adj_ret"]).abs()
    lvl = (m["close_tsx"] / m["close"] - 1).abs()
    # independent source: TS price return vs Yahoo price return on non-dividend days
    px = m.assign(r_y=m["close"] / m["close"].shift(1) - 1, r_t=m["close_tsx"] / m["close_tsx"].shift(1) - 1)
    px = px[(px["div"] == 0) & (px["div"].shift(1).fillna(0) == 0)].dropna(subset=["r_y", "r_t"])
    ts_check = (px["r_y"] - px["r_t"]).abs()
    audit = {
        "sym": sym, "asset_class": ASSET_CLASS[sym], "first": str(ts["date"].min().date()),
        "years_before_cutoff": round((pd.Timestamp(CUTOFF) - ts["date"].min()).days / 365.25, 1),
        "rows": len(ts), "dup_dates": int(ts["date"].duplicated().sum()),
        "nonpos_close": int((ts["close"] <= 0).sum()),
        "big_moves_gt25pct": [str(d.date()) for d in ts.loc[ts["tr_ret"].abs() > 0.25, "date"]],
        "unmatched_dividends": unmatched_div,
        "daily_absdiff_vs_yf_median": float(diff.median()), "daily_absdiff_vs_yf_p99": float(diff.quantile(0.99)),
        "monthly_te_median": float(te_m.median()), "monthly_te_p95": float(te_m.quantile(0.95)),
        "monthly_te_max": float(te_m.max()),
        "level_absdiff_vs_yf_close_median": float(lvl.median()),
        "missing_in_ts": int(m["close_tsx"].isna().sum()),
        "ts_indep_daily_absdiff_median": float(ts_check.median()),
        "ts_indep_daily_absdiff_p99": float(ts_check.quantile(0.99)),
    }
    ts["sym"], ts["asset_class"] = sym, ASSET_CLASS[sym]
    return ts[["date", "sym", "asset_class", "close", "div", "volume", "tr_ret"]], audit


def main() -> int:
    frames, audits = [], []
    for s in ALL_SYMBOLS:
        f, a = build_symbol(s)
        frames.append(f)
        audits.append(a)
    panel = pd.concat(frames).sort_values(["date", "sym"]).reset_index(drop=True)
    cal = set(panel.loc[panel["sym"] == "SPY", "date"])
    for a in audits:
        d = panel.loc[panel["sym"] == a["sym"], "date"]
        a["off_spy_calendar"] = int((~d.isin(cal)).sum())
        span = [x for x in sorted(cal) if x >= d.min()]
        a["missing_vs_spy_calendar"] = int(len(set(span) - set(d)))
    rf = pd.read_csv(RAW / "yahoo" / "IRX.csv", parse_dates=["date"]).dropna()
    rf["rf_daily"] = rf["yield_pct"] / 100.0 / 252.0

    dev, hold = panel[panel["date"] < CUTOFF], panel[panel["date"] >= CUTOFF]
    OUT.mkdir(parents=True, exist_ok=True)
    dev.to_csv(OUT / "panel_dev.csv", index=False)
    rf[rf["date"] < CUTOFF].to_csv(OUT / "rf_dev.csv", index=False)
    if HOLDOUT.exists():
        os.chmod(HOLDOUT, 0o644)
    hold_rf = rf[rf["date"] >= CUTOFF][["date", "rf_daily"]]
    hold.merge(hold_rf, on="date", how="left").to_csv(HOLDOUT, index=False)
    os.chmod(HOLDOUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    per_class = {}
    for a in audits:
        per_class.setdefault(a["asset_class"], []).append(a["years_before_cutoff"])
    classes_15y = sorted(c for c, ys in per_class.items() if max(ys) >= MIN_YEARS_GATE)
    bad_tr = [a["sym"] for a in audits if a["monthly_te_p95"] > 0.002 or a["ts_indep_daily_absdiff_median"] > 0.001]
    gate = "GO" if len(classes_15y) >= 6 and len(bad_tr) <= len(audits) // 5 else "STOP"
    if len(classes_15y) < 4:
        gate = "STOP"

    inputs = sorted(RAW.rglob("*.csv"))
    manifest = {"cutoff": CUTOFF, "script_sha256": sha(Path(__file__)),
                "inputs": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
                "outputs": {str(p.relative_to(ROOT)): sha(p) for p in
                            (OUT / "panel_dev.csv", OUT / "rf_dev.csv", HOLDOUT)},
                "panel_dev_rows": len(dev), "holdout_rows": len(hold),
                "dev_first": str(dev["date"].min().date()), "dev_last": str(dev["date"].max().date())}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

    rep = ROOT / f"_bmad-output/etftm1_data_audit_{date.today():%Y%m%d}.md"
    L = [f"# ETFTM-1 data audit ({date.today()})", "",
         f"Built by `tools/build_etf_panel.py` (sha `{manifest['script_sha256'][:12]}`). "
         "Data-quality statistics only: no signal or strategy return was computed.", "",
         f"- Development panel: {len(dev):,} rows, {manifest['dev_first']} → {manifest['dev_last']}.",
         f"- Holdout: {len(hold):,} rows from {CUTOFF}, written to `{HOLDOUT.relative_to(ROOT)}` (mode 444).",
         f"- Asset classes with ≥ {MIN_YEARS_GATE:.0f} years before the cutoff: {len(classes_15y)} ({', '.join(classes_15y)}).",
         f"- Symbols failing a check (monthly TE p95 vs adjclose > 0.2%, or median daily price-return gap vs TradeStation > 0.1%): {bad_tr or 'none'}.",
         f"- **Gate A0: {gate}**", "",
         "Total return = Yahoo close + Yahoo dividends (reason in the builder docstring). TE columns compare it "
         "with Yahoo adjclose; the TS columns compare daily PRICE returns with TradeStation, an independent source, on non-dividend days.", "",
         "| Sym | Class | First | Yrs<cutoff | Rows | Dup | ≤0 | >25% days | Unmatched divs | Monthly TE p95 vs adjclose | TS indep. daily diff med / p99 | TS level diff med | Missing in TS | Missing vs SPY cal |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in audits:
        L.append(f"| {a['sym']} | {a['asset_class']} | {a['first']} | {a['years_before_cutoff']} | {a['rows']} | "
                 f"{a['dup_dates']} | {a['nonpos_close']} | {len(a['big_moves_gt25pct'])} | {a['unmatched_dividends']} | "
                 f"{a['monthly_te_p95']:.2e} | {a['ts_indep_daily_absdiff_median']:.1e} / {a['ts_indep_daily_absdiff_p99']:.1e} | "
                 f"{a['level_absdiff_vs_yf_close_median']:.4f} | {a['missing_in_ts']} | {a['missing_vs_spy_calendar']} |")
    big = {a["sym"]: a["big_moves_gt25pct"] for a in audits if a["big_moves_gt25pct"]}
    L += ["", f"Days with a total-return move > 25%: {big or 'none'}.",
          "", "Raw files under `data/etf_daily/raw/` contain the holdout period too. They are inputs to this "
          "builder only; analysis code must read `panel_dev.csv` (enforced in `research/etf_trend/data.py`, A3)."]
    rep.write_text("\n".join(L) + "\n")
    (OUT / "audit.json").write_text(json.dumps(audits, indent=1))
    print("\n".join(L[:9]))
    return 0 if gate == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
