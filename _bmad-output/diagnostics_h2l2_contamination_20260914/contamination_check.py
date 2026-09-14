"""Roll-splice contamination check for the H2/L2 and wedge/MM power gates (diagnostic, no outcomes).

Question: did the dollar-aggregated CSVs those gates read contain contract-interleaved roll
weeks, and did that inflate their event counts or risk?

Reads: the same CSVs via the H2/L2 gate's own frozen loader, and /root/mnq_historical.json
(pre-2026-03-01 records only, via the ATR-band gate's parser) for contract labels.
Computes: session purity, 5-min bar ranges, and each event's risk (known at order time).
Never computes price after a real event.

Run:  .venv/bin/python _bmad-output/diagnostics_h2l2_contamination_20260914/contamination_check.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


h2 = load("power_gate", OUT / "diagnostics_h2l2_power_gate_20260913/power_gate.py")   # wedge/MM imports this name
wm = load("wedge_mm_gate", OUT / "diagnostics_wedge_mm_power_gate_20260913/power_gate.py")
atr = load("atr_gate", OUT / "diagnostics_atr_band_long_power_gate_20260914/power_gate.py")

# data/ is gitignored, so read the main checkout's copies, and prove they are the bytes the gate hashed
MAIN = Path("/root/Silver-Bullet-ML-BMAD")
h2.INPUTS = [MAIN / p.relative_to(h2.REPO) for p in h2.INPUTS]
_recorded = json.loads((OUT / "diagnostics_h2l2_power_gate_20260913/results.json").read_text())["inputs"]
for p in h2.INPUTS:
    assert h2.sha(p) == _recorded[str(p.relative_to(MAIN))], f"input differs from the gate's: {p}"


def main() -> int:
    raw = atr.parse_raw()
    r = raw[(raw.index >= pd.Timestamp(h2.START, tz=atr.NY)) & (raw.index < h2.CUTOFF)]
    mc = r.index.hour * 60 + r.index.minute
    r = r[(mc > 9 * 60 + 30) & (mc <= 16 * 60)]
    ncon = r.groupby(r.index.date)["contract"].nunique()
    interleaved = set(ncon[ncon > 1].index)

    # which CSV rows (the gates' 1-min input) mix contracts: raw minutes in (prev_ts, ts]
    csv = pd.read_csv(h2.INPUTS[0], usecols=["timestamp"])
    cts = pd.to_datetime(csv["timestamp"], utc=True, format="ISO8601").dt.tz_localize(None).to_numpy()
    rall = raw[(raw.index >= pd.Timestamp("2025-01-01", tz="UTC")) & (raw.index < pd.Timestamp("2026-01-01", tz="UTC"))]
    row = np.searchsorted(cts, rall.index.tz_convert("UTC").tz_localize(None).to_numpy(), side="left")
    grp = pd.Series(rall["contract"].to_numpy()).groupby(row)
    mixed_rows = grp.nunique()
    n_mixed_rows = int((mixed_rows > 1).sum())
    # multi-minute (dollar-aggregated) rows among the gates' RTH rows in 2025
    ct_ny = pd.DatetimeIndex(cts).tz_localize("UTC").tz_convert(atr.NY)
    in_rth = ((ct_ny.hour * 60 + ct_ny.minute > 9 * 60 + 30) & (ct_ny.hour * 60 + ct_ny.minute <= 16 * 60)
              & (ct_ny.year == 2025))
    sizes = grp.size()
    rth_rows = np.flatnonzero(in_rth)
    multi = sizes.reindex(rth_rows).fillna(0)
    n_rth_rows, n_rth_multi = int(len(rth_rows)), int((multi > 1).sum())

    # is the 2026 CSV (Jan-Feb) the front month? compare its closes with raw front-month closes
    c26 = pd.read_csv(h2.INPUTS[1], usecols=["timestamp", "close"])
    c26.index = pd.to_datetime(c26["timestamp"], utc=True, format="ISO8601").dt.tz_convert(atr.NY)
    c26 = c26[c26.index < h2.CUTOFF]
    j = c26[["close"]].join(raw[["close", "contract"]], rsuffix="_raw", how="inner")
    back_month = {"csv_rows_jan_feb_2026": int(len(c26)), "matched_timestamps": int(len(j)),
                  "raw_contract_at_matched": j["contract"].value_counts().to_dict(),
                  "raw_front_month_minutes_jan_feb_2026": int(((raw.index >= pd.Timestamp("2026-01-01", tz=atr.NY))
                                                               & (raw.index < h2.CUTOFF)).sum()),
                  "exact_close_match_share": float((j["close"] == j["close_raw"]).mean()),
                  "median_abs_close_diff_pts": float((j["close"] - j["close_raw"]).abs().median())}

    b = h2.load_5min()
    sessions = sorted(b["session"].unique())
    flag = np.array([s in interleaved for s in sessions])
    rng = (b["high"] - b["low"]).groupby(b["session"]).max()
    rng_i, rng_c = rng[rng.index.isin(interleaved)], rng[~rng.index.isin(interleaved)]

    def split(ev: pd.DataFrame, label: str) -> dict:
        r_usd = h2.PV * ev["r_pts"].to_numpy()
        inter = flag[ev["session"].to_numpy()]
        out = {}
        for name, m in (("all", np.ones_like(inter)), ("interleaved", inter), ("clean", ~inter)):
            x = r_usd[m]
            out[name] = {"N": int(m.sum()),
                         "R_usd_mean": float(x.mean()) if len(x) else None,
                         "R_usd_median": float(np.median(x)) if len(x) else None,
                         "R_usd_p90": float(np.percentile(x, 90)) if len(x) else None,
                         "n_R_over_150": int((x > 150).sum()), "n_R_over_400": int((x > 400).sum())}
        print(label, json.dumps(out))
        return out

    res = {
        "window_sessions": len(sessions), "interleaved_sessions": int(flag.sum()),
        "interleaved_dates": [str(d) for d in sessions if d in interleaved],
        "csv_2025_rows_mixing_contracts": n_mixed_rows,
        "csv_2026_back_month_check": back_month,
        "csv_2025_rth_rows": n_rth_rows, "csv_2025_rth_rows_spanning_multiple_minutes": n_rth_multi,
        "clean_session_largest_5min_bars": {str(k): float(v) for k, v in rng_c.nlargest(5).items()},
        "session_max_5min_range_pts": {
            "interleaved_median": float(rng_i.median()), "interleaved_min": float(rng_i.min()),
            "clean_median": float(rng_c.median()), "clean_p99": float(rng_c.quantile(0.99)),
            "clean_max": float(rng_c.max())},
        "events": {"H2L2": split(h2.detect(b), "H2L2"),
                   "W_wedge_s1": split(wm.detect_wedge(b, 1), "W_wedge_s1"),
                   "M_mm_fade_s1": split(wm.detect_mm(b, 1)[0], "M_mm_fade_s1")},
    }
    (HERE / "contamination_results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "events"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
