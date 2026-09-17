"""Verify the refetched HG/SI front-month files against the defective originals.

Checks:
  1. the 2026-03-01 → 03-11 window is now liquid (front month), not thin;
  2. span and row counts;
  3. price level in that window moved to the liquid contract;
  4. roll-boundary jumps are disclosed (the stitch is unadjusted, like es/gc).

Run: .venv/bin/python verify_frontmonth_refetch.py
"""
import json
from pathlib import Path

import pandas as pd

MAIN = Path("/root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute")
NEW_DIR = Path("data/processed/dollar_bars/1_minute")
CAL = json.loads(Path("metals_roll_calendar.json").read_text())
WINDOWS = [("2026-03-01..03-11 (defective window)", "2026-03-01", "2026-03-12"),
           ("2026-03-12..04-01", "2026-03-12", "2026-04-01"),
           ("2026-02-01..02-24", "2026-02-01", "2026-02-24")]


def load(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, usecols=["timestamp", "volume", "close"])
    d["ts"] = pd.to_datetime(d["timestamp"], utc=True, format="ISO8601")
    return d.sort_values("ts")


def med_vol(d: pd.DataFrame, lo: str, hi: str) -> float:
    s = d[(d["ts"] >= lo) & (d["ts"] < hi)]["volume"]
    return float(s.median()) if len(s) else float("nan")


def main() -> int:
    out = {}
    for k in ("hg", "si"):
        o = load(MAIN / f"{k}_1min_2025_2026.csv")
        n = load(NEW_DIR / f"{k}_1min_2025_2026_frontmonth.csv")
        rec = {"old_rows": len(o), "new_rows": len(n),
               "old_span": [str(o["ts"].min().date()), str(o["ts"].max().date())],
               "new_span": [str(n["ts"].min().date()), str(n["ts"].max().date())],
               "median_volume": {}, "segments": CAL[k]["segments"]}
        print(f"== {k}: old {len(o):,} rows {rec['old_span']} | new {len(n):,} rows {rec['new_span']}")
        for lbl, lo, hi in WINDOWS:
            mo, mn = med_vol(o, lo, hi), med_vol(n, lo, hi)
            rec["median_volume"][lbl] = {"old": mo, "new": mn}
            print(f"   {lbl}: median vol/min old={mo:>7,.0f}  new={mn:>7,.0f}")
        ow = o[(o["ts"] >= "2026-03-02") & (o["ts"] < "2026-03-03")]["close"].median()
        nw = n[(n["ts"] >= "2026-03-02") & (n["ts"] < "2026-03-03")]["close"].median()
        rec["close_2026_03_02"] = {"old": float(ow), "new": float(nw), "diff": float(nw - ow)}
        print(f"   2026-03-02 median close: old {ow:,.4f} | new {nw:,.4f} | diff {nw - ow:+,.4f}")
        jumps = []
        for s in CAL[k]["segments"][1:]:
            b = pd.Timestamp(s["start"], tz="UTC")
            before = n[n["ts"] < b]["close"]
            after = n[n["ts"] >= b]["close"]
            if len(before) and len(after):
                jumps.append({"roll_to": s["symbol"], "date": s["start"],
                              "gap_pts": round(float(after.iloc[0] - before.iloc[-1]), 4)})
        rec["roll_boundary_gaps"] = jumps
        print(f"   roll-boundary gaps (unadjusted stitch): {jumps}")
        pre = rec["median_volume"][WINDOWS[0][0]]
        rec["verdict"] = ("FIXED: defective window now liquid"
                          if pre["new"] >= 5 * max(pre["old"], 1) else "NOT FIXED — investigate")
        print(f"   VERDICT: {rec['verdict']}")
        out[k] = rec
    Path("verify_frontmonth_refetch.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
