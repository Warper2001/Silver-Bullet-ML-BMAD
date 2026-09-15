"""Compare the G0 / C-ml / C-noml census replays (replay_plan.md). Descriptive only.

Run: .venv/bin/python _bmad-output/diagnostics_tier2_contamination_20260914/compare_replays.py <out_g0> <out_c>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CENSUS = Path("/root/Silver-Bullet-ML-BMAD/_bmad-output/diagnostics_yank_entry_mechanics_20260913")
JAN = pd.Timestamp("2026-01-01", tz="UTC")


def trades(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p)
    d["t"] = pd.to_datetime(d["entry_time"], utc=True)
    d["month"] = d["t"].dt.strftime("%Y-%m")
    return d


def pf(x: pd.Series) -> float | None:
    gl = -x[x < 0].sum()
    return float(x[x > 0].sum() / gl) if gl > 0 else None


def stats(d: pd.DataFrame) -> dict:
    return {"n": int(len(d)), "pnl": float(d["pnl"].sum()), "pf": pf(d["pnl"]), "wins": int((d["pnl"] > 0).sum())}


def main(out_g0: str, out_c: str) -> int:
    runs = {"census_ml050": trades(CENSUS / "census_trades_ml050.csv"),
            "census_noml": trades(CENSUS / "census_trades_noml.csv"),
            "G0_ml050": trades(Path(out_g0) / "census_trades_ml050.csv"),
            "C_ml050": trades(Path(out_c) / "census_trades_ml050.csv"),
            "C_noml": trades(Path(out_c) / "census_trades_noml.csv")}
    cols = ["entry_time", "exit_time", "direction", "entry_price", "exit_price", "exit_type", "bars_held", "pnl"]
    g0_ok = runs["G0_ml050"][cols].reset_index(drop=True).equals(runs["census_ml050"][cols].reset_index(drop=True))
    res: dict = {"G0_reproduces_census_row_for_row": bool(g0_ok)}
    for a, b in (("G0_ml050", "C_ml050"), ("census_noml", "C_noml")):
        pa, pb = runs[a][runs[a]["t"] < JAN][cols], runs[b][runs[b]["t"] < JAN][cols]
        res[f"pre2026_identical_{a}_vs_{b}"] = bool(pa.reset_index(drop=True).equals(pb.reset_index(drop=True)))
        res[f"pre2026_counts_{a}_vs_{b}"] = [len(pa), len(pb)]
    res["whole_window"] = {k: stats(d) for k, d in runs.items()}
    res["jan_feb_2026"] = {k: stats(d[d["t"] >= JAN]) for k, d in runs.items()}
    res["pre_2026"] = {k: stats(d[d["t"] < JAN]) for k, d in runs.items()}
    months = pd.DataFrame({k: d.groupby("month").size() for k, d in runs.items()}).fillna(0).astype(int)
    pnl = pd.DataFrame({k: d.groupby("month")["pnl"].sum() for k, d in runs.items()}).fillna(0.0)
    res["by_month_trades"] = months.to_dict()
    res["by_month_pnl"] = pnl.round(2).to_dict()
    for k in ("G0", "C"):
        meta = Path(out_g0 if k == "G0" else out_c)
        res[f"meta_{k}"] = [json.loads(p.read_text()) for p in sorted(meta.glob("census_meta_*.json"))]
    (HERE / "replay_results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("by_month")}, indent=1, default=str))
    print(months.to_string())
    print(pnl.round(0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
