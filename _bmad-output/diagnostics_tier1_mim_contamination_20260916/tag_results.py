"""Tag MIM-NB (and other) saved backtest trade lists against the MNQ 1-min CSV defects.

Defects: 2025 roll weeks interleave two contracts minute by minute; Jan–Feb 2026 (and the
holdout's first ~2 weeks) in mnq_1min_2026_ytd.csv is the deferred MNQM26 contract.
See _bmad-output/diagnostics_h2l2_contamination_20260914/ and ..._tier2_contamination_20260914/.

Reads saved result CSVs only — no replay, no bars, no holdout files.

Run: .venv/bin/python _bmad-output/diagnostics_tier1_mim_contamination_20260916/tag_results.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = Path("/root/Silver-Bullet-ML-BMAD")
REPORTS = MAIN / "data/reports"
ATR_GATE = MAIN / "_bmad-output/diagnostics_atr_band_long_power_gate_20260914/power_gate.py"
# GAP-1's roll is the H26 quarterly cycle; these are the segment boundaries used throughout.
ROLL_2026 = "2026-03-12"      # CME H26 -> M26 roll date
FILES = ["mim_nb_gate1_v1_2026oos", "mim_nb_gate1_v2_2026oos",
         "mim_nb_gate0_v1_2025", "mim_nb_gate0_v2_2025",
         "mim_nb_catstop_s500_pooled", "mim_nb_catstop_s250_pooled"]


def interleaved_rth_sessions() -> set[str]:
    """RTH sessions whose raw 1-min records carry more than one contract label."""
    spec = importlib.util.spec_from_file_location("atr_gate", ATR_GATE)
    atr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(atr)
    raw = atr.parse_raw()
    mc = raw.index.hour * 60 + raw.index.minute
    rth = raw[(mc > 9 * 60 + 30) & (mc <= 16 * 60)]
    n = rth.groupby(rth.index.date)["contract"].nunique()
    return {str(d) for d, v in n.items() if v > 1}


def segment_of(day: str, inter: set[str]) -> str:
    if day >= ROLL_2026:
        return "Mar12+_front_month"
    if day >= "2026-03-01":
        return "Mar1-11_back_month"
    if day >= "2026-01-01":
        return "JanFeb26_back_month"
    return "roll_week_2025" if day in inter else "clean_2025"


def pf(x: pd.Series) -> float | None:
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else None


def main() -> int:
    inter = interleaved_rth_sessions()
    out: dict = {"interleaved_rth_sessions": sorted(inter), "files": {}}
    for name in FILES:
        p = REPORTS / f"{name}.csv"
        if not p.exists():
            out["files"][name] = {"missing": True}
            continue
        d = pd.read_csv(p)
        d["day"] = d["day"].astype(str)
        d["segment"] = [segment_of(x, inter) for x in d["day"]]
        pnl = "pnl_pts" if "pnl_pts" in d.columns else "pnl"
        rec = {"N": len(d), "total_pts": float(d[pnl].sum()), "PF": pf(d[pnl]),
               "span": [d["day"].min(), d["day"].max()], "by_segment": {}}
        for seg, g in d.groupby("segment"):
            rec["by_segment"][seg] = {"N": len(g), "pts": float(g[pnl].sum()), "PF": pf(g[pnl])}
        bad = ("roll_week_2025", "JanFeb26_back_month", "Mar1-11_back_month")
        contaminated = [s for s in rec["by_segment"] if s in bad]
        rec["contaminated_N"] = sum(rec["by_segment"][s]["N"] for s in contaminated)
        rec["contaminated_pts"] = sum(rec["by_segment"][s]["pts"] for s in contaminated)
        rec["share_of_total_pts"] = (rec["contaminated_pts"] / rec["total_pts"]
                                     if rec["total_pts"] else None)
        clean = d[~d["segment"].isin(contaminated)][pnl]
        rec["clean_only"] = {"N": int(len(clean)), "pts": float(clean.sum()), "PF": pf(clean)}
        out["files"][name] = rec
        print(f"{name}: N={rec['N']} PF={rec['PF']} | contaminated N={rec['contaminated_N']} "
              f"pts={rec['contaminated_pts']:.1f} ({rec['share_of_total_pts']:.1%} of total) | "
              f"clean-only N={rec['clean_only']['N']} PF={rec['clean_only']['PF']}")
    (HERE / "tag_results.json").write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
