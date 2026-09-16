"""Compare GAP-1 on the back-adjusted 2025 series against the frozen CSV and the
front-month rebuild, and verify that back-adjustment changed ONLY boundary sessions.

Run: .venv/bin/python _bmad-output/diagnostics_gap_fade_splice_20260916/measure_backadjusted.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = Path("/root/Silver-Bullet-ML-BMAD")

spec = importlib.util.spec_from_file_location("ms", HERE / "measure_splice.py")
ms = importlib.util.module_from_spec(spec)
sys.modules["ms"] = ms
spec.loader.exec_module(ms)

FROZEN = MAIN / "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
FRONT = HERE / "mnq_1min_2025_frontmonth.csv"
ADJ = HERE / "mnq_1min_2025_frontmonth_adjusted.csv"


def main() -> int:
    meta = json.loads((HERE / "rebuild_meta.json").read_text())
    boundaries = set(meta["contract_switches_between_sessions"])
    runs = {"frozen": ms.trades_for(FROZEN), "frontmonth": ms.trades_for(FRONT),
            "frontmonth_backadjusted": ms.trades_for(ADJ)}
    res = {k: ms.summary(v) for k, v in runs.items()}
    out: dict = {"summaries": res, "boundary_sessions": sorted(boundaries)}

    f_by = {t["date"]: t for t in runs["frontmonth"]}
    a_by = {t["date"]: t for t in runs["frontmonth_backadjusted"]}
    changed = sorted(d for d in set(f_by) & set(a_by) if f_by[d] != a_by[d])
    out["changed_frontmonth_vs_adjusted"] = [{"date": d, "is_boundary": d in boundaries,
                                              "frontmonth": f_by[d], "adjusted": a_by[d]}
                                             for d in changed]
    out["only_in_frontmonth"] = sorted(set(f_by) - set(a_by))
    out["only_in_adjusted"] = sorted(set(a_by) - set(f_by))
    out["all_changes_are_boundary_sessions"] = all(
        d in boundaries for d in changed + out["only_in_frontmonth"] + out["only_in_adjusted"])

    z_by = {t["date"]: t for t in runs["frozen"]}
    out["boundary_trades_across_versions"] = [
        {"date": d, "frozen": z_by.get(d), "frontmonth": f_by.get(d), "adjusted": a_by.get(d)}
        for d in sorted(boundaries) if d in z_by or d in f_by or d in a_by]
    clean = [t for t in runs["frontmonth_backadjusted"] if t["date"] not in boundaries]
    out["adjusted_excluding_boundaries"] = ms.summary(clean)
    (HERE / "backadjusted_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
