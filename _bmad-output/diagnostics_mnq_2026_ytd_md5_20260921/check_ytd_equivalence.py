"""Is the current data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv functionally the file the MIM-NB seals ran on?

The noise-bands seal pinned this file's md5 as 4ec175dd…; today it is 30bc05a8… (mtime 2026-06-11 23:31, after the ~04:53 seal;
header now has a `notional` column; last bar 2026-06-11 23:32). The sealed bytes are not recoverable (no prefix of the current
file hashes to 4ec175dd, and git has a single add, 744642e, on 2026-08-06). So test FUNCTION instead of bytes:
run the sealed engine (study_mim_nb_catstop.py, sha pinned) on the current file and compare its 2026 trades with the ones the
sealed run wrote to data/reports/mim_nb_catstop_s{250,500}_pooled.csv on 2026-06-11, before the file changed.

Print-only: writes nothing, never imports the sealed script (its module level loads data and writes data/reports/), reads nothing from
data/sealed_holdout/. The engine functions are lifted verbatim by AST.

Run: .venv/bin/python _bmad-output/diagnostics_mnq_2026_ytd_md5_20260921/check_ytd_equivalence.py
"""
import ast
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/root/Silver-Bullet-ML-BMAD")
SRC = REPO / "study_mim_nb_catstop.py"
FILE = REPO / "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"
assert hashlib.sha256(SRC.read_bytes()).hexdigest().startswith("210518d63a76374f"), "sealed script changed"

tree = ast.parse(SRC.read_text())
keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
        or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("BASE", "COST_PTS", "PT_VAL", "ET", "LOOKBACK"))
        or (isinstance(n, ast.FunctionDef) and n.name in ("load", "run_catstop"))]
ns: dict = {}
exec(compile(ast.Module(body=keep, type_ignores=[]), str(SRC), "exec"), ns)

df = ns["load"](str(FILE))
print(f"current file: md5 {hashlib.md5(FILE.read_bytes()).hexdigest()}, RTH bars {len(df)}, days {df.day.nunique()}, "
      f"{df.day.min()} .. {df.day.max()}")
ok = True
for S in (250, 500):
    mine = ns["run_catstop"](df, S)
    mine["day"] = mine["day"].astype(str)
    sealed = pd.read_csv(REPO / f"data/reports/mim_nb_catstop_s{S}_pooled.csv")
    s26 = sealed[sealed["day"].astype(str) >= "2026-01-01"].reset_index(drop=True)
    last = s26["day"].astype(str).max()
    m = mine[mine["day"] <= last].reset_index(drop=True)
    same = (len(m) == len(s26) and (m["day"].to_numpy() == s26["day"].astype(str).to_numpy()).all()
            and np.allclose(m["pnl_pts"].to_numpy(), s26["pnl_pts"].to_numpy()))
    ok &= bool(same)
    print(f"S{S}: sealed 2026 trades {len(s26)} (last day {last}) | current file, same days {len(m)} | "
          f"sum pts sealed {s26.pnl_pts.sum():.2f} vs current {m.pnl_pts.sum():.2f} | IDENTICAL={same} | "
          f"trades after {last} (bars appended post-seal): {len(mine) - len(m)}")
sys.exit(0 if ok else 1)
