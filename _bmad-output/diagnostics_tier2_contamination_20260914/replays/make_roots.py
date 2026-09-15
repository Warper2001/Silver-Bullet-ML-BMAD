"""Build two replay roots (symlink farms) for the Tier2 back-month replay."""
import os
import shutil
from pathlib import Path

T = Path("/root/.claude/jobs/960bda86/tmp")
W = Path("/root/Silver-Bullet-ML-BMAD/.claude/worktrees/tier2-contamination-check")
M = Path("/root/Silver-Bullet-ML-BMAD")
SKIP = {"data", "models", "logs", ".git"}
DB1 = Path("data/processed/dollar_bars/1_minute")

for name, csv26 in (("replay_g0", M / DB1 / "mnq_1min_2026_ytd.csv"),
                    ("replay_c", T / "front_month/mnq_1min_2026_ytd.csv")):
    root = T / name
    if root.exists():
        shutil.rmtree(root)
    (root / "models/xgboost").mkdir(parents=True)
    (root / DB1).mkdir(parents=True)
    (root / "logs").mkdir()
    for e in W.iterdir():
        if e.name not in SKIP:
            os.symlink(e, root / e.name)
    os.symlink(W / "models/xgboost/tier2_meta_labeling_model.pkl", root / "models/xgboost/tier2_meta_labeling_model.pkl")
    for f in ("tier2_threshold.json", "lr_regime_config.json"):
        os.symlink(M / "models/xgboost" / f, root / "models/xgboost" / f)
    os.symlink(M / DB1 / "mnq_1min_2025.csv", root / DB1 / "mnq_1min_2025.csv")
    os.symlink(csv26, root / DB1 / "mnq_1min_2026_ytd.csv")
    print(name, sorted(p.name for p in root.iterdir())[:12], "->", os.readlink(root / DB1 / "mnq_1min_2026_ytd.csv"))
