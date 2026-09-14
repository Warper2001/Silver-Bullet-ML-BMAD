#!/bin/bash
# Three Tier2 census replays (replay_plan.md), niced so the live trader units keep priority.
T=/root/.claude/jobs/960bda86/tmp
W=/root/Silver-Bullet-ML-BMAD/.claude/worktrees/tier2-contamination-check
PY=/root/Silver-Bullet-ML-BMAD/.venv/bin/python
common="--start 2025-05-19 --end 2026-02-28 --pin max_daily_loss=-750"
nice -n 19 $PY $W/tools/tier2_census.py --ml-threshold 0.50 --tag ml050 $common --repo-root $T/replay_g0 --out-dir $T/out_g0 > $T/g0_ml.log 2>&1 &
nice -n 19 $PY $W/tools/tier2_census.py --ml-threshold 0.50 --tag ml050 $common --repo-root $T/replay_c --out-dir $T/out_c > $T/c_ml.log 2>&1 &
nice -n 19 $PY $W/tools/tier2_census.py --ml-threshold 0.0 --tag noml $common --repo-root $T/replay_c --out-dir $T/out_c > $T/c_noml.log 2>&1 &
wait
echo "all done"
