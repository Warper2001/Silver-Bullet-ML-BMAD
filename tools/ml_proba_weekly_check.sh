#!/usr/bin/env bash
# One-shot ml_proba ordinal-hypothesis verdict check.
# Pre-registration: _bmad-output/preregistration_ml_proba_ordinal.md (seal 110a533)
# Self-removes its own crontab line after running (one-time behavior).
set -uo pipefail
cd /root/Silver-Bullet-ML-BMAD || exit 1
PY=.venv/bin/python
STAMP="$(date -u +%Y%m%d)"
OUT="data/reports/ml_proba_verdict_${STAMP}.txt"
{
  echo "=================================================================="
  echo " ml_proba ordinal-hypothesis verdict — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo " prereg: _bmad-output/preregistration_ml_proba_ordinal.md (seal 110a533)"
  echo " rule: PENDING <30; interim early-stop N>=15 if rho<=0;"
  echo "       PASS at N>=30 iff rho>0 one-sided p<0.05 AND top-tercile E\$>bottom"
  echo "=================================================================="
  echo
  echo "----- trader-s26 (PRIMARY) -----"
  $PY analyze_ml_proba_hypothesis.py --trader trader-s26 --cutoff 2026-06-20
  echo
  echo "----- trader-yank (secondary) -----"
  $PY analyze_ml_proba_hypothesis.py --trader trader-yank --cutoff 2026-06-20
} > "$OUT" 2>&1
echo "wrote $OUT"
# one-time: remove our own cron line, preserve any others
crontab -l 2>/dev/null | grep -v 'ml_proba_weekly_check.sh' | crontab - 2>/dev/null || true
