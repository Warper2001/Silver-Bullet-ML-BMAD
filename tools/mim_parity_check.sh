#!/bin/bash
# MIM-NB parity check — runs the per-session gate check and the whole-era replay,
# writes a dated report under logs/parity/.
#
# Invoked by mim-parity-check.service (systemd oneshot + timer). Runs LOCALLY because
# it needs the live data/mim_nb/*.csv that the bot appends to — a cloud runner would
# only ever see whatever was last committed.
#
# Usage: tools/mim_parity_check.sh [YYYY-MM-DD]     (default: today, ET)
set -uo pipefail

BASE=/root/Silver-Bullet-ML-BMAD
PY="$BASE/.venv/bin/python"
DAY="${1:-$(TZ=America/New_York date +%F)}"
OUT_DIR="$BASE/logs/parity"
OUT="$OUT_DIR/parity_${DAY}.txt"

mkdir -p "$OUT_DIR"
cd "$BASE" || exit 1

{
  echo "############ MIM-NB PARITY CHECK — $DAY ############"
  echo "generated: $(TZ=America/Chicago date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "HEAD: $(git rev-parse --short HEAD 2>/dev/null)"
  echo

  echo "########## 1. PER-SESSION GATES (G1/G2, includes the 16:00 mark) ##########"
  timeout 300 "$PY" tools/mim_parity_day.py "$DAY" 2>&1
  DAY_RC=$?
  echo
  echo "  per-session exit code: $DAY_RC  (0=pass 1=gate fail 2=nothing to compare)"
  echo

  echo "########## 2. WHOLE-ERA REPLAY (context; historical rows can never match) ##########"
  timeout 550 "$PY" tools/mim_parity_replay.py 2>&1 | sed -n '/mark-level diff/,/^$/p'
  echo

  echo "########## 3. BOT STATE ##########"
  systemctl is-active trader-mim-nb | sed 's/^/  trader-mim-nb: /'
  echo "  last restart: $(systemctl show trader-mim-nb -p ActiveEnterTimestamp --value)"
  grep -viE 'HTTP Request' logs/mim_nb_live.log 2>/dev/null \
    | grep -E 'Sigma restored|Sigma seeded|Sigma folded|CATCHUP_NO_RECORD|Catch-up complete|DEPTH_GATE' \
    | tail -6 | sed 's/^/  /'
  echo
  echo "############ END ############"
} > "$OUT" 2>&1

echo "parity report written: $OUT"
tail -25 "$OUT"
