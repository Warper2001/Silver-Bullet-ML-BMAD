#!/bin/bash
# YANK live PF checkpoint — runs yank_live_pf_check.py, writes a dated report under
# logs/yank_live_pf/, and optionally pushes a Telegram summary.
#
# Invoked by yank-live-pf-check.service (systemd oneshot + weekly timer). Runs LOCALLY
# because it needs data/trades.db and models/xgboost/tier2_threshold.json, which are
# gitignored and live-appended. Read-only: never touches the bot, config, or trading
# state. Same contract as tools/yank_weekly_funnel_check.sh.
#
# Usage: tools/yank_live_pf_check.sh
set -uo pipefail

BASE=/root/Silver-Bullet-ML-BMAD
PY="$BASE/.venv/bin/python"
DAY="$(date -u +%F)"
OUT_DIR="$BASE/logs/yank_live_pf"
OUT="$OUT_DIR/report_${DAY}.txt"

mkdir -p "$OUT_DIR"
cd "$BASE" || exit 1

{
  echo "HEAD: $(git rev-parse --short HEAD 2>/dev/null)"
  echo
  PYTHONPATH="$BASE" timeout 120 "$PY" tools/yank_live_pf_check.py 2>&1
} > "$OUT" 2>&1

echo "report written: $OUT"
cat "$OUT"

# Append one JSON line of history so the series is machine-readable later.
PYTHONPATH="$BASE" timeout 120 "$PY" tools/yank_live_pf_check.py --json 2>/dev/null \
  | "$PY" -c 'import json,sys; print(json.dumps(json.load(sys.stdin)))' \
  >> "$OUT_DIR/history.jsonl" 2>/dev/null || echo "WARN: history line not appended" >&2

# Telegram: the verdict, not the whole report. Silent no-op without credentials.
[ -f "$BASE/.env.telegram" ] && . "$BASE/.env.telegram"

notify_telegram() {
  [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ] || {
    echo "telegram: no credentials — report-only run"; return 0; }
  local resp
  resp=$(curl -s --max-time 15 \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=$1" \
    --data "disable_web_page_preview=true" \
    "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage")
  case "$resp" in
    *'"ok":true'*) echo "telegram: verdict pushed" ;;
    *) echo "WARN: telegram send failed (report is still at $OUT): ${resp:0:200}" >&2 ;;
  esac
}

trim() { sed 's/^ *//;s/ *$//'; }
ALL=$(grep -E '^LIVE_ALL:' "$OUT" | head -1 | trim)
SINCE=$(grep -E '^LIVE_SINCE_ML_OFF:' "$OUT" | head -1 | trim)
STOP=$(grep -E '^FORWARD_STOP:' "$OUT" | head -1 | trim)
CFG=$(grep -E '^CONFIG:' "$OUT" | head -1 | trim)
HEALTH=$(grep -E '^BOT_HEALTH:' "$OUT" | head -1 | trim)
SUMMARY=$(grep -E '^SUMMARY:' "$OUT" | head -1 | sed 's/^SUMMARY: *//' | trim)

MSG="YANK live PF checkpoint — ${DAY}
${SUMMARY:-(report did not produce a summary line — see file)}

${SINCE:-LIVE_SINCE_ML_OFF: n/a}
${ALL:-LIVE_ALL: n/a}
${STOP:-FORWARD_STOP: n/a}
${CFG:-CONFIG: n/a}
${HEALTH:-BOT_HEALTH: n/a}

report: logs/yank_live_pf/report_${DAY}.txt"

notify_telegram "$MSG"
echo "--- telegram payload ---"
echo "$MSG"
