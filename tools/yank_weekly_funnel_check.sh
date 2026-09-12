#!/bin/bash
# YANK weekly funnel checkpoint — runs yank_weekly_funnel_check.py, writes a dated
# report under logs/yank_weekly_funnel/, and optionally pushes a Telegram summary.
#
# Invoked by yank-weekly-funnel-check.service (systemd oneshot + one-shot OnCalendar
# timer). Runs LOCALLY because it needs data/trades.db and logs/*.log, which a cloud
# runner would only ever see as of the last commit — this data is gitignored and
# live-appended. Read-only: never touches the bot, config, or trading state.
#
# Usage: tools/yank_weekly_funnel_check.sh
set -uo pipefail

BASE=/root/Silver-Bullet-ML-BMAD
PY="$BASE/.venv/bin/python"
DAY="$(date -u +%F)"
OUT_DIR="$BASE/logs/yank_weekly_funnel"
OUT="$OUT_DIR/report_${DAY}.txt"

mkdir -p "$OUT_DIR"
cd "$BASE" || exit 1

{
  echo "HEAD: $(git rev-parse --short HEAD 2>/dev/null)"
  echo
  PYTHONPATH="$BASE" timeout 180 "$PY" tools/yank_weekly_funnel_check.py 2>&1
} > "$OUT" 2>&1

echo "report written: $OUT"
cat "$OUT"

# ---------------------------------------------------------------------------
# Telegram push — the verdict, not the whole report. Same contract as
# tools/mim_parity_check.sh / tools/combine_ops_alert.sh: credentials live in
# .env.telegram (gitignored), and this is a silent no-op when absent, so the
# script stays runnable on a machine without them.
# ---------------------------------------------------------------------------
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
ENTRIES=$(grep -E '^REAL_ENTRIES:' "$OUT" | head -1 | trim)
SHADOW=$(grep -E '^SHADOW_LEDGER:' "$OUT" | head -1 | trim)
SEALABLE=$(grep -E '^SEALABLE:' "$OUT" | head -1 | trim)
HEALTH=$(grep -E '^BOT_HEALTH:' "$OUT" | head -1 | trim)
SUMMARY=$(grep -E '^SUMMARY:' "$OUT" | head -1 | sed 's/^SUMMARY: *//' | trim)

MSG="YANK weekly funnel checkpoint — ${DAY}
${SUMMARY:-(report did not produce a summary line — see file)}

${ENTRIES:-REAL_ENTRIES: n/a}
${SHADOW:-SHADOW_LEDGER: n/a}
${SEALABLE:-SEALABLE: n/a}
${HEALTH:-BOT_HEALTH: n/a}

report: logs/yank_weekly_funnel/report_${DAY}.txt"

notify_telegram "$MSG"
echo "--- telegram payload ---"
echo "$MSG"
