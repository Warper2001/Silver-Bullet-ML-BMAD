#!/usr/bin/env bash
# One-shot S25-prep reminder: compare invvol SIM giveback vs combine -> Telegram.
# Scheduled via invvol-giveback-report.timer (fires ~2026-07-21). Set 2026-06-19.
set -uo pipefail
BASE="/root/Silver-Bullet-ML-BMAD"
[ -f "$BASE/.env.telegram" ] && . "$BASE/.env.telegram"
PY="$BASE/.venv/bin/python"

body="$("$PY" "$BASE/tools/invvol_giveback_report.py" 2>&1)"
hdr="⏰ S25 prep — invvol SIM vs combine giveback"
msg="$(printf '%s\n\n%s' "$hdr" "$body")"
echo "$msg"

if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
  curl -s --max-time 20 \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=${msg}" \
    --data "disable_web_page_preview=true" \
    "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" >/dev/null \
    || echo "WARN: telegram send failed" >&2
else
  echo "WARN: no telegram creds (.env.telegram) — printed only" >&2
fi
