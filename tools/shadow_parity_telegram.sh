#!/usr/bin/env bash
# Daily ProjectX shadow-parity check -> Telegram.
#
# Runs tools/analyze_shadow_parity.py against both bots' live shadow logs and pushes
# a compact summary (the OVERALL + GATE lines) to Telegram. Stage-1 cutover-gate
# tracking for the data-feed migration — fires until ≥10 clean sessions/bot, then the
# GATE line flips to PASS. Read-only; never touches the live traders.
set -uo pipefail

BASE="/root/Silver-Bullet-ML-BMAD"
PY="$BASE/.venv/bin/python"
[ -f "$BASE/.env.telegram" ] && . "$BASE/.env.telegram"

summarize() {  # $1=csv  — prints "(N sessions)\n  OVERALL\n  GATE"
  local csv="$1"
  if [ ! -f "$csv" ]; then printf '(no log yet)\n'; return; fi
  local out; out="$("$PY" "$BASE/tools/analyze_shadow_parity.py" --csv "$csv" 2>&1)"
  local sessions overall gate
  sessions="$(printf '%s' "$out" | grep -oE '[0-9]+ sessions' | head -1)"
  overall="$(printf '%s' "$out" | grep -A1 '^OVERALL:' | tail -1 | sed 's/^ *//')"
  gate="$(printf '%s' "$out" | grep -E 'GATE \(' | sed 's/^ *//')"
  printf '(%s)\n  %s\n  %s\n' "${sessions:-?}" "${overall:-n/a}" "${gate:-n/a}"
}

msg="$(printf '📊 Shadow parity — %s\n\nYANK %s\nMIM %s' \
  "$(date -u '+%Y-%m-%d %H:%MZ')" \
  "$(summarize "$BASE/logs/yank_shadow_parity.csv")" \
  "$(summarize "$BASE/data/mim_nb/shadow_parity.csv")")"

echo "$msg"   # also to journal

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
