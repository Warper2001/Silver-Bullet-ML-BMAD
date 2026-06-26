#!/usr/bin/env bash
# Daily ProjectX shadow-parity check -> Telegram, ALERT-ONLY.
#
# Runs tools/analyze_shadow_parity.py on both bots' live shadow logs and pushes to
# Telegram ONLY when something needs attention:
#   PROBLEM  — parity degraded (median > tick / <95% within tick / coverage gaps /
#              fetch errors / contract mismatch)
#   PASS     — a bot cleared the cutover gate (≥10 clean sessions) -> ready to pre-register
# Silent while both bots are just ACCUMULATING clean sessions. Read-only.
set -uo pipefail

BASE="/root/Silver-Bullet-ML-BMAD"
PY="$BASE/.venv/bin/python"
[ -f "$BASE/.env.telegram" ] && . "$BASE/.env.telegram"

YCSV="$BASE/logs/yank_shadow_parity.csv"
MCSV="$BASE/data/mim_nb/shadow_parity.csv"

ystat="$("$PY" "$BASE/tools/analyze_shadow_parity.py" --status --csv "$YCSV" 2>&1)"
mstat="$("$PY" "$BASE/tools/analyze_shadow_parity.py" --status --csv "$MCSV" 2>&1)"
ytok="${ystat%% *}"; mtok="${mstat%% *}"

echo "$(date -u '+%Y-%m-%dT%H:%MZ')  YANK: $ystat | MIM: $mstat"   # journal trail

# Alert only if either bot is PROBLEM or PASS.
notable() { case "$1" in PROBLEM|PASS) return 0;; *) return 1;; esac; }
if ! notable "$ytok" && ! notable "$mtok"; then
  echo "both ACCUMULATING/NODATA — no alert."
  exit 0
fi

hdr="📊 Shadow parity"
notable "$ytok" && [ "$ytok" = PROBLEM ] && hdr="⚠️ Shadow parity PROBLEM"
notable "$mtok" && [ "$mtok" = PROBLEM ] && hdr="⚠️ Shadow parity PROBLEM"
{ [ "$ytok" = PASS ] || [ "$mtok" = PASS ]; } && [ "$hdr" = "📊 Shadow parity" ] && hdr="✅ Shadow parity GATE PASS"

msg="$(printf '%s — %s\n\nYANK: %s\nMIM:  %s\n\n(silent while both just accumulate clean sessions)' \
  "$hdr" "$(date -u '+%Y-%m-%d %H:%MZ')" "$ystat" "$mstat")"
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
