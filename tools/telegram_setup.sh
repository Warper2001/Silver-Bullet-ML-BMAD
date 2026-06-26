#!/usr/bin/env bash
# One-shot Telegram setup for combine ops alerts.
#
# Prereq (you do this in the Telegram app, ~2 min):
#   1. Message @BotFather -> /newbot -> follow prompts -> copy the HTTP API token.
#   2. Open YOUR new bot and send it any message (e.g. "/start"). This is required:
#      Telegram won't reveal your chat_id until you've messaged the bot first.
#
# Then run:  tools/telegram_setup.sh <BOT_TOKEN>
# It discovers your chat_id via getUpdates, writes .env.telegram (gitignored, 0600),
# and sends a confirmation message so you can verify delivery end-to-end.
set -uo pipefail

BASE="/root/Silver-Bullet-ML-BMAD"
PY="$BASE/.venv/bin/python"
ENVFILE="$BASE/.env.telegram"

TOKEN="${1:-${TELEGRAM_BOT_TOKEN:-}}"
if [ -z "$TOKEN" ]; then
  echo "usage: tools/telegram_setup.sh <BOT_TOKEN>   (or set TELEGRAM_BOT_TOKEN)" >&2
  exit 2
fi

echo "Discovering chat_id from getUpdates …"
updates="$(curl -s --max-time 15 "https://api.telegram.org/bot${TOKEN}/getUpdates")"

chat_id="$(printf '%s' "$updates" | "$PY" -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    print(""); sys.exit()
if not d.get("ok"):
    print(""); sys.exit()
ids = []
for u in d.get("result", []):
    msg = u.get("message") or u.get("edited_message") or u.get("channel_post") or {}
    c = (msg.get("chat") or {}).get("id")
    if c is not None:
        ids.append(c)
print(ids[-1] if ids else "")
')"

if [ -z "$chat_id" ]; then
  echo "✗ No chat_id found." >&2
  if printf '%s' "$updates" | grep -q '"ok":false'; then
    echo "  The bot token looks invalid — getUpdates returned ok:false:" >&2
    printf '  %s\n' "$updates" >&2
  else
    echo "  Send a message to your bot in the Telegram app first (e.g. /start), then re-run." >&2
  fi
  exit 1
fi

echo "✓ chat_id = $chat_id"
umask 077
cat > "$ENVFILE" <<EOF
# Telegram alert credentials for tools/combine_ops_alert.sh (auto-generated $(date -u +%Y-%m-%dT%H:%M:%SZ))
TELEGRAM_BOT_TOKEN=$TOKEN
TELEGRAM_CHAT_ID=$chat_id
EOF
chmod 600 "$ENVFILE"
echo "✓ wrote $ENVFILE (chmod 600, gitignored)"

echo "Sending test message …"
resp="$(curl -s --max-time 15 \
  --data-urlencode "chat_id=${chat_id}" \
  --data-urlencode "text=✅ Combine ops alerts wired to Telegram. You'll get a ping on WARN/CRITICAL/RECOVERED." \
  "https://api.telegram.org/bot${TOKEN}/sendMessage")"
if printf '%s' "$resp" | grep -q '"ok":true'; then
  echo "✓ Test message sent — check your phone."
else
  echo "✗ Test send failed: $resp" >&2
  exit 1
fi
