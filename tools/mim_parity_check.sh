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

# ---------------------------------------------------------------------------
# Telegram push — the verdict, not the whole report.
#
# Same contract as tools/combine_ops_alert.sh: credentials live in .env.telegram
# (gitignored), and the push is a silent no-op when they are absent, so this script
# stays runnable on a machine without them. Report-only — it never touches the bot.
# ---------------------------------------------------------------------------
[ -f "$BASE/.env.telegram" ] && . "$BASE/.env.telegram"

notify_telegram() {
  [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ] || {
    echo "telegram: no credentials — report-only run"; return 0; }
  # Check the API's own ok flag, not just curl's exit status: Telegram answers a bad
  # chat_id or revoked token with HTTP 400 and {"ok":false}, which curl reports as
  # success. A push that silently never arrives is worse than no push.
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
VERDICT=$(grep -E 'VERDICT:' "$OUT" | head -1 | trim)
MARKS=$(grep -E 'marks compared:' "$OUT" | head -1 | trim)
WSIGMA=$(grep -F 'worst |dsigma|' "$OUT" | head -1 | trim)
WBAND=$(grep -F 'worst |dband|' "$OUT" | head -1 | trim)
M1600=$(grep -E '^[[:space:]]+16:00[[:space:]]' "$OUT" | head -1 | trim)
NOTE=""
grep -q 'A FAIL before then is expected' "$OUT" && \
  NOTE=$'\nexpected FAIL: 07-29/07-31 contaminated sessions sit in the window until ~2026-08-20'

MSG="MIM-NB parity — ${DAY}
${VERDICT:-VERDICT: (not produced)}
${MARKS:-marks compared: n/a}
${WSIGMA:-worst |dsigma|: n/a}
${WBAND:-worst |dband|: n/a}
16:00 mark: ${M1600:-not evaluated}${NOTE}
report: logs/parity/parity_${DAY}.txt"

notify_telegram "$MSG"
echo "--- telegram payload ---"
echo "$MSG"
