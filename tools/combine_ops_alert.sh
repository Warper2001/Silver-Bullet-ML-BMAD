#!/usr/bin/env bash
# Combine ops alerting wrapper (local log + journal + optional Telegram push).
#
# Runs tools/combine_ops_healthcheck.py and:
#   - always prints to stdout (captured by the systemd journal: journalctl -u combine-ops-healthcheck)
#   - always refreshes a current-status snapshot (data/combine_joint/ops_status.txt)
#   - on WARN/CRITICAL, appends to logs/combine_ops_alerts.log — but only on a STATE CHANGE
#     (dedup), so a persistent condition logs once, not every 5 minutes for weeks.
#   - on that same STATE CHANGE (and on RECOVERED), pushes a Telegram message IF
#     TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are set (sourced from .env.telegram, gitignored).
#     No creds -> silently skips Telegram; the local log/journal path is unaffected.
#   - throttles WARN and RECOVERED Telegram pushes to at most one per PUSH_COOLDOWN
#     (default 3600s) PER condition, so a flapping check can't spam the phone. CRITICAL
#     pushes ALWAYS bypass the throttle (a money alert must never be muted), and the
#     local alert log records every transition regardless of whether a push was sent.
#
# Exits 0 always: the alert signal is the log/journal, not the exit code, so the
# systemd timer unit never accumulates a "failed" state that would itself need triage.
set -uo pipefail

BASE="/root/Silver-Bullet-ML-BMAD"
PY="$BASE/.venv/bin/python"
# Paths are env-overridable (defaults below) so the alerting branches can be tested
# against scratch files without polluting the real alert log.
: "${ALERT_LOG:=$BASE/logs/combine_ops_alerts.log}"
: "${STATUS:=$BASE/data/combine_joint/ops_status.txt}"
: "${SIGFILE:=$BASE/data/combine_joint/.ops_alert_sig}"
# Push throttle: at most one Telegram push per category per PUSH_COOLDOWN seconds.
: "${PUSH_COOLDOWN:=3600}"
: "${PUSHLOG:=$BASE/data/combine_joint/.ops_push_log}"

# Optional Telegram credentials (gitignored). Absent file => local-only, no error.
[ -f "$BASE/.env.telegram" ] && . "$BASE/.env.telegram"

notify_telegram() {
  # $1 = message text. No-op unless both creds are present.
  [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ] || return 0
  curl -s --max-time 15 \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=$1" \
    --data "disable_web_page_preview=true" \
    "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" >/dev/null \
    || echo "WARN: telegram send failed (alert is still in $ALERT_LOG)" >&2
}

should_push() {
  # $1 = category key. Returns 0 (allow) if this category has not been pushed within
  # PUSH_COOLDOWN seconds, else 1 (skip). On allow, records the push time and keeps the
  # history file bounded. Distinct conditions use distinct keys, so a genuinely new alert
  # is never suppressed by an unrelated one. Used only for WARN/RECOVERED — never CRITICAL.
  local cat="$1" now last
  now="$(date +%s)"
  last="$(awk -F'\t' -v c="$cat" '$2==c{v=$1} END{print v}' "$PUSHLOG" 2>/dev/null)"
  if [ -n "$last" ] && [ "$((now - last))" -lt "$PUSH_COOLDOWN" ]; then
    return 1
  fi
  printf '%s\t%s\n' "$now" "$cat" >> "$PUSHLOG"
  tail -n 200 "$PUSHLOG" > "$PUSHLOG.tmp" 2>/dev/null && mv "$PUSHLOG.tmp" "$PUSHLOG"
  return 0
}

ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
# Extra args ("$@") pass through to the healthcheck (e.g. --max-stale for a forced test).
out="$("$PY" "$BASE/tools/combine_ops_healthcheck.py" "$@" 2>&1)"; rc=$?

# Journal + always-current snapshot
echo "$out"
echo "exit=$rc"
{ echo "# combine ops status @ $ts (exit $rc)"; echo "$out"; } > "$STATUS"

# Dedup signature = the set of failing lines with variable numerics stripped (ages,
# equity, distances jitter every run). Collapses "stale 25s"/"stale 26s" to one signature
# so a persistent condition logs ONCE; a level/category change (different text) re-fires.
# Live numbers still reach the journal + ops_status.txt snapshot.
sig="$(printf '%s\n' "$out" | grep -E 'CRIT!|WARN' | sed -E 's/[0-9][0-9.,]*//g' | sort || true)"
last="$(cat "$SIGFILE" 2>/dev/null || true)"

if [ "$rc" -ne 0 ] && [ "$sig" != "$last" ]; then
  level="WARN"; [ "$rc" -ge 2 ] && level="CRITICAL"
  { echo "===== $ts  $level (exit $rc) ====="; echo "$out"; echo; } >> "$ALERT_LOG"
  # Only the failing lines in the push (keep it phone-readable); full detail is in the log.
  fails="$(printf '%s\n' "$out" | grep -E 'CRIT!|WARN' || true)"
  msg="$(printf '🛑 Combine ops %s (%s)\n%s\n\nsnapshot: data/combine_joint/ops_status.txt' "$level" "$ts" "$fails")"
  # CRITICAL is never throttled. WARN is rate-limited per condition (keyed on the dedup
  # signature) so a flapping check can't spam the phone.
  if [ "$level" = CRITICAL ]; then
    notify_telegram "$msg"
  elif should_push "warn-$(printf '%s' "$sig" | md5sum | cut -d' ' -f1)"; then
    notify_telegram "$msg"
  else
    echo "telegram push throttled (WARN within ${PUSH_COOLDOWN}s cooldown; logged to $ALERT_LOG)" >&2
  fi
fi

# Record recovery transitions too (failing -> all-clear)
if [ "$rc" -eq 0 ] && [ -n "$last" ]; then
  { echo "===== $ts  RECOVERED (exit 0) ====="; echo "$out"; echo; } >> "$ALERT_LOG"
  recov_msg="$(printf '✅ Combine ops RECOVERED (%s) — all checks OK.' "$ts")"
  # The all-clear for a CRITICAL is itself a money-relevant message — never throttle it.
  # Only recoveries from a WARN-only state are rate-limited (the flapping-noise case).
  if printf '%s' "$last" | grep -qF 'CRIT!'; then
    notify_telegram "$recov_msg"
  elif should_push recovered; then
    notify_telegram "$recov_msg"
  else
    echo "telegram push throttled (RECOVERED within ${PUSH_COOLDOWN}s cooldown; logged to $ALERT_LOG)" >&2
  fi
fi

printf '%s' "$sig" > "$SIGFILE"
exit 0
