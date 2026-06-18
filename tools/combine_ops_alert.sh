#!/usr/bin/env bash
# Combine ops alerting wrapper (LOCAL-ONLY, no external push — choice 2026-06-18).
#
# Runs tools/combine_ops_healthcheck.py and:
#   - always prints to stdout (captured by the systemd journal: journalctl -u combine-ops-healthcheck)
#   - always refreshes a current-status snapshot (data/combine_joint/ops_status.txt)
#   - on WARN/CRITICAL, appends to logs/combine_ops_alerts.log — but only on a STATE CHANGE
#     (dedup), so a persistent condition logs once, not every 5 minutes for weeks.
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
fi

# Record recovery transitions too (failing -> all-clear)
if [ "$rc" -eq 0 ] && [ -n "$last" ]; then
  { echo "===== $ts  RECOVERED (exit 0) ====="; echo "$out"; echo; } >> "$ALERT_LOG"
fi

printf '%s' "$sig" > "$SIGFILE"
exit 0
