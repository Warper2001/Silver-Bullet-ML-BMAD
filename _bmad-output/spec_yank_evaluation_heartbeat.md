# Spec: YANK Evaluation Heartbeat (v1)

**Date:** 2026-07-06
**Authors:** Winston (architecture, party-mode roundtable) → Amelia (implementation review, corrections W1–W6) → built as corrected
**Motivation:** During the 2026-06-22 → 2026-07-06 quiet period, "no qualifying signals" and "silent signal-path failure" were observationally identical on a real-money bot. Confirming YANK was alive required reading source code and journald. This project's incident ledger (06-07 401-loop, .env loss on reboot, phantom position, pre-06-23 floor-monitor blind spot) shares one signature: **quiet**. The fix is a negative-space heartbeat — absence-of-trades becomes a positive, verifiable signal.

**Design principle:** the heartbeat proves **evaluation**, not liveness. Log-file mtime stays fresh even during a 401-loop (error lines touch the file) — that exact failure burned us on 2026-06-07.

## Prereg scope statement

The S25 freeze covers strategy parameters and gate logic — anything that changes *which trades occur*. This change adds telemetry: attribute writes, counters inside `_poll_and_process` (per Amelia W3 this is **not** call-site-only instrumentation — counters and status fields are maintained inside the polling function), and one exception annotation that **re-raises into the pre-existing catch-all** so control flow is unchanged. No `strategy_config.yaml` values move; no gate logic is touched. Ops instrumentation, out of prereg scope.

## 1. What YANK emits

`logs/yank_heartbeat.json` — single latest-state file, atomically replaced (`tmp` + `os.replace`, same filesystem) once per main-loop iteration (~60s), **in both the market-open and market-closed branches** (Amelia W2: otherwise the ts-staleness check false-pages all weekend). Writer: `src/research/yank_heartbeat.py` (`HeartbeatWriter`), double-firewalled — `write()` never raises, `_write_heartbeat()` wraps again. A broken heartbeat fails **safe** (healthcheck alarms on missing/stale file), never **dangerous** (trader dies).

Fields:

| Field | Purpose |
|---|---|
| `ts` | write time UTC — staleness = loop frozen |
| `loop_seq` | monotonic; proves the event loop turns |
| `pid`, `started_at` | restart detection |
| `market_open` | bot's own `_is_market_open()` view — distinguishes "alive but closed" from "evaluating" (W2) |
| `is_backfill` | suppresses bar-lag alarms during startup replay (sharp edge: backfill spike) |
| `data_source` | `tradestation` / `projectx` — PX-cutover-proof now (W3) |
| `contract` | active symbol; catches roll anomalies for free |
| `last_bar_ts` | newest 1-min bar **processed** — the load-bearing field |
| `bars_new_this_cycle`, `bars_evaluated_total` | evaluation counters |
| `consec_poll_failures` | maintained in-process; healthcheck stays stateless. Incremented on non-200, timeout, PX fetch error; reset on successful fetch |
| `poll_http_status` | TS branch only; `null` on ProjectX (W3) |
| `poll_error` | `"HTTP 401"` / `"timeout"` / `"px: …"` / `null` |
| `detect_errors_this_cycle`, `detect_errors_total` | replaces Winston's per-cycle `detect_enter_ok` bool, ill-defined for multi-bar cycles (W4). `_total` is **sticky until restart** so a 5-min-cadence healthcheck can't miss a self-healed cycle |
| `last_exception`, `last_exception_ts` | truncated repr, tagged `_detect_and_enter:` or `poll_cycle:` |

**Cut from Winston's draft:** `halt_flag` (W6 — healthcheck already reads `data/combine_joint/HALT` directly; a copy can drift), `detect_enter_ok` bool (W4), the standalone `poll_http_status != 200 → WARN` check (redundant with the consecutive counter; status kept for forensics only).

**Corrected from Winston's draft (W1):** "record and re-raise at the run-loop call-site" was impossible — `_poll_and_process` already swallows every exception internally, and re-raising past it would convert a today-survivable detect error into a full trader shutdown (`start_streaming` breaks its loop and calls `stop()` on any escaped exception). As built: the per-bar `_detect_and_enter` call is wrapped in a try/except that annotates the heartbeat counters and re-raises **into `_poll_and_process`'s existing catch-all** — behavior bit-for-bit identical, we only observe on the way through.

## 2. What the healthcheck checks

New section 2b in `tools/combine_ops_healthcheck.py` (read-only, exit 0/1/2, 5-min systemd timer). Only judged while `trader-yank` is systemd-active (a stopped service already alarms via the existing SERVICES check — no double-paging). Pure evaluator `evaluate_heartbeat(hb, now_utc, uptime_secs)` — no filesystem/clock/systemctl inside, so fixture tests drive every branch offline.

| Condition | Threshold | Severity | Gating |
|---|---|---|---|
| file missing/corrupt while service active | after 600s startup grace (systemd `ActiveEnterTimestamp`); unknown uptime → CRITICAL (conservative) | CRITICAL | none |
| `ts` stale | >300s (Amelia: 180s was barely 2 cycles — one slow poll pages; 300s allows one lost + one slow cycle, matches the bot's own `_check_stale` convention) | CRITICAL | none — heartbeat ticks 24/7 |
| `consec_poll_failures` | ≥3 | CRITICAL | none |
| `detect_errors_total` | >0 (single occurrence; a detect exception silently skipped an entry decision and is deterministic per code path — no 3-strike grace) | CRITICAL | none; sticky until restart |
| `last_bar_ts` lag | >300s | CRITICAL | `market_open` && `!is_backfill` && `in_globex_window()` && >300s since Globex reopen |

Globex window (W5 — the existing `SERVICES` tuple windows cannot express it): dedicated `in_globex_window()` = Sun 18:00 ET → Fri 17:00 ET minus the 17:00–18:00 daily break; `secs_since_globex_reopen()` provides reopen grace (first post-break bar isn't instant). **No holiday calendar** — a false page on a half-day costs a human 30 seconds; a holiday module costs maintenance forever.

Known co-fire (documented, accepted): the bot's own `STALE_DATA` halt and the healthcheck's bar-lag CRITICAL both trigger at 300s during RTH — belt + suspenders, one incident, two alarms.

The existing log-mtime check in `SERVICES` stays untouched — defense-in-depth for the case where the heartbeat writer itself is broken.

## 3. Failure modes → detection

| Failure mode | Caught by |
|---|---|
| 401-loop / stale TS token | `consec_poll_failures` ≥3 (previously **undetected**: non-200 was a silent `return`, no counter existed — Fact C) |
| Frozen event loop / deadlock | `ts` stale >300s |
| Data-feed gap (200 OK, no bars) | `last_bar_ts` lag in Globex hours |
| Silent exception in `_detect_and_enter` | `detect_errors_total` >0 + `last_exception` |
| Cold-start failure post-reboot | heartbeat never appears within grace + systemd check |
| Wrong contract after roll | `contract` field (forensic, no alarm yet) |

## 4. Tests (all passing)

- `tests/unit/test_yank_heartbeat.py` — 10 tests: atomic write, firewall never raises, parent-dir creation, datetime serialization, payload field completeness, **heartbeat written in market-closed branch**, **detect exception recorded and swallowed by the existing handler (trader survives)**, consec-failure increment/reset across 401→401→200, timeout counted, backfill flag.
- `tests/unit/test_combine_ops_healthcheck_heartbeat.py` — 26 tests: Globex window boundaries (Sun 17:59/18:01, Fri 16:59/17:01, Sat, daily break), reopen-seconds math (Mon 10:00 → Sun 18:00 open), missing-file grace/past-grace/unknown-uptime, corrupt JSON → treated as missing, ts 299s/301s, fails 2/3, detect single-occurrence CRITICAL, bar-lag alarm + all four suppressions (closed flag, backfill, Saturday, reopen grace), no-bar-yet CRITICAL, healthy single-OK.

## 5. Rollout (human steps — NOT performed by this change)

1. Merge PR; deploy during the 17:00–18:00 ET maintenance break (restart `trader-yank`).
2. Watch one hour: `loop_seq` increments ~60s, `last_bar_ts` tracks the market, healthcheck shows the new OK line.
3. **Negative test:** `mv logs/yank_heartbeat.json /tmp/` for 6+ minutes → confirm Telegram CRITICAL fires → restore. Monitoring is not "working" until it has been seen to alarm.
4. Soak 3 trading days.

## 6. Out of scope

- Phase 2: per-gate rejection reason codes inside `_detect_and_enter` (touches strategy-path code; separate commit, revisit prereg question then).
- Filter-log redesign — stays as-is; the heartbeat makes its silence interpretable.
- MIM-NB port (after YANK soak; MIM has its own loop shape).
- Auto-remediation — the healthcheck pages, humans act.
- Holiday calendar, alert-delivery changes, ProjectX parity checks.
