---
title: Broker-reconciled daily MIM and YANK record
type: feature
created: '2026-09-19'
status: in-review
route: dispatch
baseline_commit: 523bc692a46b63991fba83709cec646a05eccc5d
review_loop_iteration: 0
context: []
---
<frozen-after-approval>
## Intent
Implement the user's approved plan: independently capture and reconcile the shared ProjectX MIM/YANK account, repair MIM fill attribution, and report accurate operational health. Engineering acceptance needs fixtures and deployed capture; operational acceptance awaits a naturally traded session. No test orders.
## Boundaries & Constraints
Always work in /root/Silver-Bullet-ML-BMAD/.claude/worktrees/broker-daily-audit. Preserve existing CSV headers/history, strategy parameters, order behavior, reset uncertainty, frozen comparison protocol and no interim efficacy analysis. Only authentication and read-only broker endpoints in capture. All reports advisory and local. No ledger rewriting, external messaging, process killing, credential commits or dependency installation. Lead handles merge/deploy/restart separately after verification. Restart MIM only in verified flat window.
## I/O & Edge-Case Matrix
|Scenario|Input/state|Expected behavior|Error handling|
|Fills|Unrelated, partial, duplicate, voided or contradictory records|Exact account/contract/order matching; preserve individual valid fills and actual costs separately from references|Reject contradictions; retain diagnostic evidence|
|Correction|Late broker fee/fill correction|New immutable snapshot/report version|Never overwrite historical evidence|
|Missing|Unknown fees, missing records, failed API, attribution ambiguity|Explicit incomplete status, no invented zero costs|Bounded retries and backoff|
|Flat gate|Broker flat but bots offset, missing/malformed state|No observation start|Explain pending evidence|
|Continuity|Account transition or unexplained balance delta|Interrupt observation continuity|Never infer account-reset epoch|
|Recovery|Restart, active child, finalization, overdue child, orphan|Deduplicated durable capture and accurate health|No automatic killing|
</frozen-after-approval>
## Code Map
- src/research/mim_nb_live.py `_order`, `_log_fill`: first trade bug; capture immutable payload identity before await. `_find_fill_for` and `_find_closing_fill` affect trading; leave unchanged.
- src/research/projectx_auth.py ProjectXAuth caches tokens; reuse in dedicated read-only capture adapter with endpoint allowlist/timeouts; no execution client.
- research/project_goals/__main__.py: existing CLI output creation unsuitable for persistent commands; route new commands separately before existing processing.
- research/project_goals/common.py: input_path excludes holdout, hashing and JSON helpers.
- research/project_goals/audit.py: existing exact attribution checks, no account inferred from accountless CSV.
- research/project_goals/scheduler.py: child_health currently always recovery; process_identity detects reused PIDs; add supervisor identity/heartbeat and finalization lifecycle.
- tools/combine_ops_healthcheck.py: add local audit health finding, no messaging.
- MIM data/mim_nb/state.json has position/cat_stop_id/symbol/saved_at; non-atomic reads must fail closed. YANK logs/active_trade_state.json active has direction and sim_*_order_id; flat is risk-only daily_pnl/daily_halted/last_trading_date. Missing or malformed is unknown. Flat files change only on events: preserve timestamps, prove producer/service account identity and require stable before/after local reads plus repeated broker-flat observations. Never treat net-zero bots as individually flat.
- Both live units currently PROJECTX_ACCOUNT_ID=26556101; explicit account argument/config required. MIM/YANK may use different contracts. YANK ProjectX order IDs appear in logs and active state; capture attributed evidence with hashes, never assign unknown orders by exclusion.
## Tasks & Acceptance
**Execution:**
- [x] src/research/mim_nb_live.py plus a focused pure fill helper: capture submission identity, log only matching valid individual fills, persistent broker-ID dedup with conflicts/voids diagnosed; actual prices/costs separate from strategy references, retain headers.
- [x] research/project_goals/capture.py: singleton persistent 60-second service, cached auth, bounded read requests/backoff; durable immutable account/orders/trades/positions responses, request windows and observed timestamps; content hashes and local attribution/state evidence; restart safety. Capture rolling prior-session corrections, use bounded windows and signal uncertainty for incomplete coverage.
- [x] research/project_goals/daily.py: deterministic reconcile-day from captured evidence; NY session dates, compare filled order quantities, fills, positions, strategy attribution, fees and balance bridge. Observation window starts only with independent flatness, stores observed balance/account/source hashes and remains distinct from reset epoch. Interrupt on identity/balance discontinuity. Aggregate broker P&L less fees without double-counting slippage. Unknown costs/totals remain null/incomplete. Correct zero-trade representation requires complete coverage. Preserve all report versions.
- [x] CLI and research/project_goals/systemd units: capture and reconcile-day; daily 16:10 America/New_York report and 08:30 prior-session refresh, DST/weekend handling and restart catchup. Output under research/project_goals/runs/broker-audit. Explicit allowlisted API endpoints; no orders may be transmitted.
- [x] scheduler.py and tools/combine_ops_healthcheck.py: distinguish supervised running, finalizing, overdue and orphaned state; local freshness/audit findings.
- [x] tests/unit/project_goals and focused MIM fixture tests: cover matrix, deterministic totals/hashes, endpoint restriction, restart/corrections, opposing positions, daily scheduling and health lifecycle.
- [x] research/project_goals/README.md: commands, report statuses, observation-vs-epoch limitations, recovery, deployment units and pending operational acceptance.
**Acceptance Criteria:**
- Given fixed snapshots, when reconciliation repeats, then economics and source hashes are deterministic and corrections preserve earlier versions.
- Given both bots individually flat and no broker orders/positions with matching account evidence, when capture observes stable agreement, then a prospective window starts with observed balance and no reset claim.
- Given missing evidence or unknown fees, when reporting, then status is incomplete and no unsupported complete total appears.
- Given successful tests and verified merge, when lead deploys, then read-only capture runs; traded-session acceptance remains pending until natural evidence exists.
## Implementation Notes
User explicitly authorized implementation/verification/deployed capture from the plan; no renewed plan permission needed. Worktree starts clean from current main HEAD preserving seven local commits; unrelated main changes untouched. Subagent implements/tests only; lead reviews, merges and deploys. Before any git operation check status and ahead/behind counts. Use /root/Silver-Bullet-ML-BMAD/.venv/bin/python and installed formatting tools, no installs. Never read sealed_holdout.
## Spec Change Log
Review hardening retains the approved scope. Preserve endpoint allowlist, immutable snapshots, exact MIM matching, local-only health, and unchanged strategy behavior while closing identity, continuity, attribution and integration-test gaps. No agent-context changes.
## Review Triage Log
Follow-up: intermediate producer-account transitions and next-day order update timestamps were reproduced, fixed, and covered by dedicated tests. MIM actual observer integration and prospective YANK short-trade log capture tests pass. All retained review findings have implementation fixes; immutable evidence retention remains a documented operator responsibility.

| Finding | Verdict | Evidence and resolution |
|---|---|---|
| Blind1 | high | Missing/wrong broker account identities were filtered out before accounting; now require explicit incomplete evidence. |
| Blind2 | high | Missing profitAndLoss was indistinguishable from documented null; preserve unknown gross. |
| Blind3 | high | MIM assignments overwrote earlier ambiguous claims; accumulate all owners. |
| Blind4 | high | Active-state sampling misses fast YANK trades; prospective process/account-bound log cursor supplies exact IDs. |
| Blind5 | medium | HTTP request/response intervals are uncertain balance cutoffs; boundary fills must not prove discontinuity. |
| Blind6 | high | Unbounded continuity included later unrelated days; bound report horizon to session closure. |
| Blind7 | medium | Event receipts prevented automatic late corrections; track changed broker evidence for report refresh. |
| Blind8 | high | Mutable observation stayed STARTED after report interruption; synchronize authoritative health state. |
| Blind9 | medium | Accountless event-only state lacks post-transition producer proof; expose unproven state rather than claiming current provenance. |
| Blind10 | medium | Reports reloaded all local blobs and historical records; cache hydrated blobs and restrict relevant report horizon; immutable retention remains operator-managed. |
| Edge1 | high | Reproduced missing account rows yielding complete zero session; same identity-validation root cause as Blind1. |
| Edge2 | high | Null fillVolume became zero for filled orders; missing filled quantity must be unknown. |
| Edge3 | high | Attribution conflict could disappear on later MIM evidence; same root cause as Blind3. |
| Edge4 | medium | Accepted numeric-string size reached integer addition; normalize validated execution quantities. |
| Edge5 | medium | JSON null/list YANK state raised during attribution; validate mapping before reading IDs. |
| Edge6 | high | Report and live observation disagreed; same root cause as Blind8. |
| Edge7 | medium | 401 retries reused cached rejected token; invalidate it before bounded reauthentication. |
| Edge8 | high | Future evidence invalidated historical sessions; same root cause as Blind6. |
| Edge9 | medium | Schedule receipts lacked account identity; account-specific receipts required. |
| Edge10 | high | Missing gross P&L was zero; same root cause as Blind2. |
| Edge11 | high | Filled quantities did not prove signed contract inventory or order-side agreement; add inventory and side reconciliation. |
| Verification1 | high | No nonempty report fixture exercised actual trade ingestion; add complete MIM/YANK traded-session and mismatch fixtures. |
| Verification2 | high | Pure helper tests missed disconnected observer scheduling; new AST-isolated actual _order/_log_fill test verifies artifacts and identity changes. |
| Verification3 | high | Real adapter failure validation untested; add MockTransport failures/malformed envelopes/401 recovery fixtures. |
## Verification
Engineering fixtures and static service validation passed. Deployment and natural traded-session acceptance are recorded separately in the deployment note. Existing Pydantic deprecation warnings are unrelated.
Run focused new tests plus tests/unit/project_goals and relevant existing MIM tests. systemd-analyze verify units. Lead verifies actual capture, read-only request inventory, local report and flat gate before MIM restart. No strategy research test, so statistical power gate is inapplicable.
