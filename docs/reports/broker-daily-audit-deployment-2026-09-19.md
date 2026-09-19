# Broker daily audit deployment — 2026-09-19

Engineering acceptance: implemented, reviewed, merged and capture deployed. Operational acceptance remains pending.

Implementation commit: `c9ffc2f4ccd476106fa947b2e6dc4da6ae0c21e2`. The follow-up `84ccc2123a9dcd184de0cedf99230d19069d8621` prevents historical refreshes from replacing current health reasons or moving the latest-session pointer backward. Both were fast-forward merged into main from `.claude/worktrees/broker-daily-audit`; the deployment was installed from main. Existing unrelated working-tree changes were retained.

## Verification

- 275 relevant worktree tests passed: project-goals audit fixtures, all MIM unit suites, ProjectX commingling and YANK recovery. After the initial merge, 123 audit tests passed from main; after the final reporting fix, all 36 affected reporting/lifecycle tests passed again from main.
- Independent blind, edge-case and verification-gap reviews covered identity loss, ambiguous attribution, incomplete costs, signed inventory, late corrections, producer transitions, HTTP timing and adapter failure paths. Reproduced findings were fixed with regression fixtures. New code passed targeted flake8; the unit passed `systemd-analyze verify`, Python compilation and shell syntax checks passed. Existing Pydantic deprecation warnings remain.
- The actual MIM `_order` and `_log_fill` methods were tested using an isolated AST fixture and real `ChainedCsv`, proving captured identity survives mutable account/contract changes, individual partial fills remain separate, repeated reads deduplicate, invalid/void/conflicting records do not append, headers remain intact, and reference prices never enter market-order payloads.
- Capture singleton refusal and restart receipt recovery were tested. A deployed capture restart preserved its snapshots and schedule receipts. Prospective YANK log-cursor fixtures cover short trades, account transitions, partial lines, bounded gaps and replay after a crash before cursor publication.

## Deployed state

`project-goals-broker-capture.service` is enabled and active for explicit account `26556101`, polling every 60 seconds. It uses cached authentication and the allowlisted read-only account, order-search, trade-search and position-search endpoints. Daily local reports run at 16:10 America/New_York; previous-session refreshes run at 08:30, with restart catch-up and changed-evidence correction refreshes.

At 2026-09-19 23:46:14 UTC, all 19 requests in snapshot `d29fdc076728a2396abbcae1172ebb101d9323f103d6139b9dfc22a7f9cc3c99` succeeded and its content hash verified. Broker positions and open orders were both empty. Artifacts reside under `research/project_goals/runs/broker-audit/`; the mutable index is `latest_report.json`, with immutable reports under `reports/<date>/<hash>.json`.

MIM was restarted after merge verification and a fresh broker-flat check. The check also confirmed individually flat persisted states, matching live producer accounts, unchanged state hashes, and no broker orders/fills since either current producer started. The immutable restart proof is `deployment/2ee625ea1a3aa2acf2c54eb26e1ad490c4945f7e09fe7a80bca91e17cc97eefa.json` beneath that audit directory. MIM changed from PID 1194607 to 1257423; its 23:45:21 UTC startup log confirmed its own state was flat and the polling loop started. YANK remained PID 1174576. Post-restart capture again observed no positions or open orders. No test trade was placed.

The frozen collector health query reported `SUPERVISED_POLLING`, a living child and supervisor, and no operator recovery requirement. The existing operations healthcheck printed the advisory `[LOCAL AUDIT]` record; that line is excluded from optional external notifications and does not alter trading-alert severity.

## Remaining evidence gates

The prospective window is **PENDING_FLAT_EVIDENCE**, with `MIM:state_predates_producer` and `YANK:state_predates_producer`. Their event-only state files predate current processes. Broker-flat restart safety evidence does not invent state-publication provenance or an account-reset epoch. A natural state update must supply current producer evidence before the automated window starts; no file was rewritten to force acceptance.

The 2026-09-18 report is **INCOMPLETE** because capture began later and cannot prove opening/closing inventory or an earlier observation window. It does not certify that day as a complete zero-trade session. Its current version is `77610cfa5aaa099871f76ed0ef24fa32f2b6443b67d5c3256d49696a44ccbb0c`.

Operational acceptance requires a naturally traded session with complete broker quantities, attribution, costs and balance reconciliation. Unknown commissions or missing gross P&L remain unknown. Historical reconstruction, account-reset provenance, SIM attribution and written provider confirmation remain separate dependencies. Capture and reports are advisory and do not control orders or rewrite historical ledgers. Evidence retention has no automatic deletion; storage remains an operator-managed dependency.

The broader operations healthcheck still reports unrelated existing warnings: S26-combine API distress, inactive SIL capture, floor headroom and historical ledger-chain defects. Those were not repaired by this audit work; no ledger history was rewritten.

Provider references checked during implementation: [ProjectX trade search](https://gateway.docs.projectx.com/docs/api-reference/trade/trade-search/), [order search](https://gateway.docs.projectx.com/docs/api-reference/order/order-search/), and [Topstep API rules](https://help.topstep.com/en/articles/11187768-topstepx-api-access). Written account-specific confirmation remains outstanding.
