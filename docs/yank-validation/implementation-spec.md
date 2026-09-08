---
title: YANK deployed validation and research-to-live fidelity
type: feature
created: 2026-09-08
status: in-progress
route: dispatch
baseline_commit: ceac044a5ec4e34435e99534ccbcb9d629185b23
context: []
---

<frozen-after-approval reason="user supplied implementation plan authorizes all four stages">

## Intent

Implement the user's four-stage YANK validation plan in `/root/Silver-Bullet-ML-BMAD-yank-validation`, branch `feat/yank-deployed-validation`. The target is installed TradeStation one-minute signal logic with ProjectX execution. Provide reproducible offline evidence and disabled prospective instrumentation. Keep engineering, data suitability, execution evidence and economics separate; retain `HOLD_VALIDATION` everywhere.

## Boundaries & Constraints

Always preserve main checkout changes, archived outputs, native evidence and frozen P&L. Snapshot the exact main checkout source INCLUDING its uncommitted shadow logger fix. Capture Git identities and file hashes. Installed/on-disk bytes are not authenticated process-loaded bytes. Preserve deployed behavior and record defects; no silent fixes/retuning. Commit each of four stages separately with conventional local commit messages. No push, merge, purchase, requests, auth, service restart, trading change, holdout reads, training or parameter search. Already-inspected 2025 is development evidence only.

## I/O & Edge-Case Matrix

| Scenario | Input/state | Expected behavior |
|---|---|---|
| Timing | partial/future, duplicate, revised, out-of-order, delayed complete event | retain source/receipt evidence; baseline keeps original polling behavior; readiness flags unsuitable timing |
| Coverage | missing minutes, session/DST, contract transitions, insufficient warm-up | explicit attribution and blocked/qualified readiness, no invented bars or default account evidence |
| Lifecycle | pending expiry, active holding, restart/recovery | original counters and state; 240 full opportunities for audit; unknown broker evidence remains unknown |
| Shadow | both existing features plus new capture; logger failure/overflow | separate attributes; disabled default, bounded nonblocking observation, explicit invalid coverage |
| Isolation | offline invocation or attempted network/auth/order/persistence | no real capability; fresh outputs; pin failure closed |

</frozen-after-approval>

## Code Map

- Main source: `/root/Silver-Bullet-ML-BMAD/src/research/yank_streaming_working.py`; main HEAD differs from base and source dirty. Constructor infrastructure must be stubbed before use. `_poll_and_process` accepts label <= clock, ignores completion flags and drops <= watermark; reuse exact source privately to preserve this. Never call `initialize`.
- Dependencies to snapshot: strategy_core.py, config_loader.py, data/models.py, lr_channel.py, ml/regime_detection/lr_channel_detector.py, shadow_parity.py, projectx_bars.py plus execution/recovery source dependencies as needed. Pin model, tier2_threshold.json, lr_regime_config.json, shared YAML, installed `/etc/systemd/system/trader-yank.service`, relevant versions. Never snapshot credentials.
- Installed service: SYMBOL=MNQU26, YANK_CONTRACTS=2, YANK_MAX_GAP_ATR_RATIO=.426, DATA_SHADOW=1, TS mirror and SIM_INVVOL enabled; signal default TradeStation. YAML quantity 5 is superseded at execution. Actual ML threshold tier2_threshold.json=.5; LR JSON threshold stored separately.
- Existing audit `src/research/yank_execution_pilot/{audit,core}.py`, `docs/reports/yank-execution-pilot/final-run1/`; preserve command unchanged. Native root main `data/yank/databento-pilot-20260907`. 19 unassessable = case1/end/0ms plus all 18 cases3–5. Capped extracts insufficient for May28: scan needed native case days from daily snapshot, completed events only. Separate invalid books/native gaps from observed 20:20 pause and 21:00–22:00 closure. No missing scheduled minutes reported; queue inherently unknown.
- Existing minute artifacts sibling `Silver-Bullet-ML-BMAD-yank-minute/data/yank/native-minute-reviewed-a`; verify exact location. Existing `yank_native_minute` builder/replay pins and reports establish 13,440 bars/17,280 coverage rows; do not substitute frozen replay engine for deployed logic.
- Auth imports, module logging, TradeDatabase, state files, equity/canary/filter logs and broker methods need private memory-only substitutes. Clock/config/model/execution state injected; raw response ordering preserved.

## Tasks & Acceptance

- [ ] Stage1 `docs/yank-validation/` and versioned snapshot directory: exact bytes/hashes/provenance, reproducible effective config, historical comparison, detailed input contract covering parsing/polling/revision/session/contract/H1/M15/warm-up/counters/availability. Record unverified provider semantics and process identity.
- [ ] Stage2 `src/research/yank_deployed_validation/gaps.py`, separate CLI, tests and incremental report: inventory all 30 conditions with overlapping gaps for 19; native file/hash/record/event/time, status/recovery and full-window intersections. Narrow native scans from valid reconstruction boundaries when needed. Explicit dispositions (evidence resolved, correction, data acquisition, unobservable). Patch confirmed audit defect separately only if proven; otherwise unchanged outcomes.
- [ ] Stage3 private offline adapter, dataset admission schema, replay CLI, tests and readiness report. Exact snapshotted methods, safe injected infrastructure, causal ordered inputs, deterministic decision traces and fidelity against independently invoked snapshot at equal states. Manifest source/request/hashes/contracts/adjustments/interval/availability/session/coverage/warm-up fields; no silent defaults. Run eligible native 2025 diagnostics twice. Separate source, coverage, warm-up, fidelity, account/risk and execution readiness. Concrete TradeStation 2025 acquisition proposal with explicit contracts, warm-up, request metadata, coverage and cost estimation, no requests.
- [ ] Stage4 separate disabled decision-time hooks/capture format, offline comparator CLI, tests and prospective installation/runbook/rollback proposal. Keep settled parity logger and consumers. Capture safe request identity, hashes, receipt/order/revisions, identities/state/readiness/decisions/intentions; actual observed broker acknowledgements/fills separately by venue. Replay receipt order through adapter; exact discrete/quantity/tick price agreement, separate float diffs; unavailable state/feed differences unassessable. Bounded nonblocking logging, failure/overflow coverage invalidation. No hooks enabled or collection launched.

Given pinned inputs, repeated runs must be byte deterministic. Given all original artifacts, hashes before/after must match. Given all 19 blocked scenarios, each has traceable gap evidence and precise next requirement without inferred fills or upgraded outcomes. Given captured equivalent states, exact decisions/quantities/tick prices reconcile or explain mismatch. New CLIs require explicit manifests/input directories and fresh output directories. Existing commands stay unchanged.

## Implementation Notes

User authorized implementation from the supplied plan, including isolation and local commits; no additional approval needed for this reversible work. Data/provider unknowns are required blockers, not intent questions. Use the existing main `.venv/bin/python`. Keep tracked generated findings compact; large runs outside committed directories. The parent will independently run preexisting regression suites and artifact-preservation checks while implementation proceeds.

## Spec Change Log

## Review Triage Log

## Verification

Run new edge/fidelity/isolation tests and deterministic diagnostic comparisons. Parent verifies existing provenance, audit, native-minute, config, strategy, recovery suites. Verify retained artifact hashes. Final handoff separates completed engineering, blocked evidence and future approved operations.
