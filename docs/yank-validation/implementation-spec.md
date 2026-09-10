---
title: YANK deployed validation and research-to-live fidelity
type: feature
created: 2026-09-08
status: done
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

- [x] Stage1 `docs/yank-validation/` and versioned snapshot directory: exact bytes/hashes/provenance, reproducible effective config, historical comparison, detailed input contract covering parsing/polling/revision/session/contract/H1/M15/warm-up/counters/availability. Record unverified provider semantics and process identity.
- [x] Stage2 `src/research/yank_deployed_validation/gaps.py`, separate CLI, tests and incremental report: inventory all 30 conditions with overlapping gaps for 19; native file/hash/record/event/time, status/recovery and full-window intersections. Narrow native scans from valid reconstruction boundaries when needed. Explicit dispositions (evidence resolved, correction, data acquisition, unobservable). Patch confirmed audit defect separately only if proven; otherwise unchanged outcomes.
- [x] Stage3 private offline adapter, dataset admission schema, replay CLI, tests and readiness report. Exact snapshotted methods, safe injected infrastructure, causal ordered inputs, deterministic decision traces and fidelity against independently invoked snapshot at equal states. Manifest source/request/hashes/contracts/adjustments/interval/availability/session/coverage/warm-up fields; no silent defaults. Run eligible native 2025 diagnostics twice. Separate source, coverage, warm-up, fidelity, account/risk and execution readiness. Concrete TradeStation 2025 acquisition proposal with explicit contracts, warm-up, request metadata, coverage and cost estimation, no requests.
- [x] Stage4 separate disabled decision-time hooks/capture format, offline comparator CLI, tests and prospective installation/runbook/rollback proposal. Keep settled parity logger and consumers. Capture safe request identity, hashes, receipt/order/revisions, identities/state/readiness/decisions/intentions; actual observed broker acknowledgements/fills separately by venue. Replay receipt order through adapter; exact discrete/quantity/tick price agreement, separate float diffs; unavailable state/feed differences unassessable. Bounded nonblocking logging, failure/overflow coverage invalidation. No hooks enabled or collection launched.

Given pinned inputs, repeated runs must be byte deterministic. Given all original artifacts, hashes before/after must match. Given all 19 blocked scenarios, each has traceable gap evidence and precise next requirement without inferred fills or upgraded outcomes. Given captured equivalent states, exact decisions/quantities/tick prices reconcile or explain mismatch. New CLIs require explicit manifests/input directories and fresh output directories. Existing commands stay unchanged.

## Implementation Notes

User authorized implementation from the supplied plan, including isolation and local commits; no additional approval needed for this reversible work. Data/provider unknowns are required blockers, not intent questions. Use the existing main `.venv/bin/python`. Keep tracked generated findings compact; large runs outside committed directories. The parent will independently run preexisting regression suites and artifact-preservation checks while implementation proceeds.

## Spec Change Log

## Review Triage Log

All three requested review lenses completed before triage. The third lens ran after a reviewer slot became free. Findings below were independently assessed before duplicate grouping; no finding changes the preserved strategy or frozen audit.

| ID | Finding | Individual disposition and evidence |
|---|---|---|
| B1 | Active shadow checkpoint restores nested decision as dict | patch: subsequent original exit code accesses decision attributes; restore original dataclass and test continuation |
| B2 | Checkpoint omits combine routing flag | patch: original exit routing branches on this flag; serialize and restore it |
| B3 | Failed HTTP polls disappear from capture | patch: non-200 and timeout paths are normal observable outcomes; retain explicit failure envelopes |
| B4 | Comparator invents scheduler time | patch: captured poll has already crossed scheduler boundary; replay that boundary without an invented scheduler call |
| B5 | Missing observed broker reply defaults to success | patch: require complete observed reply sequence; synthetic defaults remain explicitly synthetic |
| B6 | Missing/non-numeric price crashes comparison | patch: report exact structural/type mismatch |
| B7 | All floating differences still report MATCH | patch: exact risk/config numerics; separate feature differences with qualified discrete-match result |
| B8 | Container expansion before payload bound | patch: enforce node/container bounds before traversal/copy |
| B9 | Arbitrary account hash treated as evidence | patch: declared account evidence remains unverified unless bound and verified; cannot authorize observed equivalence |
| B10 | Reference labels treated as loaded runtime identity | patch: distinguish proven private snapshot identity from unverified live declaration; missing evidence makes comparison unassessable |
| E1 | Active shadow nested restore (independent finding) | patch: confirmed same continuation failure as B1, grouped only after this assessment |
| E2 | Oversized container expands before bound | patch: confirmed same allocation path as B8 |
| E3 | Recursive normalization precedes bounds | patch: separate pre-serializer path also unbounded; bounded normalization required |
| V1 | No successful native conversion verification | patch: valid native replay fixture must assert exact price conversion, upward receipt rounding and stable availability ties |
| V2 | Installed observer test never executes orders | patch: order-producing installed-observer roundtrip with captured replies and missing-reply negative control |

Additional findings P2–P4: capture shutdown could race with an in-flight serializer (patch: active-producer coverage invalidation and no late enqueue); live decision dataclass names needed bounded primitive normalization (patch: known shape test); label convention and HTTP hash scope needed explicit evidence (patch: conditional minute intervals and bounded raw-response hash, separate canonical Bars hash). These are observational tool changes only. All reviewed patches are covered by targeted tests; identity/account authentication remains an explicit validation-evidence blocker, not a falsely completed attestation feature.

Parent additional finding P1: the exclusive development interval rejected midnight 2026-01-01, which is a valid exclusive boundary for 2025. Patch admission to permit that boundary while rejecting any receipt/label outside admitted development coverage; add boundary tests. This does not admit or read holdouts.

## Verification

Run new edge/fidelity/isolation tests and deterministic diagnostic comparisons. Parent verifies existing provenance, audit, native-minute, config, strategy, recovery suites. Verify retained artifact hashes. Final handoff separates completed engineering, blocked evidence and future approved operations.


Final independent verification: 58 new tool tests pass within the 462-pass combined suite; the sole failure is the pre-existing SL5 expectation against pinned SL2 YAML. The separate frozen replay suite passes 133 tests. Two complete native runs produce identical full traces and reports across 13,440 polls with zero processing errors. The retained order-producing capture reconciles decisions/state exactly while correctly blocking unverified account evidence. Original source/data/artifact preservation checks pass for 177 unique files; main checkout Git status is identical. Complete evidence and final source hashes are in `docs/reports/yank-deployed-validation/verification.json`.

All four engineering stages are complete. Missing TradeStation timing, authenticated loaded runtime/account state, preinterval warm-up and historical/observed execution evidence remain explicitly blocked. Future acquisition, attestation integration and enabled observation are proposals only. No economic validation or trading authorization is inferred. Retained test logs/XML preserve their original whitespace as evidence; source/document whitespace checks pass.
