---
title: YANK frozen-order Databento execution pilot
type: feature
created: 2026-09-07
status: done
route: dispatch
baseline_commit: c4156121a63ad8c27d448459006739c4841af95b
context: []
---
<frozen-after-approval>
## Intent
Implement the user's approved conservative evidence audit of four frozen orders per archived arm against purchased MNQM5 May 19–30, 2025 events. Produce reproducible traceable execution findings, including the May 28 same-bar fill/stop ambiguity. Preserve signals, quantities, prices and strategy settings. The user explicitly requested implementation and verification of this plan in a fresh context.

## Boundaries & Constraints
Always read only the pinned acquisition and archived replay inputs, verifying hashes. Stream DBN in bounded memory, preserve native record order and completed event boundaries. Initialize books with synthetic snapshots; synthetic timestamps are never trade observations or arrival timestamps. Retain capture time as observable timing proxy and exchange timestamps as diagnostics. Distinguish T trade prints from F fill notifications; neither updates resting book (C does). Incomplete snapshots, invalid books, unsupported records, invalid timestamps and unavailable evidence are reported gaps. No network/API keys/live trader imports in audit command.

Reconcile native T minute OHLC against original bars under start-labelled and end-labelled interpretations. Report coverage and differences without shifting prices or choosing favorable timing. Retain both conditional interpretations unless independent provenance resolves them. Place short limits after signal completion with fixed extra delays 0,100,500ms. Pending lifetime comprises the next 240 actual scheduled bar opportunities from the frozen original bars, including the last opportunity, bounded by purchased interval. It is not 240 wall-clock minutes and must not stop at the historical assumed fill.

Report arrival spread, displayed size, marketability, first touch, first strictly higher trade, and volume at/through limit. Insufficient displayed size must be explicit. Touch is inconclusive. Strict trade-through is supporting evidence only under no impact assumption, not proof of fill/queue; never invent partial fills. Outcomes supported/touch-only/unsupported/unassessable must reflect gaps and coverage. Status/halts affect assessability. May 28 timelines include entry evidence and subsequent stop/target crossings in each interpretation of archived fill/exit interval. Same-event or equal-time unresolved ordering stays ambiguous. Historical P&L is reference only; historical $4 cost is not a verified broker charge. No revised return, propagation of fills, tuning, acquisition, subscriptions or automation.

## I/O & Edge-Case Matrix
| Scenario | Input | Expected behavior |
|---|---|---|
| Valid snapshot/events | R/A snapshot followed by A/M/C/R,T,F,N and F_LAST | Complete book only at event boundary; trades and fills separated |
| Invalid evidence | Incomplete snapshot, malformed updates, timestamps, unsupported flags/records, halt or missing interval | Explicit evidence gap; no fabricated certainty |
| Timing | Two label conventions and 0/100/500ms | Signal completed first; exact boundaries conservative |
| Execution | Touch, through, insufficient size, expiry | Distinct conservative evidence findings; no partial fills |
| Ordering | Entry before stop; stop before possible entry; equal time or event | Source timeline; unresolved ordering labelled ambiguous |
| Integrity | Changed/missing input or implementation failure | Nonzero exit; no PASS |
| Repeat run | Same pinned inputs and code | Identical canonical report bytes and hashes |
</frozen-after-approval>

## Code Map
- Current repository `/root/Silver-Bullet-ML-BMAD`: add isolated `src/research/yank_execution_pilot/` package, `src/cli/check_yank_execution_pilot.py`, poetry script, focused tests and audit docs. Existing workspace is dirty; preserve all unrelated work. No need to port/import historical replay code.
- `data/yank/databento-pilot-20260907/manifest.json`: hashes of acquisition files. Native daily DBN files under native job dirs, three schemas mbo/definition/status, dataset GLBX.MDP3 instrument 42009475 MNQM5. `.venv/bin/python` has databento 0.85.0. Pin manifest's own hash in code/input lock to prevent substituted acquisition. Verify every consumed input. Native files total ~3.9GB compressed; process efficiently, no full-file dataframe.
- Sibling `/root/Silver-Bullet-ML-BMAD-yank-replay/docs/reports/yank-signals/development-run1/manifest.json`: artifact hashes and declared_manifest.development_data (path and hash). Pin this manifest too. Event gz files `no-ml-events.jsonl.gz`, `ml050-events.jsonl.gz` contain ORDER/FILL/EXIT/EXPIRE kinds with order_id, sequence, timestamp and exact economics. Select ORDER signal times within purchased interval, require four per arm (five unique cases: three common, no-ml id4 signal May28 19:52Z and ml050 id4 signal May28 20:02Z); dedup economics/time while retaining arm-specific source refs and historical events.
- Current main worktree `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv` (absent sibling): original bars, manifest hash 3f20ec70885cdee6b48e6c5c7ed3254dd4cc8ce7bd8533696c5e461c75fb7822.
- Sibling `src/research/yank_signals/engine.py`: advance increments pending bars and checks high >= entry BEFORE age >=240; run calls advance before detect. Use this only as provenance, no import. Order 4 May28 19:52Z, fill/exit 20:22Z, entry21408 stop21440.5 target21278 quantity-5, historical ambiguity SAME_BAR_FILL_EXIT_ORDER_UNKNOWN. Another May28 order 3 signal18:53Z fill18:58Z exit19:09Z.
- Vendor https://databento.com/docs/standards-and-conventions/mbo-snapshot and https://databento.com/docs/examples/order-book/order-tracking: read conventions. Snapshot starts R then A, F_SNAPSHOT32|BAD_TS_RECV8; F_LAST128. Snapshot may require next live F_LAST before usable. T/F/N no book changes. Cancel subtracts size, modify replaces price/size. Inspect only complete F_LAST events.

## Tasks & Acceptance
- [x] Audit package and local-only CLI `check_yank_execution_pilot --output-dir` -- implement all above, bounded-memory processing, stable canonical JSON/Markdown/event extracts/reconciliation diagnostics and input/code hash report. Output fresh dirs only, reject writing into source trees. Include decoder version; omit wall-clock run timestamps and output-dir-dependent data from canonical reports.
- [x] Focused meaningful unit tests -- cover every matrix row and deterministic chunk processing.
- [x] Purchased-data verification -- run twice fresh directories, compare all canonical bytes/hashes, verify archived inputs unchanged, one finding per case and May28 timeline.
- [x] Existing regression tests -- run sibling's 133 replay/accounting/policy tests without modifying replay.

Given pinned inputs, when command runs, then report separates PASS_AUDIT_CHECKS from HOLD_VALIDATION and yields every conditional case finding without revised strategy returns. Given corrupt input, when run, then exit nonzero. Given unchanged inputs/code, when run twice, then canonical artifacts match exactly.

## Implementation Notes
No intent gaps or irreversible actions. Parent will independently run existing sibling regression tests and verify/review produced implementation. Implementation agent owns package, CLI, focused tests and outputs, and should execute twice efficiently. Do not stage unrelated files.

## Verification
- `.venv/bin/python -m pytest tests/unit/yank_execution_pilot -q`
- Sibling cwd: `/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/yank_signals tests/unit/yank_replay tests/unit/test_strategy_core_consistency.py tests/unit/test_strategy_core_scaling.py -q` (parent owns this run).
- `.venv/bin/python -m src.cli.check_yank_execution_pilot --output-dir docs/reports/yank-execution-pilot/run1` and run2; identical canonical artifacts.

## Spec Change Log

## Review Triage Log

## Independent review triage (2026-09-07)
Every finding was checked against callers and current code; implementation fixes preserve approved intent.

| Finding | Verdict | Evidence / disposition |
|---|---|---|
| Blind1 regression diagnostics depend on chunk | medium | Vector branch counts one per chunk and wrong location; confirmed independently by edge reviewer. Emit per-record gaps once. |
| Blind2 invalid-event reconciliation contamination | medium | Aggregate runs before book validity. Preserve raw observed T OHLC but explicitly mark contaminated minutes and reasons. |
| Blind3 non-case stateless order fields unchecked | medium | Inactive path skips Book.apply. Validate A/M/C side/price/size globally. |
| Blind4 minute-bucket gap overlap | medium | Inclusive minute comparison includes outside-lifetime gaps. Track precise timestamps. |
| Blind5 event-end-only coverage | medium | Trade can belong to previous minute, ending after expiry. Credit validated live record minutes on event completion. |
| Blind6 timeline constrained by pending window | medium | Same outer conditional filters timelines. Process windows independently of pending lifetimes. |
| Blind7 timeline gaps do not qualify ordering | medium | Gaps stored but ordering only checks crossing endpoints. Qualify entire timeline interval. |
| Blind8 early PASS publication | high | Reports precede final input verification; KeyboardInterrupt is not caught. Stage/publish only after integrity checks and cleanup interruption. |
| Verification1 Scenario.result coverage missing | medium | No test exercised assembled final result with status/native/arrival/coverage gaps. Add boundary-level cases. |
| Verification2 reconciliation mapping untested | medium | Aggregate tests cannot catch end-label offset removal. Assert emitted mappings and counts. |
| Verification3 final-check cleanup untested | high | No test invokes run with second verification failure. Add failure/interruption artifact assertions. |
| Edge1 regression diagnostics | medium | Same verified root cause as Blind1; test multiple malformed chunk sizes. |
| Edge2 expiry-minute gap contamination | medium | Same root cause as Blind4; assert exact arrival and expiry exclusions. |
| Edge3 event coverage attribution | medium | Same root cause as Blind5; test assembled final outcome too. |
| Edge4 interrupted PASS publication | high | Same root cause as Blind8; test KeyboardInterrupt. |

## Implementation Notes (review)
The full-data trial was intentionally interrupted before any PASS artifact to apply these findings. Keep frozen cases, economic fields, both timing conventions, native ordering, and all prior tests. A sorted price-level map is warranted to avoid repeated full-level scans at each of millions of completed events; this changes lookup cost only. Final two runs must use final code and fresh directories.

Final-data inspection caught SDK StatusMsg Optional[bool] values versus string fixture assumptions. Normalized native flags before status assessment and added three actual databento_dbn.StatusMsg regression cases. Superseded reports were removed; final two full runs must be regenerated. Prior trial reproducibility alone was not accepted as correctness.

## Final verification
48 audit tests plus 133 existing replay/accounting/policy tests passed. Both corrected full runs returned PASS_AUDIT_CHECKS with HOLD_VALIDATION; 237146989 native MBO records per run, byte-identical five canonical artifacts, unchanged pinned inputs, independent OHLC counts and May28 trade references matched. 11 supported and 19 unassessable conditional scenarios. All review findings addressed; none deferred. Results and acceptance evidence: docs/reports/yank-execution-pilot/.
