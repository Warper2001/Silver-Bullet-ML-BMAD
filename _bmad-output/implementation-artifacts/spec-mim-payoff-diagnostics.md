---
title: MIM-NB payoff attribution and data feasibility
type: feature
created: 2026-09-12
status: done
route: dispatch
baseline_commit: 5f179cb6ef203dae3528e0833b8b7aef0ef65682
context: [AGENTS.md]
---

<frozen-after-approval>
## Intent

Explain the unchanged baseline's existing payoff, distinguish entry timing from profit accumulation, and assess closing-demand and scheduled-information data feasibility. Deliver reproducible diagnostics, verified CSV/JSON ledgers, readable Markdown/standalone HTML reports and research specifications. The user's supplied implementation plan authorizes implementation without another approval checkpoint.

## Boundaries & Constraints

Work only in `/root/Silver-Bullet-ML-BMAD/.claude/worktrees/mim-diagnostics`. Inputs live in the original checkout; use its `.venv/bin/python` without installing dependencies. Create `research/mim_diagnostics/`; runtime writes go exclusively to fresh direct children of its `runs/`, refusing overwrites and preserving failures. Freeze hashes, source snapshots, versions and definitions. Read existing artifacts/bars only. No strategy simulation, historical entrypoint execution, broker/collector imports, dataset acquisition, holdout access, live edits, service changes, deployment, parameter tests, filters or thresholds. All history is exposed development evidence. No power gate is appropriate for these descriptions; future strategy tests require an appropriate gate and committed preregistration. Preserve prospective A/B and FOMC protocols.

Primary is arm A, recorded delay=2 fills (open one full minute after decision), $2.24 roundtrip, one MNQ at $2/point. Display already recorded timing/cost alternatives separately, never new benchmarks. Preserve all eligible sessions, flat days, selected contracts and exclusions; reject inconsistent joins. Retain source execution order within each minute. Distinguish true reversals from opposite entries after flat, retaining original labels. Reconstruct realized P&L, event costs, minute-close marked equity and position.

Use fixed cash-session half-hours for entry-time versus P&L-accrual attribution. Attribute gross/net, counts, losing-trade dollars/share and exposure by direction, exit label, calendar year, first/subsequent entry and entry half-hour; accrual partitions reconcile too. Report concentration, sampled drawdown depth/duration and the previously disclosed best-67-session attribution. Include fills/closes in sampled excursions; use highs/lows only over fully held intervals. Exclude stop-minute subsequent prices and label incomplete coverage. Preserve uncertain stop intervals without invented timestamps. Costs belong to recorded executions; time attribution uses completed-minute event labels and retains actual modeled fill timestamps.

Inventory economic, policy-shock and forward FOMC calendars: coverage, duplicates, timezone, official support, availability evidence. Dates alone are insufficient provenance; shock labels remain retrospective absent contemporaneous evidence. Inventory Nasdaq-relevant options positioning and leveraged ETF AUM/leverage: observed versus proxy, delays, history, assumptions; SPX is not Nasdaq inventory. Classify usable / requires verification/acquisition / unavailable with specific missing evidence. Read prospective collection status with observation timestamp and coverage, no efficacy analysis. Correct the September 12 innovation assessment for rejected policy throttle, failed impulse following and separate FOMC fade, preserving verdicts/protocols. Future simpler benchmark definitions/proposals need prerequisites and falsifiers, with no returns or promotions.

## I/O & Edge-Case Matrix

| Given | When | Then |
|---|---|---|
| Frozen source/data | audit or run | New sealed evidence directory, primary reconciles 1323 sessions/801 trades/$21889.76 within 1e-8 |
| Missing/duplicate/corrupt input or unsafe output | validate | Fail closed, preserve failed-run evidence |
| Long/short, reversal, same-minute entry/stop, adverse gap, EOD | reconstruct | Correct ordering, costs once, uncertainty explicit |
| Flat session, DST, contract transition | reconstruct | Exact source grid, ET labels, no join loss |
| Changed post-exit prices | excursions | Closed trade excursions unchanged |
| Incomplete inventory or repeated identical input | verify | Reject missing inventory; deterministic analytical outputs |
</frozen-after-approval>

## Code Map

- Original root: `/root/Silver-Bullet-ML-BMAD`; completed source `research/mim_robustness/runs/20260912T151842-run-f8608e71fb`, completion SHA256 `b747ceff679c3874a6236d71eb960c4a68890aa5267aaa0575070b34b097a9a7`; data `data/mim_x/mnq_1min_by_contract.csv`, SHA256 `ff76aefca405dd94359b15223c57710f4e7f01f245880426a60d0f934c6f5bea`.
- Source `ledger.csv` has event_timestamp, modeled_fill_timestamp, fill_time_basis, fill, position_after, quantity (turnover), costs, reason, delay, cost_scenario. Primary ledger has 1595 events. Preserve file row order; compare reconstructed trades with `trades.csv`, session totals with `daily.csv`, preserve `exclusions.csv`.
- `research/mim_lifecycle/analysis.py`: useful session validation and sampled path conventions; net marks there reserve all fees, unsuitable for event-cost equity. `research/mim_comparison/data.py` loads end-labelled ET bars without simulation. Do not invoke engines. Source manifest binds original comparison completion and data.
- `research/mim_robustness/RESULTS.md`: existing best 67 = $39552.92; remainder -$17663.16. Exit labels: CAT_STOP 71/-$35331.04, EOD_CLOSE_PROXY 723/$59824.48, REVERSAL 7/-$2603.68. 544 flat sessions.
- Prior evidence under original `_bmad-output/`: `option_c_retrospective_20260703.md`, `option_b_gate0_verdict_20260703.md`, `preregistration_evfade_fomc_prospective.md`; find September 12 innovation report and calendar files read-only. Parent has independent provenance investigation available.

## Tasks & Acceptance

- [x] `research/mim_diagnostics/{__init__,__main__,artifacts}.py`: CLI audit/run --source-run --data, verify --run; defaults resolve original checkout evidence from worktree; exclusive safe output, frozen inventory, independent reconciliation verification.
- [x] `research/mim_diagnostics/analysis.py`: validated ledger reconstruction, all partitions/excursions/equity/exposure and original scenario display; reject all missing joins.
- [x] `research/mim_diagnostics/{feasibility,report}.py`: evidence-backed inventory/status, corrected assessment, proposals and standalone charts/report.
- [x] `research/mim_diagnostics/{README,RESULTS,definitions}.md`: commands, explicit semantics and verified conclusions; all implementation and tests confined to package and `tests/unit/mim_diagnostics/` plus this spec.
- [x] `tests/unit/mim_diagnostics/`: cover matrix and reconciliation; run relevant existing comparison/lifecycle/robustness accounting/data tests and full diagnostics + independent verify. Long jobs (>30s) use nohup/logs.

Given primary source, when outputs verify, then minute/daily/trade totals and every complete partition agree within $1e-8, including pinned exit labels/contributions; full inventory and data binding pass. Given feasible proposals, when read, then none contains a new return test, efficacy claim or candidate promotion.

## Implementation Notes

First full final-source run reconciled all pinned totals, with 26 new tests and 223 distinct existing tests passing. Review identified verifier and provenance hardening patches; no intent gap or change to frozen behavior.

No intent gaps or irreversible changes. This is one diagnostic interface with new standalone modules/tests; existing modules remain unchanged. User explicitly authorized isolated worktree despite unrelated root untracked files; root is 2 ahead/0 behind origin/main.

## Spec Change Log

## Review Triage Log

| Finding | Verdict | Evidence and route |
|---|---|---|
| B1 | high | Calendar/journal/event readers bypass permitted() before opening; a resolved holdout symlink is reachable. Patch reader guards. |
| B2 | high | Source manifest keys are joined without relative containment validation; traversal can hash outside snapshot. Patch containment before reads. |
| B3 | medium | Verifier tests partition sums only; swapped labels/offsetting allocations preserve sums. Patch keyed group verification. |
| B4 | medium | Accrual rows lack trade/minute membership checks; orphan IDs or timestamps preserve day totals. Patch exact joins and row accounting. |
| B5 | medium | Excursion/coverage/duration output fields lack independent checks. Patch bar/fill recomputation. |
| B6 | medium | scenarios table is loaded but not compared against source arm A. Patch complete scenario comparison. |
| B7 | medium | Narrative summary fields beyond payoff and minute depth are unchecked. Patch counts, concentration, transitions, drawdown clocks. |
| B8 | medium | Required inventory omits imported dependencies and two promised prospective protocols. Patch complete required path sets. |
| B9 | medium | SQLite SELECT statements lack explicit read transaction, so concurrent appends can mix snapshots. Patch BEGIN around observation reads. |
| B10 | medium | Inventory schema permits missing evidence fields and arbitrary truthy official/correction payloads. Patch category-specific validation. |
| B11 | medium | 120-session status text is unconditional despite queried count. Patch count-derived observation text with no efficacy analysis. |
| E1 | high | Source-directory guard precedes completion child symlink validation; digest can read prohibited target. Patch guard every read path. |
| E2 | high | Same calendar/journal/prospective reader gap as B1 confirmed at actual callers. Patch same reader guard. |
| E3 | medium | Accrual output trade/minute joins are not independently validated. Patch with B4. |
| E4 | medium | Per-bucket membership/value correctness not implied by partition total. Patch with B3. |
| E5 | medium | Source scenario semantics unchecked after inventory reseal. Patch with B6. |
| V1 | medium | Pre-verified gap: no test runs cross-half-hour accrual; replacing accrual bucket with entry bucket passes existing suite. Add exact timing fixture and semantic verifier. |
| V2 | medium | Pre-verified gap: no scenario aggregation test; dropping arm-A filter passes existing checks. Add distinct-arm scenario fixture and verifier. |
| V3 | high | Reader guards absent on concrete feasibility CSV/SQLite paths, same B1/E2. Add direct-reader regression coverage. |

## Verification

Use original `.venv/bin/python -m pytest tests/unit/mim_diagnostics tests/unit/mim_comparison tests/unit/mim_lifecycle tests/unit/mim_robustness -q` from worktree. Run audit, one full diagnostic and verify; independently check numerical partitions and inventory. Keep all files in worktree; no merge/deployment requested.

## Review resolution and final validation

All B1–B11, E1–E5 and V1–V3 findings were patched and covered by the expanded 127-test diagnostics suite. Reader guards precede CSV/SQLite/hash/snapshot reads; full keyed accounting and risk summaries are independently checked; dependency/protocol omissions and malformed provenance fail closed. Status queries use one read transaction. A package-local ignore protects large runs. The final combined suite passed all 350 tests in 132.52 seconds. The combined run initially exposed a duplicate test module basename; adding a diagnostics test package and relative fixture import resolved discovery, with the complete suite then passing. No findings were deferred.

Final run: research/mim_diagnostics/runs/20260912T194421-run-3346059bd9. Final audit: 20260912T194418-audit-3a00001098. All eight analytical ledgers/summary were byte-identical to the preceding completed run. Six SVGs/seven HTML tables parsed. Separate CLI verification passed and is recorded in research/mim_diagnostics/runs/20260912-independent-checks/final-verify.log. The full source inventory matched the final executing code. All acceptance criteria and matrix cases passed.
