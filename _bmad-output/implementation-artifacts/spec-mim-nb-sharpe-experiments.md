---
title: Implement and evaluate fixed MIM-NB Sharpe experiments
type: feature
created: 2026-09-11
status: done
route: dispatch
baseline_commit: 8434a705dfd249c75e4cf50aa6427a62e340f4b0
context:
  - _bmad-output/specs/spec-mim-nb-sharpe-experiments/SPEC.md
  - _bmad-output/specs/spec-mim-nb-sharpe-experiments/protocol.md
---
<frozen-after-approval>
## Intent
Implement approved research comparison of unchanged MIM-NB versus four isolated entry gates, run historical experiments, report risk/profit trade-offs and uncertainty. User already approved complete plan and implementation. Canonical contract is in context files.
## Boundaries & Constraints
Always follow protocol; preserve exact baseline timing, risk and session conventions. Never touch production, frozen comparison sources, original data/reports or activate services. No parameter search or live sizing. Keep unrelated work untouched.
</frozen-after-approval>

## Code Map
- research/mim_comparison/engine.py: simulate is reference; copy/adapt in new package only. Read it fully including pending fill and reference guard bookkeeping.
- research/mim_comparison/data.py: reuse read-only load/audit_select. historical.py shows same14-session moves/sigma construction and primary data grid; don't invoke its side-effectful run.
- research/mim_comparison/artifacts.py: use design patterns, but own isolated output and source freeze. Existing make_run is tied to old protocol: don't call it.
- research/mim_comparison/runs/20260910T210224-historical-f3950efb68: authoritative manifests,completion,daily,decisions,ledger. Baseline daily primary Sharpe1.138494516892445,total21889.76,MDD2437.26,1323days2021-01-15..2026-08-27. Data hashff76aefca405dd94359b15223c57710f4e7f01f245880426a60d0f934c6f5bea.
- tests/unit/mim_comparison/test_comparison.py: fixtures/patterns; existing tests should remain unchanged.

## Tasks & Acceptance
- [x] research/mim_robustness/: implement source/config-frozen CLI audit/run/evaluate and immutable manifests, safe output isolation and verified baseline reconciliation.
- [x] research/mim_robustness/: implement per-minute features plus copied causal execution engine for A/R/E/F/P. Include enough decision context to explain gate/exit and paired trade outcomes.
- [x] research/mim_robustness/: implement efficient synchronized stationary bootstrap, metrics, screen classification, feature/loss attribution, complete ledgers and Markdown/self-contained HTML reports.
- [x] tests/unit/mim_robustness/: test protocol acceptance, no lookahead, stop/reset semantics, accounting, deterministic artifacts and protected paths. Given future bars changed, earlier decisions stay unchanged. Given gate rejects reversal, old position exits. Given source/input drift, command fails closed in new run. Given existing complete outputs, never overwrite.
- [x] README.md in new package: exact commands and study limitations. Root ran actual full historical experiment/evaluation and published RESULTS.md; implementation agent did not run full data or commit.

## Implementation Notes
Dispatch implements new package and tests only; root owns specs, real research runs, final report and commit. No subagents from implementation agent. Use .venv/bin/python for tests. Stable artifacts before expensive real run. Primary screen full precision and highcost semantics from protocol. Ledger additions may use new fieldnames but retain original reference fields for baseline reconciliation.

Implementation subagent stopped on usage limit after saving engine/features/artifact/statistics modules. Root resumed implementation locally, completed workflow/reporting and37 tests. User's intervening greeting did not cancel the approved implementation. All real candidate computations remain pending review. No production or frozen source edits.

## Spec Change Log

## Review Triage Log

Three independent review lenses completed. Edge-case review returned no findings. All actionable findings below are resolved; no implementation issues deferred.

| Finding | Disposition and verification |
|---|---|
| Claimed mandatory power gate | Withdrawn by reviewer: unsupported by repository instructions or approved protocol. Do not introduce an unapproved study prerequisite. |
| Queued reversal after catastrophe stop | Preserve exact baseline behavior; corrected misleading comment, disclosed limitation and added reference-agreement regression. |
| Evaluation used default paths | Freeze original input bindings and reuse them; evaluation passes with nonexistent defaults. |
| Trade costs/net not reconciled | Verify per-trade costs, gross/net and fill turnover against every daily cost scenario; corruption tests fail closed. |
| Nonfinite accounting accepted | Reject NaN/infinity across daily, fill and trade accounting; mutation fixtures cover failures. |
| Configuration disconnected from behavior | Feature, risk, sizing denominator, bootstrap and screen calculations now consume frozen configuration; changed threshold/window fixtures verify linkage. |
| Missing runtime versions | Freeze and verify Python, NumPy, pandas and platform metadata; version-drift fixture fails closed. |
| Daily guard rejection not explicit | Record guard state, projected gross, risk block and entry disposition; boundary fixture verifies daily-guard attribution. |
| Data/output verification gaps | Real loader fixtures cover missing/duplicate/early-close/zero-volume sessions, DST and rolls; workflow asserts output isolation. |
| Baseline mismatch not tested end-to-end | Mutate baseline daily/fill/decision artifacts independently; CLI seals failure before candidate outputs. |

## Verification
.venv/bin/python -m pytest tests/unit/mim_robustness -q
Root: full relevant suite and real audit/run/evaluate then independent ledger/manifest checks.

Completed 2026-09-12: 167 tests passed in 102.32 seconds (112 existing, 55 new). Real audit `20260912T151751-audit-210f6b1b24` reproduced every original column in 7,938 daily scenario rows, 31,752 decisions and 3,190 fills. Full fixed experiment `20260912T151842-run-f8608e71fb` and separate evaluation `20260912T152100-evaluate-0918c79842` completed, including 20,000 bootstrap draws for each of three block lengths. None of R/E/F/P meets the predefined screen. See research/mim_robustness/RESULTS.md. No prospective collection or live change was authorized by the results.

Independent verification passed: three sealed inventories, original baseline/data/source hashes, 14 byte-identical re-evaluation artifacts, all 30 scenario metrics and daily cost reconciliation. Git whitespace inspection reports only pre-existing extra EOF blank lines in the newly tracked canonical SPEC/protocol; preserved the frozen protocol bytes intentionally. No code whitespace findings.
