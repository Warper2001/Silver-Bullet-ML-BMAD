---
title: Diagnose MIM-NB trade lifecycle and early-loss recovery
type: feature
created: 2026-09-12
status: done
route: dispatch
baseline_commit: faba8ca2c8f2171bcfc07b324360c4dfb572d54d
context: []
---

<frozen-after-approval>
## Intent
User authorized the proposed trade-lifecycle study: explain favorable/adverse movement, time to profit, recovery from early losses, and dependence on large winners before proposing a small number of future hypotheses. Deliver a reproducible analysis of unchanged A, primary delay2, one contract, from the completed fixed-filter study. Existing history is exposed. This is descriptive observation, not testing new stops, entry filters, holding rules or returns of alternative strategies.

## Boundaries & Constraints
Always preserve frozen research sources, runs, data, live code and services. New isolated research/mim_lifecycle package only, tests/unit/mim_lifecycle, root-owned spec and results. No dependencies installed. No broker imports, sealed holdout, parameter sweep, strategy changes, orders or deployment. A future strategy test requires the repository's power gate and preregistration; this diagnostic makes no superiority or deployability inference. Observation checkpoints are not trading thresholds. User's authorization to do the study already covers implementation; unrelated untracked work is preserved.

## I/O & Edge-Case Matrix
| Scenario | Expected behavior |
|---|---|
| Long / short path | Direction-adjusted dollars at $2/point; roundtrip friction $2.24; terminal P&L equals source trade. |
| Entry / open exit | Entry minute included after entry open; opening exit minute OHLC excluded. Never use pre-entry or post-exit price. |
| Stop exit | Exclude entire ambiguous stop-minute OHLC, include actual exit fill; report excursions as observed lower bounds, stop duration as minute interval. |
| EOD exit | Include final completed bar and close-proxy fill. |
| Landmark after exit | Excluded from at-risk denominator, count exited trades separately. |
| Missing/duplicate/nonfinite data or integrity drift | Fail closed; no partial success report. |
| Repeat input | Identical numeric output; new immutable output directory, never overwrite. |
</frozen-after-approval>

## Code Map
- research/mim_robustness/runs/20260912T151842-run-f8608e71fb is frozen input. Completion SHA256 b747ceff679c3874a6236d71eb960c4a68890aa5267aaa0575070b34b097a9a7. trades.csv has day,contract,arm,delay,direction,entry_event_timestamp (end-labeled entry bar),entry_fill_timestamp (opening time),exit_event_timestamp,exit_fill_timestamp (blank for stop),exit_fill_time_basis,entry_fill,exit_fill,quantity,gross,costs,net,exit_reason. Select A,delay2:801 trades. daily.csv same arm/delay/cost2.24:1323 sessions, total21889.76.
- data/mim_x/mnq_1min_by_contract.csv hashff76aefca405dd94359b15223c57710f4e7f01f245880426a60d0f934c6f5bea. research/mim_comparison/data.py load(path,'end') reads normalized ET RTH bars; use read-only, never edit/import old executable scripts. Join exact day AND contract; all390 minutes on selected trade sessions required.
- research/mim_robustness/artifacts.py verify(run,complete=True) and verify_baseline(DATA,BASELINE) can verify original integrity read-only. Do not add files to that package (its source hash covers every *.py). Own lifecycle artifact helper must bind/snapshot its source and numerical runtime, input hashes and this spec/protocol before analysis.

## Tasks & Acceptance
- [x] New package: CLI `.venv/bin/python -m research.mim_lifecycle` produces unique sealed runs/ with manifest, source snapshot, completion SHA inventory, paths.csv, lifecycle.csv, landmarks.csv, summary.json, report.md and standalone plots in report.html. Catch failures into failure.json; output stays in package runs/.
- [x] Trace each trade's valid completed bars, price high/low excursions and net liquidation close path. Include entry fill and terminal fill as known points. Full roundtrip fee reserved in net marks (hypothetical liquidation); primary excursions are gross dollars relative actual fill. Record MFE>=0, MAE>=0, minutes to first positive net close, first negative close and later nonnegative recovery (including terminal fill). Censor time-to-profit if never observed. Report stop-minute uncertainty and never count its full high/low as experienced. Stop fill is still known loss/profit. Duration for stop [exit_event-1min-entry_open,exit_event-entry_open]; open/EOD exact.
- [x] Observe checkpoints5,15,30,60,120 elapsed minutes. Only exact completed close while trade still open/known survives that close; use same timestamp entry_fill+checkpoint. Compute current net mark, cumulative known excursions, finalwinner/net, remaining net change (final net minus current net), recovery after red checkpoint from strictly later valid closes or terminal fill. Include at-risk and already-exited counts; never interpret eventual-winner conditioning as causal entry information.
- [x] Summarize counts, quantiles and net contribution by final winners/nonwinners, long/short, year, exit reason. Show landmark red/green cohorts with sample sizes, eventual winner rate, mean final net and remaining change. Show time-to-first-positive distributions WITH never-positive counts. Select top ceil(5%*801) trades by final net and baseline top67 daily net dates for descriptive large-winner recovery/MAE/contribution analysis; label hindsight selection clearly. No optimized cutoff or trade recommendation. Include aligned-path chart by eventual outcome (survivorship disclosed) or excursion/first-profit scientific plots using standalone HTML SVG/no external assets.
- [x] Tests cover every matrix row and independent hand-calculated paths/landmarks, no future leakage into checkpoint features, recovery sequencing, stop uncertainty, source/daily reconciliation and immutable outputs. No full historical run or git staging/commit by implementer; root runs real study and owns final RESULTS.md and spec updates.

Acceptance: Given pinned baseline input, when full analysis runs, then all801 trades sum to21889.76 within1e-8, and the common1323 daily grid with flats reconciles. Given altered later bars, earlier landmark features remain identical; labels may change. Given ambiguous stop bar, no false intrabar recovery or excursion claim appears. Given completed study, user can distinguish robust observations from future untested hypotheses.

## Implementation Notes
Implementation agent owns new package/tests only, no subagents. Root owns independent source investigation, actual run, final report and local commit. Runtime >30s launched with nohup and polled per AGENTS.md. No open user choices or irreversible actions.

## Review Triage Log

| Layer / finding | Verdict and disposition |
|---|---|
| Blind 1: checkpoint terminal recovery at equal timestamp omitted | Medium; reproduced red preceding close followed by profitable opening exit. Patch observation ordering and regression test. |
| Blind 2: terminal loss counted as negative-close history | Medium; search includes terminal marks. Patch negative-close origin and explicit fixture. |
| Blind 3: failed run verifies as successful completion | Medium; completion verification needs explicit failure rejection. Patch and test. |
| Blind 4: incomplete success artifact set can seal/verify | Medium; require named success artifacts, with explicit failed-invocation exception. Patch and test. |
| Blind 5: nested completion marker excluded from inventory | Medium; path-name filter skips nested content. Exclude only root marker and test. |
| Blind 6: writable directories allow sealed-file replacement | Low; hashes detect replacement but write bits alone do not prevent it. Clarify tamper-detection guarantee in report/README instead of changing directory access semantics. |
| Blind 7: tamper test depends on root privileges | Low; test directly writes read-only file. Explicit chmod before intentional mutation. |
| Blind 8: binding/verification not exercised | Medium; real verification functions lack tests. Add isolated binding and drift mutations. |
| Edge 1: same-time opening terminal recovery omitted | Medium; same reproduced defect as Blind 1, same patch/test. |
| Edge 2: first-negative-close mixes terminal loss | Medium; same defect as Blind 2, same patch/test. |
| Verification 1: provenance drift not tested through run | Medium; pre-verified test gap. Add post-analysis CLI drift fixture that seals failure without success reports, alongside real binding mutations. |
| Verification 2: top cohorts cannot detect reversed ranking | Medium; one-trade fixture is nondiscriminating. Add cutoff/tie/membership/contribution fixture exceeding both selection cutoffs. |

## Verification
Focused pytest, real CLI, independent accounting/inventory validation; recheck source run/data hashes after completion.

Completed 2026-09-12. All actionable review findings fixed; none deferred. Verification: 111 relevant tests passed in 7.61 seconds (56 lifecycle, 55 robustness); scoped Black check passed. Full run 20260912T154857-diagnostic-b54c02459db4 completed with 801 trades, 1323 sessions including 544 flat, 4005 landmarks, net21889.76. Independent per-trade net/MFE/MAE/first-positive-close calculations and all five checkpoint cohort statistics matched. Sealed inventory and original source/data hashes verified; standalone SVG inspected. See research/mim_lifecycle/RESULTS.md for findings and the untested profit-giveback hypothesis. No strategy parameter, original protocol, data, service or live file changed.
