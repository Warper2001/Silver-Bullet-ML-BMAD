---
title: 'MIM prospective collection readiness and benchmark design'
type: 'chore'
created: '2026-09-12'
status: 'done'
route: 'oneshot'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Completed payoff diagnostics found zero eligible prospective MIM sessions, no stored FOMC events, and no evidence of an operating MIM polling schedule. The operator authorized the recommended next task: prospective collection-readiness audit followed by a preregistered simple-benchmark design.

**Approach:** Inspect collection coverage, timestamps, contracts, protocol integrity, warmup and scheduling read-only. Produce timestamped evidence, a clear readiness verdict and a concrete remediation sequence. Commit a separate cash-session long-exposure benchmark design with explicit execution, paired inference and pre-execution power/activation gates. This task prepares research; it does not run a new strategy test, compute benchmark returns, acquire data, modify services/collectors/trading, access sealed holdout, reset existing protocols, or invent an economically meaningful effect size. Existing historical outcomes remain exposed development evidence. A committed design is not an activated experiment or a passing power gate.

</frozen-after-approval>

## Implementation Notes

- Use the existing clean research/mim-diagnostics worktree; preserve divergence from current origin/main. No merge or push.
- Reuse the existing diagnostics audit for immutable input/source/inventory evidence, with a separate fresh run under research/mim_diagnostics/runs/. Operational checks add timestamped read-only evidence and an explicit scope/limitations statement.
- Read research/mim_comparison/{README,FEED_STATUS}.md, poll.sh, feed_adapter/README.md, governing freeze and journal metadata, and the separate FOMC preregistration. Existing comparison/statistics.py is reference only; its A/B sample size and dollar threshold cannot authorize a different benchmark test.
- New outputs: research/mim_diagnostics/READINESS.md, _bmad-output/preregistration_mim_cash_session_benchmark_design.md, and versioned compact readiness evidence under research/mim_diagnostics/evidence/. Link them from README.md. Keep large audit snapshots ignored.
- No intent gaps or irreversible actions are needed for this design-only deliverable. An executable seal requires a later reviewed power artifact and calibrated design; until those exist activation remains denied.
- Verification: independently check recorded hashes, protocol boundaries, read-only queries, evidence-to-report agreement, relative links and no strategy/collector/service mutations. Run existing audit verify. Have a context-free reviewer inspect changed documents before commit.
- Fresh audit completed at runs/20260912T211038-audit-847291ba09; separate verify returned verified=true, command=audit. Readiness agent found missing detected MIM scheduler, no timely forward coverage, missing next-contract coverage, and a FOMC endpoint-only validation mismatch with the full-window gap rule.
- Created READINESS.md, inactive benchmark design, compact readiness JSON and independent verification JSON; updated README links. Independent verification rehashed 26 files and two consumed live-source prefixes without printing raw logs, and validated document links. No code changes or application tests were necessary for this documentation/evidence-only follow-up.

## Review Triage Log

The oneshot workflow used its single active Blind Hunter layer; no other layer was required. All eight findings were checked against the cited text/source and patched; none deferred.

| Finding | Verdict and evidence | Resolution |
|---|---|---|
| Interval construction unspecified | Medium: stationary resampling alone did not define interval endpoints | Specify percentile intervals, linear quantiles and invalid-draw handling, bound in activation record |
| Composite null underspecified | Medium: joint-zero-only calibration could miss either branch of the union null | Require both null branches, their boundaries/intersection and nuisance cases |
| Irregular eligible-day blocks | Medium: blocks could silently be interpreted as calendar days | Define eligible-session index, retain date-gap metadata and calibrate clustered exclusions |
| Conditional population unclear | Medium: common exclusions can correlate with volatile days | Limit conclusions to eligible sessions and require outcome-blind exclusion-clustering reports |
| Audit linkage lacked completion hash | Medium: local path and stated verification were insufficient durable linkage | Independently rerun verify and record command, timestamps, parsed result, manifest/completion hashes and scope separately from manual assertions |
| Failed FOMC attempts not preserved | Medium: None returns are retried with no existing event row | Include failed/ineligible attempts, source/coverage/revision history in concrete engineering repair requirements |
| N=15 interim sample not pinned | Medium: catch-up loop can cross interim target before human review | Require immutable first-15 accumulation sample, one-time look record and backlog recovery checks |
| FAIL versus insufficient data ambiguity | Medium: blanket FAIL language conflicted with deadline disposition | Define precedence and mutually exclusive final statuses; distinguish pre-activation power states |

## Verification

- Fresh diagnostics audit independently verified twice; the final verification record binds completion and manifest hashes, command, UTC timing and parsed success result.
- All 26 evidence file hashes and both consumed source-prefix hashes independently matched; final document links checked after review patches. Benchmark remains design_only_inactive / NOT_ASSESSED; no strategy/collector/code/service changes.
- Bounded reviewer recheck confirmed all eight fixes, no unresolved material contradiction, and independently checked six hashes linking final documents, readiness evidence and audit manifest/completion.
