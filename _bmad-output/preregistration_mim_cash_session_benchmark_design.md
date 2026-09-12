---
experiment: MIM-CASH-REFERENCE
created: 2026-09-12
status: design_only_inactive
power_status: NOT_ASSESSED
activation_authorized: false
strategy_tests_executed: false
---

# MIM versus scheduled cash-session exposure: prospective design

This committed document registers the research question and the proposed design. **It is not an executable experiment seal.** Collection readiness, power calibration and the final activation record remain prerequisites. No benchmark returns were computed, no paired outcome was inspected, and no existing experiment was amended by this work.

## Question and scope

Does the unchanged MIM arm A deliver better net performance per unit of daily variability than simply holding one MNQ contract long through the cash session? This tests the incremental value of the complete MIM decision process against a simple exposure reference. It does not identify which mechanism causes any difference. The estimand is conditional on the common eligible-session grid; it cannot establish unconditional operating performance. Input outages and unavailable A decisions may cluster on volatile days, so excluding them can change the population materially.

The September 12 diagnostics exposed all 1,323 historical sessions and their outcomes. Every inspected historical outcome remains development evidence. The best-67 attribution and profitable half-hours are excluded from benchmark construction. No direction, time-window, stop or re-entry filter is proposed. The fixed cash session is an instrument convention, not a selected profitable window.

This is independent of the existing prospective A/B comparison and EVFADE-FOMC. Their frozen sources, eligibility rules, deadlines and look schedules remain unchanged. Existing A/B power estimates and its $5 hurdle do not apply to this question. Shared future market observations may create dependence between experiments; disclose that overlap and never choose which result to report or revise one experiment in response to the other.

## Arms and execution specification

| Item | A: unchanged MIM | L: scheduled cash-session long reference |
|---|---|---|
| Instrument / quantity | One MNQ, $2 per point | Same selected MNQ, one long contract |
| Entry decision | Frozen source decision rules | Scheduled before the cash-session open; no current-session signal |
| Entry fill | Primary recorded-model convention: open one full minute after a completed-minute decision | 09:30 America/New_York opening price, the open of the bar completed at 09:31 |
| Exit | Frozen catastrophe-stop, reversal, guard and EOD behavior | Entire position at the 16:00 cash-session close proxy |
| Overnight | None | None |
| Primary fees | $1.12 per contract-side execution | $1.12 at entry and $1.12 at exit |
| Secondary fees | Existing $3.24 and $6.24 round-trip scenarios from actual turnover | Same per-side conversion, one round trip |
| Sizing | Fixed one-contract accounting | Fixed one-contract accounting; no volatility scaling |

The scheduled reference's opening order is known in advance, so it has no signal-processing delay. Its opening price and the closing proxy are modeled prices, not observed broker executions. This assumption must be prominently reported. Any eventual realized-slippage study is separate. No opportunistic alternative opening time or stop is selectable in this experiment. Intraminute stops in A retain interval-valued timing and adverse-gap handling; do not fabricate exact timestamps.

Bind A to the nine source hashes in the governing freeze snapshot at `research/mim_comparison/runs/20260910T214803-shadow-47055eba62/freeze-snapshot.json` in the original checkout. Before activation, an isolated benchmark implementation must prove recorded-baseline parity with the diagnostics within $1e-8 accounting tolerance. That verification reuses existing outcomes; it does not compute new historical L returns. Preserve the original exit ordering and first/subsequent entries. Do not add L to, or edit, the frozen A/B collector.

## Paired session grid and first observations

Use the existing prior-session-volume contract-selection, complete 14-session warmup, same-contract prior-close, quarterly expiry and exclusion rules without revisions. Preserve all eligible A-flat sessions: A is zero while L still holds its scheduled position. Never drop a paired session based on either payoff or whether A traded.

Eligibility is determined from common market-input and decision-availability requirements. Full cash sessions contain all 390 completed bars from 09:31 through 16:00 ET. Use the IANA `America/New_York` zone for DST. Early closes, unresolved closures, missing bars, duplicate/corrected first observations, incomplete candidate-contract coverage, missing warmup or prior close, and unavailable A decisions are common exclusions. A missing winner never triggers a fallback contract. Report every excluded session and reason; do not repair it retrospectively into the confirmatory sample. Before efficacy is opened, report scheduled versus eligible counts, excluded-run lengths, contract transitions and calendar locations of outages, using availability metadata only. Report the same coverage alongside final results without attributing returns to excluded days.

At activation, freeze the exact candidate-contract universe and the provenance of its availability; it must support the inherited selection rule through a quarterly transition. Explicitly identified contracts with trustworthy first-receipt and collector timestamps are required. An authenticated omission is different from an unknown feed outage. Inferred contract identity is disclosed and accepted only if the final evidence standard explicitly supports it; the current operational log cannot authenticate response payload identity.

Require both actual receipt and durable collection within the existing 60-second completed-minute budget. Preserve permanent first observations and invalid-row tombstones; replayed old bars never become prospective. The future standalone collector must retain the prescheduled L entry intent before the opening interval and sufficient common data to evaluate A without later reconstruction of missed decisions. Do not retrofit L intents into the ongoing A/B journal.

No observation at or before the final activation cutoff is confirmatory. Select the first complete ET session whose opening is strictly after that committed activation timestamp. All data observed during readiness checks, power planning or implementation validation are excluded from the future confirmatory sample. The existing A/B deadline cannot be reset or extended by this new cutoff.

## Estimands and inference

For each common eligible session, let a and l be each arm's net one-contract dollar outcome. Define S(x) = sqrt(252) * mean(x) / sample_sd(x), with sample SD using n−1. This is a daily dollar-outcome Sharpe statistic under fixed one-contract sizing; it is not a return on account equity. The 252 multiplier is a reporting convention, not a claim of independent days.

The primary comparative estimand is **Delta S = S(a) − S(l)**. Require positive expected net a as a separate joint condition, so being less negative than a losing reference cannot support the hypothesis. Undefined SD, too few usable observations or failed numerical checks produce no support; never substitute zero or infinity.

Proposed final inference resamples the entire paired (a, l) daily sequence with stationary blocks, preserving same-day pairing and zero A days. Use the existing research method's block lengths 5, 10 and 20 as a predeclared dependence-sensitivity set, 20,000 draws and seed 7 for reproducibility. The resampling index is the chronological sequence of eligible paired sessions, with the original date and preceding calendar gap retained as metadata; blocks can therefore span excluded dates. Block length means eligible observations, not calendar days. Recompute both Sharpe statistics on each draw. The proposed construction is percentile intervals using the 0.025 and 0.975 empirical quantiles with linear interpolation (NumPy method="linear"). Retain the total attempted draws and invalid-draw reasons. Any undefined resampled statistic denies this proposed construction an evaluable result; no invalid draw is silently discarded or replaced. Calibration must check confidence-interval coverage for this nonlinear statistic, serial dependence, heavy tails, rare winners, unequal exposure, clustered outages, holidays and contract-transition exclusions on this irregular sampling grid; the existing mean-return bootstrap's validation does not establish these properties. If it fails, revise the design before activation rather than changing inference after outcomes.

The proposed support rule requires lower two-sided 95% interval endpoints above the mathematical zero null for both Delta S and mean(a), with agreement at all three block lengths. Requiring both conditions is an intersection-union decision; neither endpoint is independently a promotion test. Final statuses have explicit precedence: if the deadline arrives before the fixed eligible target, return INSUFFICIENT_SAMPLE without an efficacy look; at the target, missing/inconsistent evidence or undefined statistics/invalid bootstrap draws return UNEVALUABLE; otherwise return SUPPORT_FOR_FURTHER_VALIDATION only when both lower-endpoint conditions hold at every block length, and FAIL for all other evaluable results, including disagreement across block lengths. These are mutually exclusive. NOT_ASSESSED and UNDERPOWERED are pre-activation power states, never statistical failure verdicts. Every status except support denies evidence of incremental value; ambiguous evaluable evidence is FAIL. No status authorizes deployment.

Report paired mean-dollar difference, each arm's mean, gross/net/costs, exposure, turnover, annualized daily dollar SD and sampled minute/daily drawdown as secondary descriptions. The $3.24/$6.24 cost scenarios are sensitivity disclosures, never alternative winning primary cells. Do not claim equal maximum risk or capital needs from a Sharpe comparison. No year, direction, event or half-hour subgroup can rescue a failed primary result. No statistical support authorizes trading or deployment.

## Power planning before an executable seal

**Current verdict: NOT_ASSESSED; execution denied.** Neither the historical A/B detectable increment nor an arbitrary sample target is a valid power gate for a paired Sharpe comparison. The XSMOM gate is a firewall pattern only; do not run its unrelated commodity experiment.

The next gate implementation must:

1. Accept a predeclared planning grid of plausible paired dependence, marginal variability, tail/concentration structure and effect sizes, with provenance for each range. Use synthetic joint panels or explicitly permitted, outcome-blinded nuisance evidence. Do not compute L's historical strategy returns as a shortcut, and do not expose confirmatory outcomes to the planner.
2. Separate the zero statistical null from economically worthwhile alternatives. Zero defines the null; it is not a plausible effect for power. Sweep effect assumptions and report minimum detectable Delta S jointly with A-mean sensitivity across attainable session counts. Never select an attractive effect because it makes the gate pass. Economic relevance needs independently documented support, not the exposed $21,889.76 baseline outcome.
3. Calibrate type-I error and interval coverage over both branches of the composite union null: Delta S <= 0 with positive A expectancy, and nonpositive A expectancy with positive Delta S. Include their boundaries, intersection and nuisance-dependence cases; simulating only both-zero panels is insufficient. Calibrate joint power under alternatives satisfying both conditions, including block-length conflicts and zero-SD failures. Record Monte Carlo uncertainty, invalid-draw rates and dependence sensitivity. Do not silently discard invalid draws to improve coverage or power.
4. Produce a hash-bound sweep artifact and gate verdict. The final gate acceptance criteria, chosen feasible session count, calendar deadline and decision calibration must cite that artifact and its full commit, meeting repository policy on derived thresholds. Do not import the ongoing A/B study's 120-session target or nine-month deadline merely for convenience.
5. Return UNDERPOWERED if an independently supported effect cannot be detected within the operationally attainable sample, or retain NOT_ASSESSED if effect support/calibration is absent. Neither result permits a strategy test. The final policy must spell out the treatment of any borderline power classification before activation.

No numerical economic hurdle, minimum power acceptance threshold or confirmatory horizon is hand-set in this design. Their derivation is a required deliverable of the next, separately versioned calibration stage. Consequently this document deliberately cannot be passed to an evaluator as authorization.

## Activation contract and stopping

Activation requires one committed, reviewed record containing: this design's full revision; benchmark and accounting implementation hashes; passed parity and data-integrity checks; verified collection/provenance/roll readiness; the cited power-sweep artifact and verdict; the derived inference and sample decision settings; an immutable future start timestamp; and a fixed final eligible-session target and calendar stop date. The gate and evaluator must verify that record and refuse missing fields, mismatched hashes or denied power. No placeholder or date inferred at evaluation time is acceptable.

The sample stops at the first of the fixed target or deadline. There are no interim efficacy looks, optional extensions, reset after bad outcomes, pooling of inspected history or additional winning comparator choices. Before the final look, report collection counts and integrity only. If the deadline arrives short of the target, close INSUFFICIENT_SAMPLE; do not call it evidence for the hypothesis. Any design revision before activation starts a new version; any revision after activation ends confirmatory use under this design and must preserve the original record and outcome obligations.

## Evidence and current disposition

- [Collection readiness report](../research/mim_diagnostics/READINESS.md): operational blockers and their resolution evidence.
- [Completed payoff diagnostics](../research/mim_diagnostics/RESULTS.md): exposed historical accounting, concentration and sampled-risk limitations.
- [Existing comparison protocol documentation](../research/mim_comparison/README.md): inherited accounting and availability conventions, not authorization to reuse its efficacy rule.
- [Separate FOMC protocol](preregistration_evfade_fomc_prospective.md): unchanged.

This stage is complete when the readiness evidence and inactive design are reviewable and committed. The experiment remains inactive until the activation contract is fulfilled.
