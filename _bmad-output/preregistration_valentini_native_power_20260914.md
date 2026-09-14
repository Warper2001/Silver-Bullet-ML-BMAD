---
id: valentini-native-power-20260914
status: preregistered-planning-gate
created: 2026-09-14
strategy_parameters_changed: false
aligned_strategy_outcomes_authorized: false
---

# Native Globex reclaim: power feasibility gate

This registration precedes the new gate's numerical calculations. The previous measurement study already disclosed six eligible sessions, 8,274 dependent profile comparisons, and profile differences. Those facts are reused openly; they are not new independent evidence of an edge. No strategy signals, fills or aligned returns have been computed in this work.

## Question and information boundary

Can the existing native pilot support an adequately powered test of positive mean net session profit for the frozen one-contract Globex reclaim strategy? Existing instrument, profile, entry and exit rules are unchanged. This gate uses session counts and source provenance only. It does not decode prices, generate signals, compute returns, estimate win rate or infer absorption.

The inspected local native archives are under `data/yank` and `data/tick`. The May2025 pilot has six measurement-eligible regular sessions. Other discovered native requests are short June22 windows and 28 windows chosen around previous parity fills; these do not supply a second admitted full-session calibration sample. Raw TradeStation minute OHLCV is not traded volume at price and cannot substitute for this native study. The search is limited to these inspected sources; it is not a claim that no other dataset exists.

Use the committed measurement manifest SHA-256 `23f2205518d7be751fa147274441efe0af89c2a2aec52f76a8810e443a4b1667`, its artifact bindings, session ledger and provenance. Validate every listed artifact's bytes; parse only report, sessions and provenance, not histogram/snapshot/observed-bar rows. File hashing does read bytes but computes no price-path statistic. Preserve all original exclusions. Read no sealed holdout, trade ledger, credentials or live trader. No network or purchases during the gate.

## Statistical model and prespecified calculations

The unit for the proposed test is one complete session's net dollars at one contract, including zero-trade sessions. This gate does not construct that quantity. Define hypothetical standardized net effect d = mean net session dollars / population standard deviation of net session dollars. Neither numerator nor denominator is estimated here. No conversion from gross effect or per-trade R is justified, and no borrowing of another strategy's assumed effect is allowed.

For sensitivity calculations only, assume independent identically distributed normal session outcomes with unknown variance. Test H0: mean <=0 versus mean >0 at one-sided alpha0.05, with target power0.80. These are prospective statistical design settings, not trading thresholds or data-selected acceptance filters. Under that model, df=n-1, critical=t.ppf(0.95,df), and power=nct.sf(critical,df,d*sqrt(n)). This is conditional model power, not a measured property of the strategy. Dependence, nonnormality and selection are not calibrated; iid results are not asserted as a universal bound.

Report:
- The d yielding80% model power for each integer independent-session count from2 through the six observed eligible sessions. n<2 is unassessable for this model.
- A complete declared sensitivity table for d in {0.10,0.20,0.30,0.50,1.00}: power at the observed eligible count and minimum integer n>=2 reaching80% model power. These are hypothetical mathematical scenarios, not pessimistic/central/optimistic effect estimates, trading thresholds or a recommendation to adopt an effect.
- Required n is solved by bracketing/binary search, checking power(n)>=0.80 and power(n-1)<0.80. Cap the computational search at1,000,000 sessions; report an explicit unresolved-above-cap result if necessary. Detectable d uses a bracketed numerical root with finite checks.
- A known-variance normal formula cross-check, clearly separate from the unknown-variance finite-sample t result. Synthetic tests independently verify the nct calculation by integrating over a chi-square variable and verify the minimum-n boundary.

## Gate decision and next work

No transferable effect, independent calibration, calibrated dependence model or validated net-cost mapping exists for this exact construct in the inspected evidence. Therefore the operational verdict is POWER_UNDETERMINED and evaluation_allowed=false regardless of any hypothetical sensitivity cell. Do not label unknown power UNDERPOWERED and do not issue POWERED from these assumptions. The gate must stop before strategy testing.

The result must identify concrete inputs needed to proceed: representative full native sessions with contract/calendar/feed evidence; separate calibration and validation roles fixed before outcomes; defensible net effect and cost assumptions; and a suitable dependence-aware test. The conditional sample-size table assists planning but is not a data-purchase authorization or guaranteed sufficient horizon. A subsequent preregistration is required before any strategy outcome test. No calendar restriction will be selected based on favorable returns.

## Sources and reproducibility

NIST's sample-size discussion explains why effect, variance, alpha and beta must be specified and gives the known-variance cross-check: https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm . SciPy's primary documentation defines the noncentral t parameterization used here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html . Accessed2026-09-14. Local SciPy1.17.1 is already installed; install nothing.

Commit this registration before gate calculations, and commit reviewed gate code/tests before its actual metadata-driven run. Bind registration, code, measurement and inventory hashes in the result. Preserve hypotheses and every sensitivity cell; do not revise assumptions after seeing calculated output. Synthetic fixtures are allowed during implementation. No existing market-evaluation refusal is removed.
