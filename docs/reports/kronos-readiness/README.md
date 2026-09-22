# Kronos TradeStation SIM readiness — 2026-09-22

**HOLD_EVALUATION. Power UNASSESSABLE.** `strategy_test_permitted=false`; `trading_authorized=false`.

The [readiness decision](run-20260922-reviewed/report.md), [evidence register](run-20260922-reviewed/evidence-register.json), [candidate protocol](run-20260922-reviewed/candidate-protocol.json) and [conditional power analysis](run-20260922-reviewed/power.json) are fresh artifacts. The assessment snapshots its verified documentary inputs and binds them in [COMPLETE.json](run-20260922-reviewed/COMPLETE.json). The delivery manifest additionally binds the source archive, original probe/timing artifacts and implementation.

The milestone did not score historical strategy outcomes, reopen sealed data, submit orders, start an ongoing collector or restart services. The synthetic replay and existing preflight remain unchanged.

The single finite SIM probe completed 180 latest-three-minute-bar GETs plus one symbol-metadata GET between 14:20:29 and 14:35:25 UTC, taking 895.533 seconds. Every request returned HTTP 200. SIM metadata identified MNQZ26, CME, expiration December 18, 2026, point value $2 and tick size 0.25. The [probe summary](sources-20260922/probe-summary.json) links its original raw artifact directory and manifest. There were 18 distinct provider minute timestamps and 181 changes across repeated records. Of 17 changes after a previous Closed status, 15 changed only IsEndOfHistory; two also changed price/volume/tick fields. No gaps were observed between collected timestamps. This is a small current-feed observation, not historical completeness or execution validation.

The initial probe sampled its receipt times after local response processing. Its delay values are logger receipt upper bounds, not precise network arrival or adopted latency. A formatting-only pass shortly after launch also means finalization hashes describe formatted source rather than the exact source bytes loaded at launch. Both limitations are preserved in the [provenance note](sources-20260922/probe-provenance-note.json). Review fixes were developed in a second worktree, with no further changes to the running probe's files. The final implementation records receipt immediately after transport and tests that boundary. The capture was not repeated or extended beyond its registered request budget.

Pinned offline inference used the existing synthetic fixture at the first eligible decision, March 9 at 09:45 New York, with a four-bar horizon ending 10:45. Startup took 7.464 seconds; three complete three-seed decisions took 1.544, 1.401 and 1.350 seconds. All nine paths passed candle validation. These repetitions measure computation only and add no independent market observations. No latency value was adopted. Original artifacts are in `_bmad-output/kronos-readiness-timing-20260922/`.

Official [TradeStation pricing](https://www.tradestation.com/pricing/) and [exchange fees](https://www.tradestation.com/pricing/exchange-execution-and-clearing-fees/) were archived on September 22. Published micro commissions vary from $0.25 to $0.50 per side, with tier-dependent clearing charges; the MNQ nonmember exchange fee is $0.35 per side. The account tier and exact NFA side convention remain unresolved. [SIM documentation](https://api.tradestation.com/docs/fundamentals/sim-vs-live/) describes instant simulated fills, which do not establish market slippage. The [cost worksheet](sources-20260922/fee-assessment.json) separates these components without adopting a total cost.

The frozen future outcomes are session-level, one-contract Kronos net PnL, momentum net PnL and their paired difference. Both Kronos expectancy and incremental expectancy must be positive; Sharpe and drawdown remain separate outcomes. Conditional planning allocates family alpha 5% equally to two one-sided comparisons and targets 90% marginal power, giving at least 80% joint power by the union bound. Detectable standardized effects depend on assumed session count and SE inflation. These statistical choices are not economically useful effects or admission thresholds.

The exact remaining blockers are:

- Original acquisition records binding historical files to provider requests and responses.
- Authenticated historical minute-label convention, timezone, completion, first availability and revisions.
- Dated MNQ session calendars, early closes and DST boundaries for every eligible session.
- Causal per-session contract selection and correctly matched prior closes.
- An untouched evaluation population, research-exposure audit and pretraining-exposure limitations.
- Account-specific costs, independently supported slippage and a justified availability/latency protocol.
- Independently justified useful effects, variance/covariance, dependence and eligible session count for both comparisons.

No sessions have been admitted. If acquisition-linked historical evidence cannot be recovered, the next step is a separately preregistered [prospective evidence collection](../../kronos-prospective-collection.md), followed by an economic protocol and experiment-specific power gate. This milestone cannot authorize a real-data scoring run.

Verification: 201 Kronos/readiness tests passed in the fixes worktree; flake8 and mypy passed. Three independent review lenses identified issues, all accepted fixes were implemented, and follow-up review confirmed the credential, receipt and nested-response fixes. No implementation findings were deferred. See the committed implementation spec for per-finding triage. Main-checkout verification is recorded in the delivery manifest.

Reproduction commands and source-pack schema are in [the command guide](../../kronos-readiness.md). Use a fresh output directory. The reviewed assessment's `sources/` contains hash-named copies of every verified input; `evidence-register.json` maps original identities to those copies, allowing an independent documentary root/source pack to be reconstructed without opening price files. Raw probe and timing runs remain in the local `_bmad-output/` paths bound by the delivery manifest.
