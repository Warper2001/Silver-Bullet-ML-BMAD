# Prospective evidence specification for Kronos

Status: PROPOSED_ONLY. `strategy_test_permitted=false`; `trading_authorized=false`.
No collector has been launched by this document. The current readiness probe is finite and does not supply an evaluation population.

Historical admission remains unresolved. If original acquisition-linked timestamps, completion/arrival records, causal contract selection and untouched exposure cannot be established, collect a new population under a separately committed protocol. Current-feed observations cannot repair historical provenance.

Before collection, archive the dated exchange calendar and the chosen cash-equity RTH research window, including holidays, early closes and DST conversions. Obtain provider confirmation of minute labels, completion flags, timestamp timezone and revision behavior. Maintain a per-session schedule with its source hashes. Unexpected closures and missing final-minute data must produce incomplete sessions, not inferred exits.

Define and commit the eligible contract universe and causal roll policy before any scoring. Archive symbol metadata and expiration for each candidate. Base any volume-derived roll decision on information received before that session; preserve candidate-contract volume evidence, its acquisition times, the actual decision and its source hashes. Verify each prior close belongs to that session's chosen contract. A successful MNQZ26 request establishes neither a front-month policy nor historical roll dates.

The collection implementation should write original ordered responses and immutable request identities, explicit contract, request and receipt UTC times, monotonic elapsed times, provider timestamp/status fields and transport outcomes. Retain revisions as separate observations. Store raw collection separately from derived bars; transformations must reference original observation hashes. Do not overwrite duplicates, fill gaps, backdate receipt times or treat an old bar as completed without evidence. Authentication renewal would require a separately reviewed isolated design; do not reuse the readiness probe to mutate shared token files.

For each derived completed 15-minute bar, retain the constituent minute identities, minute-label interpretation, completion evidence and latest required receipt time. Decision availability must include input availability and actual computation completion. Archive quotes needed to evaluate strict-after-availability execution, scheduled flattening and adverse slippage. TradeStation SIM fills alone cannot validate market slippage. Confirm the account's commission tier, clearing charges, exchange/NFA treatment and any applicable charges independently before adopting costs.

Keep warmup observations distinguishable from eligible outcome sessions. Preserve the frozen model, source/checkpoint hashes, three seeds, mean-terminal forecast rule, momentum and flat arms. Contract changes reset the context and require a flat account, no pending orders and no partial bucket. Do not inflate sample size with seeds, overlapping forecasts or multiple arms.

Commit access controls and an exposure log before evaluation begins. Unscored integrity checks may inspect data quality; strategy outcomes must remain hidden until the economic protocol and power gate admit a fixed untouched population. Freeze missing-session handling, stopping rules and the dependence estimator before examining outcomes. Do not repeatedly stop when PnL becomes favorable.

Obtain economically useful dollar effects independently for both Kronos net session PnL and its paired advantage over momentum. Establish variance/covariance and dependence evidence using a separate, disclosed population or external evidence. If these inputs cannot be justified, keep power UNASSESSABLE and do not select an effect merely because the collected sample can detect it. Two one-sided tests allocate 0.025 alpha each and target 0.90 marginal power; this provides at least 0.80 joint power by the union bound under the stated design. Statistical targets are not economic usefulness thresholds.

The next milestone may propose a real-data replay only after those blockers are resolved in a separately committed economic preregistration and experiment-specific power gate. This specification grants no permission to score, submit orders, restart services or run an ongoing collector.
