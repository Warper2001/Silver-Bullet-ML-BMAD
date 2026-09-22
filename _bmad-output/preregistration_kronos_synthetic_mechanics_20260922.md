---
id: kronos_synthetic_mechanics_20260922
scope: SYNTHETIC_MECHANICS_ONLY
economic_evaluation: false
trading_authorized: false
strategy_test_permitted: false
power_status: NOT_APPLICABLE_SYNTHETIC_SOFTWARE_VERIFICATION
---

# Kronos standalone replay mechanics preregistration

Record before implementation. This is software verification with generated prices, not a strategy test or adoption of live strategy parameters. No historical or sealed data, credentials, training, trader imports, service changes, or orders are authorized. Existing HOLD_EVALUATION / UNASSESSABLE findings remain in force. Any economic test requires its own admitted sample, justified costs/target effect, frozen comparisons, committed preregistration and experiment-specific power gate first. No economic acceptance thresholds are set here.

Candidate rule specified by the user: pinned Kronos-small and tokenizer, 128 completed 15-minute bars, four future bars, seeds 0/1/2, temperature 1, top-p .9, top-k 0, sample_count 1. Average all three terminal closes; above/below/equal latest observed close requests +1/-1/0 MNQ. Any invalid path requests flat. Compare fixed trailing-one-hour momentum and always-flat arms through identical accounting and eligibility. No filters, optimization, variable sizing, leverage or stops.

Use explicit timezone-aware New York RTH session schedules and normalized contract-labelled minute-open bars. Context only includes completed scheduled bars, resets on contract change, and never splices identities. No new forecast with horizon beyond session close. Fill at first valid minute open strictly after close plus explicit modeled latency. Pending targets execute in availability order; scheduled flatten at final-minute open overrides and cancels remaining orders. A contract change with outstanding exposure aborts rather than inventing a close. Missing/malformed minutes abort with positions and pending orders preserved. Unavailable final-minute data cannot create an exit.

Accounting uses $2/point and .25-point MNQ ticks, one contract, adverse tick slippage and fees per filled side. Repeat targets hold; reversals close then open and charge both sides. Report realized and unrealized PnL, costs, turnover, equity and drawdown. Execution fixtures (including latency, fees, slippage) are deliberately synthetic software examples, not market assumptions or economic thresholds.

Verification: hand-computed entries/holds/reversals/exits and mark-to-market; causality/delay; DST/early close; warmup/gaps/duplicates/malformed candles/roll; invalid paths and all-seed averaging; deterministic mock and cached-only model integration; existing readiness and inference regressions; independent review. Fresh hash-manifested runs retain protocol, input fixture, raw forecasts, decisions, fills, equity and completion status with SYNTHETIC_MECHANICS_ONLY and authorization false. CLI offers bundled synthetic fixtures only, stub default, optional offline cache adapter, never arbitrary historical input.
