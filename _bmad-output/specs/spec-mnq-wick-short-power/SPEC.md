# MNQ wick-short detectability gate

## Contract

This is an outcome-blind, RTH-only feasibility gate for a reconstructed,
unfiltered wick-short model. It reports the minimum **net** dollars per trade a
future test could detect; it does not calculate actual strategy returns.

Use right-labelled five-minute candles made from close-stamped RTH minutes
09:31 through 16:00 America/New_York.  A candle signals when
`B = abs(close-open) > 0`, `U = high-max(open,close) >= 2B`, and
`D = min(open,close)-low <= 0.1U`. Candle colour and trend are not filters.
For a signal bar, the hypothetical position holds from the next bar's open to
its close. Signals through 15:50 are allowed, ending at 15:55. One MNQ
contract has point value $2; stops, discretionary exits, and sizing are absent.

The input is `/root/mnq_historical.json`, restricted while parsing to timestamps
strictly before 2026-03-01T00:00:00Z. This is a reconstruction assumption, not
a claim to reproduce an unidentified video. A red-day filter is a separate
future gate.

## Firewall

The gate may count actual signal times but must never pair an actual signal to
its own next-bar open/close movement. For every full RTH session circularly
shifted by k sessions, k=5 through N-5 inclusive, each actual signal's holding
slot is instead read from the shifted session. Identity pairing is rejected.
The shifts estimate transferred dispersion only; they are sensitivity draws,
not additional observations.

## Decision

For each shift, calculate hypothetical short dollars as
`2 * (next_open - next_close)`. Cluster the mean standard error by original
signal session and by its ISO calendar week, use the larger SE, and report its
median and 10th--90th percentile over shifts. Approximate MDE is
`(z_0.95 + z_0.80) * SE`; show point equivalents and gross movement after costs
$1.22, $2.22, and $3.22. Normal approximation, transferred dispersion and cost
scenarios are assumptions, not validated power, fills, or account fees.

Every completed valid gate returns `POWER_UNDETERMINED` and
`evaluation_allowed=false`: no independent effect estimate or calibrated
execution model exists. Stop there: no profitability backtest, holdout access,
optimization, live-code changes, or deployment.
