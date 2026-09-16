---
id: mnq-wick-short-power-20260916
status: preregistered-power-gate
created: 2026-09-16
strategy_parameters_changed: false
aligned_strategy_outcomes_authorized: false
---

# MNQ wick-short detectability gate registration

The frozen contract is `_bmad-output/specs/spec-mnq-wick-short-power/SPEC.md`
with its mechanics, data-handling and statistical-assumptions companions. This
registration precedes all market-data gate calculations. The gate admits only
pre-cutoff, complete, single-contract RTH sessions; it detects the specified
wick geometry only to obtain event timing and frequency.

It must never calculate a signal's own following-bar return. Instead it must
use circular whole-session shifts from five through N minus five sessions and
refuse an identity pairing. For each shift it transfers the same holding slot
to another session, calculates `2 * (next_open-next_close)`, and uses the
larger of session- and ISO-week-clustered standard errors. The reported MDE is
`(z0.95+z0.80)*SE`, with $1.22, $2.22 and $3.22 cost scenarios. These are
normal-approximation and transferred-dispersion scenarios, not a calculation of
actual return, validated power, fills or fees.

No independently supported expected effect or calibrated execution model is
available. Therefore a syntactically valid completed gate is always
`POWER_UNDETERMINED`, with `evaluation_allowed=false`. It cannot authorize a
strategy evaluation. Stop after the JSON and Markdown reports. Do not access
sealed holdout data, optimize parameters, change live code, or deploy.
