# Kronos evaluation design

**PARK_PENDING_EVIDENCE.** `strategy_test_permitted=false`; `trading_authorized=false`.

The [offline comparison](run-20260923-reviewed/comparison.md) covers 3, 6 and 12 calendar months from **2026-09-23**, the next hypothetical full cash-RTH session after report generation. No horizon has been accepted for collection. The [machine-readable decision](run-20260923-reviewed/report.json), [prospective protocol](run-20260923-reviewed/protocol.json), [session ledger](run-20260923-reviewed/scenarios.json) and [source-pack evidence](run-20260923-reviewed/evidence.json) are bound by [COMPLETE.json](run-20260923-reviewed/COMPLETE.json). Reproduce with the [command guide](../../kronos-design.md) and a fresh destination.

| Horizon | Weekday ceiling | Cash-calendar capacity after initial warmup | With assumed quarterly resets | With resets, dated losses and 20 calibration sessions |
| --- | ---: | ---: | ---: | ---: |
| 3 months | 65 | 59 | 54 | 15 |
| 6 months | 129 | 119 | 109 | 46 |
| 12 months | 261 | 246 | 226 | 119 |

These are conditional planning counts, not observed or admitted samples. The cash-calendar proxy excludes published holidays and includes 13:00 early closes. A dated, product-specific MNQ calendar still needs verification. Warmup uses 128 completed same-contract 15-minute bars, with four bars remaining for the forecast horizon. Assumed rolls are December 10, March 11, June 10 and September 9; actual front-contract decisions must use contemporaneous evidence. The loss scenario resets context after each dated gap and reserves 20 eligible sessions for calibration. Those are stress assumptions, not measured failure rates or a powered calibration plan. Zero-calibration columns assume independent calibration is available before the horizon; its preparation time is additional and unknown. Missing flatten prices or unresolved positions may invalidate an evaluation population; the capacity calculation does not authorize dropping those outcomes.

Under quarterly-reset assumptions and independent-session standard errors, the detectable standardized effects are **0.4411, 0.3105 and 0.2156** for 3, 6 and 12 months. Apply these separately to the standard deviations of Kronos session net PnL and its paired difference from momentum. Multiplying standard errors by 1.5 or 2 increases these requirements proportionally. These known-variance normal approximations target 90% marginal power at one-sided alpha 0.025 each, with only an 80% joint-power lower bound. They are not dollar effects, empirical power, or economic usefulness thresholds.

The [bounded documentary inventory](sources-20260922/bounded-inventory.md) found no historical bar acquisition archive in the cited acquisition run. The source identity checks passed, but historical completion, first receipt and revision provenance remain unresolved. The selected future-shadow route would create its own provenance; current observations cannot repair historical files. No independent useful effects, Kronos/momentum variances, paired covariance or serial-dependence estimates were established. Another strategy's results were not transferred.

Reconsider when there is independently justified economic usefulness for both comparisons; defensible variance, covariance and dependence evidence or a separately preregistered and power-gated calibration route; account-specific costs and executable slippage evidence; validated product calendar and causal contract policy; qualified bar/quote collection, recovery, clock and isolated-authentication behavior; and a committed untouched evaluation population with exposure controls. Each item needs documentary support and independent review. The command intentionally cannot certify that a citation resolves a substantive evidence gap; a later supported review may recommend CONTINUE_TO_PREREGISTRATION without authorizing execution.

The protocol freezes input versions at each decision cutoff, records actual forecast timing, treats recovered observations as newly received, preserves strict-after-availability minute-open fills and scheduled flattening, and blocks ambiguous clock ordering. Proposed polling budgets and quote coverage remain unvalidated launch blockers. Integrity reports must conceal forecasts, positions and strategy outcomes. A published unrelated-strategy appendix encountered during inventory is disclosed in the exposure log; its values are excluded from the evidence and power inputs.

No collection, real-data forecasts, strategy outcomes, orders, service changes or paid purchases occurred. The design preregistration was committed before implementation; the frozen model, strategy, comparison arms, replay and earlier published readiness artifacts remain unchanged.

Verification in the isolated worktree: **255 tests passed**, with targeted flake8 and mypy clean. All 13 findings from three independent reviewers were corrected and confirmed closed; none deferred. See [review disposition](review-disposition.json), [verification record](verification-worktree.json) and [delivery manifest](DELIVERY.json).

Merged-main verification also passed **255 tests**, lint, and type checks. A fresh offline run reproduced all reviewed artifact content except its generation timestamp. See [main verification](verification-main.json). No service restart or other operational launch occurred.
