# Provisional account and capital comparison

These are synthetic software demonstrations, not performance forecasts. See [input assumptions](account-inputs/README.md) and the [provider question packet](../../research/project_goals/account-questions.md). No account-specific applicability or invoice is confirmed. Automated Live remains unavailable.

The CLI overall insufficient-data status refers to the absent portfolio inputs in these account-only invocations; the account-scenario component statuses below are separate.

## Synthetic positive path

| Scenario | Model status | Ending phase | Ending balance | Operator withdrawals | Modeled operating cost |
|---|---|---|---:|---:|---:|
| combine | PROVISIONAL_PATH_ONLY | xfa_standard | 2800.0 | 1800.00 | 227.00 |
| xfa_standard | PROVISIONAL_PATH_ONLY | xfa_standard | 6000.0 | 1800.00 | 29.00 |
| xfa_consistency | PROVISIONAL_PATH_ONLY | xfa_consistency | 6000.0 | 1800.00 | 29.00 |
| self_funded | PROVISIONAL_PATH_ONLY | self_funded | 18000.0 | 0.00 | 0.00 |
| direct_capital_5000.0 | PROVISIONAL_PATH_ONLY | self_funded | 13000.0 | 0.00 | 0.00 |
| direct_capital_20000.0 | PROVISIONAL_PATH_ONLY | self_funded | 28000.0 | 0.00 | 0.00 |

[Full phase/floor/payout paths](account-hypothetical_positive/evidence.json). Unknown ending balances after breach must not be replaced with the pre-loss balance.

## Synthetic adverse path

| Scenario | Model status | Ending phase | Ending balance | Operator withdrawals | Modeled operating cost |
|---|---|---|---:|---:|---:|
| combine | INCOMPLETE_OR_STOPPED | combine | None | 0.00 | 78.00 |
| xfa_standard | INCOMPLETE_OR_STOPPED | xfa_standard | None | 0.00 | 29.00 |
| xfa_consistency | INCOMPLETE_OR_STOPPED | xfa_consistency | None | 0.00 | 29.00 |
| self_funded | INCOMPLETE_OR_STOPPED | self_funded | None | 0.00 | 0.00 |
| direct_capital_5000.0 | INCOMPLETE_OR_STOPPED | self_funded | None | 0.00 | 0.00 |
| direct_capital_20000.0 | PROVISIONAL_PATH_ONLY | self_funded | 10000.0 | 0.00 | 0.00 |

[Full phase/floor/payout paths](account-hypothetical_adverse/evidence.json). Unknown ending balances after breach must not be replaced with the pre-loss balance.

## Capital limitations

The $5,000/$10,000/$20,000 direct cases hold cash P&L and quantity fixed. They measure sensitivity to initial cash only. Margin requirements, intraminute loss excursions, execution capacity, taxes, agreements and actual hosting/data charges are unresolved, so no minimum safe capital or income target is certified.

Published Combine/XFA/API fees are modeled assumptions. Actual broker fill fees are separately reconciled in the [audit](audit-reviewed/report.md). Written provider answers and agreement versions belong in the question packet register before route decisions.
