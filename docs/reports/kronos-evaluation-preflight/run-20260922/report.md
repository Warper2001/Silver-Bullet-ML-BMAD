# Kronos evaluation preflight

**HOLD_EVALUATION — power UNASSESSABLE. No strategy test permitted.**

Candidate post-revision interval: 2025-09-10 through 2026-08-27; at most 252 weekdays before exchange-calendar, data and research-exposure exclusions.

Admitted untouched sessions: 0; actual eligible count and strategy power: unknown.

## Illustrative planning, not a measured edge

One-sided normal known-variance test; illustrative family alpha 5%, target power 80%, one comparison, no SE inflation. Net daily-return Sharpe annualized with sqrt(252). These are sensitivity assumptions, not adopted admission thresholds.

| Assumed annual net Sharpe | Required days | 252-day years | Power at weekday ceiling |
| --- | ---: | ---: | ---: |
| 0.5 | 6233 | 24.73 | 12.6% |
| 1.0 | 1559 | 6.18 | 26.0% |
| 1.5 | 693 | 2.75 | 44.2% |
| 2.0 | 390 | 1.55 | 63.9% |

Full sensitivity (SE inflation and comparison families) is in report.json.

## Qualifications

- Planning scenarios are unsealed assumptions, not adopted thresholds or Kronos estimates.
- 252 days/year is a scaling convention, not a historical exchange calendar.
- Post-revision weekday ceiling includes holidays and unknown gaps; not effective N.
- Revision dates come from public commit metadata, not verified training cutoffs.
- Later dates reduce temporal pretraining concerns but do not erase local research exposure.
- Zero admitted sessions means none certified here, not proof that none could qualify.
- Three seeds and overlapping forecast bars do not multiply market evidence.
- Normal known-variance calculations do not replace a dependence-aware strategy power gate.
- No prices, forecasts, returns, PnL, credentials or sealed data were read.
