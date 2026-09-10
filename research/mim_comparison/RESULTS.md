# Published strategy versus MIM-NB: current assessment

The published guarded rules (B) changed mean net P&L by **−$3.99 per eligible session relative to deployed-rule A** in the primary historical model, over **1,323 paired sessions**. Both had positive standalone historical expectancy. This is exposed-history diagnosis, not a prospective replacement decision.

[Immutable run report](runs/20260910T210224-historical-f3950efb68/report.md) · [Manifest](runs/20260910T210224-historical-f3950efb68/manifest.json) · [Frozen protocol](runs/20260910T210224-historical-f3950efb68/protocol.json) · [Commands and implementation](/root/Silver-Bullet-ML-BMAD/research/mim_comparison/README.md) · [Reference differences](evidence/reference-audit.md)

## One-contract historical results

Primary fills use the open one full minute after the decision close; friction is $2.24 per round trip. A/B retain the deployed signal-price catastrophe anchor and realized-gross daily guard. C/D remove those controls. All include eligible no-trade sessions as zero outcomes.

| Arm | Mean net/session | Total net | Maximum daily-equity drawdown | Contract sides traded |
|---|---:|---:|---:|---:|
| A | $16.55 | $21,889.76 | $2,437.26 | 1,602 |
| B | $12.55 | $16,605.46 | $4,463.82 | 2,242 |
| C | $19.80 | $26,201.72 | $3,007.44 | 1,594 |
| D | $12.56 | $16,617.46 | $4,463.82 | 2,242 |

Contract expiry is conservatively inferred; missing or unverified closed weekdays, incomplete sessions and warmup exclusions are recorded. The existing acquisition ends August 28, 2026, with that last day incomplete. Recorded operational bars are reconciliation evidence, not added performance history.

## Execution and cost sensitivity

| Fill timing | Round-trip cost | A mean net | B mean net | Paired B−A |
|---|---:|---:|---:|---:|
| One full minute later | $2.24 | $16.55 | $12.55 | $-3.99 |
| One full minute later | $3.24 | $15.94 | $11.70 | $-4.24 |
| One full minute later | $6.24 | $14.12 | $9.16 | $-4.96 |
| Next-bar open (optimistic) | $2.24 | $18.76 | $15.19 | $-3.57 |
| Next-bar open (optimistic) | $3.24 | $18.15 | $14.34 | $-3.81 |
| Next-bar open (optimistic) | $6.24 | $16.34 | $11.80 | $-4.53 |

These costs are scenarios, not measured execution costs. Turnover includes both sides of reversals. The ledgers distinguish signal references, modeled fill prices and fill-time uncertainty.

## Mechanisms

Relative to A, gap anchoring alone changed mean net by $+0.08/session, VWAP construction/confirmation by $-0.06, and published exit behavior by $-4.83. The combined-minus-singles residual was $+0.81. The neutral-sample exit accounts for most of the historical shortfall. The residual also contains the disclosed first eligible-day author sigma warmup difference.

Earlier V1/V2 work already studied related exits; this report does not present those mechanisms as new discoveries. Large-winner removal and yearly results are in the immutable report; neither arm's aggregate profit is robust to removing its largest five percent of session winners.

## Sizing adaptation, separately

These are published unguarded signals with $100,000 initial equity and integer MNQ sizing. The full-notional comparator compounds its own equity at 1×, matching the authors' executable convention. The 2% target and 4× reference limit apply before integer rounding; actual rounded leverage is disclosed in `sizing.csv`.

| Sizing | Total return | Annualized realized volatility | Maximum daily-equity drawdown | Contract sides traded |
|---|---:|---:|---:|---:|
| Author full-notional (own-equity 1×) | 70.46% | 8.87% | $15,914.14 | 9,298 |
| Author volatility target | 104.93% | 14.04% | $50,696.22 | 18,644 |

These outcomes cannot satisfy the one-contract promotion hurdle or authorize larger live size.

## Prospective status and power

**No prospective replacement verdict is available.** The existing live bar CSV lacks contract identity; collection requires a research-side mapping or identified append-only record without changing production. No prospective sessions or deployment authorization are claimed.

The protocol was frozen at `2026-09-10T21:02:24.363841+00:00`. Its first-session rule and original nine-month deadline remain binding when shadow collection starts; delayed readiness does not move the horizon. The endpoint is 120 eligible paired sessions, with unavailable sessions reported and no interim efficacy decisions.

At 120 sessions, historical paired variability gives an approximate **$28.51/session detectable increment** at 80% power and two-sided 5% alpha; the estimated true mean needed to clear the $5 hurdle with that power is $33.51. The $5 effect is underpowered under these assumptions. The agreed horizon remains unchanged.

Final evaluation uses 20,000 stationary-bootstrap resamples, seed 7, mean block length five, with ten- and twenty-session sensitivity. Favorable intervals authorize only a recommendation for subsequent execution validation. Insufficient coverage or conflicting dependence conclusions remain inconclusive.

## Verification

The implementation includes source-extracted broker fixtures, recorded-log precision reconciliation, reference sequence checks, deterministic P&L/turnover tests, 120-session synthetic finalization, restart/correction tests and sandbox isolation tests. Synthetic test sessions are not prospective market observations. See the [verification record](VERIFICATION.md) for the exact final test result and [incomplete final decision report](runs/20260910T210450-evaluate-967b6ea2cb/report.md).
