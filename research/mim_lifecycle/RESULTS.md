# MIM-NB trade-lifecycle study

Completed 2026-09-12. Early losses and temporary recoveries are both common. The study does not identify a justified early-exit rule. The clearest next research question is whether a causal measure of weakening follow-through can reduce profit giveback while retaining the strategy's large winners.

## Findings

| Observation | Historical result |
|---|---:|
| Final winning trades | 462 of 801 (57.7%) |
| Winners with a negative net close at some point | 415 of 462 (89.8%) |
| Winners with a negative net close in the first 30 minutes | 387 of 462 (83.8%) |
| Winners' median gross adverse excursion | $71.00 |
| Losing trades with a positive net close before exit | 298 of 339 (87.9%) |
| Losing trades that recovered to nonnegative after a negative close, then still lost | 273 of 339 (80.5%) |
| Net profit from the largest 41 trades (hindsight top 5%) | $29,530.66 |
| Net profit from the remaining 760 trades | -$7,640.90 |
| Largest-trade group with a negative net close | 34 of 41 (82.9%) |

The biggest winners require room to fluctuate. Among the 41 largest trades, median gross adverse excursion was $59.50, the 90th percentile was $386.50, and the maximum was $658.50. These are descriptions of known winners, not proposed stop distances or a way to recognize winners in advance. The existing stop is anchored to the signal price; a 250-point signal-anchored stop is not a guarantee of a $500 maximum loss relative to the modeled entry fill.

Breakeven recovery is a weak basis for confidence: 80.5% of eventual losing trades also recovered to nonnegative at some point after being red. Likewise, the fact that 89.8% of winners spent time red does not give a red trade an 89.8% chance of winning. At five minutes, 192 of 402 red trades ultimately won (47.8%).

## Early-red checkpoint results

Each row conditions on trades with an observable completed close at that elapsed minute, then selects those with negative net liquidation marks. Earlier exits are excluded.

| Minutes after entry | All observable trades | Red trades | Eventually profitable | Mean final net | Mean change after checkpoint |
|---:|---:|---:|---:|---:|---:|
| 5 | 801 | 402 | 47.8% | -$16.54 | +$20.04 |
| 15 | 800 | 383 | 47.5% | -$23.68 | +$43.46 |
| 30 | 775 | 369 | 43.9% | -$40.66 | +$46.78 |
| 60 | 748 | 353 | 35.7% | -$79.67 | +$23.66 |
| 120 | 695 | 294 | 32.0% | -$105.67 | +$29.21 |

These cohorts often finish with losses, but their average modeled P&L improves after the observation. Final profitability and the value of continuing from the current loss are different questions. The continuation column compares the observed close mark with the eventual baseline fill; it does not model an executable early exit or prove continuation is optimal.

## Trades that have never reached net profit

This cohort uses only the completed closes available by the checkpoint. A trade that was previously profitable and has fallen back into the red is outside this group.

| Minutes after entry | Never-positive-yet trades still observable | Eventually profitable | Mean final net | Mean change after checkpoint |
|---:|---:|---:|---:|---:|
| 5 | 203 | 45.8% | -$27.94 | +$17.11 |
| 15 | 126 | 39.7% | -$64.09 | +$22.03 |
| 30 | 95 | 31.6% | -$95.81 | +$17.76 |
| 60 | 60 | 23.3% | -$140.97 | +$9.91 |
| 120 | 45 | 22.2% | -$146.55 | +$12.64 |

The probability of a profitable finish declines with prolonged failure to reach profit. However, average remaining change stays positive at every displayed checkpoint. A simple time-without-profit exit therefore has no affirmative support from these descriptive averages. Later cohorts are small and selected by survival; the study attaches no significance claim to these differences.

Time to first positive completed close also overlaps substantially: the median is one minute among winners and two minutes among losers that ever reached a positive close. Forty-one losing trades never had a positive close. Winner versus loser distributions alone cannot provide a causal intervention rule.

## Recommended follow-up

Keep baseline trading and risk rules unchanged in simulation. These observations provide no basis for discretionary stop overrides when price appears to recover.

The next bounded diagnostic should examine **profit giveback after an initial advance**, measuring the state at completed bars: favorable excursion already achieved, current retreat from that excursion, elapsed time, and recent directional progress. Any predictive target should be remaining payoff and adverse risk from that moment, since final win/loss can be misleading after losses are already incurred. Track the same conditions on large winners to quantify the opportunity cost of interference.

This is an untested hypothesis, not a recommendation to install a trailing stop. Before evaluating a specific trading intervention, freeze the proposed rule and execution conventions, run the required power gate, account for selection and dependence, and use validation data the choice did not see. A different continuation profile for long and short trades is a secondary hypothesis; current side totals alone are insufficient to choose different rules.

## Study question

Which early losses are normal for eventual winners, and which observations deserve a separate test as potential signs of a failed trade?

This study describes the unchanged baseline's 801 one-contract trades across 1,323 eligible sessions, 2021-01-15 through 2026-08-27. The source is the frozen primary execution scenario: completed-bar decisions, market fill at the open one full minute later, $2.24 roundtrip friction. No new strategy, stop, filter or position size is evaluated.

## How to read the evidence

Minute checkpoints describe trades still observable at that elapsed minute. They exclude earlier exits and uncertain stop-minute closes. Final winners and large-winner groups are selected with hindsight; their behavior cannot identify winners in advance. Net marks reserve the full roundtrip cost and represent hypothetical liquidation values, not additional trades.

Favorable and adverse excursions use only bars during the modeled holding period and known fills. An opening exit excludes its bar's later prices. A catastrophe-stop minute has unknown price ordering, so its entire OHLC range is excluded and only its known exit fill is used. Excursions for those trades are observed lower bounds. Time to a stop is an interval rather than a fabricated exact timestamp.

## Research boundaries

Early loss, breakeven recovery and final profitability answer different questions. A trade can recover temporarily and still lose; a profitable trade can spend time red first. Remaining P&L after an observation is a descriptive continuation outcome, not an executable alternative-exit backtest.

Existing history is exposed. This study performs no parameter search or hypothesis test and makes no superiority claim. Any strategy experiment arising from it needs a separate preregistration, power gate, execution model and validation on data the choice did not see. The original A/B prospective protocol and all live behavior remain unchanged.

## Artifacts and verification

- [Sealed report](runs/20260912T154857-diagnostic-b54c02459db4/report.md), [standalone HTML and charts](runs/20260912T154857-diagnostic-b54c02459db4/report.html), and [full summary](runs/20260912T154857-diagnostic-b54c02459db4/summary.json).
- [Gross excursion plot](runs/20260912T154857-diagnostic-b54c02459db4/excursions.svg) and [first-positive-time plot](runs/20260912T154857-diagnostic-b54c02459db4/first_positive.svg).
- The run includes full observed paths, 801 lifecycle records, 4,005 checkpoint records, source/spec snapshots and a manifest. Its 1,323-session grid includes 544 flat sessions and reconciles to $21,889.76 net. There are 71 stop exits with intraminute uncertainty.
- Independent formulas reproduced all 801 net/MFE/MAE/first-positive-close values, all five checkpoint sample sizes, winner rates, recovery counts and mean continuation changes, and terminal-fill accounting. Source/data hashes and the sealed inventory passed verification. The SVGs parsed and the excursion chart was visually inspected.
- All 111 relevant tests passed: 56 lifecycle tests and 55 robustness tests. Three review lenses completed; every actionable finding was fixed and tested, with none deferred.
- Run completion SHA256: `5dba0f781b0dac1c155223a9cddc82440006aeb54f973f7b7539e13484d69e71`. Large run artifacts remain local in ignored `runs/`; code, specification and this summary are versioned.
