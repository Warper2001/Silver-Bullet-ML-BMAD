# MIM-NB fixed-filter evaluation — 2026-09-12

None of the four predefined filters qualifies for the historical shortlist. Every candidate reduced daily Sharpe. Keep the unchanged baseline in the existing simulation/prospective comparison; this study supports no strategy replacement or size increase.

The comparison covers 1,323 common eligible sessions from 2021-01-15 through 2026-08-27, at one MNQ contract. Primary fills use the open one full minute after the decision close, with $2.24 per round trip. Sharpe uses daily net P&L divided by fixed $10,000, sample standard deviation and sqrt(252) annualization. All existing history was previously exposed.

| Arm | Daily Sharpe | Change vs A | Total net | Daily closing-equity max drawdown | Trades |
|---|---:|---:|---:|---:|---:|
| A: unchanged MIM-NB | 1.1385 | — | $21,889.76 | $2,437.26 | 801 |
| R: regression agreement | 1.0710 | -0.0675 | $19,736.12 | $3,069.52 | 762 |
| E: path efficiency | 0.9083 | -0.2302 | $15,757.66 | $5,392.42 | 616 |
| F: fresh-breakout reset | 1.1194 | -0.0191 | $21,544.96 | $2,429.52 | 796 |
| P: breakout persistence | 0.9931 | -0.1454 | $18,714.82 | $2,526.56 | 782 |

The agreed screen required Sharpe at least 1.3385, drawdown no more than $1,949.81, and profit at least $16,417.32. No candidate meets the Sharpe or drawdown hurdle. E also fails profit retention. At $6.24 per round trip, every arm retains positive mean daily net P&L, but every candidate still has lower Sharpe than A.

For each expected stationary-bootstrap block length (5, 10 and 20 sessions), 20,000 synchronized paired resamples used seed 7. All four candidates' adjusted 98.75% Sharpe-difference intervals include zero under every block length; there were no undefined draws. These results do not establish that every filter is harmful, but they provide no qualifying evidence of improvement. Full intervals, both timing models and all cost scenarios are in the reports below.

## What the experiments explain

- Regression agreement reduces trading modestly but worsens drawdown and retains only 90.2% of baseline profit. Its stronger 2021–2022 results do not persist across the remaining years.
- Path efficiency cuts trades from 801 to 616, but drawdown more than doubles. Lower exposure alone does not deliver a smoother profitable equity curve.
- Fresh-breakout reset changes very few trades. It sacrifices $344.80 of total profit for only $7.74 less daily drawdown, with slightly lower Sharpe.
- Breakout persistence delays or rejects entries without improving the observed risk/profit balance.
- Large winners matter: A earns $39,552.92 on its best 67 sessions and loses $17,663.16 across the rest. R and E retain 93.2% and 83.1% of net P&L on those same winning dates. Their improvement outside those dates is too small to compensate. This is descriptive attribution, not a rule for identifying winners in advance.

## Recommended next work

1. Continue the frozen A/B prospective simulation and collect the agreed eligible sessions without changing parameters. Historical baseline profits are not proof of future edge.
2. Prioritize execution reconciliation: compare recorded decision times, actual data availability and simulated fills, including the inherited queued-reversal-after-stop convention. Any execution change needs its own frozen comparison.
3. Measure intraday equity excursions and realistic execution friction before evaluating capital or scaling. The reported daily drawdown does not bound intraday losses or establish that a $6,000 intervention threshold is safe.
4. Do not tune these thresholds or combine the filters in response to these results. A future experiment should have a distinct rationale and a predeclared validation protocol that accounts for this exposed history.

These are research next steps, not additional work activated by this evaluation. No production code, orders, services, live sizing or original prospective protocol changed.

## Reproducibility and verification

- [Frozen protocol](../../_bmad-output/specs/spec-mim-nb-sharpe-experiments/protocol.md).
- [Baseline audit](runs/20260912T151751-audit-210f6b1b24/report.md): all original columns reconcile across 7,938 daily scenario rows, 31,752 decisions and 3,190 fills; accounting tolerance $1e-8, indicators 1e-9, discrete values exact.
- [Full historical report](runs/20260912T151842-run-f8608e71fb/report.md) and [standalone HTML with equity/drawdown plots](runs/20260912T151842-run-f8608e71fb/report.html).
- [Separate evaluation](runs/20260912T152100-evaluate-0918c79842/report.md) reproduced the completed experiment from its sealed artifacts.
- Independent verification passed all three sealed inventories, reproduced all 30 scenario metrics and daily cost accounting, and found 14 reports/ledgers byte-identical between run and evaluation. The original baseline inventory, source hashes and input data hash remain unchanged. Completion SHA256: run `b747ceff679c3874a6236d71eb960c4a68890aa5267aaa0575070b34b097a9a7`; evaluation `eb3a8807462eef3479d756e6e54f78ec0a81144bf7254a64e0876a1116e1e869`.
- The run contains immutable manifests, source/protocol snapshots, exclusions, feature/decision/fill/trade ledgers, daily pairs, yearly metrics and diagnostics. Large run artifacts remain local under ignored `runs/`; source and this summary are versioned.
- All 167 relevant tests passed, including the original comparison suite and 55 new experiment tests. Three review lenses completed; actionable findings were fixed and tested. The baseline's accepted queued reversal can still fill after a catastrophe stop during latency unless the daily guard has deactivated trading; preserving it is necessary for exact reference compatibility.
