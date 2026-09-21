# 2025 commodity curve coverage and reported activity

Scope: descriptive development evidence. Verdict: **HOLD-DATA**. No strategy signals, rankings, returns or P&L were calculated.

Verified source: 214 contracts, 40,375 bars, 3,007 root/timestamp observations. All 3,059 source requests and six reports passed integrity verification.

## Definitions and limits

- A candidate pair has the same root and exact broker timestamp, with broker expirations 90–180 calendar days apart. Rows after the broker expiry date are excluded from pair counting. This does not establish notice/last-trade safety.
- Active means both contracts report volume > 0 AND open interest > 0. It is a minimal activity screen, not evidence of fillability, spreads, executable liquidity, or timely open-interest publication.
- Repeat-filtered additionally excludes a leg at its third or later identical close on consecutive root-observed timestamps. Missing observations reset the run. This deliberately conservative sensitivity check does not prove staleness; legitimate unchanged prices can fail it.
- Fixed pair uses the earliest-expiring observed unexpired contract and its earliest 90–180-day deferred contract (symbol breaks expiry ties), selected BEFORE the activity screens. No alternative pair is substituted after a failure. This is an availability-based diagnostic, not the protocol’s calendar-safe nearby selection.
- Internal missing counts absent contract observations between its first and last bar against the root’s union of observed timestamps. Outside-span absence is excluded because listing dates are unknown. These are not missing exchange sessions; absence shared by every contract is invisible to this measure.
- Monthly snapshots use each root’s last observed timestamp in the month. Different sector clocks and missing publication timestamps prevent interpreting them as a synchronized, causal rebalance.
- All filters are descriptive sensitivity checks, not optimized thresholds or modifications to the frozen universe. Source artifacts and the research protocol remain unchanged.

## How much pair coverage survives?

Counts are root/timestamp observations, not independent samples.

| Root | Observed timestamps | Any pair | Both volume > 0 | Both OI > 0 | Both screens | + repeat filter | Fixed pair: both screens | Fixed pair: + repeat filter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CL | 251 | 251 | 251 | 251 | 251 | 251 | 251 | 251 |
| NG | 251 | 251 | 251 | 251 | 251 | 251 | 251 | 251 |
| RB | 251 | 251 | 251 | 251 | 251 | 251 | 251 | 251 |
| HO | 251 | 251 | 251 | 251 | 251 | 251 | 251 | 251 |
| HG | 251 | 251 | 251 | 251 | 251 | 251 | 251 | 251 |
| ZC | 250 | 250 | 250 | 250 | 250 | 250 | 250 | 250 |
| ZW | 250 | 250 | 250 | 250 | 250 | 250 | 250 | 250 |
| ZS | 250 | 250 | 250 | 250 | 250 | 250 | 250 | 250 |
| ZM | 250 | 250 | 250 | 250 | 250 | 250 | 250 | 250 |
| ZL | 250 | 250 | 250 | 250 | 250 | 250 | 249 | 249 |
| LE | 251 | 251 | 251 | 251 | 251 | 251 | 251 | 251 |
| HE | 251 | 251 | 251 | 251 | 251 | 251 | 251 | 251 |

Any-pair activity coverage: 3,007/3,007; with the repeated-close sensitivity filter: 3,007/3,007. Fixed-pair activity coverage: 3,006/3,007; with repeat filtering: 3,006/3,007.

## Pair depth and data-quality concentrations

| Root | Pairs per timestamp min / median / max | Zero-volume rows | Zero-OI rows | Internal missing | Repeat-flag rows | Longest equal-close run | Fixed-pair changes |
|---|---|---:|---:|---:|---:|---:|---:|
| CL | 23 / 42 / 59 | 3 | 0 | 3 | 0 | 2 | 12 |
| NG | 24 / 42 / 59 | 0 | 0 | 0 | 0 | 2 | 12 |
| RB | 27 / 42 / 59 | 514 | 0 | 8 | 0 | 2 | 11 |
| HO | 27 / 42 / 59 | 130 | 0 | 6 | 0 | 2 | 11 |
| HG | 24 / 42 / 59 | 695 | 102 | 3 | 0 | 2 | 12 |
| ZC | 4 / 8.0 / 10 | 0 | 0 | 0 | 2 | 3 | 5 |
| ZW | 4 / 8.0 / 10 | 36 | 0 | 5 | 0 | 2 | 5 |
| ZS | 7 / 12.0 / 17 | 52 | 0 | 2 | 0 | 2 | 7 |
| ZM | 10 / 20.0 / 26 | 71 | 0 | 6 | 0 | 2 | 8 |
| ZL | 10 / 20.0 / 26 | 98 | 0 | 0 | 0 | 2 | 8 |
| LE | 5 / 7 / 8 | 7 | 7 | 1 | 0 | 2 | 7 |
| HE | 10 / 15 / 18 | 238 | 141 | 18 | 2 | 3 | 10 |

Total zero-volume rows: 1,844; zero-OI rows: 250; internal missing observations: 52; rows after broker-expiry date: 0. These categories can overlap. Fixed-pair changes compare consecutive observations; they are not trade or roll counts.

## Last-observation monthly snapshots

| Month | Roots: any pair | Roots: active pair | Roots: repeat-filtered pair | Roots: fixed active pair | Fixed inactive roots | Observed UTC dates |
|---|---:|---:|---:|---:|---|---|
| 2025-01 | 12 | 12 | 12 | 12 | none | 2025-01-31 |
| 2025-02 | 12 | 12 | 12 | 12 | none | 2025-02-28 |
| 2025-03 | 12 | 12 | 12 | 12 | none | 2025-03-31 |
| 2025-04 | 12 | 12 | 12 | 12 | none | 2025-04-30 |
| 2025-05 | 12 | 12 | 12 | 12 | none | 2025-05-30 |
| 2025-06 | 12 | 12 | 12 | 12 | none | 2025-06-30 |
| 2025-07 | 12 | 12 | 12 | 12 | none | 2025-07-31 |
| 2025-08 | 12 | 12 | 12 | 12 | none | 2025-08-29 |
| 2025-09 | 12 | 12 | 12 | 12 | none | 2025-09-30 |
| 2025-10 | 12 | 12 | 12 | 12 | none | 2025-10-31 |
| 2025-11 | 12 | 12 | 12 | 12 | none | 2025-11-28 |
| 2025-12 | 12 | 12 | 12 | 12 | none | 2025-12-31 |

## Files and reproducibility

- `summary.json`: metrics, source hashes, analyzer hash and integrity results.
- `daily.csv`: every root/timestamp, pair counts and fixed-pair identities/activity outcomes.
- `contracts.csv`: contract-level missingness, reported activity and repeated-close diagnostics.
- `transitions.csv`: changes in the fixed diagnostic pair. No executions are implied.
- `analyze.py`: run from the repository root with `PYTHONPATH=. .venv/bin/python docs/reports/commodity-curve-feasibility-2025/analyze.py --output /tmp/curve-feasibility-new`. Use an output directory without existing report files; writes fail rather than overwrite.

Settlement provenance, publication/revision history, exchange calendars, delivery safety, account costs and executable quotes remain unresolved. Broad activity coverage does not change HOLD-DATA.
