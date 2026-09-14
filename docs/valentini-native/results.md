# Native MNQM5 profile measurement — May 2025 pilot

The native reconstruction reconciles exactly, and six complete regular Globex sessions support the specified profile measurement. The uniform OHLCV approximation differs from native traded-volume profiles enough that it should remain a separately labeled approximation in future research. This study does not establish signal differences, an edge, expected returns, or statistical power.

The scan processed 237,146,989 MBO records, including 7,526,752 nonsnapshot T prints and 13,318,999 contracts of volume. All 13,440 native minute bars reconciled on OHLCV, trade count, native-record digest and availability; status, definitions, coverage and exchange-clock diagnostics also matched the frozen reconstruction. A separate byte-offset oracle matched price-level histograms for 120 minutes containing 24,164 T prints. Reconciliation establishes consistency with the captured dataset, not independent exchange-feed completeness.

| Profile level | Exact agreement | Median absolute difference | Mean absolute difference | 95th-percentile absolute difference |
|---|---:|---:|---:|---:|
| VAL | 13.17% | 3 ticks | 13.14 ticks | 53 ticks |
| VAH | 15.23% | 3 ticks | 12.97 ticks | 41 ticks |
| POC | 2.44% | 11 ticks | 35.71 ticks | 112.20 ticks |

Each row has 8,274 comparisons. One tick is 0.25 index points. Comparisons use each eligible minute's preceding available profile, the same lower-price tie rules and 70% value area. Each session's empty opening prefix is counted separately. These repeated snapshots are dependent observations within six sessions, not 8,274 independent trials. Quantiles are descriptive, with linear interpolation; they are not adoption thresholds or confidence bounds.

Session variation is substantial; May 29 contributes the largest mean boundary differences. All six admitted sessions are shown below, each with 1,379 comparisons. This variation limits how far the pooled means can be generalized.

| Session | Mean absolute VAL difference | Mean absolute VAH difference | Mean absolute POC difference |
|---|---:|---:|---:|
| May 20 | 4.67 ticks | 4.77 ticks | 18.23 ticks |
| May 21 | 8.77 ticks | 10.32 ticks | 23.88 ticks |
| May 22 | 4.22 ticks | 3.47 ticks | 41.07 ticks |
| May 23 | 10.69 ticks | 10.40 ticks | 23.03 ticks |
| May 29 | 46.49 ticks | 45.15 ticks | 84.17 ticks |
| May 30 | 4.01 ticks | 3.71 ticks | 23.85 ticks |

Eligible dates are May 20, 21, 22, 23, 29 and 30, covering 8,280 minute bars. The remaining 5,160 observed bars are retained as excluded-session records. May 19 lacks its first two Globex hours and contains status conflicts. May 26 and 27 have unresolved holiday session-reset evidence; May 28 contains an unscheduled status interruption. Exclusions were determined from coverage and calendar evidence without calculating strategy outcomes. They should not be adopted as favorable trading-day filters.

The [dated calendar evidence](calendar-sources.md) corrects the previous reliance on a living FAQ: CME eliminated the afternoon equity-index pause in 2021. Regular May 2025 sessions therefore contain no such scheduled pause. Seven exact-reference opening exceptions explain status-message receipt after the scheduled exchange boundary; they do not allow other interruptions to be ignored. Eight sourced closing exceptions corroborate the verified candidates' end boundaries. The audit does not certify every intervening maintenance or weekend interval.

One completed event became available 2,948 nanoseconds after its minute boundary, on May 27. That session is already excluded by the calendar gate. The event remains in the audit report; synthetic tests verify that an unavailable prefix cannot be compared retrospectively and that a later eligible boundary can use the completed prefix.

Use native volume at price for the next properly gated strategy study when suitable data is available. Retain the proxy only as an explicit approximation comparison; these measured boundary differences cannot quantify trade-selection or P&L effects. Before testing signals or returns, resolve calendar/feed admission, pre-register the methodology and cost assumptions, and run the repository's power gate using appropriate independent calibration. This reused pilot supplies measurement evidence, not independent validation of a selected strategy.

The [audit command and artifact contract](../valentini-reclaim.md#native-measurement-audit) document reproduction. The committed [final machine report](../../_bmad-output/valentini-native-20260914/run-final/report.json), [artifact manifest](../../_bmad-output/valentini-native-20260914/run-final/manifest.json), and [independent verification](../../_bmad-output/valentini-native-20260914/verification-final.json) identify the measured output. Full outputs, input/code hashes, oracle scripts and execution logs are retained alongside them. The final report remains `market_evaluation: NOT_ADMITTED`; no live strategy or service was changed.

Validation:119 focused tests passed, together with Black, flake8 and strict mypy. The full native audit and independent oracle passed after review fixes. The final manifest SHA-256 is `23f2205518d7be751fa147274441efe0af89c2a2aec52f76a8810e443a4b1667`.

The subsequent [conditional power feasibility gate](../valentini-power-results.md) is complete: power remains undetermined and strategy evaluation is not admitted. It supplies hypothetical sample-size planning without calculating signals or returns.
