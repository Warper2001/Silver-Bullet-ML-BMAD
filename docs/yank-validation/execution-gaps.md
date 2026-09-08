# Incremental execution-gap assessment

**HOLD_VALIDATION.** This command supplements the frozen five-case audit. It preserves its eight archived arm orders, all 30 timing scenarios, both timestamp interpretations, 0/100/500 ms delays, and every 240-opportunity pending lifetime. It never replaces frozen outcomes or computes revised P&L. Blocker labels overlap and must not be summed as independent failed scenarios.

```sh
cd /root/Silver-Bullet-ML-BMAD-yank-validation
/root/Silver-Bullet-ML-BMAD/.venv/bin/python src/cli/check_yank_execution_gaps.py \
  --manifest docs/yank-validation/execution-gaps-manifest.json \
  --input-dir /root/Silver-Bullet-ML-BMAD/data/yank/databento-pilot-20260907 \
  --output-dir /tmp/yank-gap-assessment-new
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest -q tests/unit/yank_deployed_validation/test_gaps.py
```

The explicit manifest pins the frozen report and the necessary May 22/May 28 MBO and May 28 status files. The report location is relative to the manifest or absolute; native filenames are relative to the input directory and cannot escape it. The frozen report binds those native hashes to the original acquisition. Source and report directories are protected, existing/symlink destinations are refused, and publication uses an exclusive atomic rename after rechecking inputs and implementation. A failed scan publishes no successful report. Linux `renameat2` support is required. No credentials, API requests, live trader, archive replay, or original audit command are invoked.

The private loader reuses the unchanged audit's completed-event scanner and Book. An observational Book subclass records exact native start/terminator/sequence/event references, and the original Gaps callback contributes validation failures. It begins with each daily reset/snapshot; it never seeks into an unreconstructed book. Runs of consecutive invalid completed events retain the first event, last event, count and first following valid completed event. Native event identities count the same snapshot/live F_LAST boundaries as the frozen audit. A next valid event demonstrates recovery only; it cannot repair earlier invalid books or shorten a scenario's pending lifetime.

Case 1/end/0 ms is the sole unassessable case-1 scenario. Its incomplete arrival event is identified with the strictly preceding completed event and later terminator. That evidence explains the missing arrival book; it cannot retroactively establish one at arrival. Cases 3–5 contain the other 18 unassessable outcomes. Each retains distinct native invalid-book and observed nontrading evidence. The May 28 market-event pause and scheduled closure are observed status, not inferred acquisition holes. The frozen reports contain no missing scheduled minutes for these scenarios. Coverage for the remainder of each full window, including next-day expiry, is retained from the pinned original audit; this incremental scan does not claim a new complete audit of unscanned days.

Every scenario in `report.json` has its original outcome, full interval and schedule. Each blocker contains source SHA256, native record/event references or source status records, precise interval intersection, recovery evidence, disposition and next requirement. The separate queue disposition remains unobservable even for supported scenarios: a hypothetical historical order never had an observed queue position. Independent provider recovery evidence and documented channel/book semantics are required for invalid native intervals; prospective broker acknowledgements cannot manufacture historical hypothetical fills.

The new scanner is tested for a delayed terminator spanning arrival, native chunk splits, two consecutive invalid completed events, first valid recovery, exact nanosecond status boundaries, overlapping labels, full-window retention and fresh output protection. Existing audit commands and canonical outputs remain unchanged. A local PASS refers to engineering checks and traceable attribution only.

## Measured result

The [retained report](../reports/yank-execution-gaps/report.md) covers all 30 scenarios; its complete machine evidence is losslessly compressed in `docs/reports/yank-execution-gaps/report.json.gz`. The manifest records its uncompressed canonical SHA256; the original local output is `/tmp/yank-gap-assessment-b`. Decompress to inspect every scenario and native reference. Independent array decoding verified all 156 unique MBO references, including exact native event ordinal, record index, sequence, flags and both clocks.

Case 1's event 3050592 starts at native record 3531144, `2025-05-22T11:59:59.999855656Z`, and completes at record 3531145, `12:00:00.000269993Z`. Arrival at `12:00:00Z` splits that event. The strictly preceding completed event ends at record 3531143, but the event already in progress prevents that book from being an available completed arrival observation under the frozen audit policy. No audit defect was found.

May 28 has **6,880 invalid completed books in 48 consecutive-event runs**. Forty-seven runs intersect the market-event pause; the final such run recovers at `20:20:05.063493652Z`. The scheduled pre-open run starts at `21:45:00.314813250Z`, has 2,062 invalid completed books, and recovers at `22:00:00.014597030Z`. Status resumes trading at `22:00:00.012420840Z`, so status alone would not establish the book recovery boundary. All 18 case-3/4/5 scenarios retain these invalid-book and nontrading blockers across their entire original pending window.

The two inspected files contain 48,562,656 native records. Their 8,600 locked/crossed completed books include 1,720 May 22 events outside the case-1 pending windows. This scoped count does not replace the frozen audit's 11,819 global events; the other 3,219 remain in the retained audit of unscanned days. Later crossing evidence, recovered books and narrower scans leave **11 supported / 19 unassessable** unchanged.
