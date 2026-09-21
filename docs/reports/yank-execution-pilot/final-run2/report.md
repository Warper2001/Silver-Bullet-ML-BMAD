# YANK frozen-order execution pilot

PASS_AUDIT_CHECKS — HOLD_VALIDATION

Five frozen cases; both bar-label interpretations and 0/100/500 ms delays are retained. Capture time is an observable proxy, not broker arrival time. Strict trade-through supports a no-impact hypothesis; it does not prove a fill or queue position.

| Case | Signal UTC | Interpretation | Delay ms | Outcome | Arrival spread | At / through volume |
|---|---|---|---:|---|---:|---:|
| case-1 | 2025-05-22T12:00:00+00:00 | start | 0 | supported | 0.75 | 380 / 659040 |
| case-1 | 2025-05-22T12:00:00+00:00 | start | 100 | supported | 0.75 | 380 / 659040 |
| case-1 | 2025-05-22T12:00:00+00:00 | start | 500 | supported | 0.75 | 380 / 659040 |
| case-1 | 2025-05-22T12:00:00+00:00 | end | 0 | unassessable | unavailable | 380 / 656761 |
| case-1 | 2025-05-22T12:00:00+00:00 | end | 100 | supported | 0.75 | 380 / 656761 |
| case-1 | 2025-05-22T12:00:00+00:00 | end | 500 | supported | 0.75 | 380 / 656761 |
| case-2 | 2025-05-27T00:24:00+00:00 | start | 0 | supported | 0.5 | 96 / 8712 |
| case-2 | 2025-05-27T00:24:00+00:00 | start | 100 | supported | 0.5 | 96 / 8712 |
| case-2 | 2025-05-27T00:24:00+00:00 | start | 500 | supported | 0.75 | 96 / 8712 |
| case-2 | 2025-05-27T00:24:00+00:00 | end | 0 | supported | 0.75 | 96 / 8712 |
| case-2 | 2025-05-27T00:24:00+00:00 | end | 100 | supported | 0.75 | 96 / 8712 |
| case-2 | 2025-05-27T00:24:00+00:00 | end | 500 | supported | 0.5 | 96 / 8712 |
| case-3 | 2025-05-28T18:53:00+00:00 | start | 0 | unassessable | 0.5 | 931 / 223219 |
| case-3 | 2025-05-28T18:53:00+00:00 | start | 100 | unassessable | 0.25 | 931 / 223219 |
| case-3 | 2025-05-28T18:53:00+00:00 | start | 500 | unassessable | 0.5 | 931 / 223219 |
| case-3 | 2025-05-28T18:53:00+00:00 | end | 0 | unassessable | 0.5 | 931 / 222735 |
| case-3 | 2025-05-28T18:53:00+00:00 | end | 100 | unassessable | 0.5 | 931 / 222735 |
| case-3 | 2025-05-28T18:53:00+00:00 | end | 500 | unassessable | 0.25 | 931 / 222735 |
| case-4 | 2025-05-28T19:52:00+00:00 | start | 0 | unassessable | 0.5 | 12 / 233427 |
| case-4 | 2025-05-28T19:52:00+00:00 | start | 100 | unassessable | 0.5 | 12 / 233427 |
| case-4 | 2025-05-28T19:52:00+00:00 | start | 500 | unassessable | 0.25 | 12 / 233427 |
| case-4 | 2025-05-28T19:52:00+00:00 | end | 0 | unassessable | 0.25 | 12 / 232399 |
| case-4 | 2025-05-28T19:52:00+00:00 | end | 100 | unassessable | 0.5 | 12 / 232399 |
| case-4 | 2025-05-28T19:52:00+00:00 | end | 500 | unassessable | 0.25 | 12 / 232399 |
| case-5 | 2025-05-28T20:02:00+00:00 | start | 0 | unassessable | 0.5 | 22 / 262736 |
| case-5 | 2025-05-28T20:02:00+00:00 | start | 100 | unassessable | 0.5 | 22 / 262736 |
| case-5 | 2025-05-28T20:02:00+00:00 | start | 500 | unassessable | 0.5 | 22 / 262736 |
| case-5 | 2025-05-28T20:02:00+00:00 | end | 0 | unassessable | 0.75 | 34 / 262075 |
| case-5 | 2025-05-28T20:02:00+00:00 | end | 100 | unassessable | 0.5 | 34 / 262075 |
| case-5 | 2025-05-28T20:02:00+00:00 | end | 500 | unassessable | 0.75 | 34 / 262075 |

Reconciliation counts (full diagnostics retain both conventions):

```json
{
  "capture/end": {
    "contaminated_minutes": 3,
    "covered_minutes": 12697,
    "different_OHLC_minutes": 2066,
    "exact_OHLC_minutes": 10631,
    "native_T_minutes_without_original_bar": 743,
    "original_minutes": 12697
  },
  "capture/start": {
    "contaminated_minutes": 1,
    "covered_minutes": 12688,
    "different_OHLC_minutes": 12688,
    "missing_native_T_minutes": 10,
    "native_T_minutes_without_original_bar": 752,
    "original_minutes": 12698
  },
  "exchange_diagnostic/end": {
    "contaminated_minutes": 1,
    "covered_minutes": 12697,
    "different_OHLC_minutes": 2053,
    "exact_OHLC_minutes": 10644,
    "native_T_minutes_without_original_bar": 743,
    "original_minutes": 12697
  },
  "exchange_diagnostic/start": {
    "contaminated_minutes": 1,
    "covered_minutes": 12688,
    "different_OHLC_minutes": 12688,
    "missing_native_T_minutes": 10,
    "native_T_minutes_without_original_bar": 752,
    "original_minutes": 12698
  }
}
```

case-1: no-ml order 1, ml050 order 1
- start +0ms gaps: none detected
- start +100ms gaps: none detected
- start +500ms gaps: none detected
- end +0ms gaps: arrival_book_unavailable, arrival_during_incomplete_event
- end +100ms gaps: none detected
- end +500ms gaps: none detected
case-2: no-ml order 2, ml050 order 2
- start +0ms gaps: none detected
- start +100ms gaps: none detected
- start +500ms gaps: none detected
- end +0ms gaps: none detected
- end +100ms gaps: none detected
- end +500ms gaps: none detected
case-3: no-ml order 3, ml050 order 3
- start +0ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- start +100ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- start +500ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- end +0ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- end +100ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- end +500ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
case-4: no-ml order 4
- start +0ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- start +100ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- start +500ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- end +0ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- end +100ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- end +500ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
case-5: ml050 order 4
- start +0ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- start +100ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- start +500ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- end +0ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- end +100ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval
- end +500ms gaps: invalid_completed_book_or_event, native_evidence_gaps_in_pending_interval, non_trading_or_unknown_status_in_pending_interval

May 28 crossing evidence:

- case-3 start interval 2025-05-28T18:58:00.000000000Z to 2025-05-28T19:10:00.000000000Z: entry_evidence_before_barrier_not_fill_proof.
  Qualifications: none detected.
  first_entry: 2025-05-28T18:58:00.002201411Z, price 21463.0, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 15748037, event glbx-mdp3-20250528.mbo.dbn.zst:13912883.
  first_strict_entry: 2025-05-28T18:58:00.791097467Z, price 21463.5, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 15748432, event glbx-mdp3-20250528.mbo.dbn.zst:13913248.
  first_stop: 2025-05-28T19:08:26.217205523Z, price 21511.5, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 16246690, event glbx-mdp3-20250528.mbo.dbn.zst:14359856.
  first_target: unavailable in this interval.
- case-3 end interval 2025-05-28T18:57:00.000000000Z to 2025-05-28T19:09:00.000000000Z: entry_evidence_before_barrier_not_fill_proof.
  Qualifications: none detected.
  first_entry: 2025-05-28T18:57:23.049187224Z, price 21463.0, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 15715565, event glbx-mdp3-20250528.mbo.dbn.zst:13883081.
  first_strict_entry: 2025-05-28T18:57:23.049200341Z, price 21463.25, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 15715570, event glbx-mdp3-20250528.mbo.dbn.zst:13883081.
  first_stop: 2025-05-28T19:08:26.217205523Z, price 21511.5, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 16246690, event glbx-mdp3-20250528.mbo.dbn.zst:14359856.
  first_target: unavailable in this interval.
- case-4 start interval 2025-05-28T20:22:00.000000000Z to 2025-05-28T20:23:00.000000000Z: ambiguous_same_event_or_equal_capture_time.
  Qualifications: none detected.
  first_entry: 2025-05-28T20:22:00.001735518Z, price 21449.25, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 19668669, event glbx-mdp3-20250528.mbo.dbn.zst:17422926.
  first_strict_entry: 2025-05-28T20:22:00.001735518Z, price 21449.25, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 19668669, event glbx-mdp3-20250528.mbo.dbn.zst:17422926.
  first_stop: 2025-05-28T20:22:00.001735518Z, price 21449.25, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 19668669, event glbx-mdp3-20250528.mbo.dbn.zst:17422926.
  first_target: unavailable in this interval.
- case-4 end interval 2025-05-28T20:21:00.000000000Z to 2025-05-28T20:22:00.000000000Z: entry_evidence_before_barrier_not_fill_proof.
  Qualifications: none detected.
  first_entry: 2025-05-28T20:21:27.942583595Z, price 21408.0, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 19638095, event glbx-mdp3-20250528.mbo.dbn.zst:17398289.
  first_strict_entry: 2025-05-28T20:21:27.942601066Z, price 21408.25, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 19638105, event glbx-mdp3-20250528.mbo.dbn.zst:17398289.
  first_stop: 2025-05-28T20:21:40.073985304Z, price 21440.75, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 19652120, event glbx-mdp3-20250528.mbo.dbn.zst:17409415.
  first_target: unavailable in this interval.
- case-5 start interval 2025-05-28T20:03:00.000000000Z to 2025-05-28T20:20:00.000000000Z: entry_evidence_before_barrier_not_fill_proof.
  Qualifications: none detected.
  first_entry: 2025-05-28T20:03:06.933964693Z, price 21359.75, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 19338554, event glbx-mdp3-20250528.mbo.dbn.zst:17139930.
  first_strict_entry: 2025-05-28T20:03:06.933979868Z, price 21360.0, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 19338561, event glbx-mdp3-20250528.mbo.dbn.zst:17139930.
  first_stop: 2025-05-28T20:18:09.107028150Z, price 21397.75, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 19538007, event glbx-mdp3-20250528.mbo.dbn.zst:17314048.
  first_target: unavailable in this interval.
- case-5 end interval 2025-05-28T20:02:00.000000000Z to 2025-05-28T20:19:00.000000000Z: entry_evidence_before_barrier_not_fill_proof.
  Qualifications: none detected.
  first_entry: 2025-05-28T20:02:26.151776820Z, price 21359.75, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 19322750, event glbx-mdp3-20250528.mbo.dbn.zst:17125779.
  first_strict_entry: 2025-05-28T20:02:26.151790213Z, price 21360.0, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 19322753, event glbx-mdp3-20250528.mbo.dbn.zst:17125779.
  first_stop: 2025-05-28T20:18:09.107028150Z, price 21397.75, native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst record 19538007, event glbx-mdp3-20250528.mbo.dbn.zst:17314048.
  first_target: unavailable in this interval.

May 28 native crossing timelines are in report.json and event-extracts.jsonl. Equal capture times or a shared event remain ambiguous. Actual fills remain unobserved. Status interruptions and coverage gaps affect assessability.

Minute-by-minute unchanged-price comparisons appear in reconciliation.jsonl. Empirical agreement cannot establish independent bar-label provenance.

Archived order, fill and exit economics are retained as historical references. The historical $4 cost is not a verified broker charge. No revised strategy return is computed.

Input and implementation hashes and decoder versions appear in report.json. Artifact hashes appear in artifacts.json. A passing audit reports successful integrity/implementation checks, not strategy validation.
