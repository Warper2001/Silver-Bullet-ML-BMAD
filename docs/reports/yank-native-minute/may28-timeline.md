# May 28 native trade-print timeline

The native prints place conservative entry-supporting evidence before the stop crossing for all three distinct trades in the reviewed minute replay. This includes the no-ML trade whose modeled fill and exit share the 20:21 minute. This resolves the observed trade-print sequence; hypothetical fills, queue position and stop execution prices remain unverified. Research status remains **HOLD_VALIDATION**. The replay, its ambiguity annotations and modeled P&L are unchanged.

All timestamps below are capture-clock UTC on May 28, 2025. Record indices are zero-based positions after the DBN metadata in `native/GLBX-20260907-NWS3PA9QPX/glbx-mdp3-20250528.mbo.dbn.zst`.

| Case | First trade exactly at entry limit | First trade strictly above entry limit | First trade at or above stop | Through-to-stop elapsed |
|---|---|---|---|---|
| Both arms: limit 21,463; stop 21,511.50 | 18:57:23.049187224 at 21,463; record 15,715,565 | 18:57:23.049200341 at 21,463.25; record 15,715,570 | 19:08:26.217205523 at 21,511.50; record 16,246,690 | 663.168005182 seconds |
| ML only: limit 21,359.75; stop 21,397.75 | 20:02:26.151776820 at 21,359.75; record 19,322,750 | 20:02:26.151790213 at 21,360; record 19,322,753 | 20:18:09.107028150 at 21,397.75; record 19,538,007 | 942.955237937 seconds |
| No ML only: limit 21,408; stop 21,440.50 | 20:21:27.942583595 at 21,408; record 19,638,095 | 20:21:27.942601066 at 21,408.25; record 19,638,105 | 20:21:40.073985304 at 21,440.75; record 19,652,120 | 12.131384238 seconds |

For the no-ML 20:21 case, entry-touch and entry-through records share an event ending at record 19,638,126. The stop-crossing print belongs to a later event ending at record 19,652,122. The entry event completes before the stop event begins. Capture timestamps, exchange timestamps and native record order all place the through print first. The first stop-crossing print is **0.25 points above** the modeled stop; this is an observed print, not a replacement execution price or a slippage estimate.

Orders were assessed only after their signal bars completed: 18:53:00 for the common trade, 20:02:00 for the ML-only trade and 19:52:00 for the no-ML-only trade. Fixed additional delays of 0, 100 and 500 milliseconds produce the same supporting records in all cases. No stop-level print precedes the first through print within these arrival-to-exit windows, and no target-level print appears anywhere in the windows. Each window ends at the end of the original modeled exit minute; this follow-up does not reassess the full pending lifetime.

| Case | Volume exactly at entry limit | Volume strictly above entry limit |
|---|---:|---:|
| Both arms | 232 | 26,800 |
| ML only | 28 | 11,963 |
| No ML only | 12 | 2,947 |

Volumes are observed T-print contract totals across each entire arrival-to-exit window, including prints after the first stop crossing through the exit minute's end. They are not displayed liquidity at arrival or an allocation to our five-contract orders. Touch alone remains inconclusive. Trading strictly above a short limit supplies supporting evidence under a no-market-impact assumption; it does not prove a fill. This focused inspection does not reconstruct the book or assess spread, marketability, queue position or stop-order matching.

## Reproduction and verification

[The JSON evidence](may28-timeline.json) contains all nine case/delay combinations, raw bytes and both timestamps for nine selected T prints, six LAST event endpoints, source indices and sequences. The script scanned all 22,081,340 records in the pinned May 28 MBO file, excluding snapshots and F notifications from trade evidence. All 15 selected records were independently decoded with the official DBN decoder. Acquisition and reviewed-replay hashes were checked before and after each run.

Two runs using 65,536-record and 32,768-record chunks produced byte-identical JSON; the second ran with optimized Python. Output SHA256: `7b1f8546feb84d76b81466fe204a2b58b52e917164aecc7eeacbebe871de060d`.

```sh
cd /root/Silver-Bullet-ML-BMAD-yank-minute
/root/Silver-Bullet-ML-BMAD/.venv/bin/python docs/reports/yank-native-minute/inspect-may28.py > /tmp/yank-may28-reproduced.json
cmp docs/reports/yank-native-minute/may28-timeline.json /tmp/yank-may28-reproduced.json
```

The [read-only inspection script](inspect-may28.py) pins the reviewed replay and uses the existing acquisition pins. No strategy code, frozen orders, model, historical result or input data was changed. The earlier 198-test result applies to the unchanged minute-builder/replay implementation; this follow-up was verified directly against native bytes and repeated with a different chunk size.
