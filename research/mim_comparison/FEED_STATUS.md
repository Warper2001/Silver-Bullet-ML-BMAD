# Existing-project feed integration — 2026-09-10

The contract adapter is implemented, reviewed, and initialized against the existing project files. The actual shadow collector accepted its output and produced an **incomplete/inconclusive** evaluation with **zero eligible prospective sessions**. These are historical replay observations. Continuous polling has not been started; no production service was changed.

## Verified result

- **Initial audit: 78,648 records mapped to MNQU26; 7,814 records excluded.**
- Subsequent bounded polls accepted ten additional after-hours bars: **78,658 mapped in total**, with exclusions unchanged. Two were timely at the adapter; no eligible prospective session resulted.
- All initial 78,648 emitted rows retained their original OHLCV and event/receipt timestamps. Independently checked 314,029 unique evidence byte offsets against the original runtime log.
- The shadow journal recorded 21,626 RTH observations as outside the frozen horizon and rejected 4,307 weekend rows. It contains no unidentified-input flags that would disqualify the collector state.
- A real adapter restart preserved the exact feed bytes and cursor, with no duplicate rows.
- **112 tests passed in 97.52 seconds**, including fresh appended observations, timing boundaries, duplicate roll messages, source mutation, partial writes, crash recovery, actual worker sandbox restrictions and adapter-to-shadow ingestion.
- Both adapter poll inventories and 38 files inventoried by the actual shadow/evaluation runs passed independent SHA256 checks.
- The original trading/collector source hashes, historical results, and governing protocol freeze `2026-09-10T21:02:24.363841+00:00` remain unchanged. Deadline: `2027-06-10T21:02:24.363841+00:00`.

| Exclusion reason | Records |
|---|---:|
| chain_break | 3 |
| duplicate_or_correction_first_observation_permanent | 1,604 |
| future_receipt_or_event | 1 |
| unknown_or_contradictory_signal_context | 6,206 |

## Evidence and outputs

- [Initial adapter report](runs/20260910-contract-feed/poll-20260910T213656053995-9a997a3a/report.md) and [manifest](runs/20260910-contract-feed/poll-20260910T213656053995-9a997a3a/manifest.json).
- [Restart report](runs/20260910-contract-feed/poll-20260910T214726717201-c532b5dc/report.md).
- [Independent row/evidence verification](runs/20260910-contract-feed/independent-verification.json).
- [Identified feed](runs/20260910-contract-feed/feed.csv), with causal log evidence and inferred-provenance labels.
- [Actual shadow output](runs/20260910T214803-shadow-47055eba62/report.md) and [governing freeze snapshot](runs/20260910T214803-shadow-47055eba62/freeze-snapshot.json).
- [Current decision report](runs/20260910T220846-evaluate-31b9429d4b/report.md).
- [Bounded shadow output](runs/20260910T220816-shadow-624c704d41/report.md) and [timing/integrity verification](runs/20260910-contract-feed-poll/initial-poll-verification.json).
- [Restarted bounded shadow output](runs/20260910T221058-shadow-1a7e830b36/report.md) and [evaluation](runs/20260910T221121-evaluate-f762ffc872/report.md).

Artifacts remain local under ignored `runs/` directories. The implementation is version-controlled.

## Subsequent collection polls

Use the finite wrapper, retaining the existing adapter and collector states:

```bash
bash research/mim_comparison/poll.sh
```

The full-feed resumed shadow scan took 218.41 seconds, too slow for the 60-second arrival budget. The wrapper maintains an event/contract lookup index in the isolated collector journal and passes a verified window of the last 500 complete feed records. The complete append-only feed remains authoritative; prior collector observations persist across windows. The real bounded poll completed in **22.339 seconds**; its two run inventories passed 38 additional file-hash checks, and all prior observations, invalid rows and session flags remained identical. This timing is one observed run, not a latency guarantee. This changes operational input preparation and database indexing only, preserving the frozen trading and collector sources.

The wrapper does not schedule itself. Actual collection must occur within the frozen 60-second timeliness window for a session to qualify. Do not change the frozen wrapper, adapter or core Python sources during collection. See the [adapter instructions](feed_adapter/README.md) for provenance, locking and preparation details.

Contract identity is inferred from causal runtime-log evidence; the log does not authenticate response payload identity. Broken hash links were excluded, and later segments remain explicitly unanchored. No missing contract, candidate volume, prior close or warmup session was invented. Existing warmup ends August 28; gaps and roll coverage remain subject to the original eligibility checks. The adapter does not add operational bars to historical performance, move the original deadline, authorize deployment, or increase live size.
