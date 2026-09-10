# YANK native one-minute pilot

This separate diagnostic dataset covers purchased MNQM5/instrument 42009475 events from 2025-05-19T00:00:00Z through 2025-05-31T00:00:00Z. It preserves the original dollar-bar CSV, acquired files, historical orders, model and strategy settings. `PASS_DATA_CHECKS` and `REPLAY_COMPLETE` always retain `HOLD_VALIDATION`; neither establishes execution quality or profitability.

Run from the isolated product worktree with the existing main environment:

```sh
cd /root/Silver-Bullet-ML-BMAD-yank-minute
/root/Silver-Bullet-ML-BMAD/.venv/bin/python src/cli/check_yank_native_minute.py --output-dir data/yank/native-minute-example
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest -q tests/unit/yank_native_minute
```

The command accepts only a fresh output directory. It checks static acquisition, model, dependency, timezone and frozen-source pins before and after processing. No credentials, downloads, purchases, live-trader imports, CSV warm-up, retraining or parameter selection are involved. The source pins refer to the fixed sibling replay checkout at revision `b6626622e3468c932722409530120d60f29c6ce6`; changing it requires an explicit future contract change. Minimal frozen modules execute privately with their source bytes unchanged. Only the two pure independent reconciliation functions are extracted from its evidence module; its historical CSV execution entrypoint is never called.

## Native contract

DBNv3 MBO records stream in bounded NumPy chunks (65,536 records by default). Every record must have the expected 56-byte framing, instrument and publisher. Daily metadata must identify GLBX.MDP3, MNQM5, the requested schema and daily UTC interval. All 33 native daily MBO/definition/status files, their batch metadata and the acquisition manifest have tracked static SHA256/size pins. Definition records independently establish the 0.25 point tick and 2 USD point value.

Only non-snapshot `T` prints contribute OHLCV. `F` fill updates do not add traded volume. Snapshot records and the recognized `A/C/M/R/F/N` non-trade actions are counted separately. Prices remain exact integer billionths of a point; volume is integer contracts. A zero-size trade still produces an observed trade bar, while a minute without prints has null OHLC and no strategy bar. The open and close follow native file/record order, not a sort by exchange time.

Capture-clock bars use `ts_recv` UTC intervals `[start,end)` and retain the interval start as the strategy timestamp. Exchange-clock aggregation is a diagnostic comparison. All 56 bytes are represented in the NumPy record dtype so filtered copies preserve the raw order ID, channel, side and capture delta as well as OHLCV fields. Each bar records trade count, a SHA256 over the concatenated raw included 56-byte trade records, and first/last file, record index, sequence and both timestamps. Capture bar availability is at least its end and extends to the LAST record terminating every included native trade event. The latest limiting terminator retains its source reference. A null `completion_ref` means the interval end remains the limiting availability after validation of every used terminator; it does not mean completion evidence was absent. Normal completion can be reconstructed from pinned native bytes and the trade lineage. Pending event completion survives chunk boundaries; an unresolved daily boundary, invalid trade time, invalid referenced terminator or capture regression makes replay unassessable. Corrupt framing, unknown actions or changed pins fail the command.

`coverage.jsonl` contains every one of the 17,280 minutes, including empty periods. It separates native MBO source-file presence, start/end observed status, whole-interval status (including mixed/unknown intervals), within-minute status transitions and null/traded OHLCV. No Saturday file is present in the pinned purchase; that absence is recorded separately from carried nontrading status. Status captured exactly at the beginning of a daily file with an earlier exchange timestamp is initial-state evidence, not proof of a new arrival then. An observed nontrading status is not an independent exchange-feed completeness guarantee.

## Causal diagnostic replay

The unchanged frozen engine receives completed, start-labelled observed minute bars. H1/M15/session alignment therefore remains unchanged; H1/M15 updates occur conservatively on a following completed minute. Every decision records its interval end, its own complete-event availability and a causal prefix watermark: the latest availability of all observed bars it consumes. Existing pending orders advance before the current completed bar is used to detect a new order. A new order may use only the following observed bar's OHLC interval, and its causal prefix watermark must not overlap that interval. If either arm violates this check, the published replay is HOLD with no arm results. A delayed bar that generates no order is explicitly reported and can remain safe: any later decision consumes it only after that later minute completes.

Both fresh arms use the exact frozen configuration: `no-ml` uses `model=None`; `ml050` uses the pinned model. The model passes an inference smoke check. Results include gates, events, trades, modeled interval annotations, terminal inventory and the independent cash/inventory/fee reconciliation. The frozen strict reconciler runs before adapter annotations are attached; all original accounting fields remain unchanged. Rechecking a published ledger should compare its monetary fields independently or remove annotation-only fields before calling that frozen strict row-equality reconciler. There is no terminal liquidation. The historical 4 USD charge per closed trade is a modeled assumption. Same-bar fill/exit and unknown intrabar order remain qualifications even when no explicit ambiguity flag fires.

No pre-purchase history exists. Each decision discloses observed closes, LR full-history readiness at 1,950 closes, preceding observed days for the 20-day ADR and actual positive ATR observations for volatility's minimum 20/full 120 history. Existing limited-history gates remain unchanged, including permitted LR WARMUP. H1 and day histories begin with partial purchased periods. Pending expiry and holding durations count observed bars, not elapsed minutes. Empty rows are never forward-filled.

## Artifacts and failure behavior

- `bars.jsonl`, `exchange-diagnostic.jsonl`: integer OHLCV and deterministic trade lineage. Availability fields in the exchange diagnostic are not replay timing evidence.
- `coverage.jsonl`, `status.json`, `definitions.json`: all-minute coverage and native auxiliary evidence.
- `delayed-events.json`: every capture bar whose complete-event availability overlaps the next observed minute interval, even if it creates no order.
- `replay.json`: both frozen arms or explicit HOLD reasons.
- `report.json`: pins, versions, executed code hashes, counts and qualifications.
- `manifest.json`: hashes for all other canonical artifacts.

Canonical files contain no output paths or run timestamps. Linux atomic `renameat2(RENAME_NOREPLACE)` publishes only after verification. Existing paths, symlinks and racing directory collisions are never replaced. Caught exceptions, SIGTERM and interruptions remove staging files; a force-killed process can leave an unpublished hidden staging directory, but no success manifest at the requested destination. Acquisition/source roots cannot be used as output directories. Large generated directories are ignored locally; compact findings live under `docs/reports/yank-native-minute/`.

## Source documentation

The official [MBO schema](https://databento.com/docs/schemas-and-data-formats/mbo) documents action semantics, native prices and event/capture timestamps. [Common fields, enums and types](https://databento.com/docs/standards-and-conventions/common-fields-enums-types) defines timestamp conventions, undefined values and record flags. [MBO snapshots](https://databento.com/docs/standards-and-conventions/mbo-snapshot) describes initial synthetic book state, and [order tracking](https://databento.com/docs/examples/order-book/order-tracking) provides native order-event context. These pages were checked on 2026-09-08; the implementation requires no network access.
