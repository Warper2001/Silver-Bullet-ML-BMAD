# May 28 displayed-book evidence

All three short-limit entries were passive at their observable arrival times, including fixed additional delays of 0, 100 and 500 milliseconds. No displayed bid was at or above any requested short limit. This adds book context to the earlier trade-through timeline; it does not establish hypothetical queue positions or fills. **HOLD_VALIDATION** remains, and modeled P&L is unchanged.

Each observation uses the last complete native event strictly before the arrival boundary, or before the first entry-through event. All 12 requested observations had complete event boundaries and uncrossed, two-sided books. No equal-capture-time arrival boundary occurred. Snapshot records initialized the book and were not treated as genuine arrivals.

## Book at arrival

Times are UTC on May 28, 2025. Prices are index points; sizes are contracts. The bid/ask columns show best price × displayed size. “Sells at limit” is existing same-side displayed quantity, not liquidity available for an immediate sale.

| Case | Arrival | Delay ms | Short limit | Best bid | Best ask | Spread | Sells at limit |
|---|---|---:|---:|---|---|---:|---:|
| Both arms | 18:53:00.000 | 0 | 21,463.00 | 21,445.25 × 3 | 21,445.75 × 4 | 0.50 | 6 |
| Both arms | 18:53:00.100 | 100 | 21,463.00 | 21,444.50 × 6 | 21,445.00 × 3 | 0.50 | 6 |
| Both arms | 18:53:00.500 | 500 | 21,463.00 | 21,443.00 × 2 | 21,443.25 × 1 | 0.25 | 6 |
| No ML only | 19:52:00.000 | 0 | 21,408.00 | 21,392.50 × 2 | 21,392.75 × 1 | 0.25 | 12 |
| No ML only | 19:52:00.100 | 100 | 21,408.00 | 21,391.75 × 3 | 21,392.25 × 3 | 0.50 | 12 |
| No ML only | 19:52:00.500 | 500 | 21,408.00 | 21,393.75 × 1 | 21,394.00 × 1 | 0.25 | 10 |
| ML only | 20:02:00.000 | 0 | 21,359.75 | 21,345.75 × 3 | 21,346.50 × 3 | 0.75 | 4 |
| ML only | 20:02:00.100 | 100 | 21,359.75 | 21,351.25 × 1 | 21,351.75 × 2 | 0.50 | 4 |
| ML only | 20:02:00.500 | 500 | 21,359.75 | 21,351.25 × 2 | 21,352.00 × 3 | 0.75 | 5 |

The immediately executable displayed quantity at or above each limit is zero. This is consistent with a passive limit waiting for price to rise; it is not evidence that the order could never fill. All hypothetical orders are five contracts.

## Before the first trade-through event

| Case | Best bid | Best ask | Existing sells at entry limit | Last complete record | First unapplied event record |
|---|---|---|---:|---:|---:|
| Both arms | 21,462.75 × 2 | 21,463.00 × 3 | 3 | 15,715,564 | 15,715,565 |
| ML only | 21,359.00 × 3 | 21,359.50 × 1 | 2 | 19,322,747 | 19,322,748 |
| No ML only | 21,406.75 × 1 | 21,408.00 × 10 | 10 | 19,638,094 | 19,638,095 |

Displayed quantity at the entry limit can change between arrival and the first through event. These snapshots do not track which orders would be ahead of a hypothetical submission, infer matching priority, or allocate observed trades to it. The later trade-through evidence remains conditional on no market impact. No partial fills or revised execution prices are invented.

## Method and verification

The pinned May 28 stream starts with a clear and completes its 6,638-record snapshot at zero-based record 6,637. The inspection applies 19,638,095 native records through the last requested pre-entry observation. Adds insert orders; modifications replace price/size; cancels remove the specified quantity; clears empty the book. T/F/N records do not change resting quantity. Book state is inspected only at complete LAST boundaries, following Databento’s [order-tracking conventions](https://databento.com/docs/examples/order-book/order-tracking) and [snapshot conventions](https://databento.com/docs/standards-and-conventions/mbo-snapshot), checked September 8, 2026.

Missing orders, duplicate adds, invalid sides/sizes/timestamps, unsupported records and incomplete initialization fail explicitly. An arrival inside an incomplete event would be reported unassessable. No such failure occurred through the requested observations. This inspection does not certify the entire daily book or independent exchange-feed completeness.

Eighteen hand-built checks cover snapshot completion, clear/add/modify/partial and full cancel, T/F/N separation, event boundaries, insufficient opposing size, missing/duplicate orders, malformed updates and crossed books. All pass. Twenty-four raw records bracketing the observations were also decoded with the official DBN decoder. The full native source hash was checked before and after both runs.

Two runs with 65,536-record and 32,768-record chunks produced identical JSON; the latter used optimized Python. Raw boundary bytes, source indices, timestamps and hashes are retained in [may28-book.json](may28-book.json). The earlier [trade timeline](may28-timeline.md) and frozen replay remain unchanged.

```sh
cd /root/Silver-Bullet-ML-BMAD-yank-minute
/root/Silver-Bullet-ML-BMAD/.venv/bin/python docs/reports/yank-native-minute/inspect-may28-book.py > /tmp/yank-book-reproduced.json
cmp docs/reports/yank-native-minute/may28-book.json /tmp/yank-book-reproduced.json
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/yank_native_minute/test_book_inspection.py -q
```
