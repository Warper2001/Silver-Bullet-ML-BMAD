# Native one-minute MNQM5 diagnostic

The purchased May 19–30, 2025 pilot yields **13,440 true capture-clock one-minute trade bars** from **7,526,752 T prints** in **237,146,989 native MBO records**. The complete requested interval has 17,280 coverage rows: 13,440 traded, 3,830 without prints under observed nontrading status throughout the minute, and 10 empty minutes containing trading/nontrading status transitions. Coverage records the absent May 24 native file separately from carried status. No OHLCV was forward-filled or borrowed from the historical dollar-bar CSV.

The data checks pass and both unchanged frozen replay arms complete, subject to **HOLD_VALIDATION**. These are diagnostic accounting outputs, not profitability or fill validation.

| Arm | Closed modeled trades | Modeled net P&L | Terminal equity | Terminal contracts | Closed-trade modeled fees |
| --- | ---: | ---: | ---: | ---: | ---: |
| No ML | 2 | −$818 | $49,182 | 0 | $8 |
| ML 0.50 | 2 | −$873 | $49,127 | 0 | $8 |

Both arms independently reconcile cash, inventory, marks, fees and closed trades. Both are flat at the final mark. The no-ML arm's second trade has an explicit same-bar fill/exit ambiguity; all OHLC fills and exits lack exact intrabar execution evidence. The $4 charge per closed trade remains the historical modeled assumption.

Exactly one trade event extends beyond its capture minute: the May 27 T print at record 4,106,157 has capture timestamp `1748349299999994093`; its LAST terminator at record 4,106,159 has capture timestamp `1748349300000002948`. The terminator arrives 2,948 ns into the next minute. Neither arm creates an order from that minute. Decision eligibility uses the maximum complete-event availability across the entire observed prefix, and every actual order is eligible before its following modeled OHLC interval starts. This delayed event remains in the output and is not filtered to obtain a replay result.

The diagnostic starts with no pre-purchase history. LR WARMUP remains permitted until 1,950 observed closes. Per-decision annotations report actual positive ATR history for volatility's minimum 20/full 120 observations and preceding observed days for the 20-day ADR. Early H1/day histories are partial; the purchase cannot supply a full 20-day ADR. H1/M15 updates remain conservative, and frozen pending/holding counters advance in observed bars rather than wall-clock minutes.

Two fresh reviewed builds produced nine byte-identical canonical files. All 65 new tests and 133 frozen legacy tests pass; three independent review layers completed with accepted findings fixed and nothing deferred. Verification and canonical artifact hashes are recorded in [verification.json](verification.json), with [independent reproduction commands](verification.md) and [full parent evidence](parent-verification.json). The independent historical native oracle matched every capture/exchange OHLCV bar and first/last trade reference; a separate full event scan identified the same sole delayed terminator. The parent also freshly passed the 133 frozen legacy replay/accounting/policy tests. Original acquired files, dollar-bar CSV, archived orders and model remain historical reference inputs.

See the [contract and reproduction command](../../yank-native-minute.md) for source documentation, static pins, interval semantics, limitations and output handling.

Follow-up: [May 28 native trade-print timeline](may28-timeline.md) establishes entry-through prints before stop crossings in all three distinct cases, including separate events 12.13 seconds apart within the no-ML 20:21 minute. Hypothetical execution remains unverified; the original modeled results and ambiguity annotations are preserved.

Further evidence: [May 28 displayed-book observations](may28-book.md) show all three short limits were passive at arrivals with 0/100/500 ms additional delay. Same-side displayed quantities and event-boundary references are retained; queue positions and hypothetical fills remain unverified.
