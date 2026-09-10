# Reference audit, 2026-09-10

This audit precedes calculation of candidate returns. `source-manifest.json` identifies the source snapshot and the historical input bytes. Python files in this directory are evidence only: do not import or execute them. Prior V1/V2 studies run at import time and can overwrite historical reports.

## Specification differences

| Aspect | Deployed source | Published executable | Research treatment |
|---|---|---|---|
| Timestamps | End labels 09:31–16:00 ET | Start labels with one-based minute counter | Normalize to end labels; samples coincide |
| Upper/lower bands | Open-scaled sigma plus additive gap | Gap anchor multiplied by sigma factor | Preserve each formula at full precision |
| VWAP | Close weighted by volume; logging only | HLC3 weighted by volume; entry confirmation | Separate VWAP/confirmation diagnostic |
| Neutral sample | Hold until opposite band breached | Flat | Exit-only diagnostic |
| Threshold equality | Strict breakouts and exits | Strict directional conditions | Equality can retain A but flatten B |
| Warmup | 14 complete prior sessions | Rolling 14, minimum 13; first-day moves absent | Independent reference; common eligible sample |
| Catastrophe | Signal-close anchor ±250 points | Absent | Retain on A/B, remove on C/D |
| Daily guard | Realized reference gross P&L ≤−$1,000 | Absent | Same reference accounting on A/B |
| Fills | Order events; reference-price bookkeeping | Shifted exposure on close differences | Common delayed-open model, separately disclosed |
| EOD | 16:00 closure before mark stop-detection | Final exposure appended flat | Scheduled close proxy |
| Sizing | One MNQ | Prior equity, volatility target, leverage cap | Separate integer MNQ adaptation |
| Volatility | Minute absolute open-return means | Sizing uses 14 returns excluding yesterday | Preserve exact lag, not prose's nominal 15-day window |
| Rolls | Broker active-contract flag | SPY has no futures roll | Previous-session volume adaptation, same for arms |

Published source: [Concretum Group executable article](https://concretumgroup.com/python-backtesting-beat-the-market-an-effective-intraday-momentum-strategy-for-the-sp500-etf-spy/), captured as `authors-page.html`. Executable quirks govern the reference: first-day indicator omission, minimum thirteen observations, and a lagged fourteen-return sample standard deviation. Missing sizing volatility invokes maximum leverage. Python rounding is ties-to-even. These are diagnostic adaptations, not a SPY performance replication.

## Deployed operational subtleties

`_enter` sets `entry_px` to the completed decision close and places the protective stop from that reference. `_record_trade` books gross signal-price P&L without friction and deactivates only after a realized close crosses the static guard. Consequently a research executable-fill ledger cannot also serve as the guard ledger. The stop is activated after the modeled entry; stop gaps affect executable P&L but retain deployed nominal accounting for the guard.

Intramark reconciliation distinguishes an own-stop fill, canceled unfilled stop, replacement rejection and external closing fill. The mark-level disappearance check has weaker corroboration. Recorded operational actions also include rejects, restarts and external events; their differences from friction-only historical modeling must not be called signal disagreement.

`_prev_close_for_symbol` reads untagged live bar records before a symbol-specific fallback. The volume-selected historical adaptation must instead require a close belonging to its newly selected contract. The live record cannot establish contract identity on its own.

## Existing exposed evidence

V1 already studied a tight VWAP/band exit; V2 studied the opposite-band exit. Neither is the exact published implementation. This comparison attributes mechanisms without claiming their rediscovery, searching parameters or calling existing history out of sample.

The contract CSV contains 2,028,965 rows across 23 contracts (MNQH21 through MNQU26), from 2020-12-18T00:01Z to 2026-08-28T15:19Z. A raw RTH inventory finds 1,466 contract-session groups, 1,415 with 390 unique minute labels and no duplicated contract/timestamp keys. Those counts are before contract selection, expiry, warmup and calendar eligibility. The last acquisition day is partial.

## Recorded-log precision

Live sigma is serialized to six decimals and bands/VWAP to two. `tools/mim_parity_day.py` nevertheless applies a 1e−9 sigma tolerance. That can reject a correct full-precision computation. New comparison ledgers retain precision; operational reconciliation should compare to the serialization interval (and identify threshold ambiguity), not claim exact source-value disagreement from rounded logs.

## Prospective provenance constraint

`data/mim_nb/bars_raw.csv` has event and receipt timestamps but no symbol. `orders.csv` also has no contract column. A current state value alone cannot retroactively label an entire stream across rolls. Prospective collection needs a frozen, dated contract mapping with supporting provenance; missing mapping must make both arms unavailable. No code or service change is authorized to add this metadata to production.
