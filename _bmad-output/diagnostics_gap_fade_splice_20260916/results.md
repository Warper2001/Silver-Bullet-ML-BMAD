# GAP-1 roll-splice sensitivity, measured properly (2026-09-16)

**The open item** from `_bmad-output/diagnostics_gap_fade_parity_20260916/results.md`: dropping the 27 interleaved 2025 sessions moved the replay from PF 2.017 to 1.885, but that also removes each dropped day's role as the *next* day's prior-close reference, so it bounded the effect instead of measuring it.

## Method: same writer, same extract, one contract per session

1. **Rebuild with identical construction.** The frozen `mnq_1min_2025.csv` was produced by a dollar-bar writer from raw TradeStation records. That same writer (sha `8f603a91…`, retained under `docs/reports/yank-provenance-closure/`) was re-run on the same hash-pinned raw extract (sha `baeb1a06…`), so bar construction is held constant — no swap to fixed minute bars.
2. **Gate.** Re-running it on the *unfiltered* extract reproduced the frozen CSV **byte for byte** (sha `3f20ec70…`). Without that, nothing below would be interpretable.
3. **The one change.** Within each Globex session, keep only the contract holding the most minutes and drop the other's. **Every session is kept**, so the prior-close chain is intact. That removed 10,148 of 351,628 raw minutes across the 40 mixed sessions; the corrected CSV has 281,645 rows against 289,230.
4. Both CSVs were replayed through the strategy's own loop. The frozen run reproduces the sealed baseline (N=77, PF 2.017), so the comparison starts from parity.

## Result: the splices are worth −0.03 PF, and change no trade count

| | N | WR | PF | Net |
|---|---|---|---|---|
| Frozen CSV (sealed baseline) | 77 | 64.9% | **2.017** | $7,861 |
| Front-month rebuild | 77 | 64.9% | **1.987** | $7,622 |
| Difference | 0 | 0 | **−0.030** | **−$238.50 (−3.0%)** |

- **No trade is added or removed.** 76 of the 77 trades are identical in date, direction, outcome and P&L.
- **Exactly one trade changes:** 2025-03-03, from a target fill (+$740.50) to a 13:00 time exit (+$502.00).
- **Only one of the 77 trades falls on an interleaved session at all**, and it is unchanged by the correction (+$45.50 in both). GAP-1 keys on the RTH open and the prior RTH close, and the splices did not disturb those.

**The earlier estimate overstated the effect about fourfold.** PF 1.885 came from breaking the prior-close chain, not from the splices.

## The finding that matters more: the March roll-boundary trade

The single changed trade is not on an interleaved session — it is on the **first session of a new contract**. In 2025 the series switches contract between sessions four times (2025-03-03, 06-02, 09-02, 12-01), and on such a day the "overnight gap" is measured against **the previous contract's close**, so it includes the calendar spread.

- 2025-03-03 is recorded in the sealed trade list with a **370.25-point (1.768%) gap**, short, +$740.50. March 3 was a genuine selloff, but a quarterly MNQ spread of roughly 200 points is folded into that number, which also inflates the target distance and the 2×gap stop.
- **This artifact exists in both versions.** Keeping one contract per session does not fix a gap measured across two contracts; only a spread-adjusted series would.
- The other three boundaries produced no trade.

**Excluding both defect kinds** (roll-boundary and interleaved sessions), the two versions agree exactly:

| Clean sessions only | N | WR | PF | Net |
|---|---|---|---|---|
| Frozen and corrected (identical) | 75 | 64.0% | 1.916 | $7,075 |

So of the sealed $7,861: **$7,075 is clean**, $740.50 comes from the roll-boundary trade (9.4% of net, geometry distorted by the contract change), and $45.50 from the one interleaved-session trade.

## What this means

- **GAP-1's sealed 2025 edge is not an artifact of the roll splices.** Correcting them moves PF 2.017 → 1.987.
- **The sealed number does lean ~9% on one distorted trade.** That is a data-construction artifact nobody had identified, and it is worth knowing before the strategy's promotion gate is judged — especially since that gate is already UNDERPOWERED at N=30 (`_bmad-output/diagnostics_gap_fade_power_gate_20260913/`).
- **No seal, parameter or live setting is changed by this.** The sealed Gate-0 figure (2025+2026 combined, N=117, PF 1.761) still includes the back-month Jan–Feb 2026 rows; re-scoring that needs its own pre-registration.
- **A future clean rebuild should also span the roll**, e.g. by back-adjusting each contract segment by the spread measured at the switch, so gaps are never taken across contracts.

## Outputs

- `rebuild_2025_frontmonth.py`, `rebuild_meta.json`, `mnq_1min_2025_frontmonth.csv` (the corrected 2025 bars, sha `f1fe5b36…`)
- `measure_splice.py`, `splice_results.json`
- this file
