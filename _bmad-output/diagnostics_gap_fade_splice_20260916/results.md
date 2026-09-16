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
- **The roll boundary still needs handling** so gaps are never taken across contracts. *(Superseded the same day — see the second half of this file. Back-adjustment was tried and rejected as not neutral for a percentage-triggered strategy; the fix is to take the prior close from the session's own contract, as live already does.)*

## Outputs

- `rebuild_2025_frontmonth.py`, `rebuild_meta.json`, `mnq_1min_2025_frontmonth.csv` (the corrected 2025 bars, sha `f1fe5b36…`)
- `measure_splice.py`, `splice_results.json`
- this file

---

# The roll-boundary artifact, fixed (2026-09-16, same day)

## Live GAP-1 never had this bug

`gap_fade_live.py` fetches `barsback=3000` bars for **one symbol** (`_fetch_bars`, `self.symbol`), so the prior RTH close and today's RTH open always come from the same contract. The live path cannot measure a gap across a contract change. On the first day after the operator switches `GAP_FADE_SYMBOL`, the new contract's own prior session is thin, and `MIN_RTH_BARS = 300` makes the bot skip the day rather than trade a bad reference.

**The artifact is a property of the research series only** — an unadjusted continuous splice of four contracts.

## Whole-series adjustment was tried and rejected

**Point (Panama) back-adjustment** (`backadjust_2025.py`): each segment shifted by the cumulative roll spread, estimated from adjacent raw minutes whose contract label changes (e.g. MNQZ25→MNQH26 = 253.0 pts, n=1,673 pairs, IQR 250.5–255.25). The shift is provably constant within each segment (asserted), and it does fix the March gap: 1.768% → 0.726%, the real overnight move.

**But it is not neutral for this strategy.** GAP-1 triggers on 0.5% **of the prior close**, and shifting levels changes that denominator. Replaying the back-adjusted series gives N=77, PF 1.968, $7,481 — which includes two unintended changes: 2025-03-27 drops out and 2025-06-02 appears, purely because their gap percentages crossed the threshold on rescaled levels. Ratio adjustment has the mirror-image flaw: it preserves percentages but rescales historical point P&L, and GAP-1 is paid in points. **Neither is a clean fix for a percentage-triggered, point-paid strategy.**

## The fix: take the prior close from the same contract, as live does

`same_contract_prior_close.py` keeps the front-month bars and the strategy's own loop, and changes exactly one input: each session's prior RTH close is read from **that session's own contract** in the pinned raw extract. Both contracts trade during roll weeks, so the incoming contract has its own prior session whenever it traded.

**Result — only boundary sessions are touched (asserted, and true: one session):**

| 2025 | N | WR | PF | Net |
|---|---|---|---|---|
| Frozen CSV (sealed baseline) | 77 | 64.9% | 2.017 | $7,861 |
| Front-month rebuild (splices fixed) | 77 | 64.9% | 1.987 | $7,622 |
| **+ same-contract prior close (boundary fixed)** | **76** | **64.5%** | **1.922** | **$7,120** |

- **2025-03-03 drops out entirely.** Its incoming contract (MNQM25) has no prior-session RTH close in the data, so there is no honest gap to measure — exactly the case where live skips the day. The distorted +$740.50 (frozen) / +$502.00 (front-month) trade is gone.
- 2025-06-02 is likewise skipped for want of a same-contract prior close; it produced no trade in the sealed run either.
- 2025-09-02 and 2025-12-01 were never affected.
- **No other session changes.** Every non-boundary trade keeps its date, direction, outcome and P&L.

## The corrected 2025 figure

**N=76, WR 64.5%, PF 1.922, Net $7,120.** Against the sealed $7,861 that is **−$741 (−9.4%)**: −$239 from the roll splices and −$502 from removing the boundary trade.

**Cite this for GAP-1's 2025 in-sample performance.** The strategy's edge survives the correction; it is simply ~9% smaller than the sealed number.

## What deliberately did not change

- **`SEALED_PARITY_2025` in `gap_fade_live.py` stays as it is.** It exists to detect drift against the frozen sealed artifact, and it must keep matching that artifact (N=77, PF 2.017) to do its job. The corrected figure is a separate research number, not a new parity target.
- **No strategy rule, parameter or live setting changed**, so no pre-registration was required. The harness mirrors live behaviour; it does not alter it.
- **The sealed Gate-0 headline (2025+2026 combined, N=117, PF 1.761) is untouched** and still carries the back-month Jan–Feb 2026 rows. Re-scoring it needs its own prereg, and would want the same same-contract treatment at the 2026 roll.

## Added outputs

- `backadjust_2025.py`, `backadjust_meta.json`, `measure_backadjusted.py`, `backadjusted_results.json` — the rejected whole-series adjustment, kept as the evidence for rejecting it
- `same_contract_prior_close.py`, `same_contract_results.json` — the fix
