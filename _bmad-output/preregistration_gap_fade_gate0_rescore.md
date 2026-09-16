# Pre-Registration: GAP-1 Gate-0 re-score on corrected bars

**Registered:** 2026-09-16
**Status:** SEALED at commit time. No method amendments after this commit.
**Re-scores:** the Gate-0 figure of `_bmad-output/preregistration_gap_fade_panic_open.md` (2026-06-25), whose trade list is `data/reports/gap_fade_20260625_205328.csv` (sha256 `a65a6cce…`): **N=117, WR 62.4%, PF 1.761, Net $9,878**, spanning 2025-01-06 → 2026-06-11.

**This is a re-measurement of an existing result on corrected data. It tests no new hypothesis, changes no parameter, and can neither promote nor retire GAP-1.** The strategy's promotion gate is prospective live N≥30 and is untouched (and already UNDERPOWERED per `_bmad-output/diagnostics_gap_fade_power_gate_20260913/`).

## 1. Why

The Gate-0 window is priced from the two defective MNQ CSVs (`_bmad-output/diagnostics_h2l2_contamination_20260914/`, `..._tier2_contamination_20260914/`) and from an unadjusted contract splice:

1. **2025 roll weeks interleave two contracts minute by minute** — 40 mixed sessions.
2. **Roll boundaries**: on the first session of a new contract the "overnight gap" is measured against the *previous contract's* close, so it folds in the calendar spread. Measured for 2025 (`_bmad-output/diagnostics_gap_fade_splice_20260916/`): it distorted one trade worth 9.4% of the 2025 net.
3. **Jan–Feb 2026 of `mnq_1min_2026_ytd.csv` is the deferred MNQM26**, not the front month, until the March 2026 roll.
4. The window runs to 2026-06-11, so it crosses the **sealed holdout** (2026-03-01 → 05-19) and beyond.

The 2025 half has already been corrected and re-scored: **N=76, PF 1.922, $7,120** against a sealed 2025 subset of N=77, PF 2.017, $7,861. This pre-registration extends that correction across the whole Gate-0 window.

## 2. Method (frozen before the run)

**Bars**

| Segment | Source |
|---|---|
| 2025-01-01 → 12-31 | front-month rebuild: the original dollar-bar writer re-run on the pinned raw extract, one contract per session (`rebuild_2025_frontmonth.py`, sha `52b275f3…`). **Gate:** re-running it unfiltered must reproduce the frozen 2025 CSV byte-for-byte (`3f20ec70…`), as it did on 2026-09-16. |
| 2026-01-01 → 2026-03-11 | front-month **MNQH26** 1-minute records from `/root/mnq_historical.json`. Sessions carrying more than one contract label are dropped whole (the convention of `diagnostics_h2l2_contamination_20260914/correction_plan.md`). |
| 2026-03-12 → 2026-06-11 | `mnq_1min_2026_ytd.csv` rows unchanged. MNQM26 is the front month from the CME roll date (2026-03-12). |

**Prior close** — the fix. For every session the prior RTH close is taken from **that session's own contract**:
- 2025 and 2026-01-01 → 03-11: from the pinned extract / raw JSON by contract label.
- From 2026-03-12: the CSV is single-contract MNQM26, so its own prior close already satisfies this.
- **A session whose contract has no prior RTH session in the data is skipped**, which is what live does (it fetches one symbol and `MIN_RTH_BARS=300` rejects a thin prior session).

**Everything else is the sealed strategy**: `gap_fade_live.py`'s own replay loop (sha `ca5866ea…`) with its frozen parameters (0.5% gap, 2× stop, prior close target, 13:00 ET time stop, Fridays excluded, `MIN_RTH_BARS=300`), via `same_contract_prior_close.py` (sha `84b8b0ce…`).

**Whole-series price adjustment is excluded**, and this is pre-committed: point back-adjustment shifts levels and GAP-1 triggers on 0.5% *of the prior close* (measured 2026-09-16: it silently dropped one trade and added another), while ratio adjustment rescales point P&L.

## 3. What will be reported

- Corrected N, WR, PF, Net for the full Gate-0 window, beside the sealed figures.
- A per-segment split: 2025, Jan–Feb 2026, Mar 1–11 2026, the holdout Mar 12 → May 19, and May 20 → Jun 11.
- Every session skipped for want of a same-contract prior close, and every trade that differs from the sealed list.
- Whichever contingencies fired (below).

**No threshold is attached to the output.** The original Gate-0 decision rule (PF ≥ 1.40 strong / ≥ 1.10 weak, N ≥ 60, WR ≥ 55%, ≤ 10 consecutive losses, worst month ≥ −$600) is quoted only for context; it was applied in 2026-06 and is not re-applied here. If the corrected figure falls below those numbers, the honest statement is that **the sealed Gate-0 claim was inflated by data defects**, and any consequence is Alex's decision, not this document's.

## 4. Contingencies (pre-declared)

- **Raw coverage ends 2026-05-04.** For sessions after that, the same-contract prior close comes from the CSV itself, which is already single-contract MNQM26. If a 2026 session before 03-12 is missing from raw, it is skipped and listed.
- **Interleaved 2026 sessions** (roll week) are dropped whole, as in 2025.
- If the 2025 rebuild gate fails, nothing is interpreted and the run is abandoned.

## 5. What this run may not do

- Change any parameter, threshold, seal or live setting; restart any unit. `SEALED_PARITY_2025` stays pinned to the frozen artifact.
- Re-run with different segment rules, roll dates or prior-close rules after seeing results. Any deviation is reported as one and does not change the reported figure.
- Treat the corrected number as a new gate, a promotion, or a retirement.

## 6. Holdout access

The window crosses `data/sealed_holdout/`'s period (2026-03-01 → 05-19) through the dual-presence rows of `mnq_1min_2026_ytd.csv`. **An `ACCESS_LOG` row citing this commit is appended before the run**, and the result is recorded there afterwards regardless of outcome.

## Integrity hashes

| Item | SHA-256 |
|---|---|
| `src/research/gap_fade_live.py` | `ca5866eab72b2c9eedddf8bb…` |
| `rebuild_2025_frontmonth.py` | `52b275f300b3083f33e9310c…` |
| `same_contract_prior_close.py` | `84b8b0cebc45547b033ff45f…` |
| sealed Gate-0 trade list | `a65a6cceddf87aa659f98f5a…` |
| git HEAD at drafting | `4fd1368b25f49bb378bd8c25be0a87a8742c10ae` |
