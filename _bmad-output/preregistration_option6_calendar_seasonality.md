# Pre-registration — Option 6: MNQ calendar seasonality

**Date:** 2026-09-06
**Parent plan:** `_bmad-output/research_plan_post_r3_options_20260904.md`, Option 6.
**Sealed before running:** committed before any P&L number is computed.

## Why this needs an unusually strict seal

Calendar seasonality is the single easiest place in this project to manufacture a false positive: weekday × month × turn-of-month × time-of-day is dozens of cells, and at N≈1,440 trading days *something* will always clear an uncorrected threshold. Option 1a's null already caught exactly this shape (a lone interior cell that looked good and wasn't). So the hypothesis set, the correction, and the comparison baseline are all fixed here, before looking.

## Data

`data/mim_x/mnq_1min_2021_2024_frontmonth.csv` (2021-01-03 → 2024-12-31) + `mnq_1min_2025.csv` + `mnq_1min_2026_ytd.csv` — **~5.75 years**, ~1,440 trading days. Columns differ (`notional` present only in the later files); only `timestamp/open/close` are used.

**No holdout split.** This is an exploratory Gate-0 screen on the full available history — if it passes, a holdout/OOS design is a *separate* seal. Stated up front so a PASS is never mistaken for a validated edge.

## Unit of analysis

**RTH open → RTH close (day session only), 1 contract, $4.00 round-turn cost** — the same cost convention as Option 2.

Deliberately *not* close-to-close, which is the literature standard for these effects: close-to-close bundles in the overnight component, and Option 2 has already tested the overnight hold separately and failed it (`verdict_option2_overnight_hold_20260905.md`). Bundling them would confound a known-failing leg with the thing under test. Disclosed as a deviation from the literature convention, made for separability.

## The three pre-declared hypotheses (fixed here; no others will be tested)

Each has external grounding in the published equity-calendar literature — none was chosen by looking at this data.

1. **Turn-of-month (TOM).** Trading days at offsets {−1, +1, +2, +3} around the month boundary (last session of a month, first three of the next) out-perform all other days.
2. **Day-of-week (DOW).** Day-session returns differ by weekday (the classic weekend/Monday effect). 5 cells.
3. **Pre-holiday.** The session immediately preceding a US market holiday carries positive drift.

## The control that decides this — beat the drift, not zero

Every cell is compared against the **unconditional always-long day-session baseline over the same period**, never against zero. MNQ rose substantially across 2021–2026; against a zero benchmark every long cell "works." This is the TSMOM-1 mechanism check applied here: the question is whether the calendar cell beats simply being long every day, not whether it is positive.

## Multiple-comparison correction, fixed now

**7 tests total** (TOM in-vs-out = 1, DOW = 5, pre-holiday = 1). Bonferroni: **α = 0.05 / 7 = 0.00714**, i.e. a cell must beat the **99.29th percentile** of its null, not the 95th.

## Gate 0 — sealed

1. **N floor:** ≥ 100 day-observations in the cell. Declared now: TOM (~276) and DOW (~288 each) clear this; **pre-holiday (~52) does not** — it is reported as descriptive only and is **not gate-eligible**. Stated in advance rather than discovered afterwards.
2. **Beat the baseline:** cell mean net daily P&L > unconditional always-long mean net daily P&L.
3. **Beat the corrected null:** random day-selection of identical cell size, 2,000 draws, cell mean must exceed the 99.29th percentile (Bonferroni-adjusted per above).
4. **Fat-day robustness:** with the top 5 days removed from both cell and baseline, check 2 must still hold.

**Verdict:** PASS only if 1–4 all hold for a gate-eligible cell.

## Stopping rule

One run. FAIL closes Option 6, no re-slicing — explicitly **no** post-hoc interactions (no "TOM *and* Wednesday", no "TOM excluding Mondays", no month-of-year cuts). Those are the lucky-cell searches this seal exists to prevent. A PASS authorizes only a separately pre-registered OOS/holdout design, not a live change.

## Scope note on Option 5 (volatility risk premium)

Checked before writing this: the repository holds **no options, VIX/VX, or implied-volatility data of any kind**. Option 5 is therefore not testable on data in hand and is **not attempted here** — it is data-blocked, not failed. What it would need is recorded in the Option 6 verdict doc rather than approximated with a futures-only proxy, which would test a different hypothesis while wearing the VRP label.
