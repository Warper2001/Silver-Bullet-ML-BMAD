# Pre-Registration: Time-Series Momentum (TSMOM-1)

**Status:** SEALED before any backtest code touches contract-month price data.
**Date:** 2026-09-04
**Origin:** `academic-lit-low-frequency-futures-crypto-edge-candid-2026-09-03` deep-recon, Recommendation 2 (runner-up of 5 screened families, behind term-structure carry). Term-structure carry (TSC-1) has since FAILED Gate 0 (2026-09-03); the prop-firm-strategy follow-on line (VWAP/auction-theory → VPOC-1) has also since FAILED Gate 0 (2026-09-04); the order-flow/footprint runner-up from that line is now additionally blocked by the independent closure of the tick-infrastructure project (`project_ticksim_r3_killed_20260904`, closed 2026-09-04 in a separate session). Time-series momentum is the last untested candidate from the 2026-09-03 deep-recon.

## Hypothesis

A security's own trailing return predicts the sign of its near-term return, independent of other securities — the mechanism documented by Moskowitz, Ooi & Pedersen (2012, *JFE*, "Time Series Momentum") and extended by Hurst, Ooi & Pedersen (2017, back to 1880). This pre-registration tests whether the mechanism survives on daily closes of this shop's own accessible CME micro/small futures, after realistic per-contract costs — the same "this shop's cost structure, not the institutional-scale academic result" framing as TSC-1.

**This is not a clean test of an undisputed literature.** A same-journal rebuttal (Huang, Li, Wang & Zhou, *JFE*) re-tested the original claim with asset-by-asset regressions and found the pooled significance fails proper bootstrap critical values — critically, they show the strategy's live P&L is "virtually the same" as one built from the **historical sample mean sign alone**, a constant, no-time-variation position that requires zero genuine predictability. That dispute is built into this test directly (§5) rather than ignored.

## Universe, data, and costs (reused verbatim from TSC-1 — no new fetch, no new cost assumption)

Same 5 instruments, same data files, same per-instrument round-turn cost figures as `preregistration_term_structure_carry.md`: MGC ($6.00, estimate), SIL ($7.00, estimate), MHG ($4.00, measured), PL ($34.00, measured, non-combine-deployable standalone), MNQ ($2.24, measured). Data: `data/term_structure/{raw_contract_bars,contract_meta}.csv` (188 contract-months, already fetched 2026-09-03, re-fetchable via `tools/fetch_term_structure_data.py`).

- **Dev window:** 2021-01-01 – 2025-12-31 (identical to TSC-1, for direct comparability on the same universe/era/costs).
- **Holdout (untouched until the single evaluation run):** 2026-01-01 – 2026-09-03.
- Front-contract selection and roll mechanics (5 trading days before expiry, same-contract-diff-only P&L on non-roll days, one round-turn cost per roll while carrying a nonzero position) are reused verbatim from TSC-1's `build_front_next` / roll convention.

## Signal construction (frozen)

1. For each instrument, at each **month-end** rebalance date: compute the front contract's trailing `k`-month log return (from the front contract's own price series, using same-contract-only price relatives across any rolls within the window — a disclosed simplification, see Open Questions).
2. **Position rule (time-series momentum, no vol-scaling — a known Gate-0 simplification, same disclosed choice TSC-1 made):**
   - trailing return > 0 → **long** 1 front-month contract until the next rebalance.
   - trailing return < 0 → **short** 1 front-month contract until the next rebalance.
   - trailing return == 0 (rare) → flat.
3. Hold the position until the next month-end rebalance; roll to the new front contract intra-month exactly as TSC-1 does (5 trading days before expiry), incurring the roll cost while carrying a nonzero position.
4. Position size: fixed 1 contract per instrument (matches TSC-1's Gate-0 convention).

## The one swept knob (pre-declared grid, "one knob at a time")

`k` (trailing lookback, in months) is swept over **{1, 3, 6, 12}** — the same four horizons Moskowitz-Ooi-Pedersen themselves report in their original robustness table, not a grid invented for this test — on the **dev window only**. The value of `k` that maximizes dev-window net Sharpe across the 5-instrument equal-weight portfolio is locked and used, unchanged, for the single holdout evaluation. No other parameter (rebalance frequency, vol-scaling, instrument set, roll timing) is swept.

## Portfolio construction

Equal-weight (1 contract each) sum of the 5 instruments' daily P&L in USD. No leverage, no vol-targeting (matches TSC-1).

## Decision rule (frozen, two-gate — identical bar to TSC-1, for direct comparability)

- **Gate 0 (dev window, in-sample):** locked-`k` portfolio must show **net Sharpe > 0.5** AND **net profit factor > 1.15** over 2021-2025. Below this: **FAIL — does not survive this shop's cost structure at this scale**, no holdout spent.
- **Gate 1 (holdout, spent once):** if Gate 0 passes, evaluate the same locked configuration on 2026-01-01–2026-09-03. PASS requires **net PF > 1.0** and same-sign Sharpe — a directional check only, flagged as such regardless of outcome (short ~8-month holdout).
- A Gate 0 FAIL is terminal: no re-sweep, no new lookback grid, no vol-scaling added, no new instruments. A materially different mechanism variant (vol-scaled sizing, cross-sectional momentum once more instruments exist, crypto TSMOM) needs its own new pre-registration.

## §5 — The mechanism-check control (informational only, never gates the decision)

To operationalize the Huang-Li-Wang-Zhou critique directly rather than ignore it: alongside the primary locked-`k` TSMOM result, this run also reports two **non-tradeable, diagnostic-only** baselines computed on the exact same instruments/costs/window:

- **Always-long baseline:** constant long 1 contract, no signal, every day in the window.
- **Full-sample-mean-sign baseline:** for each instrument, take the sign of that instrument's **entire dev-window average daily return** (computed using the whole window at once — this is look-ahead by construction and is never a tradeable strategy) and hold that single constant position for the whole window. This is the exact "no genuine predictability required" comparison Huang et al. use.

If locked-`k` TSMOM's dev-window performance is materially indistinguishable from the full-sample-mean-sign baseline, that is reported as the "artifact, not predictability" signature regardless of whether Gate 0 numerically passes — a PASS that can't beat its own look-ahead-cheating control is reported as such, not presented as clean evidence of genuine predictability. This does not change the Gate 0/1 PASS/FAIL verdict itself (which is about tradeable net P&L, not mechanism), only the report's interpretation of what a PASS would mean.

## What this run will NOT do

- Will not vol-scale position sizing (matches TSC-1's own disclosed Gate-0 simplification).
- Will not test cross-sectional momentum (ranking instruments against each other) — the 5-instrument universe is too small, same reasoning TSC-1 used to rule out cross-sectional carry.
- Will not add BTC/ETH time-series momentum in this seal — the deep-recon flagged crypto TSMOM evidence as "promising but unverified" and neither primary source was read in full; a separate future pre-registration, not folded into this one.
- Will not treat platinum (PL) as combine-deployable even if the portfolio passes, per the same standing caveat as TSC-1.
- Will not re-sweep the lookback grid, add vol-scaling, or add instruments after seeing a Gate 0 FAIL.

## Open questions logged in advance (not resolved by this test)

- MGC/SIL cost estimates remain unmeasured guesses, exactly as flagged in TSC-1 — a PASS here is provisional on real fill data before live deployment.
- Fixed 1-contract sizing means portfolio risk is dominated by whichever instrument has the largest point value, same known simplification as TSC-1.
- Computing trailing return using same-contract-only price relatives (skipping the return contribution across a roll day, matching TSC-1's P&L convention) is a disclosed simplification of "true" continuous-series momentum; a back-adjusted continuous series would compute trailing returns slightly differently. Not expected to be large for monthly-horizon lookbacks but not verified in this seal.
- The 2003-2013-era MNQ-adjacent literature and this shop's own OHLCV falsification work (arXiv:2605.04004) are about pattern-based intraday signals, not this monthly-horizon cross-market mechanism — TSMOM-1's outcome says nothing about that separate, already-closed line, and vice versa.
- This is a 5-instrument universe dominated by precious metals and one equity-index micro; it is not a representative sample of the 58-67-market universes the original academic papers used, and a FAIL here does not necessarily generalize to "TSMOM is dead everywhere," only "TSMOM as specified does not survive on this shop's specific accessible universe/costs."
