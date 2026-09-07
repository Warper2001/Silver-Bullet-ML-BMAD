# Pre-registration: Futures Term-Structure / Roll-Yield Carry (TSC-1)

**Status:** SEALED before any backtest code touches contract-month price data.
**Date:** 2026-09-03
**Origin:** `academic-lit-low-frequency-futures-crypto-edge-candid-2026-09-03` deep-recon (see `research.md` in that run folder) — Recommendation 1 (the top-ranked candidate of 5 screened families).
**Author:** Claude (background session), for Alex.

## Hypothesis

The slope of a futures curve (backwardation vs. contango between the front and next contract month) predicts the sign of that instrument's near-term roll return, independent of price direction. This is the mechanism documented by Erb & Harvey (2006, *FAJ*) and Koijen-Moskowitz-Pedersen-Vrugt (2018, *JFE*, "Carry") — see the deep-recon report, dimension 1, refs [5][6]. This pre-registration tests whether a **time-series carry-timing** implementation (long/short/flat the front-month contract by the sign and magnitude of its own annualized roll yield, not cross-sectionally ranked against other instruments — the universe below is too small for a meaningful cross-sectional sort) survives on daily settlement data, after realistic per-contract costs, on the specific CME micro/small futures this shop can actually access.

This is explicitly **not** a re-test of the academic literature's institutional-scale result. It is a fresh, this-shop's-cost-structure test of the same mechanism.

## Universe (frozen — 5 instruments)

| Root | Instrument | Contract months probed | Cost/RT (1 contract) | Source of cost figure |
|---|---|---|---|---|
| MGC | Micro gold | all 12 (F–Z) | **$6.00 (estimate)** | Not previously measured by this shop — commission + 1-tick slippage estimate (tick=$1.00). Flagged as an assumption; must be replaced with a real-fill measurement before any live deployment. |
| SIL | Silver (micro/small, per TradeStation root) | H,K,N,U,Z | **$7.00 (estimate)** | Same caveat as MGC. |
| MHG | Micro copper | all 12 (F–Z) | **$4.00** | Measured, YANK cross-instrument portability study, 2026-07-05 (`project_yank_cross_instrument_copper`), contract MHGU26. |
| PL | Platinum (no micro exists) | F,J,N,V | **$34.00** | Measured, same study, contract PLV26. Already flagged combine-fit FAIL for standalone full-size sizing — included here for signal-strength evidence only; NOT a combine-deployable leg on its own. |
| MNQ | Micro Nasdaq-100 | H,M,U,Z | **$2.24** | Measured, this shop's live cost model (`preregistration_event_fade_prospective.md`). |

Instruments were chosen because they are the only ones in this shop's history with either (a) a real measured micro-futures cost figure, or (b) a clearly liquid, well-known CME metals contract — not cherry-picked post-hoc from a wider backtest. No instrument is added or dropped after this seal based on backtest performance.

## Data

- Source: TradeStation `marketdata/barcharts/{contract_symbol}` (Daily unit), one specific contract-month symbol at a time (e.g. `MGCZ23`), confirmed empirically on 2026-09-03 to serve full historical daily bars for expired contracts.
- Expiration dates per contract from TradeStation `marketdata/symbols/{contract_symbol}`.
- Sample window: as far back as each contract's listed history reaches (target 2021-01-01) through the most recent complete week before this backtest runs.
- **Dev window:** 2021-01-01 – 2025-12-31.
- **Holdout (OOS, untouched until the single evaluation run):** 2026-01-01 – present (2026-09-03). This is a short holdout (~8 months) — flagged as a real limitation, not concealed.

## Signal construction (frozen)

For each instrument, at each trading day `t`:
1. Identify the front contract (nearest unexpired) and next contract (next-nearest) from the set of contracts with valid data on `t`.
2. `roll_yield_annualized(t) = ln(F_front(t) / F_next(t)) * 365 / (expiry_next - expiry_front in days)`.
   Positive = backwardation (front pricier than next) = classic "long-rewarded" carry state.
3. Rebalance **weekly**, on each Friday's settle (or the last trading day of the week if Friday is a holiday).
4. Position rule (time-series carry timing, one pre-registered deadzone parameter `d`):
   - `roll_yield_annualized(t) > +d` → **long** 1 front-month contract.
   - `roll_yield_annualized(t) < -d` → **short** 1 front-month contract.
   - otherwise → **flat**.
5. Hold the front-month contract; roll automatically to the new front contract 5 trading days before expiry (avoids delivery/illiquidity window), at that day's settle price, incurring one round-turn cost on each roll (whether or not the position changed sign — a roll is a real trade even when the position is held through it).
6. Position size: fixed 1 contract per instrument (no vol-scaling in this first test — a known simplification, logged as an open question below).

## The one swept knob (pre-declared grid, "one knob at a time")

`d` (the deadzone, annualized) is swept over **{0%, 3%, 5%, 8%, 12%}** on the **dev window only**. The value of `d` that maximizes dev-window net Sharpe across the 5-instrument equal-weight portfolio is locked and used, unchanged, for the single holdout evaluation. No other parameter (rebalance frequency, roll-ahead days, position size, instrument set) is swept or tuned at any point — if the strategy needs a second knob to work, that is a finding to report, not a reason to add a second sweep to this seal.

## Portfolio construction

Equal-weight (1 contract each) sum of the 5 instruments' daily P&L in USD. No leverage, no vol-targeting in this first test.

## Decision rule (frozen, two-gate)

- **Gate 0 (dev window, in-sample):** the locked-`d` portfolio must show **net Sharpe > 0.5** AND **net profit factor > 1.15** over 2021-2025. Below this, the mechanism is recorded as **FAIL — does not survive this shop's cost structure at this scale**, and no holdout is spent.
- **Gate 1 (holdout, spent once):** if Gate 0 passes, evaluate the same locked configuration, unchanged, on 2026-01-01–present. PASS requires **net PF > 1.0** (i.e., does not lose money net of costs) and a same-sign Sharpe to the dev window. Given the short holdout window, this is explicitly a **directional check, not a statistical confirmation** — flagged as such in the report regardless of outcome.
- A Gate 0 FAIL is terminal for this pre-registration: no re-sweep, no new deadzone grid, no added instruments. A new mechanism variant (e.g., cross-sectional carry once more instruments are added, or vol-scaled sizing) would need its own new pre-registration.

## What this run will NOT do

- Will not test cross-sectional carry ranking (the 5-instrument universe is too small for a meaningful cross-sectional sort — this is explicitly the time-series carry-timing variant only).
- Will not vol-scale position sizing.
- Will not add BTC/ETH term-structure carry (the deep-recon report flagged this as a real but unread lead — a separate future pre-registration, not folded into this one).
- Will not treat platinum (PL) as combine-deployable even if the portfolio passes — it stays a full-size, no-micro instrument with the previously-documented combine-fit problem; a PASS here licenses a backtest-evidence conclusion about the carry mechanism, not a live-deployment decision for PL specifically.

## Open questions logged in advance (not resolved by this test)

- MGC/SIL cost estimates are unmeasured guesses — a PASS here is provisional on getting real fill data before live deployment.
- Fixed 1-contract sizing means the portfolio's risk is dominated by whichever instrument has the largest point value — this is a known simplification, not an oversight.
- The Bhardwaj et al. pre/post-2000 decay tension flagged in the deep-recon report (ref [7]) is not resolved by this backtest — a Gate 0 PASS in 2021-2025 dev data says nothing about whether the mechanism was stronger before 2000.
