# Pre-Commitment: PL (Platinum) Slippage Measurement — prospective quote capture

**Date:** 2026-06-26 (committed before any captured platinum-quote data is analyzed)
**Author:** Alex
**Parent chain:** YANK cross-instrument portability fan-out (batch 2) → PL = a
SECOND orthogonal candidate (net PF 1.274, corr-to-MNQ +0.104) alongside HG.
**Sibling precedents:** `precommit_hg_slippage_measurement_2026-06.md`,
`precommit_sil_slippage_measurement_2026-06.md` (same method).

## Purpose

The frozen YANK engine (bearish-FVG + H1-sweep + M15-CHoCH, ML-off, structural
mode) run on platinum produced an orthogonal, positive-skew edge. Its survival
rests on an *assumed* cost. Platinum is THINLY traded — a live post-close probe
showed a 13-tick ($65/RT) spread — so this measurement is make-or-break. This
addendum replaces the assumption with a prospective empirical measurement of the
real platinum bid-ask spread. **No historical data is re-tested; the exploratory
backtest result is frozen as-is.**

## Frozen exploratory result (cost-tested, NOT re-run)

- Instrument: FULL platinum PL, 50 troy oz, **$50/pt, tick $0.10 = $5/contract**
  (no CME micro platinum exists).
- N = 101 trades, gross PF = 1.344, gross net = +$6,265, **gross avg = $62.03/trade**
  (per 1 full contract), window 2025-05-19 → 2026-02-28, ML-off, structural mode,
  bearish-only.
- Trade list: `data/reports/backtest_1year_20260626_025416.csv` (frozen).

## Decision threshold (derived from the frozen result, fixed before capture)

Decision rule for an exploratory candidate to earn a holdout slot: **net PF ≥ 1.10
AND N ≥ 40** at the measured cost. (Mary's `net $/trade ≥ $2` clause is non-binding
for a full $50/pt contract — it is satisfied at any plausible cost — so net PF is
the binding clause here, not net $/trade.) Applying per-trade all-in cost `c` to
the frozen trade list (computed, not hand-set):

- max all-in cost for **net PF ≥ 1.10: c\* = $41.71/RT (8.3 ticks)** ← binding
- pure breakeven (net PF = 1.0): $62.02/RT (12.4 ticks)

`c = measured_RTH_median_spread_usd + COMMISSION_USD`, with stated assumption
**COMMISSION_USD = $4.00/RT** (TradeStation full-size futures commission + exchange
+ NFA fees, RT). Equivalent measured-spread ceiling: **median spread ≤ $37.71/RT
(≤ 7.54 ticks).** Slippage estimator: empirical slip_RT ≈ quoted bid-ask spread
(market order pays ask in, sells bid out). Platinum tick = $5.

## Frozen measurement protocol

- **Binding symbol:** the platinum contract (PLN26 July / PLV26 Oct) with the
  GREATER number of valid RTH samples pooled across qualifying sessions (the
  actively-traded front; robust to the imminent Jul→Oct roll). The other is
  captured as context.
- **Capture:** poll TradeStation `/v3/marketdata/quotes/PLN26,PLV26` every 5 s
  during 09:25–16:00 ET, Mon–Fri, via `capture_pl_quotes.py` (nohup daemon,
  auto-stops 2026-07-08). Rows appended to `data/quotes/pl_quote_capture.csv`.
- **Valid sample:** binding symbol, bid and ask present, ask > bid, RTH
  09:30–15:55 ET.
- **Qualifying session:** ≥ 3,000 valid binding-symbol RTH samples.
- **Minimum evidence:** ≥ 5 qualifying sessions.

## Frozen decision rule

- **Primary metric:** pooled median binding-symbol bid-ask spread ($/contract)
  over valid RTH samples across qualifying sessions; all-in
  `c = median_spread + $4.00`.
- **PASS** iff `c ≤ $41.71/RT` **AND** no single qualifying session's all-in
  median exceeds **$62.02/RT** (the pure-breakeven guard).
- **PASS consequence:** authorizes WRITING a Gate-1 pre-registration for the PL
  structural strategy on its sealed holdout
  (`data/sealed_holdout/`, if a PL slice exists; else acquire one under seal),
  using the *measured* spread as cost basis. Combine-fit (full-contract notional
  ~$78K vs 50K account, per-trade SL $ risk) is a SEPARATE downstream gate.
  Nothing deploys from this measurement.
- **FAIL consequence:** PL reclassified gross-only; no holdout spent; logged as
  "orthogonal structural edge real but not cost-survivable at full-contract size
  given platinum illiquidity."
- Context stats reported but non-binding: p75/p90 spread, % at ≤8 ticks, median
  bid/ask sizes (1-lot adequacy), the non-binding symbol comparison, spread by hour.

Analysis is performed only by `analyze_pl_quotes.py` against these frozen
thresholds.
