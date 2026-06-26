# Pre-Commitment: HG (Copper) Slippage Measurement — prospective quote capture

**Date:** 2026-06-26 (committed before any captured copper-quote data is analyzed)
**Author:** Alex
**Parent chain:** YANK cross-instrument portability sweep (exploratory, ML-off,
structural mode) → HG = standout (net PF 1.183 at an *assumed* $3.50/RT cost).
**Sibling precedent:** `precommit_sil_slippage_measurement_2026-06.md` (same method).

## Purpose

The frozen YANK engine (bearish-FVG + H1-sweep + M15-CHoCH, ML-off) was run on
4 non-MNQ instruments + an MNQ reference over the exploratory window
2025-05-19 → 2026-02-28 (sealed holdouts untouched; regression gate passed,
MNQ byte-identical). **HG (micro copper, MHG) was the only non-MNQ instrument
that both survived a rough cost model and was orthogonal to MNQ.** Its
survival, however, rests entirely on an *assumed* round-trip cost. This
addendum replaces the assumption with a prospective empirical measurement of
the real micro-copper bid-ask spread. **No historical data is re-tested; the
exploratory backtest result is frozen as-is.**

## Frozen exploratory result (the thing being cost-tested — NOT re-run)

- Instrument in backtest: micro copper economics, point_value $2,500/pt,
  tick $0.0005 = **$1.25/contract**.
- N = 95 trades, gross PF = 1.439, gross net = +$630, **gross avg = $6.63/trade**
  (per 1 micro contract), window 2025-05-19 → 2026-02-28, ML-off, structural
  mode (no daily breaker, no $-gap ceiling, zero commission), bearish-only.
- Trade list: `data/reports/backtest_1year_20260625_225218.csv` (frozen).

## Decision threshold (derived from the frozen result, fixed before capture)

Decision rule = the roundtable's pre-agreed bar for an exploratory candidate to
earn a sealed-holdout slot: **net PF ≥ 1.10 AND net $/trade ≥ $2.00 AND N ≥ 40**,
all at the measured cost. Applying per-trade all-in cost `c` to the frozen
trade list (derive-don't-assert; computed, not hand-set):

| all-in c ($/RT) | net PF | net $/trade |
|---|---|---|
| 4.00 | 1.151 | 2.63 |
| **4.63** | **1.121** | **2.00**  ← net$/trade binds |
| 4.84 | 1.10 | 1.84 |
| 6.63 | 1.000 | 0.00  (pure breakeven) |

- Binding **all-in RT cost ceiling: c\* = $4.63/RT** (the `net $/trade ≥ $2`
  clause binds before the `net PF ≥ 1.10` clause, which allows up to $4.84).
- `c = measured_RTH_median_spread_usd + COMMISSION_USD`, with stated assumption
  **COMMISSION_USD = $1.50/RT** (TradeStation micro futures commission + exchange
  + NFA fees, RT, conservative). Equivalent measured-spread ceiling:
  **median spread ≤ $3.13/RT (≤ 2.50 ticks).**

Slippage estimator (same as SIL precedent): a market order pays the ask on entry
and exits at the bid, so **empirical slip_RT ≈ the quoted bid-ask spread in
dollars.** Micro copper tick = $1.25, so the ceiling sits between 2 ticks
($2.50 spread → comfortable pass) and 3 ticks ($3.75 spread → fail).

## Frozen measurement protocol

- **Instrument (binding):** front-month micro copper **MHGN26**. **MHGU26**
  captured alongside as context. **Roll clause (objective):** if MHGN26's valid
  RTH sample count, pooled across qualifying sessions, falls below 50% of
  MHGU26's (i.e. liquidity has rolled to Sep), the binding symbol switches to
  MHGU26. This is decided by sample density, not by which is cheaper.
- **Capture:** poll TradeStation `/v3/marketdata/quotes/MHGN26,MHGU26` every 5 s
  during 09:25–16:00 ET, Mon–Fri, via `capture_hg_quotes.py` (nohup daemon,
  auto-stops 2026-07-08, before the copper Jul→Sep liquidity roll completes).
  Rows appended to `data/quotes/hg_quote_capture.csv`.
- **Valid sample:** binding symbol, bid and ask present, ask > bid, RTH window
  09:30–15:55 ET (matches the backtest session).
- **Qualifying session:** ≥ 3,000 valid binding-symbol RTH samples.
- **Minimum evidence:** ≥ 5 qualifying sessions (expected Jun 26 – Jul 2, 2026).

## Frozen decision rule

- **Primary metric:** pooled median binding-symbol bid-ask spread ($/contract)
  over all valid RTH samples across qualifying sessions; all-in
  `c = median_spread + $1.50`.
- **PASS** iff `c ≤ $4.63/RT` **AND** no single qualifying session's all-in
  median exceeds **$5.63/RT** (the pure-breakeven guard, net PF = 1.0).
- **PASS consequence:** authorizes WRITING a Gate-1 pre-registration for the
  HG (copper) structural strategy on its sealed holdout
  (`data/sealed_holdout/hg_1min_holdout_20260301_plus.csv`), using the *measured*
  spread as the cost basis — NOT this assumption. Nothing deploys from this
  measurement; the holdout run + Alex's decision remain separate gates.
- **FAIL consequence:** HG is reclassified gross-only; no holdout is spent, and
  the cross-instrument portability claim is logged as "structural edge real but
  not cost-survivable at micro size."
- Context stats reported but non-binding: p75/p90 spread, % at ≤1/≤2 ticks,
  median bid/ask sizes (1-lot adequacy), MHGU26 comparison, spread by hour.

Analysis is performed only by `analyze_hg_quotes.py` against these frozen
thresholds.
