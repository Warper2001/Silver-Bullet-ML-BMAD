# Research Queue: ES/MNQ Divergence Filter (S29 Candidate)

**Added:** 2026-05-25
**Status:** QUEUED — blocked until S25 reaches N≥20 live trades; runs after S27+S28
**Priority:** Medium-High (signal quality) / High friction (requires second instrument feed)

---

## Motivation

MNQ (Micro Nasdaq-100) and ES (E-mini S&P 500) are closely correlated but can diverge
short-term during sector-specific flows, ETF rebalancing, or index-specific news.

When the H1 liquidity sweep is genuine broad-market structure, ES and MNQ should move
in the same direction. A bearish H1 sweep on MNQ that is **not confirmed by ES** may be:
- A Nasdaq-specific move (tech/semi sector rotation)
- A temporary dislocation that mean-reverts quickly
- A lower-quality sweep that does not carry the same predictive weight

Adding ES confirmation as a gate on H1 sweep validity could reduce false sweeps and
improve the quality of the trade universe — at the cost of requiring a live ES data feed.

---

## Concept (Bearish example)

```
1. H1 bearish sweep detected on MNQ
2. ES divergence check: does ES also show a bearish H1 swing high violation
   within the same lookback window?
   → Yes: sweep confirmed, proceed normally
   → No: sweep flagged as "unconfirmed", skip or require wider filter margins

Divergence metric options:
  A. Same directional sweep required on ES H1 (binary)
  B. MNQ/ES return correlation over last N H1 bars (continuous, threshold-gated)
  C. ES must not be in a bullish H1 trend (prevents counter-trend entries)
```

---

## Hypothesis (pre-registration candidate)

> Requiring ES H1 directional confirmation of the MNQ sweep (within same lookback
> window) improves Profit Factor vs S25 baseline by filtering low-quality sweeps,
> without reducing trade count by more than 35%.

**H₀ (null):** ES-confirmed sweep PF ≤ S25 baseline PF
**H₁ (alternative):** ES-confirmed sweep PF > S25 baseline PF + 0.05

---

## Implementation Sketch

1. **ES data feed:** TradeStation REST API already supports ES/ESM26 bars with the
   same 1-minute OHLCV endpoint. A secondary polling request alongside the MNQ poll
   is straightforward to add.

2. **ES H1 state:** Mirror `_update_h1_structure()` logic for ES bars; detect ES sweep
   direction independently using the same `detect_liquidity_sweep()` pure function.

3. **Confirmation gate:** In `_detect_and_enter()`, add a check:
   `es_h1_bearish_sweep_active` must be True when MNQ bearish sweep is True.

4. **Config parameters:** `enable_es_confirmation: bool = False`,
   `es_symbol: str = "ESM26"` in `StrategyConfig`.

5. **Backtest data:** Requires a historical ES 1-minute CSV alongside MNQ. Need to
   source and align this before the backtest can run.

6. **Parity gate:** With `enable_es_confirmation=False`, output byte-identical.

---

## Pre-Registration Requirements

Per methodology (AR6–AR8):
1. Commit planning doc (done)
2. Source ES historical 1-minute data (2025-05-19 → 2026-05-19) before backtest
3. Run `prereg_seal.py` with exact divergence metric and threshold BEFORE backtesting
4. Reference S25 (or latest active baseline) as comparison anchor

---

## Sequencing Constraint

Highest implementation friction of the current queue — requires new data source.
Run after S27 (IFVG) and S28 (news calendar) are evaluated. Pre-registration
cannot begin until ES data is sourced and the metric is fully specified.

---

## Open Questions

1. **Divergence metric:** Binary sweep match (simple) vs return correlation (continuous)?
   Binary is easier to pre-register with a clean H₁; correlation requires a threshold
   chosen before the backtest.

2. **Lag tolerance:** ES and MNQ sweeps don't always happen in the same H1 bar. Allow
   ±1 H1 bar tolerance? Tighter = higher quality but fewer confirmed sweeps.

3. **Data sourcing:** TradeStation has ES data but download volume/rate limits apply.
   May need to batch-download before the backtest run.

4. **Roll-over handling:** ES futures roll quarterly (ESM26 → ESU26 in June 2026).
   Backtesting across the roll requires stitching or adjusted continuous contract.

5. **Interaction with vol filter:** If ES is diverging, the vol filter may already
   be catching the regime. Worth measuring marginal contribution.
