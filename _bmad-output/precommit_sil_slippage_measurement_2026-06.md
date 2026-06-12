# Pre-Commitment Addendum 3: SIL Slippage Measurement (prospective quote capture)

**Date:** 2026-06-12 (committed before any captured data is analyzed)
**Parent chain:** pair survey `b54fb08` → 5m extension `023d0de` → results `98cd4ad`.

## Purpose

The SI–GC 5m LONG divergence fade passed all five Gate 0 criteria
(N=462, WR 57.8%, PF 1.27, +$12.60/trade, 2.21/day, worst-month 45.3%) and
failed qualification on exactly one clause: the ASSUMED slippage stress of
$10/RT (1 tick/side on SIL). This addendum replaces the assumption with a
prospective empirical measurement. **No historical data is re-tested; the
backtest result is frozen as-is.** This was authorized by Alex on 2026-06-12.

## Decision threshold (derived from the frozen result, fixed before capture)

The frozen measured WR (57.8%) clears its cost-adjusted breakeven iff

```
(80 + 3.74 + slip_RT) / 160 ≤ 0.578  →  slip_RT ≤ $8.74 per round trip
```

Slippage estimator: a market order pays the ask and exits at the bid relative
to the mid/last-based simulation price, so **empirical slip_RT ≈ the quoted
bid-ask spread in dollars**. SIL tick = $0.005 = $5/contract, so the bar sits
between 1 tick ($5 → comfortable pass) and 2 ticks ($10 → fail).

## Frozen measurement protocol

- **Instrument:** SILN26 (front-month micro silver; SIN26 captured alongside
  for context only — not part of the rule). Capture ends before the ~Jul 17 roll.
- **Capture:** poll TradeStation `/v3/marketdata/quotes/SILN26,SIN26` every
  5 s during 09:25–16:00 ET, Mon–Fri, via `capture_sil_quotes.py` (nohup
  daemon, auto-stops 2026-06-23). Rows appended to
  `data/quotes/sil_quote_capture.csv` (UTC timestamp, bid, ask, sizes, last).
- **Valid sample:** both bid and ask present, ask > bid, RTH window
  09:30–15:55 ET (the exact backtest session).
- **Qualifying session:** ≥ 3,000 valid SILN26 RTH samples (full session at
  5 s ≈ 4,600).
- **Minimum evidence:** ≥ 5 qualifying sessions (expected: Jun 15–19, 2026).

## Frozen decision rule

- **Primary metric:** median SILN26 bid-ask spread ($/contract) over all valid
  RTH samples pooled across qualifying sessions.
- **PASS** iff median spread ≤ **$8.74** AND no single qualifying session has a
  median spread > $10 (session-stability guard).
- **PASS consequence:** authorizes the WRITING of a Gate 1 pre-registration for
  SI–GC 5m LONG (which must use the measured spread, not $10, as its cost
  basis, and remains subject to the full Gate 1/Gate 2 pipeline and Alex's
  deployment decision). Nothing deploys from this measurement.
- **FAIL consequence:** the family closure from `98cd4ad` is confirmed final.
- Context stats reported but non-binding: p75/p90 spread, % samples at 1 tick,
  median bid/ask sizes (1-lot adequacy), SIN26 comparison, spread by hour.

Analysis is performed only by `analyze_sil_quotes.py` against these frozen
thresholds.
