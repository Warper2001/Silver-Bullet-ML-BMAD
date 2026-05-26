# Research Queue: Hold-Period Exit Management (S27-revised Candidate)

**Added:** 2026-05-26
**Status:** QUEUED — blocked until S25 reaches N≥20 live trades
**Replaces:** IFVG fallback (S27-original), which was de-prioritized after exploratory backtest

---

## Motivation

In the true S25+CHoCH backtest (62 trades, PF=1.411):

| Exit type | Count | % |
|---|---|---|
| SL (stop-loss) | 12 | 19% |
| TP (take-profit) | 14 | 23% |
| Time-stop after fill | 36 | **58%** |

58% of filled trades were held for 60 bars (1 hour of M1 bars) and exited flat — price
did not reach TP or SL within the hold window. These are likely trades where the
directional move was real but slow, or where price oscillated around entry without
committing. A flat 60-bar exit leaves substantial unrealised potential on the table.

The TP multiplier is 6× gap size (aggressive). A trade that moves in the right direction
but not far enough hits the time-stop rather than TP. Possible improvements:

1. **Reduce max_hold_bars** (e.g., 40 or 30) — exit sooner on stalled trades, reduce
   capital tie-up, potentially improve per-trade quality.
2. **Trailing stop** — lock in partial gains as price moves toward TP; prevents giving
   back a profitable position that stalls before reaching the full TP target.
3. **Partial TP** — take half at 3× gap size, trail the rest to 6×. Improves win rate
   on partially-profitable trades.
4. **Breakeven stop** — after price moves X× gap size in our favour, move SL to entry.
   Converts losers that briefly went green into breakevens.

---

## Hypothesis (pre-registration candidate)

To be specified before backtest. Candidate:

> Adding a breakeven stop (move SL to entry when unrealised P&L ≥ 2× gap size) improves
> Profit Factor vs S25 baseline (PF=1.411) by reducing the average loss on time-stopped
> trades that briefly went profitable, without reducing win rate below 50%.

**H₀:** Modified exit PF ≤ S25 baseline PF (1.411)
**H₁:** Modified exit PF > S25 baseline PF + 0.05

---

## Implementation Notes

All exit logic is in `strategy_core.check_exit()` and `Tier2StreamingTrader._advance_active_trade()`.
Changes are isolated to exit management — entry logic and filters unchanged.

New `StrategyConfig` fields (all off by default):
- `enable_breakeven_stop: bool = False`
- `breakeven_trigger_mult: float = 2.0`  (move SL at 2× gap move in our favour)
- `enable_trailing_stop: bool = False`
- `trailing_stop_mult: float = 1.5`  (trail at 1.5× gap behind peak)

Parity gate: with all new flags False, output byte-identical to S25 baseline.

---

## Sequencing Constraint

Do not begin pre-registration until S25 reaches N≥20 live trades.

---

## Open Questions

1. **Which mechanism first?** Breakeven stop is simplest and most conservative.
   Trailing stop is more complex but addresses a wider class of stalled trades.

2. **Interaction with max_hold_bars:** If a breakeven stop is active, should
   max_hold_bars be extended (since we're protected on the downside)?

3. **Commission impact:** Partial TP adds a second round-trip commission per trade.
   At $4/RT × 5 contracts, this is non-trivial on small gaps.
