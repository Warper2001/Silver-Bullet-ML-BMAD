# Research Queue: IFVG Fallback Entry (S27 Candidate)

**Added:** 2026-05-25
**Status:** ❌ DE-PRIORITIZED — exploratory backtest (2026-05-26) shows zero IFVG candidates armed; hypothesis rests on a flawed baseline
**Source:** F.7 in `technical-h1-liquidity-sweep-confluence-research-2026-05-20.md`

---

## ⚠️ Findings (exploratory backtest 2026-05-26)

**IFVG never fired. Zero additional trades vs baseline.**

Root cause: the original motivation ("47% pending expiries" in the 191-trade pre-CHoCH run)
was based on a flawed baseline. The 191-trade backtest was generated WITHOUT the M15 CHoCH
filter active in the backtest loop — the script was missing the `_update_m15_choch()` call.

The true S25+CHoCH baseline is **62 trades, PF=1.411**. In this config:
- All 62 limits filled — zero pending expiries.
- The CHoCH requirement selects only high-conviction setups where price reliably reaches
  the FVG midpoint. The "limit never fills" problem does not exist.
- IFVG candidates were never armed → IFVG never triggered.

The 36/62 (58%) "time" exits are time-stops **after fill** (held 60 bars without TP/SL).
That is a hold-period management problem, not an entry-fill problem. See S27-revised below.

**Decision: do not pre-register IFVG as S27. The code infrastructure is harmless
(flag=false, zero runtime cost) but the S27 experiment slot should go to hold-period
management.**

---

## Original Motivation (superseded)

In the S25 1-year backtest (191 trades, pre-CHoCH), **47% of exits were time-stops**
(91/191) — the limit order at FVG midpoint never filled within 240 bars. This motivated
the IFVG fallback concept. The 191-trade result has since been identified as a flawed
baseline (CHoCH not applied in backtest loop). The correct S25+CHoCH baseline is 62
trades with zero pending expiries.

---

## Concept (Bearish S25 example)

```
1. H1 sweep active + M15 CHoCH fired → M1 FVG detected (bearish gap: bar[0].low > bar[2].high)
2. Primary entry: SELL LIMIT at FVG midpoint (50% of gap)
3. Primary entry fills → normal TP/SL/time-stop management
   OR
3b. Primary entry does NOT fill within MAX_PENDING_BARS (240 bars)
    → cancel pending limit
    → mark gap zone as "IFVG candidate" (original gap high/low stored)

4. IFVG trigger: price later closes ABOVE the original FVG high (gap fully violated)
   → the zone inverts: original bearish gap now acts as resistance
   → place new SELL LIMIT at the IFVG midpoint (same 50% entry logic)
   → same TP/SL multipliers, same time-stop rules

5. If IFVG also misses → discard, move on
```

---

## Hypothesis (pre-registration candidate)

> Adding an IFVG fallback entry (triggered when primary FVG limit expires unfilled and
> price subsequently violates the gap) improves annualised Profit Factor vs S25 baseline
> by ≥ 0.10 PF points on the same 1-year backtest window, without increasing max drawdown
> by more than 20%.

**H₀ (null):** IFVG fallback PF ≤ S25 baseline PF (no improvement or degradation)
**H₁ (alternative):** IFVG fallback PF > S25 baseline PF + 0.10

---

## Implementation Sketch

Changes required in `src/research/strategy_core.py`:

1. **New dataclass:** `IFVGCandidate(frozen=True)` — stores original gap high/low, direction,
   expiry time (e.g., same H1 sweep window), bar timestamp of original FVG detection.

2. **New function:** `check_ifvg_trigger(bars, candidate, config) -> Optional[FVGSignal]`
   — returns an entry signal if current bar closes beyond the original gap boundary.

3. **Trader integration:** `_detect_and_enter()` checks for pending IFVG candidates
   after primary entry path. IFVG entry uses identical bracket order logic.

4. **Config parameter:** `enable_ifvg_fallback: bool = False` in `StrategyConfig`
   (off by default, activated explicitly for the S27 backtest run).

---

## Pre-Registration Requirements (before any backtest)

Per methodology (AR6–AR8):

1. Commit this planning doc now (done)
2. When ready to test: run `prereg_seal.py` with full hypothesis, exact config snapshot,
   and data range BEFORE running `backtest_tier2_1year_validation.py`
3. S27 pre-reg must reference S25 baseline PF as the comparison anchor
4. Use the same 1-year CSV window as S25 (2025-05-19 → 2026-05-19) for comparability

---

## Sequencing Constraint

**Do not start S27 pre-registration until S25 reaches N≥20 live trades.**

Reason: S25 is the active OOS experiment. Running a new backtest variant now risks
contaminating the research record if S25 conclusions later need to be revisited. S27 is
purely in-sample exploration on the training window — that's fine — but it should not
begin until S25's live collection phase is independently progressing.

Estimated S25 N=20 timeline: ~4–8 weeks from 2026-05-26 (Monday market open)
at the observed ~2–4 trades/week rate.

---

## Open Questions

1. **IFVG expiry:** Should the IFVG candidate expire when the H1 sweep expires (6 bars),
   or have its own independent window? Shorter window = higher quality, fewer trades.

2. **IFVG entry price:** Midpoint of original gap (same as primary) or midpoint of the
   violated zone (slightly different)? ICT canonical is the midpoint of the original gap.

3. **Multiple IFVG candidates:** Can there be more than one IFVG active at a time?
   Simplest rule: max one IFVG candidate per H1 sweep window.

4. **Interaction with M15 CHoCH:** Should M15 CHoCH still be required to be active at
   IFVG trigger time, or does the original CHoCH fire count?
