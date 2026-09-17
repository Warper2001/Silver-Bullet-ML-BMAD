# Pre-Commitment: how to read GAP-1's N=30 decision, written before trade 30

**Date:** 2026-09-17. **Status:** committed BEFORE trade 30 lands. No parameter, threshold,
or config changes — the sealed decision rule in `preregistration_gap_fade_panic_open.md`
is unchanged and will still fire exactly as written. This document only says, in advance,
how the result should be read, so the reading isn't decided after seeing it.

**Live state today:** `trader-gap-fade`, N=**29** realtime trades, first trade 2026-06-25
(84 calendar days ago — the 30-day clock cleared long ago; only N binds). PF **1.517**,
net **+$1,944**, 14W/15L. At the current pace (29 trades / 84 days ≈ 2.4/wk) trade 30 is
imminent — days, not weeks.

## What the sealed rule literally does at N=30

`preregistration_gap_fade_panic_open.md`, OOS/Live Decision Rule:

> PF > 1.20: **scale to 2ct, continue.** PF 1.00–1.20: continue at 1ct, re-evaluate at
> N=60. PF < 1.00 at N ≥ 30: **STOP. Archive strategy.**

Gross win/loss today: $5,706.50 / $3,762.50. Trade 30 would need to be a loss of **≥$993**
to drop PF below the 1.20 SCALE line (the worst single trade to date is −$1,032, so this
is possible but not the base rate). **Absent an unusually bad trade 30, the literal rule
outputs SCALE to 2ct.**

## Why SCALE at N=30 is not proof of edge

`_bmad-output/diagnostics_gap_fade_power_gate_20260913/results.md` (sealed plan
`7a55e71f…`, ran against the Gate-0 trade log only, no live outcome read):

| At N=30 | Rate | Standard |
|---|---|---|
| STOP when the Gate-0 edge is real | 7.1% | ≤20% — passes |
| **SCALE when there is no edge at all** | **35.5%** | ≤5% — **fails** |
| Doesn't STOP a strategy actually losing $51/trade | 23.3% | — |

Under zero edge, SCALE stays above 30% all the way out to N=60 (30.3%) and only reaches
17.3% at N=200 — waiting a little does not fix this, because the PF>1.20 bar is small
relative to trade-to-trade noise (2025: mean +$102/trade, SD $384).

A real one-sided confirmatory test (α=0.05, 80% power) needs:

| Edge size | Trades needed | 5%-false-scale PF threshold |
|---|---|---|
| Full Gate-0 edge (+$102/trade) | 88 | 1.62 |
| Half edge (+$51/trade) | 351 | — |

The sealed log's actual 2026 rows (N=40 in the power-gate sample, PF 1.384) already
**behave like the half-edge case**, not the full one — and today's live PF (1.517) does
not clear even the N=60 exploratory threshold (1.79), let alone N=30's (2.30). By the
standard the strategy's own Gate-0 edge would need to clear to be confirmatory, a PF>1.20
SCALE reading at N=30 is weak evidence, indistinguishable from noise about a third of the
time.

## The backtest basis is also 16% smaller than when this rule was sealed

`_bmad-output/diagnostics_gap_fade_gate0_rescore_20260916/results.md` (prereg `151f1d05`,
logged access): the Gate-0 window was partly priced off the deferred MNQM26 contract.
Corrected: **N=115, PF 1.646, $8,281** vs. the sealed N=117/PF 1.761/$9,878 — a fifth of
the sealed net came from 11 pre-roll days that don't exist on front-month bars. The edge
still clears this document's own 1.40 "strong" bar, but the number the N=30 rule's PF
thresholds were calibrated against was overstated when they were chosen.

## What we will do

1. **Trade 30's literal output stands** — the sealed rule is not being overridden. If it
   says SCALE, position sizes to 2ct as written; if STOP, archive as written.
2. **A SCALE outcome will not be reported or treated as "the strategy is proven."** It
   will be logged as "did not fail the STOP bar," which is what the power gate showed it
   actually means. No document should cite N=30 SCALE as evidence of edge going forward.
3. **Data collection continues past N=30 regardless of outcome**, toward N=88 — the point
   a real confirmatory mean test has 80% power at the full Gate-0 edge. At the live rate
   (~2.4 trades/week) that's roughly **6 months out** (~2027-03); at the 2025 backtest rate
   (1.48/wk) it would be ~2027-07. Either way, this is the actual cost of a rule that would
   distinguish edge from noise, versus the sealed rule's N=30, which mostly doesn't.
4. **No new hand-set threshold is adopted here.** The exploratory 5%-false-scale PF
   thresholds (2.30@30, 1.79@60, 1.62@88) are cited as candidates for a future
   pre-registration, not adopted now — consistent with the "derive from a sweep, don't
   hand-set" policy.
5. If PF drops below 1.00 and STOP fires, that outcome is trustworthy as stated (E1: only
   7.1% false-STOP on a real edge) — no additional scrutiny needed on a STOP.

## What would actually justify scaling

Not a single PF snapshot at N=30. Either:
- **N≈88** with PF still clearing a properly-derived threshold (~1.62 exploratory,
  pending its own pre-registration), or
- An earlier pre-registered stronger test (e.g., adopting the 2.30@N30 / 1.79@N60
  exploratory thresholds via a sealed follow-up) — not decided today, flagged as the
  option if 6 months is judged too long to wait at 1ct.

## What we will NOT do

- Will not read a bare PF>1.20 at N=30 as confirmation of edge.
- Will not commission a new decision rule after seeing trade 30's result (that would be
  the same "decide after seeing the result" pattern this document exists to avoid).
- Will not extend the 1ct→2ct sizing decision beyond what the sealed rule already
  authorizes without a fresh pre-registration.
