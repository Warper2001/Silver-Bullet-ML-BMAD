# Pre-Registration: GAP-1 — Panic-Open Mean-Reversion Fade on MNQ

**Registered:** 2026-06-25
**Authored by:** Alex (warper2001@gmail.com)
**Status:** SEALED — frozen at commit time. No parameter amendments after this commit.

---

## Transparency Disclosure

This pre-registration was authored AFTER exploratory analysis of both the 2025
full-year and 2026 YTD datasets. Both datasets have been seen. The parameters
below are motivated by prior theory and confirmed directionally by exploration;
they are NOT tuned to optimize reported results.

The critical finding that changed the hypothesis: the published "77.8% fill rate
for tiny gaps" does NOT apply to MNQ futures. Tiny gaps (<0.5%) show PF < 1.0
in the data. Large gaps (>=0.5%) show meaningful IS edge. This hypothesis
inversion is documented here before any formal backtest run.

The parameters below are locked at commit time. The true prospective OOS test
begins the trading day after this commit is sealed.

---

## Hypothesis

> The MNQ "Panic-Open Mean-Reversion Fade" strategy — going SHORT when RTH open
> is >=0.5% above prior RTH close, going LONG when RTH open is >=0.5% below prior
> RTH close, with 2x gap stop, prior-RTH-close as target, and 13:00 ET time-stop,
> excluding Fridays — produces a profit factor > 1.20 on the full 2025 IS dataset
> (Gate 0) and maintains PF > 1.00 prospectively from 2026-06-26 onward.

**Mechanism:** Large overnight dislocations (>=0.5% ~100+ MNQ points) at RTH open
represent panic or euphoria reactions in thin pre-market conditions. Institutional
order flow in the first 3.5 RTH hours tends to absorb the dislocation and partially
revert toward the prior session's equilibrium price. The 13:00 ET time-stop captures
this morning mean-reversion window without requiring a complete gap fill.

---

## Frozen Parameters

All parameters below are locked at commit time. No amendments permitted.

| Parameter | Frozen Value | Prior Rationale |
|---|---|---|
| Gap threshold | >= 0.5% of prior RTH close | Separates dislocation from noise; below this Globex already digested the move |
| Direction | Fade (gap up = SHORT, gap down = LONG) | Mean-reversion hypothesis |
| Entry price | RTH open: open of first 1-min bar at 09:30 ET | At-the-open market order; highest liquidity of session |
| Target | Prior RTH close (100% gap fill) | The equilibrium anchor |
| Stop | 2.0x gap_abs beyond RTH open | Gap that extends 2x is momentum not panic |
| Time stop | Close at market at 13:00 ET bar open | Morning reversion window only; afternoon is directionally driven |
| Friday exclusion | YES | Weekend news risk + position squaring weakens fade thesis |
| Contract size | 1 MNQ ($2/point) | Conservative sizing for combine; scale post-validation |
| Max trades/day | 1 | One gap per session |
| Prior RTH close | Close of last 1-min bar before 16:00 ET (15:59 bar) | Official RTH session end |
| RTH open | Open of 09:30 ET 1-min bar | Official RTH session start |
| Data source | data/processed/dollar_bars/1_minute/mnq_1min_2025.csv (IS) | Confirmed path |
| Min RTH bars | 300 per prior session to compute prior close | Filters half-days and data outages |

---

## Gate 0 Decision Rule (2025 calendar year, IS plausibility check)

Primary metric: Gross Profit Factor on the full 2025 dataset.

| Condition | Verdict |
|---|---|
| N < 60 | INSUFFICIENT_SAMPLE — investigate data or threshold |
| N >= 60 AND PF <= 1.10 | NO_EDGE — strategy CLOSED, do not access OOS |
| N >= 60 AND 1.10 < PF <= 1.40 | WEAK_EDGE — proceed to live paper at 1ct; N>=40 live for decision |
| N >= 60 AND PF > 1.40 | STRONG_EDGE — proceed to live paper at 1ct; N>=30 live for decision |

Secondary checks (all required at WEAK_EDGE or above):
- Win rate >= 55%
- Max consecutive losses <= 10
- No single calendar month worse than -$600 gross (1ct)

---

## OOS / Live Decision Rule (prospective, from 2026-06-26)

Decision after N >= 30 live trades AND >= 30 calendar days from first live trade:
- PF > 1.20: scale to 2ct, continue
- PF 1.00-1.20: continue at 1ct, re-evaluate at N=60
- PF < 1.00 at N >= 30: STOP. Archive strategy.

---

## What We Will NOT Do

1. No DOW filter changes after seeing results (Friday exclusion is pre-committed above).
2. No gap threshold tuning after seeing Gate 0 results. 0.5% is locked.
3. No stop multiplier changes after seeing results. 2.0x is locked.
4. No time-stop changes after seeing results. 13:00 ET is locked.
5. No data subsetting ("strategy works better in trending/volatile months").
6. No direction asymmetry split ("only take longs" or "only take shorts").

---

## Early Stop Rule

Abandon Gate 0 if N < 30 qualifying trades appear before July 2025 in the IS data.
This indicates a data or threshold problem, not a strategy failure.

---

## Relationship to Existing Strategies

Gap fade entry is at 09:30 ET; YANK/MIM earliest entries are post-09:30. No
simultaneous position overlap. This is a standalone strategy with no modifications
to YANK or MIM-NB.

---

## Backtest Script

backtest_gap_fade.py in repo root.

Run: .venv/bin/python backtest_gap_fade.py

Output: data/reports/gap_fade_<timestamp>.csv and console summary.
