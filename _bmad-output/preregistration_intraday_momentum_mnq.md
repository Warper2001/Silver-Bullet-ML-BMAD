# Pre-Registration: MIM-X — Market Intraday Momentum, MNQ Cost Screen

**Registered:** 2026-08-28
**Authored by:** Alex (warper2001@gmail.com), drafted in BMAD session
**Status:** SEALED at commit time. Append-only amendments.

---

## 1. The question, and why it is worth one test

Baltussen, Da, Lammers & Martens, *"Hedging demand and market intraday momentum"*,
**Journal of Financial Economics 142 (2021) 377–403**, report that across 60+ futures
1974–2020 the return in the **last 30 minutes before the close** is positively predicted
by the **return over the rest of the day** (previous close → last 30 min). Annualised
Sharpe 0.87–1.73 at asset-class level. Mechanism: short-gamma hedging demand from options
market makers and leveraged-ETF rebalancers, which mechanically trades *with* price
direction and concentrates at day's end.

**The authors explicitly decline to net transaction costs** (verbatim): *"Note that we do
not consider transaction costs. Given that trading on market intraday momentum requires
frequent rebalancing, the strategy as presented might not be exploitable to many investors
after accounting for transaction costs."*

This document does exactly one thing: **apply this project's own friction screen to their
rule on MNQ.** It is not a search for a variant that works.

## 2. Why this needs no parameter selection

The rule is **fully specified by an external, peer-reviewed source** published before this
project examined it. There is no tuning freedom, and that is the point — this is the
cleanest test available to this shop, because the hypothesis was not generated here.

## 3. Frozen specification

| Item | Value |
|---|---|
| Instrument | MNQ, 1 contract, $2.00/point |
| Session | RTH, 09:30–16:00 ET |
| `r_ROD` (signal) | close(15:30 ET bar) / close(prior session 15:59 ET bar) − 1 |
| `r_LH` (traded) | close(15:59 ET bar) / close(15:30 ET bar) − 1 |
| Position | LONG if `r_ROD > 0`, SHORT if `r_ROD < 0`, flat if exactly 0 |
| Entry | close of the 15:30 ET bar |
| Exit | close of the 15:59 ET bar |
| Trades/day | exactly 1 |
| Session validity | a day is used only if BOTH a 15:30 and a 15:59 ET bar exist, AND the prior session has a 15:59 ET bar |
| **Primary friction** | **$3.00 round trip** (commission $1–2 + ~1 tick, per broker schedules) |
| Secondary friction | $2.00 RT (this project's measured booked commission) reported alongside |
| Data | `mnq_1min_2025.csv` + `mnq_1min_2026_ytd.csv` (2025-01-01 → 2026-06-11) |

## 4. Primary metric and decision rule

**Primary:** mean **net** P&L per trade at $3.00 friction, with a two-sided t-test on the
per-trade net P&L series.

| Condition | Verdict | Pre-committed action |
|---|---|---|
| mean net ≤ 0 | **FAILS COST SCREEN** | The JFE effect does not survive MNQ friction at 1 contract. Record; close. No variant search. |
| mean net > 0, t < 2.0 | **POSITIVE, UNPROVEN** | Record. Eligible for a prospective seal. No deployment. |
| mean net > 0, t ≥ 2.0 | **SURVIVES COST SCREEN** | Record. Triggers §7. Still no deployment. |

**Secondary, reported but never decision-bearing:** the paper's own predictive regression
`r_LH = α + β·r_ROD + ε` (β, t-stat, R²); win rate; edge in bps of notional against the
0.51 bps friction benchmark; and the same figures at $2.00 friction.

## 5. What we will NOT do

1. No change to the 30-minute window if it fails. The window is the paper's.
2. No substitution of a different predictor (e.g. first-half-hour) if `r_ROD` fails.
3. No volatility, day-of-week, or regime filter added after seeing results.
4. No dropping of months, or of any qualifying session.
5. No re-running at a lower friction assumption to rescue a negative result. $3.00 is primary.
6. No sizing above 1 contract to make dollar figures look larger.

## 6. Disclosed limitations, stated before the run

- **The sample overlaps the paper's only at its very start.** JFE covers 1974–2020; this
  test covers 2025-01-01 → 2026-06-11. It is therefore **an out-of-sample test in time**
  — which is a strength for validity and a weakness for power (N ≈ 300 sessions).
- No post-2020 replication of the effect exists in the literature (recorded as an open
  question in `research/technical-hf-order-flow-strategies-2026-08-27/research.md`). If
  this test fails, decay since 2020 and MNQ-specific friction are **not separable** here.
- MNQ is one instrument. The paper's claim is cross-market; a single-instrument failure
  does not refute it.

## 7. Successor trigger

A **SURVIVES COST SCREEN** verdict authorises drafting a prospective pre-registration only.
It does not authorise deployment, sizing, or a combine promotion. Note in advance: entry
15:30 / exit 16:00 ET is inside the RTH session and flat by the close, so unlike LRC this
strategy carries **no overnight or weekend exposure** — it is venue-compatible in principle.

## 8. Values fixed at seal time

| Item | Value |
|---|---|
| git HEAD at seal | `b7c838f` |
| Source | JFE 142 (2021) 377–403, doi via Elsevier; local extract retained |
| Expected N | ~311 qualifying sessions (counted from data before the run; no outcome inspected) |

---

# Amendment 1 — Result (2026-08-28)

Append-only. Original sealed text unedited.
Script: `study_intraday_momentum_mnq.py` | Report: `data/reports/intraday_momentum_mnq_20260828_150839.txt`
Ledger: `data/reports/intraday_momentum_mnq_trades.csv`. Run under seal `e865655`.

## A1.1 Verdict — FAILS COST SCREEN (as pre-committed)

N = 303 sessions, 2025-01-03 → 2026-06-11.

| | total | mean/trade | t | win rate | PF |
|---|---|---|---|---|---|
| GROSS (no costs) | −$2,739.00 | **−$9.04** | −1.280 | 48.5% | 0.817 |
| **NET @ $3.00 RT (primary)** | −$3,648.00 | **−$12.04** | −1.704 | 47.5% | 0.764 |
| NET @ $2.00 RT (secondary) | −$3,345.00 | −$11.04 | −1.563 | 47.9% | 0.782 |

Mean net at primary friction is ≤ 0, so per §4 the verdict is **FAILS COST SCREEN**.
Recorded; thread closed; no variant search (§5).

## A1.2 But this is NOT a cost failure — and my own seal framed it wrongly

**Gross is already negative (−$9.04/trade).** Friction never got the chance to matter.
The seal was written as a *cost screen* on the assumption the effect would be present and
the question would be whether $3.00 ate it. That assumption was wrong: on MNQ over this
window there is no gross effect for costs to consume.

The paper's own predictive regression confirms it:

```
r_LH = a + b*r_ROD :   beta = +0.00069   t(beta) = +0.063   R^2 = 0.00001
```

The **sign is as JFE predicts (β > 0)** but the magnitude is indistinguishable from zero.
R² of 0.00001 is not a weak relationship; it is no relationship.

Note also that gross t = −1.280 — **the negative result is itself not significant.**
This is a null, not a reversal. Nothing here supports "intraday momentum is inverted on MNQ".

## A1.3 THE QUALIFIER THAT GOVERNS THIS RESULT: the test was underpowered by construction

At N = 303 with per-trade sd = $122.97, the smallest mean detectable at t = 2.0 is
**$14.13/trade**.

The JFE effect, translated to per-trade terms at this instrument's volatility:

| JFE annualised Sharpe | per-trade Sharpe | implied mean/trade | detectable here? |
|---|---|---|---|
| 0.87 (bottom of range) | 0.0548 | $6.74 | **NO** |
| 1.30 (mid) | 0.0819 | $10.07 | **NO** |
| 1.73 (top of range) | 0.1090 | $13.40 | **NO** |

**The entire claimed Sharpe range sits below this test's detection threshold.** Even if the
JFE effect were fully present and undecayed on MNQ, this test would most likely have
returned a null.

Sessions required at t = 2.0: **596 (~2.4 years)** for the mid of the range; 1,332
(~5.3 years) for the bottom; 337 (~1.3 years) for the top.

**Therefore this result does NOT refute Baltussen et al.** It says only: at N = 303 on MNQ,
no economically usable edge is visible and the point estimate is negative. Anyone citing
this amendment as "the JFE intraday-momentum effect was tested and failed" would be
misrepresenting it.

## A1.4 What can and cannot be separated

§6 disclosed that a failure could not separate decay-since-2020 from MNQ friction. The
gross figure resolves one of those: **friction is not the mechanism of failure.** It does
not separate the remaining three — decay since 2020, MNQ-specific absence, and insufficient
power — and A1.3 makes **insufficient power the leading explanation**, not a footnote.

## A1.5 Cross-cutting methodological finding

This is the **second** underpowered test in two days. GAP-V (2026-08-27) returned
INSUFFICIENT_SAMPLE and announced it, because a per-subgroup floor was pre-committed in its
seal. MIM-X had no such floor: it ran, produced a clean-looking verdict with a t-statistic,
and was never capable of detecting the effect it was testing.

**Rule for future seals in this project: compute the minimum detectable effect size at the
planned N BEFORE sealing, and state it in the seal alongside the decision rule.** A verdict
threshold without a power statement can manufacture a confident null. This seal should have
carried that calculation in §6 and did not.

## A1.6 Status

MIM-X on MNQ is **closed as an underpowered null**, not as a refutation. Nothing deployed,
no live system touched, no parameter changed anywhere.

The honest residual: the strongest externally-validated candidate this research produced
remains untested at adequate power on this instrument, and reaching adequate power needs
~2.4 years of MNQ sessions — or a longer historical window than the two local CSVs provide.
TradeStation serves MNQ minute bars well beyond this window (verified 2026-08-27, ≥4.3
years including expired contracts), so **a properly powered version of this test is
available without new data purchase** and would need its own seal.
