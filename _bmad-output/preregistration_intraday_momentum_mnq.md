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
