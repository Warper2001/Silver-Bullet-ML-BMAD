# Pre-Registration: PDH-S1 — Prior-Day High Sweep-and-Reject (SHORT-ONLY)

**Registered:** 2026-06-25
**Authored by:** Alex (warper2001@gmail.com)
**Experiment ID:** pdh-sweep-reject-v1
**Status:** SEALED — frozen at commit time. No parameter amendments after this commit.

---

## Required Disclosure

REJECT_BARS=5 was selected after observing that values of 1 and 3 bars gave negative
IS PF while 5 bars gave positive IS PF. All other parameters are first-principles.
This selection is disclosed here and frozen at seal. The OOS is the real test.

---

## Hypothesis

H1: When the prior RTH session High (PDH) is swept (any 1-min bar HIGH > PDH) in the
first 120 minutes of RTH (09:30-11:30 ET) and price closes back below PDH within 5 bars,
this signals a liquidity grab — institutional orders filled above retail stops clustered
at PDH — and price should revert bearishly with PF >= 1.10 at 2:1 R:R.

H0: No such edge exists. PF < 1.10 over N >= 40 IS trades.

Direction commitment: SHORT only. The LONG direction (PDL downside sweep-and-reject)
was found to be net-negative in IS exploration and requires a separate pre-registration
before testing. This pre-registration covers SHORT entries only.

---

## Frozen Parameters

| Parameter | Frozen Value | Prior Rationale |
|---|---|---|
| Direction | SHORT only | IS exploration shows LONG direction PF ~0.85; separate pre-reg required |
| RTH definition | 09:30-16:00 ET | Confirmed as 390 bars/session from data |
| PDH definition | MAX(prior RTH session bar HIGHs) | Most inclusive; catches all intraday extremes |
| Min RTH bars prior day | 300 | Filters half-days and feed outages |
| Gap skip threshold | RTH open > PDH + 200 pts | PDH not a meaningful reference after a large gap |
| Sweep window | 09:30-11:30 ET (first 120 min) | NYC open + ICT 10 AM macro kill zone |
| Sweep trigger | Any bar HIGH > PDH | Standard intrabar sweep |
| Reject window | 5 bars after sweep bar | DISCLOSED: IS-derived. Frozen at seal. |
| Reject signal | Bar CLOSE < PDH | Price returned inside prior range |
| Entry | CLOSE of reject bar (SHORT at market) | Worst-case execution assumption |
| Stop loss | sweep_bar.HIGH + 20.0 pts | Buffer above sweep extreme; $40 per MNQ contract |
| Take profit | Entry - 2.0 x (SL - entry) | 2:1 R:R. PDL-as-target failed in IS (range too wide). |
| Time stop | 60 bars from fill | Same as YANK; prevents holding through afternoon |
| Max trades/day | 1 | PDH is one level per day |
| Commission | $4.00 round-trip | Standard estimate |
| Instrument | MNQ ($2.00/point) | Front-month contract |
| Per-trade risk cap | Skip if (SL - entry) x $2 x contracts > $400 | Combine daily-loss protection |

---

## IS Evidence (2025, parameters above)

Reported here for transparency — these numbers were seen before writing the pre-reg.

| Metric | Value |
|---|---|
| N trades | 65 |
| Win rate | 38.5% (25W / 40L) |
| Profit factor | 1.510 |
| Net P&L (1ct) | +$1,597 |
| Avg win | +96.6 pts ($193) |
| Avg loss | -37.2 pts (-$74) |
| Max consecutive losses | 8 |
| Exit breakdown | 31 SL / 21 TIME / 13 TP |
| Positive months | 7/12 (58%) |

---

## Gate 0 Decision Rule (IS 2025)

All three conditions required for PASS:

| Criterion | Threshold | IS Result |
|---|---|---|
| Profit factor | PF >= 1.10 | 1.510 PASS |
| N trades | N >= 40 | 65 PASS |
| Win rate | WR >= 35% | 38.5% PASS |

Gate 0 verdict: PASS based on IS exploration. Formal backtest must confirm.

---

## OOS Data

File: data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv
Coverage: 2026-01-01 through ~2026-06-11

---

## OOS Pass Criteria

All required:
- PF >= 1.10
- N >= 20 OOS trades
- Positive cumulative net P&L

OOS Early Stop: If N >= 10 OOS trades AND PF < 0.80, stop immediately — OOS FAIL.

---

## Decision Table

| Gate 0 | OOS | Action |
|---|---|---|
| FAIL | - | Archive. Do not access OOS. |
| PASS | PASS | Proceed to combine architecture + deployment planning. |
| PASS | FAIL | Strategy is IS-regime-dependent. Archive. |
| PASS | N < 20 | Inconclusive. Resume at N=20. |

---

## What We Will NOT Do

1. Will NOT test LONG direction under this pre-registration.
2. Will NOT use PDL as take-profit target.
3. Will NOT change REJECT_BARS after OOS access.
4. Will NOT change stop buffer (20 pts) after OOS access.
5. Will NOT split by DOW or time-of-day after seeing OOS results.

---

## Combine Safety Requirements (if OOS passes)

- Start at 1ct per trade
- Per-trade risk cap: skip any setup where (SL - entry) x $2 > $400
- Combined position limit: YANK 2ct + PDH 1ct = 3ct MNQ maximum
- Update combine_floor_monitor.py to track PDH trades from trades.db

---

## Backtest Script

backtest_pdh_sweep_reject.py in repo root.

IS only (Gate 0 check):
    .venv/bin/python backtest_pdh_sweep_reject.py

IS + OOS (after Gate 0 confirms pass):
    .venv/bin/python backtest_pdh_sweep_reject.py --oos --log-access

Output: data/reports/pdh_sweep_IS_<timestamp>.csv and data/reports/pdh_sweep_OOS_<timestamp>.csv
