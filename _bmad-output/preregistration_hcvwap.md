# Pre-Registration: HCVWAP — VWAP 2σ Fade × 4-Condition Confirmation Stack

**Generated:** 2026-06-09
**Experiment ID:** hcvwap-v1
**Pre-registration commit:** (populate after `git commit`)
**Status:** SEALED — no backtest data has been read at time of this document

---

## 1. Motivation and Hypothesis

### What the prior 10 failures taught us

Ten strategy families were tested on MNQ/NQ 1-min and 5-min RTH bars; all failed Gate 0:

| Root cause | Families killed |
|---|---|
| MNQ momentum continuation (wrong direction) | VWAP ×2, POC fade, lunch oscillation |
| Stop geometry incompatible with combine trailing DD | ORB ×4 |
| Monthly variance too high at combine limits | PBC ×2 |
| Frequency too low (<1.0/day) | Vol Compression 1m, Vol Compression 15m |
| Regime instability OOS | ES/MNQ Stat Arb |
| Gap-open reversal dominance | PDH/PDL Breakout |
| No directional edge (constant PF across TP_MULT) | 5-min EMA Trend-Pullback |

**The critical diagnostic** (BMAD Round 3, 2026-06-09): every test so far evaluated a **setup
CLASS** in isolation. Systematic VWAP fade studies (study_vwap_reversion_rate.py) took
**every** 2σ signal → WR 14–37%, clear edge absent. Yet external research (2026 Topstep data)
reports funded traders citing "VWAP mean reversion at 2+ standard deviations" as their primary
edge, with ~55% WR.

**Resolution:** the edge is not in the *trigger* (2σ extension) — it is in the *confirmation
stack* that discretionary winners apply before pulling the trigger. Taking every 2σ extension
produces the 14–37% WR we measured. Taking *only* extensions with all four confirmations
satisfied simultaneously is the untested hypothesis.

### Hypothesis H₁ (alternative)

A VWAP 2σ fade filtered by a **pre-registered 4-condition confirmation stack** (extension +
time window + volume spike + HTF ranging) produces a measurably positive edge on MNQ and MES
1-minute RTH bars in the 2025-01-01 → 2026-02-28 in-sample period:

- **EV > $0** per trade net of commission
- **PF ≥ 1.20**
- **WR ≥ breakeven+5%** (breakeven ≈ 34% for 2:1 R/R)
- **Median stop ≤ $150/contract**
- **N ≥ 30** on the filtered population (sufficient for edge read)
- **Worst-month avg P&L ≥ −$50/trade**

If edge gates pass, the secondary question is path-shape: can this (deliberately selective,
low-frequency) strategy clear the Topstep $50K combine in 30 trading days without hitting the
$2,000 trailing HWM drawdown?

### Hypothesis H₀ (null)

The confirmation stack does not rescue the VWAP fade. The filtered population either (a) has
too few signals (N<30) to evaluate, or (b) has N≥30 but PF < 1.20 / EV ≤ $0 — confirming
that the 14–37% WR problem is structural to MNQ's momentum regime at any selectivity level.

---

## 2. The 4-Condition Confirmation Stack (ALL required simultaneously)

### Condition 1 — VWAP Extension ≥ 2.0 SD

- **Metric:** `(close − session_vwap) / σ_vwap` ≥ `SD_THRESH`
- **VWAP:** cumulative session VWAP reset at 09:30 ET each day; typical price = (H+L+C)/3;
  weighted by bar volume.
- **σ_vwap:** rolling intra-session standard deviation of (close − VWAP), reset daily;
  minimum 5 bars before first signal allowed.
- **Direction:** short at `+2.0σ` (price too far above VWAP), long at `−2.0σ`.
- **Primary SD threshold:** 2.0 · **Sensitivity grid:** {1.5, 2.0, 2.5}

### Condition 2 — Time Window

Trade only within these ET time bands (all others filtered out):
- **AM session:** 09:45 – 11:30 ET (CT: 08:45 – 10:30)
- **PM session:** 14:00 – 15:00 ET (CT: 13:00 – 14:00)

Excluded: 09:30–09:45 (opening range noise, gap fills, high directionality);
11:30–14:00 (lunch chop / low volume / institutional inactivity).

**Rationale (pre-commit):** funded trader research and professional discretionary accounts
consistently flag these as the two highest-quality mean-reversion windows in RTH. The 09:30
opening is where gap-open continuation dominates (killed PDH/PDL breakout). Lunch is where POC
fade was 14% WR. The PM session (into close) is a second strong mean-reversion window.

### Condition 3 — Volume Spike

Current 1-min bar volume > `VOL_MULT × rolling_20_bar_mean(volume)`.

- **Primary VOL_MULT:** 1.5 (volume at least 50% above recent average)
- **Intent:** signal must fire on genuinely elevated participation, not a slow drift to 2σ.
  Slow drifts = continuation. Spikes to 2σ on volume = exhaustion / overextension.

### Condition 4 — HTF Ranging (15-min)

15-min EMA spread: `|EMA(9, 15m) − EMA(21, 15m)| < HTF_EMA_ATR_MULT × ATR(14, 15m)`

- **Primary HTF_EMA_ATR_MULT:** 0.5
- **Intent:** only fade VWAP when the 15-minute timeframe is NOT in a sustained trend.
  When the 15-min is trending (large EMA spread), the 1-min extension is likely a
  continuation move, not exhaustion. When the 15-min is ranging (small EMA spread),
  mean reversion to VWAP is structurally justified.
- **Implementation:** resample 1-min bars to 15-min OHLCV; compute EMAs and ATR(14) on
  the 15-min series; as-of join back to the 1-min index (using last available 15-min bar
  at the time of each 1-min bar).

---

## 3. Trade Rules (Frozen)

| Element | Rule |
|---|---|
| Trigger | 1-min bar close satisfies all 4 conditions simultaneously |
| Direction | Short at +2σ; Long at −2σ |
| Entry | Close of the signal bar (same bar) |
| Stop (fixed) | 6 points from entry; `stop_p = entry − dir × 6` |
| Target (fixed) | 12 points from entry; `tp_p = entry + dir × 12` (2:1 R/R; grid varies TP) |
| Stop cap | $150/contract (6 pts × $2/pt = $12 MNQ, way inside cap; included for consistency) |
| Hold max | 60 bars; force-close at market if neither TP nor stop hit |
| Session close | 15:55 ET: force-close all open positions |
| One trade at a time | No new entry while a trade is active |
| RTH only | 09:30–15:55 ET bars used |

---

## 4. Gate 0 Thresholds (Pre-committed, Immutable After Seal)

### Primary verdict criteria (binding go/no-go)

| Criterion | Gate | Action if below |
|---|---|---|
| EV (avg net P&L/trade) | > $0 | STOP — edge absent |
| Profit factor | ≥ 1.20 | STOP — edge absent |
| WR vs breakeven | ≥ breakeven + 5pp | STOP — edge absent |
| Median stop/contract | ≤ $150 | STOP — exceeds combine hard limit |
| N (filtered population) | ≥ 30 | STOP — too rare to evaluate (like vol-compression) |
| Worst-month avg P&L | ≥ −$50/trade | WARNING → Track 3 stacking |

### Frequency gate: RELAXED by design

The standard `freq ≥ 1.0/day` gate is **intentionally not applied** to this study. The HCVWAP
confirmation stack is designed to be selective. Expected frequency: 0.2–0.4 signals/day.
Applying freq≥1.0 would be identical to the error that killed vol-compression (real edge, too
rare). Frequency is reported as an informational diagnostic.

**Instead, frequency informs the path simulation:** if edge gates pass, the combine-math
simulation (Step 4 in the study plan) determines whether a 0.3/day strategy can realistically
reach $3,000 before hitting −$2,000 trailing DD in 30 days — the actual binary question.

### WR breakeven calculation (dynamic, from primary spec)

At stop=6 pts (fixed), tp=12 pts (primary), commission=$4.80:
```
be_wr = (stop_usd + commission) / ((tp_pts/stop_pts + 1) × stop_usd)
       = (12 + 4.80) / (3 × 12)
       = 16.80 / 36 ≈ 46.7%   [MNQ, 1 contract: stop_usd = 6 × $2 = $12]
       MES, 1 contract: stop_usd = 6 × $5 = $30
       be_wr = (30 + 4.80) / (3 × 30) = 34.80 / 90 ≈ 38.7%
```
Gate WR ≥ breakeven + 5pp for each instrument independently.

---

## 5. Path-Shape Simulation (if edge gates PASS)

Per Dr. Quinn's reframe and Mary's combine-math evidence path, the binding second question is:
*Can a selective low-frequency strategy pass the combine's path-shape problem?*

Simulation parameters:
- **Sizing:** 3 contracts (conservative combine sizing; 1-contract Gate 0 P&L × 3)
- **Combine target:** $3,000
- **Trailing DD limit:** −$2,000 (HWM-based)
- **Qualifying day gate:** ≥$150 daily P&L
- **Daily consistency cap:** no single day > 50% of running total
- **Horizon:** 30 trading days (avg combine sprint)
- **Method:** bootstrap daily P&L from in-sample distribution; 10,000 Monte Carlo paths

**Advance only if:** P(ruin, 30 days) < 20% AND E[30-day P&L] > $0.

---

## 6. Data Integrity

- **In-sample period:** 2025-01-01 → 2026-02-28
- **Sealed holdout:** ≥ 2026-03-01 — DO NOT ACCESS until Gate 2 (requires separate pre-reg)
- MNQ data: `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv` +
  `mnq_1min_2026_ytd.csv`
- MES price data (= ES bars): `data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv`
  (MES point value = $5/pt, ES point value = $50/pt — only price path is shared)

---

## 7. Data-Observation Disclosure

**This is a CLEAN pre-registration.** No backtest has been run on any form of HCVWAP with
these 4 conditions prior to this commit. The prior VWAP study (`study_vwap_reversion_rate.py`)
tested an *unfiltered* 2σ fade (no time window, no volume filter, no HTF filter) and returned
WR 14–37%. That result is the MOTIVATION for adding the confirmation stack — not a prior
observation of the filtered population. The unfiltered result is disclosed here.

The 4 conditions and the primary spec were derived from:
1. External research (Topstep funded trader surveys, prop-trading community best practices)
2. BMAD party-mode brainstorm (Dr. Quinn, Victor, Carson, Mary — Round 3, 2026-06-09)
3. Structural reasoning about which bars should have mean-reversion vs continuation structure

**No parameter was tuned on in-sample data for this study.**

---

## 8. Integrity Seal

| Item | Value |
|---|---|
| hcvwap_config.yaml | (SHA-256 computed at commit time by git) |
| Git HEAD at pre-registration | (populate after `git commit`) |
| Study code (study_hcvwap.py) | NOT YET WRITTEN at time of this pre-registration |

Note: `study_hcvwap.py` is written AFTER this pre-registration commit — this is the strongest
possible pre-registration: the hypothesis and parameters are sealed before a single line of
simulation code exists. No post-hoc parameter tuning is possible.

---

## 9. Scope Constraint

- Gate 0 verdict: PASS/FAIL on in-sample edge
- If PASS: path simulation on in-sample daily P&L stream
- If PASS path-sim: pre-register OOS holdout access (Gate 2) in a NEW pre-registration commit
- Do NOT access sealed holdout until Gate 2 pre-registration exists
- S25 (`tier2_streaming_working.py`) continues running unchanged on account 23884932

---

## 10. Combine Strategy Context

This is the **eleventh** strategy family tested. All prior ten have failed Gate 0 (or passed
Gate 0/1 but failed Gate 2 OOS). The confirmation-stack hypothesis is structurally different
from all prior attempts in that it specifically addresses the "why does it work for some people
but not backtests" question by requiring multiple concurrent conditions rather than a simple
signal trigger.

See `memory/project_combine_strategy_dead_end.md` for the full failure record.
