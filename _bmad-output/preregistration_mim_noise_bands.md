# Pre-Registration: MIM-Noise-Bands — Adaptive Noise-Boundary Intraday Momentum on MNQ

**Generated:** 2026-06-11
**Experiment ID:** mim-noise-bands-mnq
**Pre-registration base commit:** 6a50670 (seal commit SHA recorded by the commit itself)
**Motivating document:** research doc a0427e7 (Candidate 2); MIM-Classic failed Gate 0 (commit 6a50670) → this is the pre-committed next test
**Published source:** Zarattini, Aziz & Barbon, "Beat the Market" (SSRN 4824172 / SFI 24-97): SPY 2007–2024, +19.6% ann. net, Sharpe 1.33
**Status:** SEALED — no noise-band statistic has ever been computed on local data. No study script exists at sealing time.

**Data files (frozen, identical to MIM-Classic seal):**
- Dev: `mnq_1min_2025.csv` md5 `3ba83a32cac3fa1284e09277259887c9`
- OOS (LOCKED): `mnq_1min_2026_ytd.csv` md5 `4ec175dfe0c18fd372fb3bf003e5fd9e`
- Secondary diagnostic (non-gating): `mnq_1min_2023_sepnov.csv`, `mnq_1min_2024_sepnov.csv`

---

## 1. Integrity Disclosure

**Observed before sealing:** published SPY results (above); CXO Advisory's rule summary; the MIM-Classic Gate 0 FAIL (gross PF 0.83 on the same dev data — first/last half-hour effect decayed on MNQ). NOT observed: any band, signal count, or P&L from this spec on any local data.

**ORB-family disclosure:** this is the one allowed ORB-family revisit per the research doc. Differences from the 4 dead ORB variants that justify it: volatility-seasonality-scaled adaptive bands (not fixed opening range), entries checked all day at half-hour marks (not an opening window), trailing exits with reversals (not fixed brackets), and independent peer-grade evidence net of costs through 2024.

**Risk declared:** MIM-Classic's decay on MNQ lowers the prior for this cousin; SPY→MNQ transfer unproven; multiple intraday round-trips multiply cost drag.

## 2. Hypothesis

**H₁:** Noise-boundary breakouts with trailing-stop management produce net positive expectancy after $2.24/contract RT on MNQ 2025–2026 and a combine MC pass% ≥ 50% at ≤10 contracts.

**H₀:** The effect is absent/decayed/below cost on MNQ. If both stop-variants fail Gate 0, MIM-Noise-Bands is closed (no parameter iterations); test order proceeds to Candidate 3 (HTF-MR) under a new seal, after which the global stop applies.

## 3. Signal Definition (Frozen)

All times ET; 1-min bars end-labeled; RTH = bars labeled 09:31–16:00. Session open O = open of the 09:31 bar; prior close C_prev = close of prior session's 16:00 bar (fallback: last bar ≤ 16:00).

| Quantity | Definition |
|---|---|
| Move from open | m(d, t) = \|close(d, t) / O_d − 1\| for each minute label t |
| Noise σ(t) | mean of m(d−i, t) over the prior **14** trading days having a bar at t (warm-up: first 14 trading days of each file untraded) |
| Upper bound | UB(t) = O·(1 + σ(t)) + max(C_prev − O, 0)  (gap-down adjustment up) |
| Lower bound | LB(t) = O·(1 − σ(t)) − max(O − C_prev, 0)  (gap-up adjustment down) |
| VWAP(t) | cumulative Σ(close·volume)/Σ(volume) over RTH bars 09:31 → t |
| Check times | bar closes at **HH:00 and HH:30**, from 10:00 to 15:30 for entries/reversals; 10:00 to 16:00 for stops |
| Entry | at a check: close > UB(t) → target LONG; close < LB(t) → target SHORT; else keep current position. Fill at next 1-min bar open |
| Reversal | allowed: opposite boundary breach at a check flips the position (one RT cost each leg) |
| EOD exit | close of the 16:00 bar (always flat) |

### Variants (exactly two — the two stop rules documented in the paper)

- **V1 (primary, tight stop):** long stop level = max(UB(t), VWAP(t)); short stop = min(LB(t), VWAP(t)). At each stop-check, if the bar close is beyond the stop (long: close < level; short: close > level), exit at next bar open.
- **V2 (wide stop):** long stop = LB(t); short stop = UB(t). Same execution.

No other parameters. The 14-day lookback, check times, and gap adjustments are from the published spec and are not tunable.

## 4. Cost Model (Frozen)

$2.24/contract per round trip = **1.12 MNQ points deducted per completed trade leg-pair** (each entry→exit, including reversal legs and stop-outs). 1 contract throughout Gate 0/1.

## 5. Decision Gates (Frozen)

### Gate 0 — Dev 2025 (this run), per variant

- N (completed trades) ≥ **100**
- Net PF ≥ **1.10**
- Net expectancy > $0/contract/trade
- Cost ≤ 25% of avg gross |win|

N below minimum → INCONCLUSIVE; metric miss with sufficient N → FAIL.
2023/2024 Sep–Nov reported as non-gating diagnostics.

### Gate 1 — OOS 2026 YTD (one shot, Gate-0 passers only)

N ≥ 40; net PF ≥ 1.05; net expectancy > $0.

### Gate 2 — Combine MC (pooled, corrected rules)

As per MIM-Classic seal §5: EOD-ratchet MLL $2,000 (locks at $50k), DLL $1,000 day-deactivation, consistency rule, $2.24/ct; pass% ≥ 50% at some size 1–10 and pass > blow.

### Stop rule

Both variants fail Gate 0 → MIM-Noise-Bands closed. Next and final: Candidate 3 (HTF-MR) under a new seal; then the research-doc global stop.

## 6. Untouched

Live bots, models, YAML configs, sealed holdout. Gate-2 pass authorizes a deployment pre-registration only.
