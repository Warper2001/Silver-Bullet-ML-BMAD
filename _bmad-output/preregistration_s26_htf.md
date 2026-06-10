# Pre-Registration: S26 HTF — Soft-FVG + Sweep on 15-Min / 1-Hour BTC Bars

**Generated:** 2026-06-10
**Experiment ID:** s26-htf-cost-viability
**Pre-registration base commit:** 385b8f8 (seal commit SHA recorded by the commit itself)
**Supersedes:** nothing — S26 1-min remains live on MBTM26 pending separate decision
**Status:** SEALED — no HTF resampled backtest has been run at the time of this document.
No study script for this experiment exists yet.

**Data file:** `data/kraken/PF_XBTUSD_1min.csv`
md5 `cd4662d609ab783df8f56d6d5b18b973`, range 2024-11-08 → 2026-05-10 (UTC)

---

## 1. Integrity Disclosure: What Was Observed Before This Pre-Registration

This experiment is motivated by a negative finding on the 1-minute version of the same
pattern. Full disclosure of what has been observed (2026-06-09/10 session):

| Observation (1-min S26, 60-day rolling WF, Jan 2025 → May 2026) | Value |
|---|---|
| Trades / WR / gross PF | 1,902 / 38.7% / **1.128** |
| Avg gross edge per trade | +10.9 pts = **+$1.09 / MBT contract** (~1.1 bps of notional) |
| Net PF at $6.00 RT/contract | **0.599** |
| Topstep 50K combine MC pass% (with costs, 1–10 ct) | **0.0–1.0%** |
| Exit autopsy | Loss mass 100% in SL exits; TIME_STOP exits net positive (74% WR) |

**Diagnosis being tested:** the 1-min pattern's gross edge is real but its per-trade
expectancy (~1 bp) is structurally smaller than fixed per-contract costs. Moving the same
pattern to higher-timeframe bars multiplies per-trade range (15-min/1-hour ATR ≫ 1-min ATR)
while costs stay fixed, so IF the pattern exists at HTF, the cost fraction collapses.

**What has NOT been observed:**
- No 15-min or 1-hour resample of this data has been backtested with this (or any S26) signal.
- No signal counts, win rates, PFs, or equity curves at any timeframe other than 1-min.
- No parameter at HTF has been tuned against outcomes — all translations below are fixed
  by mechanical rule before any HTF result exists.

**Known risk acknowledged up front:** prior MNQ work (S12/S13) found patterns can survive a
1m→15m move; prior project history also includes 15+ failed strategies. Three variants are
tested below (multiple-comparison risk); the design therefore requires OOS confirmation for
any variant that passes the dev gate, and a global stop rule if all fail.

---

## 2. Hypothesis

### H₁ (alternative)

The S26 soft-FVG + liquidity-sweep pattern, translated mechanically to 15-min (primary)
and/or 1-hour (secondary) BTC bars, has positive expectancy **net of realistic costs**
($6.00 round-trip per MBT contract), and supports a Topstep 50K combine pass probability
meaningfully above coin-flip.

### H₀ (null)

The pattern's edge exists only at 1-min granularity (or not at all); at HTF the signal
either fails to fire in sufficient quantity or its net PF ≤ 1.0. In that case the S26
pattern is declared **dead at all timeframes for the combine** and the book is closed —
no further S26 timeframe/parameter iterations without a new pre-registered economic
rationale.

---

## 3. Signal Definition (Frozen — mechanical translation of the live 1-min spec)

Bars: 1-min Kraken PF_XBTUSD resampled to 15-min ("15m") and 60-min ("1h") OHLCV
(left-labeled, left-closed; volume summed).

Translation rule, fixed before any result: **structure lookbacks preserve clock duration;
trade management preserves bar counts and ATR multiples.**

| Element | 1-min (live) | 15m | 1h |
|---|---|---|---|
| ATR length | 20 bars | 20 bars | 20 bars |
| Sweep extreme window | rolling 360 min | 24 bars (6 h) | 6 bars (6 h) |
| "Recent sweep" window | 60 min | 4 bars | 1 bar |
| Soft FVG | (low.shift(2) − high) > 0.2×ATR (bear); mirror for bull | unchanged | unchanged |
| Signal debounce | no consecutive signal bars | unchanged | unchanged |
| Entry | next bar open after signal bar | unchanged | unchanged |
| Stop loss | 2.0 × ATR | unchanged | unchanged |
| Take profit | 4.0 × ATR | unchanged | unchanged |
| Time stop | 60 bars from entry, exit at close | 60 bars (15 h) | 60 bars (60 h) |
| Position | one trade at a time, both directions | unchanged | unchanged |

The 60-bar time stop is preserved in **bar units** (not clock units) — disclosed choice:
it keeps every trade-management rule identical to the live spec in bar space; the
observed 1-min exit autopsy (time-stop exits net positive) gives no reason to shorten it.

## 4. Variants (all pre-registered; exactly three)

| Variant | Timeframe | ML filter |
|---|---|---|
| **V1 (primary)** | 15m | none — raw signal |
| **V2** | 15m | walk-forward ML (below) |
| **V3 (secondary)** | 1h | none — raw signal |

V2 ML spec (frozen): `HistGradientBoostingClassifier(max_iter=150, random_state=42)`,
features `['dir','atr','rvol','dist_ema','dist_macro_ema','hour_et','dow','is_us_session']`
computed on 15m bars (RVOL = volume / 50-bar SMA; macro EMA span 200; EMA span 20),
threshold **0.62**, monthly retrain on a rolling **180-day** window (scaled from 60 days at
1-min to keep training-set size ≥ several hundred trades; fixed here, not tuned).
No ML variant at 1h — training density is insufficient by construction (disclosed).

## 5. Data Windows and Test Sequence

| Window | Range | Role |
|---|---|---|
| Dev (Gate 0) | 2025-01-01 → 2025-12-31 UTC | first and only iteration window |
| OOS holdout | 2026-01-01 → 2026-05-10 UTC | touched once, only for variants passing Gate 0 |

Indicators warm up from full data history (resample first, then compute), so January
windows are not structurally handicapped.

## 6. Cost Model (Frozen)

- MBT (Micro Bitcoin futures): $0.10 per BTC index point per contract.
- Round-trip cost: **$6.00 per contract** = commission + exchange fees + 1-tick
  ($25-point) slippage per side. Applied as a 60-point deduction per trade.
- All PF/expectancy gates below are **net of this cost**, per contract.

## 7. Decision Gates (Frozen)

### Gate 0 — Dev window (2025), per variant

| Criterion | V1 / V2 (15m) | V3 (1h) |
|---|---|---|
| Minimum N | ≥ 100 | ≥ 30 |
| Net PF | ≥ 1.10 | ≥ 1.10 |
| Net expectancy | > $0 / contract / trade | > $0 |
| Cost fraction | $6 ≤ 25% of avg gross |win| per contract | same |

N below minimum → **INCONCLUSIVE** (not FAIL) for that variant; PF/expectancy miss with
sufficient N → **FAIL**.

### Gate 1 — OOS (2026 YTD), only for Gate-0 passers, run exactly once

- N ≥ 20 (else INCONCLUSIVE — park, await more data; no re-runs with modified specs)
- Net PF ≥ 1.05
- Net expectancy > $0

### Gate 2 — Combine viability, only for Gate-1 passers

Same Monte Carlo harness as the 1-min test (5,000 sims, ET-day block bootstrap with
replacement, 90-day cap, per-trade trailing $2,000 MLL — conservative — and $1,000 DLL,
target +$3,000 from $50,000), run on pooled dev+OOS trades net of costs:

- Pass% ≥ **50%** at some integer size 1–10 contracts, AND
- at that size Pass% > Blow%.

### Global stop rule

If **all three variants** fail (or are INCONCLUSIVE with N clearly saturated, i.e., the
signal genuinely fires less than minimum N per year), then: S26 pattern **dead at all
timeframes for combine purposes**. No V4. Any future S26 work requires a new
pre-registration with a new economic rationale, not a parameter variation.

## 8. Multiple Comparisons

Three variants are tested on the dev window. The protections are: (a) gates were set
before any HTF number existed, (b) any dev pass must survive an untouched OOS window,
(c) the global stop rule forbids variant proliferation. No per-variant alpha adjustment
is applied; the OOS gate is the de-facto correction.

## 9. What This Experiment Does NOT Change

- The live S26 1-min bot on MBTM26 and `models/s26_soft_fvg_ml_model.pkl` are untouched.
- No `strategy_config*.yaml` changes.
- A Gate-2 pass does NOT authorize deployment by itself — deployment would require its own
  pre-registration per the Epic 8 workflow.
