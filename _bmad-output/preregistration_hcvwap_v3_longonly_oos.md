# Pre-Registration: HCVWAP v3 Long-Only — OOS Validation

**Generated:** 2026-06-09
**Experiment ID:** hcvwap-v3-longonly-oos
**Pre-registration commit:** (populate after `git commit`)
**Supersedes:** hcvwap-v2 (commit fb8d094) in-sample study
**Status:** SEALED — study_hcvwap_v3_longonly.py does not yet exist at time of this document

---

## 1. Integrity Disclosure: What Was Observed Before This Pre-Registration

This pre-registration is filed AFTER observing the HCVWAP v2 in-sample results (2025-01-01 →
2026-02-28). That is unavoidable: the OOS test is motivated by an in-sample finding. Full
disclosure of what was observed:

| Metric | Long-Only (N=60) | Short-Only (N=29) | Combined (N=89) |
|---|---|---|---|
| Win rate | **38.3%** | 17.2% | 31.5% |
| Profit factor | **1.87** | 0.39 | 1.289 |
| Avg P&L/trade | **+$18.73** | -$17.61 | +$6.89 |
| Avg R/R (actual) | 3.57:1 | 3.87:1 | — |

**What was NOT observed before this pre-registration:**
- Any OOS (≥2026-03-01) trade outcome for the long-only version
- The N count in the OOS window
- Any monthly breakdown of long trades in 2026

**Why the long/short split is NOT post-hoc cherry-picking (defense of architecture):**

The long/short asymmetry has a directional economic rationale disclosed in HCVWAP v1 analysis:
MNQ has persistent upward momentum throughout 2025–2026. Fading above VWAP (shorts) opposes
structural momentum. Fading below VWAP (longs) goes with structural momentum returning to
equilibrium. This asymmetry was hypothesized in the web research phase before v2 was run.
The combined result confirming this hypothesis is consistent with the prior, not purely
exploratory. However, this reasoning does not cure the pre-registration timing problem:
the split was not pre-registered before the results were seen, so this OOS test is the
only clean validation.

---

## 2. Hypothesis

### H₁ (alternative)

The HCVWAP v2 long-side architecture (5-min false-breakout rejection below VWAP −σ bands,
with volume confirmation and HTF-ranging filter) has genuine positive expectancy on MNQ,
validated out-of-sample (2026-03-01 → 2026-05-19):

- **EV > $0** per trade net of commission
- **PF ≥ 1.10** (relaxed from in-sample 1.20 gate — OOS with small N)
- **WR ≥ avg_be_wr + 3pp** (3pp relaxed from 5pp in-sample — OOS with small N)
- **N ≥ 10** — if fewer than 10 long signals fire in the OOS window, result is INCONCLUSIVE
  (not FAIL) and we wait for more live data

### H₀ (null)

The HCVWAP v2 long-side in-sample result (WR=38.3%, PF=1.87) is an in-sample artifact. The
edge does not persist in the OOS period. OOS WR and PF degrade to losing territory or N is
too small to distinguish signal from noise.

---

## 3. Signal Definition (Frozen — identical to v2 long direction)

Architecture: **5-min false-breakout rejection candle at VWAP −σ band** (LONG direction only).

### 3a. Session VWAP + σ band

- VWAP: RTH-day cumulative Σ(typical_price·vol)/Σvol; typical = (H+L+C)/3
- σ band: rolling std of (price − VWAP) within session (reset daily)
- Extension gate: (price − VWAP) / σ ≤ **−2.0** (price below VWAP by ≥ 2 σ)

### 3b. False-Breakout Rejection Entry (LONG)

```
bar.low <= VWAP - 2.0σ   AND   bar.close > VWAP - 2.0σ
```

Price pierces the −2σ band but closes back above it within the same 5-min bar.

### 3c. Confirmation Filters

1. Time window: **09:45–11:30 ET OR 14:00–15:00 ET** (identical to v2)
2. Volume: bar volume > **1.5×** trailing 20-bar mean
3. HTF ranging: 15-min `|EMA9 − EMA21| < 0.5 × ATR15` (not in persistent trend)

### 3d. Trade Rules

| Element | Rule |
|---|---|
| Direction | **LONG only** |
| Stop | Fixed **15 pts** from entry (v2 primary spec) |
| Target | **VWAP centerline** (session VWAP at bar close) |
| Min R/R | 1.5× (direction sanity: skip if VWAP ≤ entry at signal time) |
| Hold max | TBD bars (same as v2: holds up to 12 × 5-min = 60 min) |
| Session close | 15:55 ET |
| Commission | $4.80/round-turn (per v2 spec) |
| MNQ point value | $2.00/pt |

---

## 4. OOS Window

- **Data source:** `data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv`
  (confirmed available 2026-01-01 → 2026-05-19)
- **OOS window:** 2026-03-01 → 2026-05-19 (~55 trading days)
- **Expected N:** ~15 long trades (based on v2 in-sample rate of 0.20/day × 55 days;
  could be 10–25 depending on vol regime)
- **Holdout access:** must log in `data/sealed_holdout/ACCESS_LOG.md` before running

---

## 5. Gate Thresholds (OOS — relaxed for small N)

| Criterion | OOS Gate | Rationale |
|---|---|---|
| EV per trade | > $0 | Same |
| Profit factor | ≥ **1.10** | Relaxed from 1.20 (OOS volatility) |
| WR vs breakeven | ≥ avg_be_wr + **3pp** | Relaxed from 5pp (small N penalty) |
| N | ≥ **10** | Below 10 → INCONCLUSIVE, not auto-FAIL |
| Worst-month | ≥ −$100 | Relaxed (only 3 months of data) |

**If N < 10:** Report as INCONCLUSIVE. Do not declare PASS or FAIL. Wait for live S25 MNQ
data to accumulate (S25 runs on MNQ; can observe long-side HCVWAP signals in parallel).

**If PF ≥ 1.10 AND WR ≥ be_wr+3pp AND N ≥ 10:** → PASS → proceed to combine-math path
simulation (3 MNQ contracts, 30-day horizon, P(ruin) < 20% gate).

---

## 6. Sensitivity Grid

**Primary:** SD=2.0, stop=15 pts (identical to v2 primary).

Grid (informational only — no cherry-picking from grid):
- `sd_thresh` ∈ {1.5, 2.0, 2.5}
- `stop_pts` ∈ {12, 15, 18}

Primary result is the gate verdict. Grid cells are presented for robustness context only.

---

## 7. Scope

- If PASS → combine-math path simulation → buy Topstep combine and attempt
- If FAIL or INCONCLUSIVE (N<10) → HCVWAP declared exhausted across all architectures.
  Remaining threads: S25 decision (~2026-07-23), GC CPI prospective (N=10, ~17 months)
- S25 (`tier2_streaming_working.py`, account 23884932) continues unchanged
- Sealed holdout ≥2026-03-01 accessed only through this study with ACCESS_LOG entry

---

## 8. Integrity Seal

| Item | Value |
|---|---|
| study_hcvwap_v3_longonly.py | NOT YET WRITTEN |
| hcvwap_v2_config.yaml | used as parameter source (stop=15, sd=2.0) |
| Git HEAD at pre-registration | (populate after `git commit`) |
