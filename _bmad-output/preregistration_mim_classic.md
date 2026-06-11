# Pre-Registration: MIM-Classic — Market Intraday Momentum on MNQ

**Generated:** 2026-06-11
**Experiment ID:** mim-classic-mnq
**Pre-registration base commit:** a0427e7 (seal commit SHA recorded by the commit itself)
**Motivating document:** `_bmad-output/planning-artifacts/research/technical-topstep-50k-combine-strategy-research-2026-06-10.md` (Candidate 1)
**Status:** SEALED — no MIM signal has ever been computed on any data in this project.
No study script for this experiment exists at the time of this document.

**Data files (frozen):**
- Dev: `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv` — md5 `3ba83a32cac3fa1284e09277259887c9`, 2025 full year, UTC end-labeled 1-min bars
- OOS (LOCKED until Gate 0 pass): `data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv` — md5 `4ec175dfe0c18fd372fb3bf003e5fd9e`, 2026-01-01 → 2026-05-19
- Secondary dev diagnostic (disclosed, NON-GATING): `mnq_1min_2023_sepnov.csv`, `mnq_1min_2024_sepnov.csv`

---

## 1. Integrity Disclosure

**What has been observed before sealing:**
- Published results only: Gao/Han/Li/Zhou (JFE 2018) — first half-hour return predicts last half-hour return on SPY 1993–2013, significant after costs; QuantConnect replication; Zarattini et al. 2024 (related variant, SPY, net-positive 2007–2024).
- No MNQ (or any local) MIM statistic, signal count, win rate, or P&L has been computed. The signal family is absent from this project's 17 prior strategy attempts.

**Data-reuse disclosure:** the 2025 MNQ dev file has been heavily used by prior failed strategies (FVG/sweep, VWAP, ORB, PBC families). Those are different signal families; no MIM-related quantity was ever derived from it. The 2026 OOS file region overlaps the project's sealed holdout era; it will be touched exactly once, only if Gate 0 passes.

**Known risks declared up front:** post-publication crowding decay (effect public since 2014 working paper / 2018 JFE); SPY→MNQ transferability; thin last-half-hour edge vs slippage.

## 2. Hypothesis

**H₁:** The intraday momentum effect (first half-hour return, measured from prior session close, predicts last half-hour return) exists on MNQ in 2025–2026 with net positive expectancy after realistic TopstepX costs ($2.24/contract round-trip), and supports a Topstep 50K combine pass probability ≥ 50% at some size ≤ 10 contracts.

**H₀:** The effect is absent, decayed, or below the cost floor on MNQ. If both variants fail Gate 0, MIM-Classic is closed (no parameter iterations) and the research-doc test order proceeds to Candidate 2 (MIM-Noise-Bands, requiring its own pre-registration).

## 3. Signal Definition (Frozen)

All times **America/New_York (ET)**; bars are 1-min, UTC-stamped, **end-labeled** (bar `HH:MM` covers `HH:MM−1 → HH:MM`).

| Quantity | Definition |
|---|---|
| Prior session close `P_prev` | close of bar labeled **16:00 ET** on the most recent prior trading day (fallback: last bar ≤ 16:00 ET that day) |
| First half-hour price `P_10:00` | close of bar labeled **10:00 ET** (day skipped if absent) |
| Predictor `r1` | `P_10:00 − P_prev` (includes overnight gap, per paper) |
| Penultimate half-hour `r12` | close(bar 15:30 ET) − close(bar 15:00 ET) |
| Entry | **open of bar labeled 15:31 ET** (first bar after 15:30), direction = sign of predictor; signal exactly 0 → no trade |
| Exit | **close of bar labeled 16:00 ET** (fallback: last bar ≤ 16:00 ET; day skipped if no bar in 15:31–16:00) |
| Stop | none (paper baseline; 30-min hold bounds risk) |
| Days traded | any day with all required bars (half-days naturally skip) |

### Variants (exactly two)

- **V1 (primary):** direction = sign(r1). Trades ~every full trading day.
- **V2 (agreement filter, documented in the same paper):** trade only when sign(r1) == sign(r12), direction = that sign.

No other variants, filters, or parameters. Volatility/volume conditioning is a post-pass extension requiring a new pre-registration.

## 4. Cost Model (Frozen)

- MNQ: $2.00/point/contract; TopstepX commission+fees $1.24 RT + 1-tick ($0.25 = $0.50) slippage per side → **$2.24/contract RT = 1.12 points deducted per trade**.

## 5. Decision Gates (Frozen)

### Gate 0 — Dev 2025 (this run)

| Criterion | V1 | V2 |
|---|---|---|
| Minimum N | ≥ 200 | ≥ 100 |
| Net PF | ≥ 1.10 | ≥ 1.10 |
| Net expectancy | > $0/contract/trade | > $0 |
| Cost fraction | $2.24 ≤ 25% of avg gross \|win\| | same |

N below minimum → INCONCLUSIVE; metric miss with sufficient N → FAIL.
The 2023/2024 Sep–Nov diagnostic is reported for context only and cannot pass or fail a variant.

### Gate 1 — OOS 2026 YTD (one shot, Gate-0 passers only)

- V1: N ≥ 80; V2: N ≥ 40; net PF ≥ 1.05; net expectancy > $0.

### Gate 2 — Combine Monte Carlo (Gate-1 passers, pooled dev+OOS, corrected rules)

5,000 sims, ET-day block bootstrap: **EOD-ratchet trailing MLL $2,000** (locks at $50,000), DLL $1,000 as day-deactivation, **consistency rule** (best day < 50% of total profit at pass), costs per §4:
- Pass% ≥ **50%** at some integer size 1–10 contracts AND Pass% > Blow% at that size.

### Stop rule

Both variants fail Gate 0 → MIM-Classic dead on MNQ; no V3, no parameter sweeps; proceed to research-doc Candidate 2 under a new seal.

## 6. What This Experiment Does Not Touch

Live S25 MNQ bot, live S26 crypto bot, all models, all YAML configs, the sealed holdout file. A full Gate-2 pass authorizes a deployment pre-registration, not deployment itself.
