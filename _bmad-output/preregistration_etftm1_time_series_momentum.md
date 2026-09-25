# Pre-Registration: ETFTM-1 — ETF time-series momentum, timing skill (A2)

**Registered:** 2026-09-25. **Status:** SEALED by this commit. It is committed before any statistic has been computed under the true alignment of signal and outcome, and before the Gate 0 runner exists.

**Chain:**

| Step | Record |
|---|---|
| Plan | `/root/.claude/plans/do-d-then-a-imperative-eagle.md` |
| Vehicle decision | `_bmad-output/vehicle_decision_etftm1_20260925.md`: IRA-else-cash, long/flat, $50K, timing-skill claim |
| Data audit | A0 GO (`_bmad-output/etftm1_data_audit_20260925.md`, commit a158b70) |
| Power-gate seal | `_bmad-output/preregistration_etftm1_power_gate.md` (3845bf6) |
| Power verdict | **MARGINALLY_POWERED** (`_bmad-output/etftm1_power_verdict.json`, commit bdc9f07) |
| **Alex's decision (2026-09-25)** | **"Go, Gate 0 first":** run the cheap Gate 0 test once; build the full engine and run the deployment and holdout gates only if Gate 0 passes |

## 1. Hypothesis

- **H1:** the sign of an ETF's trailing 12-month excess return predicts its next-month volatility-scaled excess return **beyond the ETF's own average return** (timing skill), pooled across the 31-ETF universe.
- **H0:** no such timing predictability.
- One-sided test. The direction (trend, not reversal) comes from Moskowitz, Ooi & Pedersen (2012) and cannot be flipped.

## 2. Statistic (identical to the A1 seal §1; code reused, not rewritten)

The timing portfolio is p_t = equal-weight mean over eligible ETFs of x̃_{i,t} · ỹ_{i,t+1}, where:
- x̃ is the sign of the trailing 12-month excess return, demeaned within asset;
- ỹ is the next-month excess return divided by (MOP EWMA volatility with centre of mass 60 days / √12), demeaned within asset;
- the sample is `data/etf_daily/panel_dev.csv` only (before 2021-10-01).

The matrices come from `build_matrices()` in `tools/etftm1_power_gate.py` (commit 09a9953), used unchanged. The statistic is **m̂ = mean(p)** under the true alignment, computed **once**.

## 3. Gate 0 — PASS only if all three hold (otherwise FAIL, final)

| # | Condition | Fixed input |
|---|---|---|
| (i) | **Significance:** z = m̂ / SE ≥ 1.645, i.e. one-sided p < 0.05 | SE = 0.03321302564569293, the conservative (larger) noise SE from the A1 verdict JSON, derived outcome-blind |
| (ii) | **Beats a persistence-preserving random null:** m̂ at or above the **95th percentile** of 1,000 null draws. In each draw, every asset's x̃ series is circularly shifted by its own random offset in [13, T−13] months, which keeps each asset's signal persistence but breaks timing and the cross-asset alignment. Seed 20260926. A result between the 50th and 95th percentiles is **ambiguous and counts as FAIL** (AGENTS.md policy). | — |
| (iii) | **Not carried by one asset class:** m̂ recomputed with each of the 8 asset classes left out must be > 0 in **at least 7 of 8** cases (k − 1 of k) | — |

## 4. What each outcome means

- **PASS:** the development data shows timing skill of about IR 0.55 or more. That is the smallest effect A1 could detect with 80% power, so a pass implies an effect near or above it. The full engine may then be built (A3), followed by the deployment gate on the development window and one holdout run (Gate 1), under the rules in §5.
- **FAIL:** no timing edge of IR ≳ 0.55 is detectable in 27.7 years. That is **not** proof of no edge. A1 gave 0.55 power at the central anchor (0.39), so a FAIL doesn't rule out a 0.3–0.4 edge. It is recorded and closed, with the conditions for revisiting: a broader and less correlated universe (effective breadth here is about 4), or a longer history. There is no re-sweep of lookback, universe or volatility window.

## 5. Later gates: rules fixed now, inputs filled before they run

Only the cost *inputs* (measured half-spreads per ETF, the clearing-fee scope once TradeStation answers TS-2) go into an amendment. It is committed **before** the engine produces any aligned P&L. No threshold below may change.

- **Portfolio rule:**
  - Monthly rebalance. The signal is taken at the month-end close; orders fill at the **next trading day's close**.
  - Long when the signal is +1, flat otherwise.
  - Weights: inverse EWMA volatility, scaled to the ex-ante portfolio volatility of an always-long inverse-volatility book of the same universe.
  - Gross exposure ≤ 1.0. No single ETF above 1/(effective breadth rounded) of equity; the effective-breadth value is from the A1 JSON.
  - Uninvested cash earns the T-bill rate (proxy for holding a T-bill ETF).
  - Whole shares at **$50K**, with $25K as a sensitivity.
- **Deployment gate (development window):** the trend book's net Sharpe must be ≥ the always-long, inverse-volatility book's net Sharpe **and** its maximum drawdown must be smaller. Otherwise it's FAIL for deployment, even if Gate 0 passed.
- **Gate 1 (holdout, used once, after an ACCESS_LOG entry):** m̂ on the holdout must be > 0 **and** the trend book's net Sharpe > 0. The holdout-only MDE is IR 1.29 (A1), so Gate 1 is **labelled a direction check, not a confirmatory test**.
- **Baselines reported beside it (informational):**
  - always-long equal-weight;
  - hindsight mean-sign (look-ahead; diagnostic only);
  - trend without volatility scaling;
  - 60/40 SPY/IEF;
  - a managed-futures ETF (DBMF) over its live overlap.

## 6. Reported with Gate 0 (descriptive, never gating)

- m̂ as an information ratio (m̂ / sd_p · √12), where sd_p is the p-series standard deviation.
- m̂ by asset class; before and after 2012 (MOP's publication); excluding 2008.
- The null draws' distribution and the percentile of m̂.

## 7. What this seal does not do

- It never reads `data/sealed_holdout/` for Gate 0.
- One run only. If the runner crashes before writing a verdict, it may be fixed and re-run, because the statistic is deterministic; the fix is disclosed.
- No additional ETFs, lookbacks, volatility windows or filters.
- No live or paper execution; that is A5, after TradeStation answers TS-1 to TS-8.
