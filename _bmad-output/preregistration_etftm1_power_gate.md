# Pre-Registration: ETFTM-1 power gate (A1)

**Registered:** 2026-09-25. **Status:** SEALED by this commit. It is committed **before**
`tools/etftm1_power_gate.py` exists, following the `tools/xsmom1_power_gate.py` precedent (seal 1ff3735).
**Plan:** `/root/.claude/plans/do-d-then-a-imperative-eagle.md` §A1.
**Decision memo:** `_bmad-output/vehicle_decision_etftm1_20260925.md`. The claim is **timing skill**; the design is long/flat at $50K.
**Data:** `data/etf_daily/panel_dev.csv` and `rf_dev.csv`, whose SHA-256 hashes are in `_bmad-output/etftm1_manifest_20260925.json` (commit a158b70). The panel covers development dates only, before 2021-10-01. `data/sealed_holdout/` is not read.

## 1. The quantity whose detectability is being measured

For each ETF *i* in the frozen 32-ETF universe (`tools/etftm1_universe.py`, commit ad0e796), at each month-end *t* (the last trading day of the calendar month):

- **Daily excess return** e_{i,d} = `tr_ret` − `rf_daily`. The risk-free rate is ^IRX / 100 / 252, forward-filled over the ETF's trading days.
- **Ex-ante volatility σ_{i,t}:** an annualized EWMA of squared demeaned daily excess returns, with centre of mass 60 days (δ = 60/61) and scaling factor 261. This is exactly Moskowitz, Ooi & Pedersen (2012) §2.4, retrieved 2026-09-25 from elmwealth.com/wp-content/uploads/2017/06/timeseriesmomentum.pdf. It uses only data up to *t*.
- **Signal x_{i,t}** = sign of the compounded excess return over the trailing 12 calendar months ending at *t*. An observation is eligible only if the ETF has at least 252 trading days before *t*, which also covers the 60-day volatility warm-up.
- **Outcome y_{i,t+1}** = the compounded excess return over month *t+1*, divided by σ_{i,t} / √12, so its unit is monthly vol-scaled.
- **Asset fixed effect:** x and y are each demeaned within asset over that asset's eligible development observations, giving x̃ and ỹ. This removes each asset's own mean return, which is the "hindsight mean-sign" effect behind the TSMOM-1 failure, so what remains is timing.
- **Timing portfolio** p_t = the equal-weighted mean over eligible assets of x̃_{i,t} · ỹ_{i,t+1}. Months with no eligible asset are dropped.
- The test statistic A2 will pre-register is **mean(p) > 0**, one-sided at α = 0.05. This is the portfolio form of the pooled fixed-effects regression with equal asset weights. Its information ratio is IR_p = mean(p) / sd(p) · √12.

## 2. Outcome-blind estimation (the firewall)

- The gate computes the p-series **only** under misaligned pairings of x̃ against ỹ. A function equivalent to `ic_under_pairing` must raise an error on the true alignment.
- **(a) Circular shifts:** x̃ is shifted in time by *s* months for every s ∈ [13, T − 13], preserving the cross-section.
- **(b) Stationary block bootstrap** of whole months of ỹ, with mean block length 12 (equal to the signal's overlap horizon), B = 10,000 draws, seed 20260925. A draw is discarded if more than 10% of its months happen to land on their true alignment.
- **Noise SE** = the standard deviation of mean(p) across pairings. The larger of the (a) and (b) estimates is used, which is the conservative choice. Also reported: sd_p, the mean of sd(p) across pairings.
- **Smallest detectable effect:** MDE_IR = (1.645 + 0.842) · SE · √12 / sd_p. **Power at an anchor** IR a is Φ(a · sd_p / (√12 · SE) − 1.645).
- **Also reported, second moments only:**
  - effective breadth: the participation ratio (Σλ)² / Σλ² of the eigenvalues of the correlation matrix of ỹ across assets, over months where all assets are eligible;
  - the same ratio for the signs x̃;
  - the number of eligible months and years, and assets per month;
  - portfolio-level years needed, 6.19 / IR², for IR 0.3 to 1.0;
  - holdout power, scaling SE by √(T_dev / T_holdout) using the holdout's month count only. The holdout file is not opened: its span, 2021-10 to 2026-09 = 60 months, is known from the manifest.

## 3. Effect-size anchors (cited, not hand-set)

These are portfolio Sharpe or information ratios of diversified trend strategies. They include some mean or beta component, so they are **upper bounds** on the fixed-effects timing effect.

| Anchor | IR | Source |
|---|---|---|
| Optimistic | **0.72** | MOP (2012) Table 3 Panel A: diversified time-series momentum monthly alpha t = 8.55, 1985–2009. Information ratio ≈ 8.55 / √25 = 1.71, then × (1 − 0.58) for post-publication decay (McLean & Pontiff, JF 2016). |
| Central | **0.39** | DBMF strategy net Sharpe, 2016-07 to 2026-06, from the iMGP DBi managed-futures presentation. Cited in `domain-vehicle-economics-modest-sharpe-book-2026-09-25/research.md` [17]. |
| Pessimistic | **0.07** | SG CTA Index Sharpe over the same window and source [17]. |

## 4. Verdict rule (fixed now)

Power is taken on the development span, DEFF 1 in the sense that the SE already comes from pairings that keep cross-asset correlation.

| Verdict | Condition | Action |
|---|---|---|
| **POWERED** | power ≥ 0.80 at the **central** anchor (0.39) | A2 may proceed |
| **MARGINALLY_POWERED** | power ≥ 0.80 at the optimistic anchor but < 0.80 at the central one | Stop and ask Alex; there is no automatic continuation |
| **UNDERPOWERED** | power < 0.80 at the optimistic anchor | Stop. The result is final, and the development data stays unused for a future seal. |

0.80 power and α = 0.05 are the repo's convention in earlier gates (XSMOM-1, VRP-1, GAP-1).

## 5. Output

`_bmad-output/etftm1_power_verdict.json`, a new filename that doesn't collide with XSMOM-1's `power_verdict.json`. It records the SHA-256 of the script and of every input, all the numbers above, and the verdict.

## 6. What this gate does not do

It computes no statistic under the true alignment, reads nothing from the holdout, sweeps no parameter, and changes nothing in any live system.
