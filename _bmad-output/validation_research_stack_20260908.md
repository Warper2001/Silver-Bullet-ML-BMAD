# Research validation stack — incorporated and validated

**Date:** 2026-09-08
**Branch:** `feat/research-validation-stack`
**Follows:** `_bmad-output/research/technical-github-day-trading-repos-for-local-incor-2026-09-08/research.md`

Adopted `arch`, `skfolio` and `statsmodels`; built and validated three tools on top of them. This document records what was installed, what each tool was tested against, what the tools then said about real data in this repo — and **three corrections to the recon report that produced the recommendation.**

---

## 1. Installation — and why it is not in the live venv

Ten services are running, including `trader-yank` and `trader-mim-nb`, both live on Topstep combine account 26556101. Both execute:

```
ExecStart=/root/Silver-Bullet-ML-BMAD/.venv/bin/python src/research/{yank_streaming_working,mim_nb_live}.py
```

Installing into `.venv` would let pip re-resolve `numpy` / `scipy` / `scikit-learn` underneath two bots trading real money. So the stack lives in a **separate interpreter**:

```
/root/Silver-Bullet-ML-BMAD/.venv-research
```

Installed: `arch 8.0.0`, `skfolio 1.0.5`, `statsmodels 0.15.0` (plus scikit-learn 1.9.0, scipy 1.18.1, xgboost 3.4.1). Declared in `pyproject.toml` under `[tool.poetry.group.research.dependencies]` with the rebuild command in a comment. **Nothing in `src/` imports any of it.** Removal is deleting the group and the venv.

---

## 2. Tools built

| Tool | Purpose | Backed by |
|---|---|---|
| `tools/validation/spa_variant_family.py` | Multiple-testing correction over a family of strategy variants — Hansen SPA, Romano–Wolf StepM, Holm, BH-FDR | `arch.bootstrap`, `statsmodels` |
| `tools/validation/cv_leakage_probe.py` | Measures how much measured model performance is manufactured by the CV scheme | scikit-learn, with a purged reference |
| `tools/validation/deflated_sharpe.py` | PSR, Deflated Sharpe, expected-max-Sharpe, Minimum Track Record Length | own implementation + `--self-test` |

Deflated Sharpe is hand-implemented because neither `arch` nor `skfolio` ships it — confirmed against skfolio's API surface, whose `measures` module has no DSR/PSR.

---

## 3. Validation of the instruments (before trusting any output)

### 3.1 Deflated Sharpe — `--self-test`, ALL PASS

| Test | Result |
|---|---|
| 1. PSR(0) on Gaussian returns must equal the classical Sharpe t-test p-value | max abs diff **0.0006** (bar 0.02) — PASS |
| 2a. Null size control: P(DSR>0.95) ≤ 0.05 | **0.000** at N=10 and N=50 — PASS |
| 2a. Null centring: mean DSR ≈ 0.5 | **0.501** (N=10), **0.493** (N=50) — PASS |
| 2b. Power: a genuine daily SR of 0.20 over T=1000 must survive | **0.972** (N=10), **0.980** (N=50) — PASS |
| 3. E[max SR] monotone increasing in the number of trials | PASS |

Test 2a's centring check is the sharp one: if `expected_max_sharpe` were mis-scaled, the mean DSR would drift off 0.5. It lands on 0.501.

**A note on how this test was arrived at.** The first version of test 2 asserted DSR should be ~Uniform(0,1) under the null and *failed* at 0.000. The implementation was right and the test was wrong: DSR is `P(true SR > E[max SR under the null])`, which is false for every strategy under the null, so DSR is deliberately conservative rather than uniform. The test was replaced with the two properties DSR actually guarantees — size control *and* centring — plus a power test, so that a degenerate always-zero implementation cannot pass.

### 3.2 Leakage probe — positive control

A leak detector reporting "no leak" is worthless until it is shown to find one. The control builds a dataset with **exactly zero** true predictability: a driftless random walk, causal past-only features, labels = sign of the forward 20-bar return. Averaged over **15 independent paths** (a single path has realised drift, so AUC = 0.5 holds only in expectation).

Mean ROC AUC, n=1500, h=20:

| model | random_split | stratified_kfold | timeseries_split | **purged_walkforward** |
|---|---|---|---|---|
| forest | **0.8543** | 0.4874 | 0.5188 | **0.5124** |
| logreg | **0.5907** | 0.4872 | 0.5027 | **0.4984** |

Standard errors across paths: 0.005–0.021.

- The **purged reference is unbiased** — 0.5124 and 0.4984 against a truth of 0.5. That is what licenses using it as the baseline.
- `random_split` fabricates **+0.342 AUC** for a random forest — roughly 20 standard errors of pure fiction.
- Leakage is **model-dependent**: a flexible model fabricates far more (+0.342) than a linear one (+0.092), because the channel is memorisation of temporal near-twins.
- On this synthetic control `stratified_kfold` did **not** inflate. It does on real data (§4.2) — so the control establishes detection power, not a universal ranking.

---

## 4. What the validated tools say about this repo

### 4.1 SPA / StepM on the DOE family — `spa_variant_family.py`

Nine DOE runs, all on the same window (2025-08-01 → 2025-12-30, 108 business days), daily P&L, benchmark = cash.

| variant | total P&L | ann. Sharpe | naive one-sided p |
|---|---|---|---|
| doe_run_08 | 2760.75 | 2.304 | 0.0672 |
| doe_run_07 | 1840.70 | 1.582 | 0.1514 |
| doe_run_09 | 1403.05 | 1.385 | 0.1834 |
| doe_run_05 | 1148.70 | 1.839 | 0.1156 |
| doe_run_06 | 173.25 | 0.241 | 0.4376 |
| doe_run_04 | 154.20 | 0.181 | 0.4529 |
| doe_run_03 | 46.80 | 0.116 | 0.4698 |
| doe_run_02 | −729.80 | −1.935 | 0.8959 |
| doe_run_01 | −1657.00 | −3.045 | 0.9756 |

- Naively significant at 5%, **uncorrected**: 0 of 9
- Holm (FWER 5%) survivors: **none**
- BH-FDR (5%) survivors: **none**
- Romano–Wolf StepM superior set: **empty**
- **Hansen SPA consistent p-value: 0.1233** → fail to reject; the family's best result is consistent with luck

The search penalty is visible: the best variant's naive p of 0.0672 becomes an SPA p of 0.1233 once you account for having looked at nine.

### 4.2 Deflated Sharpe on the same family — `deflated_sharpe.py`

| variant | ann. Sharpe | PSR vs zero | **DSR (N=9)** | MinTRL |
|---|---|---|---|---|
| doe_run_08 | 2.304 | **0.9456** | **0.3884** | 114 days |
| doe_run_05 | 1.839 | 0.8874 | 0.2825 | 198 days |
| doe_run_07 | 1.582 | 0.8593 | 0.2208 | 250 days |
| doe_run_09 | 1.385 | 0.8282 | 0.1820 | 324 days |

This is the cleanest single illustration of the shop's failure mode. `doe_run_08` has PSR 0.9456 — it looks significant at ~95% if you forget you searched. Deflated for nine trials it is **0.3884**, and does not survive.

**The punchline:** with 9 variants over 108 days, `E[max Sharpe]` under the null of *no edge at all* is **2.712 annualised**. The best observed variant is **2.304** — *below* what pure luck would be expected to produce from a search this wide.

`doe_run_09`, which MEMORY.md records as the selected DOE winner at "PF 1.524", has **DSR 0.1820** and would need a **324-day** track record to be significant. It has 108.

### 4.3 CV leakage on the real meta-labeling dataset — `cv_leakage_probe.py`

`data/ml_training/s23_meta_labels_2025.csv`, n=109, 20.2% positives, 10 features, 40 random-split repeats.

| model | random_split | stratified_kfold | timeseries_split | **purged_walkforward** |
|---|---|---|---|---|
| logreg | 0.6979 | 0.6634 | 0.5414 | **0.5414** |
| forest | 0.7309 | 0.6396 | 0.4875 | **0.4875** |

Inflation over the leak-free reference:

| model | random_split | stratified_kfold | timeseries_split |
|---|---|---|---|
| logreg | **+0.1565** | **+0.1220** | +0.0000 |
| forest | **+0.2434** | **+0.1520** | +0.0000 |

Three things follow:

1. **The honest AUC of the meta-labeling features is 0.49–0.54** — essentially no predictive signal. Consistent with the model having been disabled (`ml_threshold = 0.0`).
2. **`TimeSeriesSplit` is exactly unbiased here** (+0.0000 against purged, as it must be on a zero-overlap dataset — a self-verifying consistency check). The deployed trainer at `src/ml/train_tier2_meta_labeling.py:78` uses `TimeSeriesSplit(5)`, so **the deployed path is not leaking.**
3. The leaking patterns are elsewhere: `train_test_split(shuffle=True)` at `src/ml/retraining.py:997` and `scripts/train_premium_regime_models.py:78` (+0.16 to +0.24), and `cross_val_score(cv=5)` → StratifiedKFold at `scripts/tune_regime_*.py` and `scripts/train_regime_models_real_labels.py` (+0.12 to +0.15).

Caveat: n=109. These AUCs are individually noisy; the *differences between schemes on identical data* are the reliable part.

---

## 5. Corrections to the recon report

The validation contradicted the report that motivated it in three places. All three are now fixed in `research.md`.

**Correction 1 — the headline recommendation was wrong about its own mechanism.** The report said the meta-labeling model's CV is "near-certainly leaking" via "overlapping label horizons," and called re-running it under purged CV "the highest expected value per hour." Measured: **the dataset has exactly zero overlapping label windows.** The strategy holds one position at a time (median hold 64 minutes), so every trade closes before the next opens. Purging has nothing to remove. The predicted defect does not exist, and the reasoning — triple-barrier labels ⇒ overlapping horizons — does not hold for single-position sequential trading.

**Correction 2 — the deployed path was wrongly implicated.** The report's `[33]`-backed recommendation implied the deployed meta-labeling trainer needed purged CV. It uses `TimeSeriesSplit`, which measures as exactly unbiased on this data. The genuine leakage is in `src/ml/retraining.py` and `scripts/` — the non-deployed infrastructure — not the live path.

**Correction 3 — the grep was too blunt.** "Zero hits for purged CV" was true but did not distinguish *needing* purged CV from *not needing* it. A repo of single-position strategies legitimately does not need purging. The right check is overlap, which `cv_leakage_probe.py` now measures and prints.

**What survives unchanged:** `arch` is the right adoption and the SPA/StepM/DSR results in §4.1–4.2 are the strongest output of this work. The value was in the multiple-testing layer, not the cross-validation layer.

---

## 6. What to do next

1. **Re-examine any live-config decision that came from a variant search.** §4.2 shows a 9-variant search over 108 days cannot distinguish a 2.3 Sharpe from luck. `E[max SR]` under the null exceeded every observed variant. Candidates: the S25 `min_gap_atr_ratio = 0.25` selection and the ml threshold.
2. **Run `spa_variant_family.py` before sealing any future pre-registration that follows a sweep.** It converts "the best of N looked good" into a correctly-sized statement, and is cheaper in data than pre-registering variants one at a time.
3. **Fix or delete the leaking `scripts/`.** `src/ml/retraining.py:997` and `scripts/train_premium_regime_models.py:78` use random splits on time series; `scripts/tune_regime_*.py` use StratifiedKFold. They are not in the live path, but they are the kind of code that gets copied.
4. **Do not adopt purged CV for its own sake.** It is the right tool the moment a bar-level or multi-position dataset appears — but on the current single-position trade-level data it is a no-op, and the probe now prints the overlap count that decides it.
5. **`src/ml/retraining.py:990` deserves its own look.** Its target is `(df["close"] > df["open"])` — a same-bar quantity — while its features are engineered from that same bar. If any feature encodes the bar's close, that is direct target leakage, a worse problem than the CV scheme. Not investigated here.

---

## 7. Reproduce

```bash
V=/root/Silver-Bullet-ML-BMAD/.venv-research/bin/python

$V tools/validation/deflated_sharpe.py --self-test          # must print ALL SELF-TESTS PASSED
$V tools/validation/cv_leakage_probe.py --mode control      # detector calibration
$V tools/validation/cv_leakage_probe.py --mode real \
      --csv data/ml_training/s23_meta_labels_2025.csv --repeats 40
$V tools/validation/spa_variant_family.py
$V tools/validation/deflated_sharpe.py
```

All are read-only: they load CSVs and print. None writes to `data/`, `models/`, or `trades.db`, and none imports from a live trading path.
