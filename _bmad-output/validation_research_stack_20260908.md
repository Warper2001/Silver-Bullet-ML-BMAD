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

---

## 8. Follow-up run, 2026-09-08: the live configs and the live records

Two questions from §6, answered with the tools above.

### 8.1 Deflating the grid that chose YANK's live config

`data/reports/grid_sl_tp_ml_20260613.csv` — the SL × TP × ML-threshold sweep that
selected YANK's sealed configuration. 80 rows, 78 usable. `ir` is per-trade
Sharpe (`per_trade_sharpe(pnl)` in `grid_search_sl_tp_ml.py`), so `T = n_oos`.

The selected live config is **SL 2.0 / TP 8.0 / threshold 0.50**, named as
"current live config" in the grid script itself.

| min-T filter | variants N | E[max SR] under null | selected config DSR |
|---|---|---|---|
| none | 78 | 0.6400 | **0.0000** |
| n_oos ≥ 30 | 60 | 0.2371 | **0.0071** |
| n_oos ≥ 100 | 47 | 0.1476 | **0.0751** |

The `min-T` filter matters: a per-trade Sharpe estimated from 5 observations has
sampling sd ≈ 0.45, so an unfiltered sweep has a `Var[SR]` dominated by
estimation noise rather than genuine dispersion between configs, which inflates
`E[max SR]` and makes the deflation spuriously harsh. The filtered rows are the
fair comparison. **The conclusion is the same at every threshold.**

**The robust, filter-independent number:**

> The selected live config has a per-trade Sharpe of **0.0204 over 129 trades**,
> giving **PSR vs zero = 0.5912**.

That is the probability its *true* Sharpe is above zero, using only its own
backtest and **no multiple-testing correction at all**. 59% is close to a coin
flip. Deflated for the 47–78 variants actually searched it is 0.08 or below, and
its Sharpe sits *below* `E[max SR]` under the null of no edge at every filter
setting — so no length of backtest at this effect size would have established it.

Two things in the shop's favour: the selected config was **not** the grid's
argmax (the top-Sharpe cells are the tiny ones — the best has `n_oos = 5`), and
memory already records "wider-SL grid is a 2025 mirage." The instinct was right;
this puts a number on it.

Caveat: a summary table carries no return series, so skew and kurtosis are
unavailable and **normality is assumed**. Real trade P&L is fat-tailed and often
negatively skewed, both of which *reduce* PSR/DSR — so these are an optimistic
upper bound.

### 8.2 Live track records — `tools/validation/live_track_record.py`

Read-only against `data/trades.db`. Two exclusions come first, and they are the
bulk of the work:

**Backfilled backtest replays, excluded: 1,877 rows totalling +$103,624.64.**

| trader | rows | pnl |
|---|---|---|
| trader-yank | 1,841 | **+101,892.90** |
| trader-s26 | 17 | +1,076.10 |
| trader-s27 | 17 | −42.20 |
| trader-mim-nb | 1 | +740.50 |
| trader-btc-carry | 1 | −42.66 |

**Anyone summing `pnl` from `trades.db` gets a number that is ~98% backtest.**

**Legacy pre-column rows, excluded:** 8 trader-yank rows from May–Jun 2025
(−$217.00), NULL mode and NULL symbol, predating YANK's first `realtime` row.

Prospective records that remain (N=1, no selection, so **no deflation applies**):

| trader | mode | trades | days | P&L | Sharpe ann | PSR | PSR (normal) | MinTRL | still needs |
|---|---|---|---|---|---|---|---|---|---|
| trader-s26 | paper | 164 | 62 | +2,575.70 | 2.658 | 0.873 | 0.903 | 128 d | 66 d (~0.3 yr) |
| trader-s26-combine | **live** | 78 | 61 | +1,870.00 | 2.911 | 0.991 | 0.921 | 30 d | — (provisional) |
| trader-s27 | paper | 40 | 50 | +71.10 | 0.060 | 0.511 | 0.511 | 186,097 d | ~738 yr |
| trader-gap-fade | sim | 25 | 49 | +1,116.50 | 1.096 | 0.689 | 0.684 | 533 d | 484 d (~1.9 yr) |
| trader-mim-nb | **live** | 23 | 53 | **−440.00** | −0.472 | 0.415 | 0.415 | ∞ | never at this sign |
| trader-yank | **live** | 5 | 26 | +259.00 | 1.576 | 0.699 | 0.690 | 250 d | 224 d (~0.9 yr) |

**Not one strategy has an established live track record.** The only record
clearing PSR 0.95 is `trader-s26-combine`, and it is **provisional**: 61 days,
and the verdict flips to 0.921 when normality is imposed instead of its
*estimated* skew of 5.01 and kurtosis of 34.9 — moments that cannot be measured
from 61 observations. Declaring it established would be the same error this work
exists to prevent.

Specifics worth acting on:

- **YANK's live ledger record is 5 trades, +$259, over 26 business days** (from
  2026-07-13; the −$212 first trade matches the documented 07-13 halt). Its
  apparent +$101,893 is entirely backfilled backtest. It needs ~224 more trading
  days to establish a Sharpe of this size. **If YANK's authoritative live record
  lives somewhere other than `trades.db`, point me at it** — `data/yank/` holds
  only a Databento pilot, so on current evidence `trades.db` is it.
- **MIM-NB's live record is negative**: −$440 over 23 trades, Sharpe −0.472. This
  is consistent with the documented 2026-07-07 parity failure (sealed engine
  +$490 vs live −$1,657 on the same bars), which was never resolved.
- **gap-fade** is the healthiest paper record but still needs ~1.9 more years at
  its current 0.5 trades/day to clear the bar. Its promotion gate (N≥30 live +
  30 days) is a much weaker test than statistical significance.

### 8.3 Three bugs found in this tooling while running it

Recorded because each would have produced a confident wrong answer:

1. **pandas ≥ 2 infers a datetime format from the first row.** `trades.db` mixes
   `...T13:30:00+00:00` with `...T18:00:03.542998+00:00`; with `errors="coerce"`
   this silently coerced 23 of trader-mim-nb's 24 rows to `NaT`, dropping the
   strategy from the results entirely. Fixed with `format="ISO8601"` plus a loud
   warning on any unparseable row.
2. **NULL `write_mode` meant two opposite things.** Recent live writes *and*
   legacy 2025 rows. Treating all NULLs as live gave YANK 13 trades over 334
   days instead of 5 over 26. Fixed by classifying NULLs against each trader's
   first explicit `realtime` timestamp.
3. **`Var[SR]` contaminated by tiny cells** in the grid — see §8.1's `--min-t`.

### 8.4 Reproduce

```bash
V=/root/Silver-Bullet-ML-BMAD/.venv-research/bin/python

$V tools/validation/deflated_sharpe.py --grid data/reports/grid_sl_tp_ml_20260613.csv \
      --sr-col ir --t-col n_oos --select "sl=2.0,tp=8.0,threshold=0.5" --min-t 100
$V tools/validation/live_track_record.py --min-trades 5
```

---

## 9. `retraining.py:990` target leakage — confirmed — and the CV fixes

### 9.1 The target IS leaking. Measured, not suspected.

`src/ml/retraining.py` labelled each bar `close > open` and trained on a feature
set built from that same bar. Measured on real MNQ dollar bars
(`MNQ_dollar_bars_202401.h5`, n=3,443, 47 features, positive rate 0.510),
single-feature ROC AUC against that label:

| feature | AUC | |
|---|---|---|
| `returns` | **0.9997** | the label, restated |
| `close_position` | **0.9368** | (close−low)/(high−low) |
| `stoch_k` | 0.7348 | |
| `roc` | 0.7059 | |
| `price_momentum_5` | 0.7059 | |

`returns` is the close-to-close return. `open_t == close_{t−1}` exactly for only
**31.4%** of dollar bars — weaker than assumed, which is why this was measured
rather than argued — but `close_{t−1}` sits near `open_t` throughout, so
`sign(close_t − close_{t−1})` tracks `sign(close_t − open_t)` almost everywhere.
The result is a label a single feature reproduces at AUC 0.9997.

**A model trained here reports near-perfect accuracy and has learned nothing,
and no CV scheme fixes that.** Repairing the split without repairing the label
would have made the module *look* sound while staying meaningless.

The label was **not** silently redefined — choosing a forward-looking outcome is
a research decision. Instead the line now carries a measured warning comment and
emits a `logger.warning` at runtime, so the module cannot be trained and
believed by accident. Replacing it with a genuine forward outcome (the
docstring's own "actual trade outcomes") also requires auditing the same-bar
features — `returns`, `close_position`, `stoch_k`, `roc` — against whatever
replaces it.

### 9.2 CV fixes — 16 call sites across 6 files

| file | shuffled splits | integer-cv | now |
|---|---|---|---|
| `src/ml/retraining.py` | 1 | — | positional temporal split, after an explicit `sort_values("timestamp")` |
| `scripts/train_premium_regime_models.py` | 1 | — | temporal split; sorts by timestamp when present, warns when absent |
| `scripts/train_regime_specific_models.py` | 2 | — | shared `temporal_split()` helper (data is timestamp-indexed upstream) |
| `scripts/train_regime_models_real_labels.py` | 2 | 2 | `temporal_split()` + `TimeSeriesSplit` |
| `scripts/tune_regime_1_quick.py` | 1 | 1 | `temporal_split()` + `TimeSeriesSplit` |
| `scripts/tune_regime_1_model.py` | 1 | 4 + `GridSearchCV(cv=3)` | `temporal_split()` + `TimeSeriesSplit` throughout |

A repo-wide grep now returns **zero** live `train_test_split(` calls and zero
integer `cv=` on `cross_val_score`/`GridSearchCV` outside the validation tools.
All six files byte-compile.

Two honest limitations, recorded in the code:

- **The regime datasets carry no timestamp column at all.** The helpers sort on
  the preserved integer index from the upstream extraction, which is
  chronological *there*. That is strictly better than shuffling, but it is an
  assumption that cannot be verified from the file. The real fix is to persist a
  timestamp.
- **Those datasets contain `is_augmented` synthetic rows.** An augmented row and
  the real row it derives from are near-duplicates, so if they land on opposite
  sides of *any* split that is leakage no split scheme can repair. Flagged in
  the code; drop augmented rows from the test side before trusting a score.

### 9.3 Regression test

`tools/validation/test_temporal_splits.py` asserts, for every patched script,
that the split is correctly sized, disjoint, X/y-aligned, and that **train
strictly precedes test** — feeding a deliberately shuffled index so that a
helper which trusts caller ordering fails. It caught exactly that in
`train_regime_specific_models.py`, whose helper was assuming sorted input; it
now sorts like the others.

```bash
.venv-research/bin/python tools/validation/test_temporal_splits.py   # ALL HELPER TESTS PASSED
```

### 9.4 S26 calibration — FIXED 2026-09-09 (§10)

Originally left alone pending a decision, on the belief it fed a live model.
That belief was wrong — see §10.1. Now fixed.

---

## 10. S26 calibration fix (2026-09-09)

### 10.1 Correction: this was never in the live path

§9.4 held this back on the grounds that changing calibration would alter a
deployed model's probabilities. Tracing the artifacts shows that was wrong:

| artifact | written by | read by |
|---|---|---|
| `models/mnq_s26_xgboost_model.pkl` | `run_mnq_s26_pipeline.py` | `inspect_mnq_probs.py` — a diagnostic. **Nothing live.** |
| `models/s26_soft_fvg_ml_model.pkl` | `train_s26_soft_fvg_ml.py` | **both live traders**: `src/research/s26_soft_fvg_streaming.py`, `src/research/btc_s26_combine.py` |

The leaky calibration was in the pipeline that writes the *first* file. The
model the live S26 bots actually load comes from `train_s26_soft_fvg_ml.py`,
which already uses a positional 70/30 temporal split
(`trades_df.iloc[:split_idx]`) and performs no calibration at all — clean on
this axis. So the fix carries **no live risk**, and the caution in §9.4 was
overstated.

Neither `.pkl` was regenerated: mtimes are unchanged (2026-06-10 and
2026-06-04). Changing the script changes nothing until someone deliberately
re-runs it.

### 10.2 The fix

`CalibratedClassifierCV(estimator=clf, method="sigmoid", cv=5)` resolves to
StratifiedKFold for a classifier, so each internal model trained on folds drawn
from **both sides** of the fold it calibrated — the calibration map is fitted
using data from after the period it calibrates. That distorts probability
*levels* rather than ranking, which matters here specifically because the
pipeline then sweeps thresholds of 0.55 / 0.60 / 0.65 read off those levels.

Replaced with a held-out **later** block: fit the classifier on the first 80% of
the (chronologically ordered) 2025 trades, then fit the sigmoid calibrator on
the last 20%, so the calibrator only ever sees data *after* what trained the
classifier. Falls back to `TimeSeriesSplit(n_splits=5)` when the tail block is
too small or single-class.

### 10.3 A trap worth recording: `cv="prefit"` is gone

The textbook idiom for held-out calibration is `cv="prefit"`. **It has been
removed from sklearn** and raises `InvalidParameterError` on **both** 1.8 (the
venv the live bots run) and 1.9 (the research venv). The first version of this
fix used it and would have crashed the pipeline. `FrozenEstimator`
(sklearn >= 1.6) is the supported replacement; it is verified working on both
versions, with an ImportError guard falling back to the TimeSeriesSplit path.

### 10.4 Test

`tools/validation/test_s26_calibration.py` exercises the exact block on
synthetic data — both branches — asserting that `cv="prefit"` is still rejected
(so nobody reinstates it), that the calibration rows are strictly later than the
fitting rows, that probabilities are valid and non-degenerate, and that the
fallback splitter never trains on the future.

```
sklearn 1.9.0  ALL CALIBRATION TESTS PASSED
sklearn 1.8.0  ALL CALIBRATION TESTS PASSED     # the live venv
```

---

## 11. YANK 23-day silence — investigated 2026-09-09/10

### 11.1 Verdict: YANK is not broken

It had logged no trade since 2026-08-17. It is not blocked; the market has not
been offering its setup.

The diagnostic block only prints when H1 bearish sweep **and** M15 CHoCH are
already true, so every `qualified=YES` line is a case where both higher-timeframe
gates passed. There were 10 since 08-21 — but **seven landed within eight
seconds** (20:37:23–31 on 09-04), immediately after the 20:36:45 service restart.
Those are startup backfill replay, not live events.

That leaves **3 genuine live near-misses in 23 days**: 08-21, 08-26, 09-09.

On each, entry needs `fvg_signal.direction == BEARISH and cached_sweep is not
None`. `cached_sweep` is assigned from `self._cached_sweep`, and
`h1_bearish_sweep_active` is *defined* as that being non-None and bearish, so it
cannot be the blocker inside this branch. By elimination `detect_fvg` returned a
**bullish** FVG under bearish higher-timeframe context — correctly, no entry.
Against YANK's own rate of ~1 trade per 7 business days, 3 near-misses and zero
fills is unremarkable.

### 11.2 A wrong turn worth recording

The investigation initially concluded "the no-entry path logs nothing" from
`logs/tier2_filter_log.csv` being frozen at the restart. **That was the wrong
file.** `_log_filter_decision()` writes `logs/tier2_bar_decisions.csv`, which is
current and does log every bar. The `else` branch is not silent.

### 11.3 What the investigation actually found: a 1.66 GB shared log

`logs/tier2_bar_decisions.csv` is **24,065,642 rows / 1.66 GB**, and 2025-dated
bars are still arriving in recent appends.

**Three traders append to that one file** — `yank_streaming_working.py`,
`tier2_streaming_working.py`, `btc_combine_streaming.py` — and only YANK carried
the backfill guard, whose own comment records fixing this once already at "9.2M
rows / 631MB observed". It is now 2.6× worse. This is the "fixed on one of three
copies, unmerged" item, located.

Two consequences beyond disk: the file has **no `trader_id` column**, so rows
cannot be attributed to a bot; and any manual or backtest run of the two
unguarded scripts appends its whole history into the live trail.

### 11.4 The patch (logging only — no trading behaviour changed)

1. **`tier2_streaming_working.py`, `btc_combine_streaming.py`** — YANK's backfill
   guard added verbatim to `_log_filter_decision()`.
2. **`yank_streaming_working.py`** — the 08-21 diagnostic relabelled. It printed
   `pattern=`/`raw_gap=` recomputed from the **last three bars** alongside
   `qualified=` from `detect_fvg`'s **full window scan**, so
   `pattern=no raw_gap=0.00 | qualified=YES` read as an impossible state and cost
   real time on 09-09. Now `last3_pattern=` / `last3_gap=` / `window_fvg=` + `dir=`.
3. **`yank_streaming_working.py`** — near-miss logging. An INFO line when both
   higher-timeframe gates passed and an FVG existed but was the wrong direction,
   and a WARNING if `cached_sweep` is ever None while the sweep is active (should
   be unreachable; if it fires, the sweep state has desynchronised).

Every added line is a comment, a `logger` call, or the guard's early return in a
logging-only function. `_fvg_hit` and the entry path are byte-identical. All
three files compile on the **live** interpreter (`.venv`, sklearn 1.8). Branch
logic verified across all four cases, including that the common "no FVG" case
stays silent so this is not a per-bar log.

### 11.5 Not done — needs you

- **The 1.66 GB file is untouched.** Truncating or rotating a live trading log is
  your call. The guard stops the two unguarded scripts adding to it; it does not
  shrink what is there.
- **Nothing is deployed.** These are worktree edits; the running processes are
  unaffected. Live fixes here have historically gone out by direct file copy
  rather than git, so deployment is a deliberate step.
- **No `trader_id` column** was added — that changes the CSV schema, and
  `analyze_filter_funnel.py` and `tools/combine_ops_healthcheck.py` read it.
