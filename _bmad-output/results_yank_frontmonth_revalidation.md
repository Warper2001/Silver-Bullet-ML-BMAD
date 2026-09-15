# Results: yank_frontmonth_revalidation — seal 138cab1 re-run on front-month bars

**Date:** 2026-09-15
**Pre-registration:** `_bmad-output/preregistration_yank_frontmonth_revalidation.md`, sealing commit `da82cfc9fab8aa40b4b3d9072d0e8716a3068cfe`. It was merged to main as `759e7e7` before the run.
**Harness:** `tools/yank_frontmonth_revalidation.py`, byte-identical to the sealed copy (the harness checks this). The engine ran from the worktree at `da82cfc`.
**Holdout access:** four `ACCESS_LOG` rows, 2026-09-15 02:30 and 03:06 UTC, all citing `da82cfc`.
**Outputs:** `_bmad-output/revalidation_yank_frontmonth_20260915/`. That folder holds `score.json`, the trades, the reports, and meta (config, pinned hashes, segments, guard). `corrected_2026.csv` is not committed because it contains holdout-period bars; its sha256 is `0b09dd55…20d6` in the meta files.

## Verdict: INCONCLUSIVE, so ML disabled (`ml_threshold` 0.0)

**Applying seal 138cab1's rule to front-month bars:**

| Criterion (from 138cab1) | Corrected result | Met? |
|---|---|---|
| `N_2026(ml) ≥ 25` | **23** | **no → INCONCLUSIVE → default to ML disabled** |
| `PF_ml(2026) > PF_noML(2026)` | 1.698 > 1.389 | yes |
| `PF_ml(2026) ≥ 1.20` | 1.698 | yes |

**This supersedes 138cab1's "KEEP ml_threshold = 0.50"** for the ML question (sealed doc, section 3). On correct bars the ML arm does not have the minimum sample, and the rule's default is ML disabled.

- **Halt review:** not recommended by the rule (section 4). Both corrected arms are net positive over 2026: ML +$2,573.50, no-ML +$2,565.00.
- **Edge existence:** UNDERPOWERED by construction (section 4). Neither arm's day-clustered 95% interval for the mean excludes zero.

**Acting on this** means editing `ml_threshold` (and `tier2_threshold.json`, which is where the live gate reads 0.50 from) and restarting `trader-yank`. That is a separate operator step and was not taken.

## Gates and run facts

**G0 passed** with no fallback to the 138cab1 code. The engine at `da82cfc` with 138cab1's YAML reproduces both June trade lists **row for row**:
- `backtest_1year_20260615_181838.csv`: ML, 82 trades
- `…_185354.csv`: no-ML, 107 trades

**Config:** the effective StrategyConfig equals the 138cab1 snapshot. It differs from today's YAML only in `max_daily_loss` (−750 vs −300). The pinned model and threshold hashes matched.

**Isolation:** the guard reports no change to live files in any of the four runs.

**Bars:**

| Run | Bars |
|---|---|
| Original | 280,775 |
| Corrected | 299,582 = S0 176,418 + S1 56,100 + S2 0 + S3 67,064 |

## Results by segment (trades with entry in 2026)

| Segment | Original ML | Corrected ML | Original no-ML | Corrected no-ML |
|---|---|---|---|---|
| S1 Jan–Feb | 40, +$5,411.00, PF 1.64 | **14, −$980.50, PF 0.60** | 54, +$1,383.50, PF 1.23 | **19, −$1,163.50, PF 0.77** |
| S2 Mar 2–11 | 5, −$247.50 | **0 (no bars; see below)** | 5, −$247.50 | 0 |
| S3 Mar 12 – May 19 | 9, +$1,754.00, PF 2.13 | 9, +$3,554.00, PF 3.90 | 11, +$1,928.50, PF 2.03 | 11, +$3,728.50, PF 3.42 |
| **2026 total** | **54, +$6,917.50, PF 1.60** | **23, +$2,573.50, PF 1.70** | **70, +$3,064.50, PF 1.32** | **30, +$2,565.00, PF 1.39** |

**Day-clustered t and 95% interval of the mean, corrected 2026:**

| Arm | Mean | t | 95% interval |
|---|---|---|---|
| ML | $111.89 | 0.98 | [−$112.5, $336.3] |
| No-ML | $85.50 | 0.81 | [−$121.9, $292.9] |

## A pre-declared contingency fired (reported as the sealed doc requires)

**All 8 S2 sessions (2026-03-02 … 03-11) carry both MNQH26 and MNQM26 labels in the raw data.** The rule of section 2 dropped them all, so S2 is empty.
- **Δ could not be measured** and S3 was left unshifted, per section 2. The corrected series therefore has **one unadjusted H26→M26 roll jump** between the Feb 27 close and the Mar 12 open, across a two-week gap.
- **The effect is two S3 trades, in both arms.** The corrected runs **add** a short on 2026-03-12 13:47 UTC (09:47 ET, the first morning after that jump; time exit, +$1,471). They **lack** the original 2026-04-15 trade (stop, −$329). The other S3 trades (8 of 9 ML, 10 of 11 no-ML) are identical. The added trade is probably a product of the jump: a bearish-only strategy sees an upside "sweep".
- **The verdict does not depend on it.** Without that trade, ML has 22 trades (PF 1.30, +$1,102.50) and no-ML 29 (PF 1.17, +$1,094.00). N is still below 25, so the result is still **INCONCLUSIVE → ML disabled**. This sensitivity is descriptive and changes nothing.

## What the result means

1. **The seal's evidence for keeping the ML filter was an artifact of the deferred-contract file.** It rested mostly on 40 Jan–Feb trades that do not exist on front-month bars. On correct bars, Jan–Feb loses in both arms, and the rule defaults the filter off.
2. **The only untouched out-of-sample data is S3, and it is small and favorable.** It has 9 ML and 11 no-ML trades, identical between runs apart from the roll-jump pair. The ML filter adds nothing there: it removed 2 trades, and no-ML earned slightly more.
3. **YANK's pre-cutoff record on correct bars stays weak**, per the contamination check: census ML −$94 (PF 0.98), no-ML −$2,480 (PF 0.71). Combined with an underpowered, favorable S3, there is **no established edge in either direction**.

## Not done

- No parameter, threshold or live setting was changed, and `trader-yank` was not restarted.
- The model was not retrained, even though 5% of `doe_run_08` rows sit on 2025 roll splices.
- The live unit's `Environment=` overrides were not examined. The replay does not read them.

## Applied to live YANK (2026-09-15 23:53 UTC)

Alex authorised applying the verdict. The ML filter is now disabled on the live combine account.

| Change | Detail |
|---|---|
| `strategy_config.yaml` | `ml_threshold` 0.50 → 0.0, header now cites this seal. Committed and merged (`603575a`, merge `acf70f9`). Documentation only: `yank_streaming_working.py` builds `MetaLabelingFilter(ML_MODEL_PATH)` without it. |
| `models/xgboost/tier2_threshold.json` | `threshold` 0.5 → 0.0. **This is the live gate.** Gitignored, so edited in place with a backup at `tier2_threshold.json.pre-ml-disable-20260915T235252Z.bak`; every other field unchanged, and a `superseded` block records the reason. |
| `trader-yank` | restarted 2026-09-15 23:53:04 UTC, active, 0 restarts, no open trade at the time. |

**Verified from the startup log:** `ML threshold loaded from JSON: 0.0`, `Configuration: … | ml_threshold=0.0`, `ML Filter: ACTIVE | threshold=0.0` (the model still loads; at 0.0 every signal passes). Symbol MNQZ26, 2 contracts, account 26556101 — unchanged.

**Scope checks before the edit:**
- Among live units only YANK reads `tier2_threshold.json`. The other two readers (`btc_combine_streaming.py`, `s26_crypto_streaming_working.py`) are not imported by any running service.
- No other live entry file reads `ml_threshold` or `strategy_config.yaml`.
- The LR regime config's own `ml_threshold` field is only logged, never used for gating.

**Not changed:** every other strategy parameter, the model file, the LR regime filter, and the unit's `Environment=` lines.
