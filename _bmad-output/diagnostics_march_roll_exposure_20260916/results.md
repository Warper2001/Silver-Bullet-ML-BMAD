# Which other 2026 backtests span the March roll (2026-09-16)

**The defect:** in `mnq_1min_2026_ytd.csv` and its sealed-holdout copy, every bar before the **2026-03-12** CME roll is the **deferred MNQM26** while MNQH26 was still the front month. Any MNQ backtest covering 2026-03-01 → 03-11 priced those days off the wrong contract, and any level taken across the switch folds in the calendar spread. In GAP-1's Gate-0 those 11 days supplied **+$1,888 of $9,878** and vanished entirely on corrected bars (`_bmad-output/diagnostics_gap_fade_gate0_rescore_20260916/`).

**Method:** `scan_pre_roll_exposure.py` reads every saved MNQ trade list and ranks it by how much of its net sits in that window. Result files only — no bars, no holdout data, no replays. Full table in `pre_roll_exposure.json`.

## Verdict: nothing live is still exposed

**39 MNQ result files** touch the window. Every one that supports a live strategy has already been corrected; the rest belong to closed or superseded work.

| Family | Pre-roll exposure | Status |
|---|---|---|
| **YANK** (`backtest_1year_20260615*`, `..._20260617*`, `yank_topstep_constrained`, sharpe-levers reconstructions) | 5 trades, −$248 (−3% of net) | **Corrected** — re-validated on front-month bars under seal `da82cfc`; the ML filter was disabled as a result |
| **GAP-1** (`gap_fade_20260625_205328`) | 4 trades, +$944 (+19%) | **Corrected** — Gate-0 re-scored under seal `151f1d05`: $9,878 → $8,281 |
| **MIM-NB** (`mim_nb_gate1_v1/v2`, `mim_nb_catstop_s250/s500`) | 3 trades, −$180 to −$461 (−7% to −16%) | **Checked** — contamination works *against* it; clean-only PF is higher in every file |
| **Program C May-2026 runs** (`backtest_1year_2026051*`, `2026052*`) | 3–9 trades, but up to **−1347% of net** because the net is near zero | Pre-reset. The 2026-05-20 methodology reset already marks these tentative |
| **Silver Bullet ML 6-month** (`backtest_full_silver_bullet_ml_6months_*`, `silver_bullet_corrected_history`) | 14–1,983 rows, **+8% to +37%** of net | Superseded (PF 2.60 was in-sample; see the memory entry) |
| **S26 subgroup** (`s26_subgroup_20260528_*`) | 3 trades, −$2,677 (−33%) | S26 closed DEAD on MNQ at all timeframes |
| **Stat-arb short Gate-2 OOS** (`stat_arb_short_gate2_oos_*`) | **44 of 199 trades (22%)**, +$481 on a −$1,533 net | Closed: live stat-arb has no edge (the apparent profit was a bar-close artifact) |
| **YANK walk-forward / drift** (`tier2_wf_daily_trajectory_*`, `yank_drift_*_history`) | 10–17 trades, −$396 to −$875 (−7% to −27%) | Study data, not a seal. See the note below |
| BTC / crypto / Kraken results | — | **Not affected**: different instrument, no MNQ roll |

**Read the percentages with care.** Where a run's net is near zero, the pre-roll share explodes (`backtest_1year_20260519_215831`: 9 trades worth −$5,151 against a +$382 net). That says the run is dominated by noise, not that the pre-roll days are huge.

## Two things worth knowing

**1. The live ML model is clean.** YANK's `tier2_meta_labeling_model.pkl` was trained on `doe_run_08_fullyear_*`, which is 2025 only, so no March-2026 row reached it. The contaminated `yank_drift_*_history.csv` files feed drift *studies*, not the live filter.

**2. Other instruments' holdout files are unchecked.** `data/sealed_holdout/` also holds `es/gc/hg/pl/rty/si/ym_1min_holdout_20260301_plus.csv`, written by the same pipeline and starting on the same date. Whether they are their own front month before each instrument's March roll is **unknown**, and the repo has no raw per-contract source for them to check against. `hg` and `pl` were used by the copper and platinum Gate-1 tests — both of which **failed or were aborted**, so a defect there would not revive them. Worth resolving before any of those files is used again.

## Recommendation

- **No re-runs needed.** Everything live is corrected; everything else is closed, superseded, or noise-dominated.
- **Cite the corrected figures** for YANK and GAP-1, and MIM-NB's clean-only numbers.
- **A pitfall line was added to `AGENTS.md`** so the next backtest spanning the 2026 roll does not repeat this.
- If any of the closed studies is ever revived, rebuild its bars front-month first (`rebuild_2025_frontmonth.py` for 2025; raw MNQH26 for 2026-01-01 → 03-11).

## Outputs

- `scan_pre_roll_exposure.py`, `pre_roll_exposure.json`
- this file
