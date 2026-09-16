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

**2. Other instruments' holdout files.** *(Answered the same day — see the second half of this file. **si and hg are deferred pre-roll; es, rty, ym, gc and pl are clean.** Neither defective one is load-bearing: `hg` fed the copper Gate-1 test that already failed, `si` only the closed fan-out.)*

## Recommendation

- **No re-runs needed.** Everything live is corrected; everything else is closed, superseded, or noise-dominated.
- **Cite the corrected figures** for YANK and GAP-1, and MIM-NB's clean-only numbers.
- **A pitfall line was added to `AGENTS.md`** so the next backtest spanning the 2026 roll does not repeat this.
- If any of the closed studies is ever revived, rebuild its bars front-month first (`rebuild_2025_frontmonth.py` for 2025; raw MNQH26 for 2026-01-01 → 03-11).

## Outputs

- `scan_pre_roll_exposure.py`, `pre_roll_exposure.json`
- this file

---

# The other instruments' holdout files, checked (2026-09-16, same day)

**Open item from above:** `data/sealed_holdout/{es,gc,hg,pl,rty,si,ym}_1min_holdout_20260301_plus.csv` come from the same pipeline and start on the same date as the MNQ holdout file. Do they carry the same deferred-contract defect?

**No sealed-holdout file was opened.** Every instrument has a non-sealed twin under `data/processed/dollar_bars/1_minute/` covering the same dates — the dual presence the `ACCESS_LOG` documents for MNQ — and those are what was read.

## Method

A per-contract price reference (`data/term_structure/raw_contract_bars.csv`) was tried first and **discarded**: its daily closes are not comparable to these bars, the nearest-contract match flips between months, and the residuals exceed the inter-month spreads.

What works is that **a deferred contract is thin**. Compare median volume per minute before the 2026-03-12 roll with just after. MNQ's known-defective file is the positive control.

| File | Jan | Feb | Mar 1–11 | Mar 12+ | post/pre | Verdict |
|---|---|---|---|---|---|---|
| **mnq** (control) | 2 | 3 | 6 | 612 | **×102** | **deferred pre-roll** |
| es | 220 | 253 | 218 | 290 | ×1.3 | front-month throughout |
| rty | 28 | 30 | 44 | 40 | ×0.9 | front-month throughout |
| ym | 24 | 29 | 41 | 35 | ×0.9 | front-month throughout |
| gc | 90 | 48 | 70 | 75 | ×1.1 | front-month throughout |
| pl | 16 | 9 | 8 | 8 | ×1.0 | front-month throughout |
| **si** | 65 | 21 | **1** | 18 | **×18** | **deferred pre-roll** |
| **hg** | 22 | 14 | **2** | 16 | **×8** | **deferred pre-roll** |

Corroborating provenance: `download_es_1min.py` and `download_gc_1min.py` fetch **contract segments with explicit date ranges** (ESM25 → ESU25 → ESZ25 → ESH26 → ESM26; GCM25 → … → GCM26), i.e. a roll-aware stitch. That matches their clean volume profile. No download script survives in the repo for rty, ym, si, hg or pl.

## Result: 2 of 7 share the defect, and neither is load-bearing

- **Clean: es, rty, ym, gc, pl.** ES and GC are clean by construction as well as by volume.
- **Defective: si and hg.** Their pre-roll windows are the deferred contract — 1 and 2 contracts a minute, against 18 and 16 after the roll.

**What used them:**
- **hg** fed the **copper Gate-1 holdout test**, which **FAILED** (prereg `fbd7afe`, run `328cdaf`; N=26, gross PF 0.563, net 0.463). Its holdout window opens on 2026-03-01, so its first ~11 days ran on thin deferred-contract bars. The verdict was a failure on gross P&L, and correcting the data would not obviously flip that — but the failure is now *less cleanly attributable* to the strategy. The copper track is closed either way.
- **si** appeared in the cross-instrument fan-out, which is already closed as unreproducible, and cost-failed there.
- Neither supports a live strategy.

## Limits of this check

- It infers the sealed copies' contents from their non-sealed twins. That the holdout files are extracts of these files is **documented for MNQ and presumed for the rest** — same pipeline, same cutoff date, same naming. Confirming it directly means opening them, which needs a pre-registration and an `ACCESS_LOG` row, and there is no result that currently justifies one.
- Volume is a proxy for contract identity. It is decisive at ×102, ×18 and ×8 with a thin/thick step at the roll, and it is corroborated for ES and GC by their download scripts, but it is not a contract label.
- Only the March 2026 roll was examined. The same question applies to earlier rolls in these files, and was not asked here.

## Recommendation

- Treat `si` and `hg` bars before 2026-03-12 as **deferred-contract data**, in both the working files and their holdout copies.
- Nothing needs re-running: both instruments' tracks are closed and neither is live.
- If copper or silver is ever revived, refetch per contract with explicit date ranges as `download_es_1min.py` and `download_gc_1min.py` already do.
