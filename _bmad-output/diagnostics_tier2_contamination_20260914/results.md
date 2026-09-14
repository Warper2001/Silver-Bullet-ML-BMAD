# Tier2/YANK backtests and the MNQ 1-minute data defects: results (2026-09-14)

- **Plan:** `replay_plan.md`, committed in `470288b` before any replay.
- **Engine:** unmodified `tools/tier2_census.py`, `--pin max_daily_loss=-750` (the June seal replay), window 2025-05-19 → 2026-02-28.
  - Each run used its own replay root (a symlink farm, `replays/make_roots.py`) at nice 19.
  - All runs report `live_files_unchanged: true`.
- **Holdout:** no sealed-holdout data was opened. Section 3 re-scores only the June seal's already-published trade rows.

## Verdict

**YANK's pre-cutoff backtest profit came from bad data.**
- Jan–Feb 2026 in `mnq_1min_2026_ytd.csv` is the thin deferred MNQM26, not the front month (see `../diagnostics_h2l2_contamination_20260914/`).
- On the real front-month (MNQH26) 1-minute bars, the ML0.50 arm goes from **+$6,297.50 (PF 1.58) to −$94 (PF 0.98)** over the census window.
- The no-ML arm goes from +$67 to **−$2,480 (PF 0.71)**.

**The roll-week splices in 2025 did not affect YANK:** no census signal falls in an interleaved session.

## 1. Gates

| Check | Result |
|---|---|
| G0 (original data) reproduces the 2026-09-13 census, `census_trades_ml050.csv`, row for row | **True** (68 trades, +$6,297.50) |
| Trades before 2026-01-01 identical, G0 vs corrected ML | **True** (28 = 28) |
| Trades before 2026-01-01 identical, census no-ML vs corrected no-ML | **True** (37 = 37) |

**Bar counts:** corrected 232,518 = 205,575 − 29,157 back-month rows + 56,100 front-month rows. The front month's median volume is 542 contracts a minute, against 1–6 in the rows it replaces.

**The only thing that differs between runs is the Jan–Feb 2026 bars.**

## 2. Jan–Feb 2026: back month vs front month

| Arm | Back-month file (as used since June) | Front-month bars |
|---|---|---|
| ML0.50 | 40 trades, +$5,411.00, PF 1.64, 19 wins | **14 trades, −$980.50, PF 0.60, 4 wins** |
| No-ML | 54 trades, +$1,383.50, PF 1.23 | **19 trades, −$1,163.50, PF 0.77** |

| Month (ML0.50) | Back month | Front month |
|---|---|---|
| 2026-01 | 31 trades, +$6,387 | 7 trades, −$120 |
| 2026-02 | 9 trades, −$976 | 7 trades, −$860 |

- **Front-month frequency** is about 7 trades a month, in line with 2025 (1–7 a month).
- **The thin back month nearly tripled the trade count.** Sparse prints of 1–6 contracts leave gaps between minutes, and YANK's fair-value-gap trigger fires on them.

**Whole census window:**

| Arm | As used | Corrected |
|---|---|---|
| ML0.50 | 68 trades, +$6,297.50, PF 1.58 | **42 trades, −$94.00, PF 0.98** |
| No-ML | 91 trades, +$67.00, PF 1.01 | **56 trades, −$2,480.00, PF 0.71** |

## 3. What this does to seal `138cab1` (the ML0.50 decision, 2026-06-15)

The seal's out-of-sample window was Jan–May 2026. Its published trade rows (`data/reports/backtest_1year_20260615_{181838,185354}.csv`) split as follows:

| Period | ML0.50 | No-ML |
|---|---|---|
| Jan–Feb 2026 (back-month file) | 40 trades, +$5,411.00 | 54 trades, +$1,383.50 |
| Mar 1–12 (holdout; built by the same MNQM26 writer before the March roll, so probably back month) | 5 trades, −$247.50 | 5 trades, −$247.50 |
| Mar 13 – May 19 (holdout, front month) | 9 trades, +$1,754.00 | 11 trades, +$1,928.50 |
| **Sealed 2026 total** | **54 trades, PF 1.60, +$6,917.50** | **70 trades, PF 1.32, +$3,064.50** |

**78% of the sealed out-of-sample P&L came from the back-month Jan–Feb file.** So did all of ML's advantage over no-ML: on the front-month holdout weeks, no-ML earned more.

**Corrected composite:** replace Jan–Feb with the front-month replay and keep the published Mar–May rows.

| | ML0.50 | No-ML |
|---|---|---|
| Composite 2026 | 28 trades, **PF 1.09**, +$526 | 35 trades, PF 1.06, +$517.50 |

**Checked against the seal's pre-committed rule:**
- PF_ml > PF_noML: 1.09 > 1.06, passes barely.
- **PF_ml ≥ 1.20: fails.**
- N ≥ 25: 28, passes.

**On corrected data the seal would have taken its null branch: "revert to ML disabled".**

**This is an approximation, not a verdict:**
- The composite joins two replays whose engine state differs at March 1.
- The Mar 1–12 rows probably still sit on back-month bars.
- A faithful corrected re-run of the seal needs the sealed holdout with front-month bars for March. That requires a committed pre-registration and an `ACCESS_LOG` entry.

## 4. Other exposure found

- **ML training set:** `data/ml_training/doe_run_08_fullyear_*` has 1,019 trades from 2025, the 18 features YANK's filter uses. 52 of them (5%) sit in interleaved roll-week sessions with a contract switch in the prior 24 hours, so their features were computed on fake bars. Their P&L is about −$169, near zero.
- **ML threshold file:** the filter's `tier2_threshold.json` and `lr_regime_config.json` were validated on Nov–Dec 2025, which includes the December roll week.
- **YANK-FLOOR's 2021–2024 out-of-sample file** (`data/mim_x/mnq_1min_2021_2024_frontmonth.csv`) is not interleaved. It does carry one unadjusted roll gap per quarter, 194–275 points at the Globex open on roll day (for example 2024-03-15, 06-21, 09-20, 12-20). That is a minor effect, and YANK-FLOOR failed anyway.
- **Not re-checked:** the 2026-06-13 SL/TP/ML grid, the 2026-09-13 entry-mechanics diagnostics (40 of their 68 census trades are the back-month trades) and any other consumer of `mnq_1min_2026_ytd.csv`. Their Jan–Feb 2026 evidence is affected.

## 5. What this does not do

- It changes no strategy parameter, threshold or live setting. YANK is still live as configured.
- Whether to halt, revert the ML filter, or re-validate is a decision for the operator. Any change needs a pre-registration first (AGENTS.md).

## Outputs

- `tag_census.py` and `tag_results.json`
- `tagged_*.csv`
- `replay_plan.md`
- `build_front_month_2026.py`
- `compare_replays.py` and `replay_results.json`
- `replays/{G0,C}/`: census trades, signals and meta
- `replays/run_replays.sh`, `replays/make_roots.py` and `front_month_2026_csv.sha256`
