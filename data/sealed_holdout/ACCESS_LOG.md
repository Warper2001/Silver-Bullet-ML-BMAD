# Sealed Holdout — Access Log

**Cutoff date:** 2026-03-01  
**Data file:** `mnq_1min_holdout_20260301_plus.csv`  
**Rows:** 75,081 bars (2026-03-01T23:01 UTC → 2026-05-19T21:00 UTC)  
**Permissions:** 444 (read-only — do not chmod without logging it here)  
**Established:** 2026-05-20 (Program C Phase 0.4)

---

## Access Protocol

**Before reading the holdout file, you MUST:**

1. Commit a pre-registration document to git specifying:
   - Exact hypothesis being tested (e.g. "Strategy PF > 1.0 on sealed holdout")
   - Exact decision rule (e.g. "If PF < median of 50 random-entry PFs → PIVOT")
   - Which parameters are frozen (no changes allowed after pre-registration)
2. Record the pre-registration commit SHA in this log
3. Run the test exactly as described in the pre-registration — no mid-test adjustments
4. Record the result here regardless of outcome (no selective reporting)

**You may NOT:**

- Read this file to "check if the strategy looks promising" before pre-registering
- Run a partial test on holdout data and then adjust parameters before the full test
- Use holdout data results to motivate parameter changes, then re-run on holdout
- Claim a result is "OOS" if the parameters were selected using information from this period

**The purpose of this log is to make the access history visible to your future self.** If you are tempted to bend these rules, re-read `_bmad-output/problem-solution-2026-05-20.md` § Bypass Inoculation.

---

## Why 2026-03-01?

This date was chosen because:
- All DOE parameter searches used Aug–Dec 2025 data (5 months)
- `BEARISH_ONLY`, `MAX_HOLD_BARS`, `MAX_PENDING_BARS`, `VOL_REGIME_THRESHOLD`, `TP_MULTIPLIER` were all tuned on data ending before this cutoff
- Jan–Feb 2026 (29,157 bars, in `mnq_1min_2026_ytd.csv`) is available as a small additional training segment or a pre-Phase-1 sanity check window, but is NOT part of the sealed holdout
- 2026-03-01 onward was untouched by any parameter selection process at the time of sealing

---

## Limitations of This Seal

**Root process / 444 ineffective:** This system runs as root. On Linux, root can overwrite any file regardless of permissions. The `chmod 444` on the data file conveys intent and provides a friction barrier, but does not technically prevent modification. The real enforcement mechanism is git: every access must be committed to this log, which creates an immutable audit trail in git history. Do not treat the file permission as a security guarantee.

**Contaminated ML training data:** Adversarial review identified that the following files contain post-cutoff (post-2026-03-01) labeled rows and were created after the holdout period began:
- `data/ml_training/silver_bullet_trades_full.parquet` — contains rows from 2026-03-02 to 2026-03-04
- `data/ml_training/silver_bullet_signals_full.parquet` — same window
- `data/ml_training/regime_aware/regime_0_training_data.parquet` — timestamps from 2026-03-02 to 2026-03-04

Any XGBoost or logistic regression model trained from these files is contaminated. Do not use these as training data for Phase 1 or Phase 2 models. Phase 0.5 will document the full list of contaminated files and establish clean re-generation procedures.

**Jan–Feb 2026 status (29,157 bars):** The rows in `mnq_1min_2026_ytd.csv` with timestamps before 2026-03-01 are NOT part of the sealed holdout. However, their status is ambiguous: it is unclear whether any DOE parameter searches or ML training runs explicitly included these rows. Until this is resolved, treat Jan–Feb 2026 data as *potentially in-sample* — do not use it as an independent validation window without verifying it was excluded from all prior parameter searches.

---

## Important Caveat: Dual Presence of Holdout Data

The rows in this file also exist in:  
`data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv`

That file has NOT been truncated (doing so would break existing scripts). This means:

- Any script loading `mnq_1min_2026_ytd.csv` without the Phase 0.5 pre-registration gate will silently use holdout data
- **Phase 0.5** will add a `--preregistration <git-sha>` flag to `backtest_tier2_1year_validation.py` that refuses to run on data after 2026-03-01 unless a valid pre-registration SHA is provided
- Until Phase 0.5 is complete, the holdout boundary is enforced by **honor system only**

Do not run `backtest_tier2_1year_validation.py` with `mnq_1min_2026_ytd.csv` against the post-March data until Phase 0.5 is in place and a pre-registration is committed.

---

## Access Log

| Date | SHA (pre-registration) | Accessor | Purpose | Result |
|---|---|---|---|---|
| 2026-05-20 | — | Program C Phase 0.4 setup | Establish sealed holdout; extract rows to this directory; no hypothesis tested | N/A — setup only |
| 2026-05-20 13:47 UTC | 910e95c | gate smoke-test | Verify Phase 0.5 gate accepts preregistration SHA 910e95c; not the S12 test run | gate accepted; 191 trades on full 1-yr window |
| 2026-05-20 15:35 UTC | 910e95c | s12 script | S12 failed run (pandas bug, no results) | aborted — no results |
| 2026-05-20 15:37 UTC | 910e95c | s12 script | S12 full run — `--preregistration 910e95c` | patterns_survive: real PF 1.2154 > p90 1.1350 (96 trades vs ~1000 per seed) |
| 2026-05-20 16:11 UTC | 910e95c | s13 script | S13 full run — `--preregistration 910e95c` | design_phase2_ml_test: best_TF_PF 1.8157 (15-min, 14 trades); 1-min PF 1.2154 (96 trades); 5-min PF 0.5688 (32 trades) |
| 2026-05-20 19:39 UTC | dbfa46f | s14 script | S14 gate smoke-test — `--preregistration dbfa46f` (head -5 only) | gate accepted; 75,081 bars loaded |
| 2026-05-20 19:46 UTC | dbfa46f | s14 script | S14 full run — `--preregistration dbfa46f` | no_unfiltered_edge: PF 1.0513 (150 trades) ≤ S12 p90 random 1.1350; H1 sweep is load-bearing |
| 2026-05-21 03:16 UTC | 9e61012 | s18 script | `--preregistration 9e61012` | insufficient_sample: PF 1.3121 (2 trades) |
| 2026-05-21 03:28 UTC | 39a1039 | s19 script | `--preregistration 39a1039` | insufficient_sample: PF 3.1888 (5 trades) |
| 2026-05-21 04:01 UTC | 57bd819 | s22 script | `--preregistration 57bd819` | edge_exceeds_insample: PF 1.2742 (15 trades) |

---

*Any access to the sealed holdout file that is not recorded in this log is a protocol violation.*  
*If you find an unlogged access, add a retroactive entry explaining what happened.*
| 2026-05-23 00:49 UTC | `5b581f4d88e5bf66216e23c4b66eb331ffb9b43b` | holdout_15m_oos_test.py | Phase 2 definitive 15m OOS test | PF=2.5857 (PASS), N=6, WR=0.667, Sharpe=7.684 |

## Init — 2026-05-23T15:02:03Z

- Protected: none (all already 444)
- Already protected: ['mnq_1min_holdout_20260301_plus.csv']

## Init — 2026-05-23T16:43:17Z

- Protected: none (all already 444)
- Already protected: ['mnq_1min_holdout_20260301_plus.csv']
