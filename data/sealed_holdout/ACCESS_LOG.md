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
| 2026-05-25 00:12 UTC | a97b21c | backtest script | `--preregistration a97b21c` | pending |
| 2026-05-25 00:13 UTC | a97b21c | backtest script | `--preregistration a97b21c` | pending |
| 2026-05-25 01:06 UTC | a97b21c | backtest script | `--preregistration a97b21c` | pending |
| 2026-05-25 20:14 UTC | 69972c3 | backtest script | `--preregistration 69972c3` | pending |
| 2026-05-25 21:01 UTC | 69972c3 | backtest script | `--preregistration 69972c3` | pending |
| 2026-05-25 21:46 UTC | 69972c3 | backtest script | `--preregistration 69972c3` | pending |
| 2026-05-25 21:48 UTC | 69972c3 | backtest script | `--preregistration 69972c3` | pending |
| 2026-05-25 22:46 UTC | 69972c3 | backtest script | `--preregistration 69972c3` | pending |
| 2026-05-25 23:44 UTC | 69972c3 | backtest script | `--preregistration 69972c3` | pending |
| 2026-06-07 16:19 UTC | 69972c3 | backtest script | `--preregistration 69972c3` | pending |
| 2026-06-07 16:20 UTC | 69972c3 | backtest script | `--preregistration 69972c3` | pending |
| 2026-06-07 16:23 UTC | 69972c3 | backtest script | `--preregistration 69972c3` | pending |
| 2026-06-07 16:45 UTC | 69972c3 | backtest script | `--preregistration 69972c3` | pending |

| 2026-06-09 00:35 UTC | `2e9fb90` | backtest_stat_arb_short_oos.py | Gate 2 OOS: ES/MNQ stat arb short-only. ES OOS extracted from existing in-sample file (not new download — file covered through 2026-05-20). Pre-reg: _bmad-output/preregistration_stat_arb_short_combine.md | ❌ GATE 2 FAIL — WR=45.7% PF=0.765 AvgP&L=-$7.70. Mar=54.1% OK; Apr=42.9%, May=39.5% collapse. Max DD=-$2,610 (blows combine). Direction asymmetry does not persist OOS. |

| 2026-06-09 (CST) | `e1e153f` | study_hcvwap_v3_longonly.py | HCVWAP v3 Long-Only OOS: validate v2 long-side (WR=38.3%, PF=1.87, N=60 in-sample) on MNQ 2026-03-01→2026-05-19. N=10, WR=30.0% (gate 30.2%), PF=0.847, EV=-$3.72/trade. ❌ OOS GATE FAIL. HCVWAP hypothesis exhausted. Full output: data/reports/hcvwap_v3_longonly_oos_20260609.txt |
| 2026-06-15 17:43 UTC | 138cab1b31d064555ede4c9c07503399a743893f | backtest script | `--preregistration 138cab1b31d064555ede4c9c07503399a743893f --ml-threshold 0.50` | pending |
| 2026-06-15 18:18 UTC | 138cab1b31d064555ede4c9c07503399a743893f | backtest script | `--preregistration 138cab1b31d064555ede4c9c07503399a743893f --ml-threshold 0.00` | pending |
| 2026-06-15 20:59 UTC | 138cab1 | backtest script | `--preregistration 138cab1 --ml-threshold 0.50` | pending |
| 2026-06-17 22:30 UTC | 138cab1 | backtest script | `--ml-threshold 0.50 --preregistration 138cab1` | pending |

## Init — 2026-05-23T15:02:03Z

- Protected: none (all already 444)
- Already protected: ['mnq_1min_holdout_20260301_plus.csv']

## Init — 2026-05-23T16:43:17Z

- Protected: none (all already 444)
- Already protected: ['mnq_1min_holdout_20260301_plus.csv']

## Init — 2026-06-12T17:50:43Z

- Protected: ['gc_1min_holdout_20260301_plus.csv', 'hg_1min_holdout_20260301_plus.csv', 'pl_1min_holdout_20260301_plus.csv', 'rty_1min_holdout_20260301_plus.csv', 'si_1min_holdout_20260301_plus.csv', 'ym_1min_holdout_20260301_plus.csv']
- Already protected: ['es_1min_holdout_20260301_plus.csv', 'mnq_1min_holdout_20260301_plus.csv']

## Access — 2026-07-03 (Option B impulse-aftermath Gate 0 → OOS)

| When | Prereg SHA | Script | Test | Result |
|---|---|---|---|---|
| 2026-07-03 | f2b88505dbd7c162e401a5cb9f12f554817480cb | tools/option_b_gate0_scout.py --run-oos --cell 8,follow,60 | Single-shot OOS of IS-selected cell (K8/follow/H60) on 2026-01-01→06-11 via mnq_1min_2026_ytd.csv (window CONTAINS sealed holdout 03-01→05-19; holdout CSV itself not read). Decision rule: PASS PF≥1.10 & N≥15 & ex-top3>0; MARGINAL 1.00–1.10 → PARK; FAIL <1.00 → closed. Gate 0 (IS 2025) passed all 4 sealed criteria first (PF 1.605, null 95th pct 1.569, K-neighborhood 1.318/1.493, ex-top3 +$425). | pending — recorded below after run |
| 2026-07-04 18:02 UTC | fbd7afe | backtest script | `--instrument hg --structural --ml-threshold 0.0 --start 2026-03-01 --end 2026-06-12 --preregistration fbd7afe` | ❌ FAIL — see HG Gate-1 RESULT below |

- chmod 444→644 on this log for this entry; restored to 444 after.
- **RESULT (2026-07-03): ❌ OOS FAIL.** OOS all: N=546, net PF 0.999, EV −$0.06/trade; segments: 01-01→02-28 PF 0.980 (N=475), 03-01→06-11 PF 1.106 (N=71); ex-top-3-days PF 0.893 (−$2,596). Per sealed rule (FAIL < 1.00) Option B impulse-aftermath is CLOSED on this dataset; no re-sweeps. Note: 0.999 is boundary-adjacent to MARGINAL but the rule stands. IS event rate 99/yr exploded to 546/5.4mo in the 2026 war regime — event definition is regime-unstable; 2025's follow-edge did not transfer.

## Access — 2026-07-04 (HG copper Gate-1 one-shot holdout)

- Prereg: `_bmad-output/preregistration_hg_gate1_holdout.md`, sealed commit fbd7afe (full SHA fbd7afe5f3cb9cc41c77cd667ab9718d41c51b86). Access row auto-appended by the backtest script above (18:02 UTC).
- Step 1 reproduction gate PASSED before access: IS re-run reproduced frozen N=95 / gross PF 1.4386 / +$630.00, exact trade-list match to `backtest_1year_20260625_225218.csv`.
- Step 2 integrity PASSED: working `hg_1min_2025_2026.csv` rows ≥2026-03-01 (66,368) semantically identical to sealed `hg_1min_holdout_20260301_plus.csv` (OHLCV+volume byte-exact; diffs only timestamp `T`-vs-space separator + float-repr noise in notional column).
- **RESULT (2026-07-04): ❌ GATE-1 FAIL.** Holdout 2026-03-01→06-12: N=26, GROSS PF 0.563 (−$295), net @ $4.00/RT PF **0.463** (−$399, −$15.35/trade), WR 34.6% (IS 44%), exits 13 time / 12 SL / 1 TP (IS was TP-richer), ex-top-3-days −$563.75, max net DD −$470.50, monthly net Mar +$10 / Apr −$114 / May −$229 / Jun −$66. Sensitivity @ $5.25/RT: PF 0.435. Per sealed rule (FAIL < 1.00, N≥15 valid): **HG closed as a net candidate — "structural edge did not survive the holdout at measured cost."** Not a cost death: gross itself was negative. No re-runs, no subgroup rescue. Trade list: `data/reports/backtest_1year_20260704_180517.csv` (committed).
