# Pre-Registration: yank_frontmonth_revalidation

**Generated:** 2026-09-15
**Experiment ID:** yank_frontmonth_revalidation

---

**Sealed:** 2026-09-15. Approved by Alex from `draft_preregistration_yank_frontmonth_revalidation.md` (`396033d`) without changes. This document supersedes the draft.
**Re-validates:** seal `138cab1` (`_bmad-output/preregistration_yank_sl2tp8_ml050.md`, verdict `_bmad-output/results_yank_sl2tp8_ml050.md`: "KEEP ml_threshold = 0.50")
**Evidence that prompted it:** `_bmad-output/diagnostics_tier2_contamination_20260914/results.md` (merged to main as `043a188`)

**Sealed with this commit:**

| File | SHA-256 |
|---|---|
| `tools/yank_frontmonth_revalidation.py` | `6a0f6dc6180cc3cf40775059fc46eb0398200cff781279489ab317f9976f7ecd` |
| `tests/unit/test_yank_frontmonth_revalidation.py` | `0659f7e7e36de169ad8a5ea39236faab49001366278e614d7321dd146f4343af` |

**The harness refuses to run** unless the cited commit contains this document, and its own bytes equal the copy in that commit. It appends the `ACCESS_LOG` row before any holdout-period byte is read.

---

## 1. Why

Seal `138cab1` tested whether YANK's ML meta-label filter at 0.50 adds value out of sample. It used the 2026 rows of two faithful replays, ML 0.50 and ML 0.00. The replays read `mnq_1min_2026_ytd.csv`, whose rows before the March 2026 roll are the **deferred MNQM26 contract**, not the front month:
- closes a median 223 points from raw MNQH26
- 29,157 rows against 56,100 front-month minutes in Jan–Feb
- the sealed holdout file copies these rows

**What that did to the seal's evidence:**
- 40 of its 54 out-of-sample ML trades, and +$5,411 of its +$6,917.50, fell in Jan–Feb 2026.
- A front-month replay of Jan–Feb (gates passed) gives 14 trades and −$980.50.
- Re-scoring the seal's published trade rows with that replay gives ML PF 1.09 against no-ML PF 1.06. That **fails** the seal's PF_ml ≥ 1.20.
- That composite is an approximation. It splices two replays, and Mar 1–11 is still back month.

This pre-registration replaces the approximation with a faithful re-run of seal `138cab1`'s own test on front-month bars.

## 2. What is changed: one thing, the input bars for 2026

**Code, model and every StrategyConfig field** are as sealed in `138cab1`, pinned where today's files differ (section 5).

**Bars** for the replay window 2025-05-19 → 2026-05-19:

| Segment | Trade dates (Globex session keyed to its 16:00 ET close) | Source | Change vs the seal |
|---|---|---|---|
| **S0** | 2025-05-19 → 2025-12-31 | `mnq_1min_2025.csv`, unchanged | none; in-sample for the model and not scored |
| **S1** | 2026-01-02 → 2026-02-27 | raw TradeStation MNQH26 1-minute records from `/root/mnq_historical.json` (file sha256 in `diagnostics_tier2_contamination_20260914/replays/front_month_2026_csv.sha256`) | back month replaced by front month |
| **S2** | 2026-03-02 → 2026-03-11 | raw MNQH26 records from the same JSON, stamped at or after the cutoff | back month replaced by front month |
| **S3** | 2026-03-12 → 2026-05-19 | `mnq_1min_2026_ytd.csv` rows (MNQM26, the front month after the roll), shifted by −Δ (below) | level shift only |

**Rules that apply to the segments:**
- **Roll date:** 2026-03-12, the CME roll date for H26 → M26 (the Thursday eight days before the Mar 20 expiry). The M26 segment begins with the Globex session opening 2026-03-11 18:00 ET.
- **S2 sessions:** any session whose raw records carry a label other than MNQH26 is **dropped whole**. Roll weeks interleave contracts minute by minute. This is the convention of `diagnostics_h2l2_contamination_20260914/correction_plan.md`, and no new threshold is introduced. The number dropped is reported.
- **Δ (roll continuity):** the median of (M26 close − H26 close) over minutes present in both the S2 raw records and `mnq_1min_2026_ytd.csv` during the last retained S2 session.
  - S3 prices are shifted by −Δ so the series has no roll jump; the original seal data had no jump there. How live YANK's bar history behaves across a contract change is unverified.
  - Levels do not enter YANK's gap, ATR, stop or P&L arithmetic.
  - If there is no common minute, S3 is left unshifted and this is reported.
- **Bar schema:** S1–S3 are written in the loader's schema (timestamp, open, high, low, close, volume, notional) with notional = close × volume × 20, the convention of both existing CSVs.
- **Data access:** S2 and S3 are holdout data. They are read only by the committed harness, after sealing, with an `ACCESS_LOG` entry (section 6).

## 3. Hypotheses and decision rule: inherited unchanged from seal 138cab1

Scored on trades whose entry falls in 2026 (S1 + S2 + S3), from the corrected ML 0.50 and ML 0.00 runs.

| Criterion | Rule (verbatim from `138cab1`) |
|---|---|
| Primary: OOS ML benefit | `PF_ml(2026) > PF_noML(2026)` AND `PF_ml(2026) ≥ 1.20` → keep `ml_threshold = 0.50` |
| Null outcome | otherwise → revert `ml_threshold` to **0.0** (ML disabled) |
| Minimum OOS sample | `N_2026(ml) ≥ 25`; below that → **INCONCLUSIVE** → default to ML disabled |

- PF is aggregate: gross profit ÷ gross loss over the scored trades.
- **No threshold here is new.** 1.20 and 25 are the sealed values of `138cab1`, and this document only re-applies them to corrected bars.
- **Precedence:** if this re-run's verdict differs from `138cab1`'s, this one supersedes it for the ML question, because `138cab1`'s inputs were deferred-contract bars. The `138cab1` results file gets an appended note; its hashes and files are untouched.

## 4. Power statement (fixed before the run)

**Dispersion from the corrected pre-cutoff replays** (`diagnostics_tier2_contamination_20260914/replays/C/`):

| Arm | Trades | Per-trade sd | Day-clustered sd |
|---|---|---|---|
| ML 0.50 | 42 | $391 | $400 |
| No-ML | 56 | $363 | $355 |

**Expected N_2026(ml):** about 28 (S1 gave 14; the seal's Mar–May rows were 14).

**What that N can resolve:** a one-sided mean-P&L > 0 test at α = 0.05 with 80% power detects about **$188 a trade** at N = 28 (σ = $400). The seal's own claimed out-of-sample mean was $128 a trade.

**Consequences:**
- The PF rule of section 3 is a **decision rule, not a significance test**. It is replicated as sealed, and its verdict is the operative one for `ml_threshold`.
- **Whether YANK has an edge at all is UNDERPOWERED on this window by construction.** The run reports net P&L, PF, mean, t and a day-clustered 95% interval for both arms. Neither "edge confirmed" nor "edge refuted" may be claimed from it.
- If both corrected arms are net negative over 2026, the report says so and recommends an operator **halt review**. A recommendation, not an automatic action.

## 5. Frozen configuration

- **StrategyConfig:** the `138cab1` snapshot (`strategy_config.yaml` at `138cab1`, sha256 `faef1c74…8b52`).
  - Today's YAML differs in `max_daily_loss` (−300, sealed later in `preregistration_yank_daily_breaker_2ct.md`). The harness points the engine's `STRATEGY_CONFIG_PATH` at `138cab1`'s own `strategy_config.yaml` (sha256 re-verified), which carries `max_daily_loss = -750`. No in-memory override is used.
  - The harness **asserts** that the effective StrategyConfig equals the `138cab1` snapshot field for field before running.
- **Model:** `models/xgboost/tier2_meta_labeling_model.pkl` sha256 `f58530e1…80e2`, unchanged since `31669a7` (2026-05-31), before the seal.
- **Thresholds:**
  - `tier2_threshold.json` sha256 `f1575353…04e7e`. It was rewritten on 2026-06-17, after the seal, so the G0 gate below is what proves equivalence.
  - `lr_regime_config.json` sha256 `64106250…5a24`.
- **Engine:** today's `tier2_streaming_working.py` and `backtest_tier2_1year_validation.py`, which differ from `138cab1` by about 270 lines, **provided G0 passes**. Otherwise the code at `138cab1` is used (G0′).
- **Not in scope:** YANK's live unit-file `Environment=` overrides. The replay does not read them (AGENTS.md), so this re-validates the sealed replay config, not the live unit. The ML model's training set (`doe_run_08`; 5% of its rows sit on 2025 roll splices) is also left as sealed. Retraining is a separate decision.

## 6. Procedure

1. **Done in this commit:** the harness `tools/yank_frontmonth_revalidation.py`, tested by `tests/unit/test_yank_frontmonth_revalidation.py` (11 synthetic tests; no holdout read). Given a data mode (`original` | `corrected`) and an ML threshold, it:
   - calls `backtest_tier2_1year_validation.verify_preregistration(<sha>)` and `append_access_log(<sha>, argv)`, and refuses to run otherwise;
   - runs with a git worktree at the sealing commit as its engine root, so engine writes land there, never in the live checkout. It reads the two CSVs and the raw JSON from the main checkout by explicit path, writes corrected bars and outputs to its `--out-dir`, and appends to the main checkout's `data/sealed_holdout/ACCESS_LOG.md`. The engine root's `models/xgboost` must hold the pinned files (section 5), or it refuses;
   - builds S1–S3 per section 2 (corrected mode);
   - asserts that the engine's effective StrategyConfig equals the `138cab1` snapshot (section 5), then runs `run_backtest` unchanged;
   - writes trades, the per-segment bar counts, Δ, and the dropped S2 sessions.
2. **Done in this commit:** sealed with `prereg_seal.py --name yank_frontmonth_revalidation --config <138cab1 strategy_config.yaml>`. This document and the harness are committed together, and **the run cites this commit's SHA**.
3. **G0, the reproduction gate** (original mode, both arms). The harness must reproduce `data/reports/backtest_1year_20260615_181838.csv` (ML, 82 trades) and `…_185354.csv` (no-ML, 107 trades) **row for row**.
   - If it fails, run **G0′** with the engine at `138cab1`.
   - If both fail, **STOP**: corrected runs are not interpreted, and the failure is recorded in `ACCESS_LOG`.
4. **Corrected runs** (both arms) with the engine that passed G0.
5. **Score** per sections 3–4. Report per segment (S1, S2, S3) as well as in total, and alongside the original-data result.
6. **Record** the result in `ACCESS_LOG` regardless of outcome. Commit results, the `ACCESS_LOG` update, and the appended note on `results_yank_sl2tp8_ml050.md`.

## 7. What this may not do

- Change any strategy parameter, threshold or live setting. **Acting on the verdict** (editing `ml_threshold` and restarting `trader-yank`) is a separate operator step after results are committed.
- Re-run with a different roll date, Δ rule, segment split or scoring window after seeing results. Any deviation is reported as one and does not change the verdict.
- Retrain the model, or use holdout results to tune anything.

## 8. Known limitations (acknowledged before the run)

- S0 still contains the 2025 roll-week splices. No YANK signal fell in those sessions, and S0 is not scored, but engine state entering 2026 is computed through them.
- **Bar construction differs by segment.** S0 is dollar-aggregated; S1–S3 are 1-minute time bars, as the seal's 2026 rows already were.
- The S1 bars have already been replayed pre-cutoff (the contamination check). S1 is not a fresh window, and only S2–S3 are new holdout exposure. That makes S2–S3 (about 14 ML trades) the only untouched part, far below any power to settle the ML question on its own.
- A single small-N window. The verdict is directional, as `138cab1`'s was.

---

## Configuration Snapshot

| Field | Value |
|---|---|
| sl_multiplier | 2.0 |
| tp_multiplier | 8.0 |
| entry_pct | 0.5 |
| atr_threshold | 0.5 |
| max_gap_dollars | 60.0 |
| max_gap_atr_ratio | 0.0 |
| max_hold_bars | 60 |
| max_pending_bars | 240 |
| contracts_per_trade | 5 |
| max_daily_loss | -750.0 |
| vol_regime_lookback | 120 |
| vol_regime_threshold | 0.75 |
| min_gap_atr_ratio | 0.25 |
| ml_threshold | 0.5 |
| bearish_only | True |
| h1_sweep_lookback | 6 |
| kill_zone_start_et | 09:30 |
| kill_zone_end_et | 11:00 |
| commission_per_roundtrip | 4.0 |
| enable_kill_zone_filter | True |
| m15_confirmation | True |
| tuesday_exclusion | True |
| enable_ifvg_fallback | False |
| funding_rate_filter_enabled | False |
| funding_rate_short_threshold | 0.03 |
| funding_rate_long_threshold | -0.02 |
| enable_breakeven_stop | False |
| breakeven_trigger_r | 2.0 |
| enable_trailing_stop | False |
| trailing_stop_mult | 1.5 |

---

## Holdout Data Range

- **Directory:** `data/sealed_holdout/`
- **Start date:** 2026-03-01
- **End date:** 2026-03-01

---

## Integrity Hashes

| Hash | Value |
|---|---|
| (a) YAML config SHA-256 | `faef1c740ed753449796dc948cce15f41722e0b8b60dbb99df3d82e37d1d8b52` |
| (b) strategy_core.py SHA-256 | `96e087d8154a99b31da9a7f0239da9dbf49129126cb4553da5d8543870931201` |
| (c) Git HEAD commit | `396033d03adefe510bc84aab03dbc282c276ffbb` |

*Hash (a): SHA-256 of `strategy_config.yaml` as committed in `138cab1` (`git show 138cab1:strategy_config.yaml`). It equals that seal's own hash (a); the harness re-verifies it at run time.*
*Hash (b): SHA-256 of `/root/Silver-Bullet-ML-BMAD/src/research/strategy_core.py` source bytes.*
*Hash (c): `git rev-parse HEAD` at seal time — commit this document to make it tamper-evident.*
