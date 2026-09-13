# Feasibility Memo: PDM-PHASE1 — Portfolio Decay-Monitor Feasibility Audit

**Date:** 2026-09-12
**Pre-registration:** `_bmad-output/preregistration_portfolio_decay_monitor_phase1_feasibility.md` (sealed commit `d223e0a`)
**Git HEAD at audit:** `2e47ba19160fe864956039bec55702933df4b3c6`
**Status:** Phase 1 complete per §3–§6 of the pre-registration. **No detection-power or PASS/FAIL claim is made here** — that is out of scope by §3 and reserved for a separate Phase 2 pre-registration with its own power gate.

---

## 1. Bottom line

The candidate indicator named in §5 of the pre-registration — rolling live win rate vs. sealed
validation win rate — **is not cleanly computable, as specified, for 3 of the 4 live
strategies**, and only partially for the 4th. The live-side of the comparison (rolling
realized win rate from `data/trades.db`) computes cleanly wherever trades exist. The
*baseline* side does not exist as a citable point figure for YANK, MIM-NB, or Kraken
Thursday-short, because none of their sealed decision rules are win-rate-based — they are
PF-based (YANK, MIM-NB) or Sharpe-based (Thursday-short). This is a real design mismatch
between `DriftDetector`'s win-rate-difference API and how this portfolio's edges are actually
validated, not a data problem. Kraken Thursday-short additionally has **zero eligible live
trades** to compute anything from — its restart accrual doesn't begin until 2026-09-17.

## 2. Artifact audit (§4)

| Artifact | Path | Live-wired? | Finding |
|---|---|---|---|
| YANK meta-labeling model | `models/xgboost/tier2_meta_labeling_model.pkl`, `tier2_threshold.json` | **Yes** — imported by YANK's live entry path | Threshold 0.5, validated 2026-05-16 against `val_tail_months` 2025-11/2025-12, `gate_criteria.val_pf_min=1.2`. No win-rate figure anywhere in the artifact or its promotion doc. |
| LR channel regime filter | `models/xgboost/lr_regime_config.json`, `src/ml/regime_detection/lr_channel_detector.py` | **Yes** — `from src.ml.regime_detection.lr_channel_detector import LRChannelRegimeDetector` appears in both `src/research/yank_streaming_working.py:840` and `src/research/tier2_streaming_working.py:1012` | Confirms AGENTS.md's note that this file's presence switches the LR filter on. Same 2026-05-16 validation window and gate criteria as above; also PF-gated (`val_pf_min=1.2`), not win-rate-gated. |
| HMM regime detector | `src/ml/regime_detection/hmm_detector.py`, `features.py`, `models.py` | **No** — not imported by any currently-active `trader-*` entry file. It is imported only by the dormant async pipeline (`src/ml/pipeline.py`, `hybrid_pipeline.py`, `regime_aware_inference.py`, `regime_aware_model_selector.py`) and by two non-active legacy files (`src/research/btc_combine_streaming.py`, `src/research/s26_crypto_streaming_working.py` — neither is any current systemd unit's `ExecStart`; the live S26 units run `btc_s26_combine.py` and `s26_soft_fvg_streaming.py` instead). | The pre-registration's "not confirmed wired" is now confirmed: **not wired, to anything active.** |
| Async-pipeline drift detector | `src/ml/drift_detector.py` (`DriftDetector`) | **No** — only referenced from `src/ml/drift_detection/__init__.py` and the dormant `src/ml/pipeline.py`; no active `trader-*` file imports it. | Fully built: `check_drift(actual_win_rate, expected_win_rate)`, 10-pt drift / 5-pt recovery / 3-cycle halt. **Provenance of its three constants: none found.** Sole history is commit `ca95165` (2026-03-16, "Implement Story 3.8: Detect Model Drift"), which states them as acceptance-criteria defaults with no cited sweep, backtest, or historical distribution — no story spec doc exists in the repo beyond the commit message itself. These constants predate this program's pre-registration/power-gate policy (adopted after the 2026-05-20 methodology reset) and would not clear it today. |

## 3. Candidate indicator — per-strategy computability (§5, §6)

Computed from `data/trades.db`, filtered `write_mode='realtime'`, `timestamp` parsed
`format="ISO8601"`, at git HEAD `2e47ba1`:

| Strategy | `trader_id` | Eligible live trades (N) | Realized win rate | Sealed baseline win rate | Baseline extractable? |
|---|---|---|---|---|---|
| YANK | `trader-yank` | 5 (2026-07-13 → 2026-08-17) | 60.0% (3/5) | — | **No.** Every sealed YANK document (`preregistration_yank_sl2tp8_ml050.md`, `tier2_threshold.json`) expresses its decision rule in PF (`PF_ml(2026) ≥ 1.20`, forward stop `PF < 0.90` at N≥20). No win-rate figure exists to compare against. |
| MIM-NB | `trader-mim-nb` | 23 (2026-06-24 → 2026-09-04) | 43.5% (10/23) | — | **No.** `preregistration_mim_nb_catstop_250.md`'s decision rule is OOS PF 1.30. The only WR table in any MIM-NB-family document (`preregistration_mim_x2_powered.md`, WR 47.5–48.5%) belongs to the **X2-powered variant, which FAILED and was never deployed** — using it as "the" MIM-NB baseline would silently substitute a different, dead strategy's numbers. |
| GAP-1 | `trader-gap-fade` | 26 (2026-06-25 → 2026-09-10) | 50.0% (13/26) | 55% (Gate-0 threshold) | **Partial.** `preregistration_gap_fade_panic_open.md` §"Secondary checks" states "Win rate >= 55%" — but this is a pass/fail gate from the 2025 in-sample Gate-0 check, not a point baseline from GAP-1's actual OOS/live decision rule (which is itself PF-based: scale at PF>1.20, continue at 1.00–1.20, stop at <1.00 for N≥30). Using 55% as `expected_win_rate` in `DriftDetector.check_drift()` would compare a live rolling rate against an in-sample gate threshold from a different evaluation stage — not what the API is designed to do. |
| Kraken Thursday-short | *(not in `trades.db` — own ledger at `data/thursday_ts/trades.csv`)* | **0 eligible** | n/a | — | **No, and moot.** All 14 rows (7 Thursdays × 2 legs: MBT/MET) in `data/thursday_ts/trades.csv` predate the 2026-09-17 restart and are voided per `project_thursday_short_forward_negative_20260910.md` ("never cite the pre-2026-09-17 Thursdays as evidence either way"). The restart accrual's first countable Thursday is 2026-09-17 — five days after this audit. Its sealed decision rule is Sharpe-based (PASS if Sharpe > 0.80 at N≥30) in any case, not win-rate-based. |

**Additional computability note:** the pre-registration's window rule ("trailing 20 trades or
all available if fewer") is exercised as "all available" for all three strategies with any
data (N=5, 23, 5 < 20 in every case but GAP-1 at N=26, which is the only strategy currently
past the 20-trade trailing window). No divide-by-zero or undefined-window case occurred on the
*numerator* side for any of the three. The blocking problem throughout is the denominator.

## 4. Answer to Phase 1's two questions (§3)

1. **What do the artifacts do, and do their assumptions hold for a 4-strategy portfolio?**
   Two of four regime/ML artifacts are live (meta-labeling model, LR channel filter); two are
   fully built but wired to nothing (HMM detector, drift detector). The one that most directly
   resembles "the decay monitor" this initiative is meant to evaluate (`DriftDetector`) has
   unvalidated constants and an API shape (win-rate differencing) that does not match how any
   of the four live strategies are actually gated (PF or Sharpe). It does not hold as-is.

2. **Can the named indicator be computed cleanly from data already committed, with no
   lookahead or missing-data problem?** The live-side statistic computes cleanly wherever
   trades exist (3 of 4 strategies, N=5/23/26). The comparison it's supposed to be measured
   against does not exist in sealed form for any of the four strategies in the way
   `DriftDetector` expects it (a single point `expected_win_rate`) — YANK and MIM-NB have none
   at all, GAP-1 has a differently-sourced gate threshold, and Thursday-short currently has no
   trades to measure. **Not cleanly computable as specified.**

## 5. What this does and does not authorize going forward

This memo makes no recommendation to build, deploy, or threshold anything — that determination,
and any second candidate indicator, requires its own pre-registration per §7.5 and §8 of the
parent pre-registration. For the record, the finding in §4 suggests (not proposes) that a
future candidate indicator would need to be expressed in each strategy's **own native metric**
(PF for YANK/MIM-NB, Sharpe for Thursday-short, PF for GAP-1's live rule) rather than a single
win-rate statistic borrowed from `DriftDetector`'s existing but never-validated design — but
that is an observation for whoever writes the next pre-registration, not a decision made here.

## 6. Values fixed at memo time

| Item | Value |
|---|---|
| Git HEAD | `2e47ba19160fe864956039bec55702933df4b3c6` |
| Pre-registration commit | `d223e0a` |
| `data/trades.db` snapshot time | 2026-09-12 (session query) |
| YANK realtime | N=5, WR 60.0%, last trade 2026-08-17 |
| MIM-NB realtime | N=23, WR 43.5%, last trade 2026-09-04 |
| GAP-1 realtime | N=26, WR 50.0%, last trade 2026-09-10 |
| Kraken Thursday-short | N=0 eligible (restart accrual begins 2026-09-17) |
| `DriftDetector` constants, provenance | Not found — implementation commit `ca95165` (2026-03-16), no cited derivation |
| HMM detector, live-wiring | Confirmed not wired to any active `trader-*` unit |
