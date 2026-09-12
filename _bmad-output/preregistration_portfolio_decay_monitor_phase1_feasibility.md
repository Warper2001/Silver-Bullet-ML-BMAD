# Pre-Registration: PDM-PHASE1 — Portfolio Decay-Monitor Feasibility Audit

**Registered:** 2026-09-12
**Status:** SEALED at commit time. Append-only amendments.
**Parent document:** `_bmad-output/innovation-strategy-2026-09-12.md` (Option A, Phase 1)

---

## 1. The gap this fills

The innovation-strategy session on 2026-09-12 recommended, in place of a full "ingest every
live and historical source" probability engine (rejected — it reproduces the structure of the
9-run DOE sweep already shown indistinguishable from luck by SPA/DSR), an internal-data-only
portfolio regime/decay monitor. That document's Phase 1 read:

> "Retrospectively backtest a minimal regime/decay signal against the already-known
> Thursday-short 24-day dead period and the 2026-05-20 methodology-reset OOS results, purely
> as a feasibility check... Deliverable: a feasibility memo, not a pre-registration."

This document exists because that plan, taken at face value, is wrong on both cited cases and
would either produce a false positive or misuse voided evidence. Writing that down loosely
in a strategy brainstorm is fine; running it without correction is not. §2 states the
correction. The rest of this document pre-registers what Phase 1 can *legitimately* test,
before any indicator is computed, so the memo cannot quietly redefine "detected" after seeing
the numbers — the same discipline this program already applies to strategy tests, applied here
to a monitoring tool instead.

## 2. Correction to the parent document — neither cited case is usable as stated

**Thursday-short's 24-day gap is not a regime-decay event.** Per
`project_thursday_short_forward_negative_20260910.md`: the bot was dead because `main()`
swallowed a fatal exception and returned exit 0, which `Restart=on-failure` reads as a clean
shutdown — a **process-liveness bug**, fixed by adding `trader-thursday-short` to
`tools/combine_ops_healthcheck.py`. A statistical regime/decay monitor watching price or P&L
features has no mechanism to detect "the process is not running"; that is a heartbeat
problem, already fixed by a heartbeat solution. Citing it as a decay-monitor target would be
attributing a catch to the wrong instrument.

**The reconstructed Thursday-short ledger cannot be cited as evidence at all, in either
direction.** The same memory: *"never cite the pre-2026-09-17 Thursdays as evidence either
way"* — the 7-Thursday, PF-0.547 record was explicitly voided, accrual restarted at N=0 on
2026-09-17. Using it to validate a decay indicator's sensitivity would violate that voiding on
its first use.

**The 2026-05-20 methodology reset is a no-edge-ever-existed finding, not a decay event.**
Per `project_methodology_reset_20260520.md`, the full 1-year OOS backtest showed the base
Tier2 pattern had Sharpe ≈ 0 from the start — there is no "was working, then drifted" moment
for a decay monitor to have flagged. It is evidence for the power-gate-before-testing policy,
not a labeled decay incident.

**Conclusion:** this operation has **zero retrospective incidents that are simultaneously
(a) a genuine statistical regime/decay event, (b) live-traded (not backtest-only), and
(c) not already voided as evidence.** Phase 1 is redefined below to not depend on one.

## 3. What Phase 1 actually is

Phase 1 is a **code and data audit plus one candidate-indicator computability check** — not a
predictive-power test, and not a backtest. It answers two questions only:

1. What do the ML/regime artifacts already in this codebase do, exactly, and do their
   internal assumptions hold for a 4-strategy portfolio?
2. Can a single, named decay indicator be computed cleanly from data already committed to
   `data/trades.db`, with no missing-data or lookahead problems — without yet claiming it
   detects anything?

No PASS/FAIL/detection-rate claim is authorized from Phase 1. That is reserved for a
separate, later pre-registration covering a prospective shadow period (parent doc's Phase 2),
which will define its own power gate against genuinely new data.

## 4. Audit scope — named artifacts, frozen now

| Artifact | Path | What it actually is |
|---|---|---|
| YANK meta-labeling model | `models/xgboost/tier2_meta_labeling_model.pkl` + `models/xgboost/tier2_threshold.json` | XGBoost entry filter, threshold 0.5, validated 2026-05-16 against `val_tail_months` 2025-11/2025-12, gate criteria `val_pf_min=1.2` |
| LR channel regime filter | `models/xgboost/lr_regime_config.json` + `src/ml/regime_detection/lr_channel_detector.py` | Counter-trend LR-channel filter (fast_len=390, slow_len=1950), same 2026-05-16 validation, gates YANK's LR filter live |
| HMM regime detector | `src/ml/regime_detection/hmm_detector.py`, `features.py`, `models.py` | Built, present in the repo; not confirmed wired to any live bot — audit must establish this, not assume it |
| Async-pipeline drift detector | `src/ml/drift_detector.py` (`DriftDetector`, used by `src/ml/pipeline.py`) | A **complete, already-coded** win-rate-drift monitor: flags drift at >10 pts below expected win rate, recovery at ≤5 pts, recommends a halt after 3 consecutive drift cycles. **Not wired to any live bot** — `ml/pipeline.py` is confirmed dormant per `AGENTS.md`. Its three constants (0.10 / 0.05 / 3 cycles) carry no cited derivation in the module docstring or code — they read as hand-set, exactly what AGENTS.md's monitor policy forbids putting in a sealed doc without a sweep citation. |

The existence of `DriftDetector` changes the shape of this initiative: it is not "build a
decay monitor from scratch," it is "determine whether an **already-built** decay monitor,
currently unused and carrying unvalidated thresholds, is fit to extend to a 4-strategy
portfolio, or whether its win-rate-only design is the wrong shape for edges as sparse as
YANK's (0.125 trades/day)."

## 5. The one candidate indicator — named now, not after looking

Per AGENTS.md's "change one parameter at a time" policy, Phase 1 prototypes exactly **one**
indicator, not a menu:

> **Rolling realized win rate per live strategy, trailing 20 trades or all available trades if
> fewer, computed from `data/trades.db` filtered to `write_mode = 'realtime'` with
> `timestamp` parsed as `format="ISO8601"`, compared against that strategy's own sealed
> validation win rate** (YANK: per `tier2_threshold.json`'s validation window; MIM-NB, GAP-1,
> Kraken Thursday-short: per their respective sealed pre-registration documents).

This is chosen because it is the same statistic `DriftDetector` already uses — testing it
first is a direct feasibility check of the existing artifact, not a new design. No other
candidate indicator (PF-based, Sharpe-based, regime-conditional) is in scope for Phase 1;
if the win-rate indicator proves unworkable (§6), a *new* pre-registration names the next
candidate rather than silently trying several here.

## 6. Phase 1 completion criteria — fixed before computation

The feasibility memo is complete, and only complete, when it states, for each of the four
live strategies:

- Whether ≥1 completed live trade at `write_mode='realtime'` exists to compute the indicator
  at all (Kraken Thursday-short: only trades on or after 2026-09-17 count, per the
  restart-void rule in `project_thursday_short_forward_negative_20260910.md`).
- Whether that strategy's sealed validation win rate is unambiguously extractable from an
  existing pre-registration or promotion artifact (cite the document and figure — do not
  compute a new one).
- Whether the indicator, once computed, is well-defined (no divide-by-zero, no undefined
  comparison window) — a computability check, not a report of whether it looks predictive.
- An explicit statement of `DriftDetector`'s three constants' provenance (found derivation
  or found none) — this is a factual lookup, not a judgment call.

Nothing else. In particular the memo **may not** report an opinion on whether the indicator
"seems to work," "looks promising," or "would have caught" any incident — §2 already
forecloses the only candidates that framing could point to.

## 7. What is explicitly out of scope for Phase 1

1. No new backtest run, no `data/sealed_holdout/` access (no committed pre-registration or
   ACCESS_LOG entry exists for this activity, and none is being sought here).
2. No hard-coded threshold set anywhere, on `DriftDetector` or otherwise — until a sweep with
   a cited artifact exists, per policy, monitors observe and report only.
3. No edit to any file a live `trader-*` unit imports (`strategy_core.py`, `auth_v3.py`,
   `models.py`, `trade_db.py`, execution modules, or anything under `src/ml/regime_detection/`
   or `src/ml/`) — this is a read-only audit plus a standalone script writing to
   `_bmad-output/`, never to a path a live bot loads.
4. No claim about whether the four-strategy portfolio "needs" this monitor — that is a
   decision for after Phase 1's factual audit, not before it.
5. No second candidate indicator, no parameter variant of the one named in §5 — one knob,
   this phase, per policy.

## 8. What this pre-registration authorizes

Only: producing the feasibility memo described in §3–§6, as a new file under `_bmad-output/`
or `docs/reports/`, using data already committed to `data/trades.db` and artifacts already
present in the repo at the commit below. It does **not** authorize Phase 2 (shadow deployment
of any monitor), does not authorize wiring `DriftDetector` or any regime module to a live bot,
and does not authorize a config or threshold change anywhere. Phase 2 requires its own
pre-registration and its own power gate, per the parent document's roadmap and per AGENTS.md's
standing policy ("run a power gate before any new strategy test").

## 9. Values fixed at seal time

| Item | Value |
|---|---|
| git HEAD at seal | `5f179cb6ef203dae3528e0833b8b7aef0ef65682` |
| Live strategies in scope | YANK, MIM-NB, GAP-1, Kraken Thursday-short (post-restart only) |
| Candidate indicator | Rolling realized win rate, trailing 20 trades, vs. sealed validation win rate |
| Retrospective test cases rejected (§2) | Thursday-short 24-day gap (ops bug, not decay); Thursday-short pre-09-17 ledger (voided evidence); 2026-05-20 methodology reset (no-edge finding, not a decay event) |
| `DriftDetector` constants found in code, provenance TBD by audit | `DRIFT_THRESHOLD=0.10`, `RECOVERY_THRESHOLD=0.05`, `MAX_CONSECUTIVE_CYCLES=3` |
