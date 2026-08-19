# Pre-Registration: YANK Compressed-Cascade Timeframe Test

**Status:** SEALED — frozen at commit time. No modifications after commit.
**Date:** 2026-08-19
**Authored by:** party-mode round-table (Winston / Mary / Amelia / Dr. Quinn / Grumbal / Victor), at Alex's direction
**Lineage:** informally continues the Program C Phase-1-falsification method used in
`preregistration_s12_random_entry.md` and `preregistration_s13_timeframe.md`. Does **not**
claim an "S##" slot in that numbered sequence — S27/S28/S29 are already reserved for the
Tier2 track (IFVG, news filter, ES/MNQ divergence) per `project_research_queue.md`. Alex to
assign a number if this should join that registry.
**Touches:** nothing live. No modification to `strategy_config.yaml`, `strategy_core.py`,
`yank_streaming_working.py`, or `tier2_streaming_working.py`. `trader-yank` is not restarted
by this document or anything it authorizes.

---

## 0. What question this actually answers

Alex asked to research YANK "on a shorter time frame." The room split that into three
readings (see §0.1) and Alex picked: **test whether the H1→M15→M1 structure cascade
survives being compressed one rung finer, as a genuinely new pre-registered strategy** —
not a parameter tweak to the live bot, and not a re-litigation of whether the already-shipped
gap-ceiling fix (`preregistration_yank_gap_ceiling_denomination.md`, PF `max_gap_atr_ratio`
0.25→0.426) restored trade frequency, which is a separate, already-answered question.

### 0.1 Why this isn't just S12 again, and why it still has to clear S12's bar

S12 tested the *original* single-resolution 1-minute pattern (H1 sweep + M1 FVG, no M15
CHoCH) against a random-entry null and got **AMBIGUOUS = PIVOT** (PF 0.937 at the 70th
percentile of 100 random-entry sims). That result is about a different, simpler
architecture than what's live today. The current YANK/S25 cascade — H1 sweep → M15 CHoCH
confirmation → M1 FVG entry — was never itself tested against a random-entry null; it was
validated via the OOS holdout (Phase 2, PF=2.586, N=6, "weak, N=6 caution" per
`phase1_verdict_20260523.md` lineage) and via live trading, not via S12's method.

So compressing the cascade one rung finer (H1→M15 sweep, M15→M5 CHoCH) is a **new
hypothesis**, not a repeat of a known-failed one. But it inherits the exact risk S12 exists
to catch — finer-resolution structure detection fires more often and looks more
"significant" purely from higher event count, not real signal. It must clear the same bar
S12 set, not get a pass because the current architecture happens to already be live.

### 0.2 The hard constraint the room found: there is no finer FVG

`data/processed/` (2025 full-year, 2026 YTD) has **1-minute OHLCV as the finest available
resolution** — no tick data, no sub-minute bars, nothing under `data/` matches `*tick*`.
`detect_fvg` (`strategy_core.py:323`) already runs at M1. **The FVG entry leg cannot be
compressed further** — it's already at the data floor. "Shorter timeframe" can only mean
compressing the *structure* legs (sweep, CHoCH), not the entry trigger.

---

## 1. The candidate, precisely

One knob, following S13's precedent exactly: **candlestick resolution of the structure
legs is the only manipulated variable.** Every `StrategyConfig` bar-count constant stays
numerically unchanged — S13 did this for `max_hold_bars=60` (300 min at 5m, 900 min at
15m) and "accepted" the wall-clock deltas rather than re-deriving them; this experiment
does the same for the structure-lookback constants.

| Leg | Live YANK (S25) | Candidate | Same-numbered constant, new meaning |
|---|---|---|---|
| Sweep detection | `resample_to_h1` (1h bars), `h1_sweep_lookback=6` | resample to **15min** bars, lookback stays `=6` | 6 H1 bars (6 hr) → 6 M15 bars (1.5 hr) |
| CHoCH confirmation | resample to 15min, `SWING_R=2`, `CHOCH_ATR_MULT=0.3` | resample to **5min**, same `SWING_R=2`, `CHOCH_ATR_MULT=0.3` | 2-bar swing pivot on 15m (30 min either side) → on 5m (10 min either side) |
| FVG entry | M1, `min_gap_atr_ratio=0.426` | **unchanged — M1, 0.426** | data floor, cannot compress |

All other `StrategyConfig` fields (sl_multiplier=5.0, tp_multiplier=6.0, entry_pct=0.5,
contracts_per_trade=5, max_daily_loss=-750.0, vol_regime_lookback/threshold, bearish_only,
tuesday_exclusion, ml_threshold=0.0) are **frozen at current YANK values, unmodified.**

**Engineering constraint the room is pre-committing to:** `resample_to_h1` and the M15
resample inside `_update_m15_choch`-equivalent logic in `strategy_core.py` are **shared
code that both YANK and the still-live Tier2 architecture import.** This experiment must
be built as a **new, isolated function** (`resample_to_timeframe(bars, rule)`, parameterized)
called only from a new offline backtest script — never by editing `resample_to_h1` or the
hardcoded `"15min"` resample in place. Zero edits to the function bodies deployed bots call.
Winston's rule from this session: a shared-engine change for one bot's experiment is not
allowed to become a live blast-radius risk to the other.

---

## 2. Hypothesis

**H₀ (null):** The compressed cascade's PF over the 2025 training window is consistent
with a direction-matched random-entry control run under the same compressed structure
gates and the same exit rules (i.e., no real signal above what the gate frequency alone
would produce).

**H₁ (alternative):** The compressed cascade's PF lies above the 90th percentile of its
own random-entry null distribution, indicating real pattern survival at the finer
resolution — not just more trades from looser gates.

---

## 3. Phase 1 — Random-null falsification test (generalizes `s12_random_entry_control.py`)

- **N = 100** simulations, seeds 0–99, `np.random.default_rng(seed)` — same protocol as S12
- **Entry gates shared with the candidate:** M15 sweep active (within last 6 M15 bars),
  M5 CHoCH confirmed, vol regime filter passes, daily circuit-breaker not tripped, not Tuesday
- **Entry decision:** uniform random coin flip, `p_enter` calibrated to the *candidate's own*
  measured per-bar entry rate — **not S12's original 129-trade/2025 rate**, since the
  compressed cascade will fire at a different frequency and reusing S12's calibration would
  silently smuggle in a second manipulated variable
- **Data:** `data/processed/mnq_1min_2025.csv` (training window only). Sealed holdout **not**
  touched in Phase 1.
- **Decision rule (verbatim structure from S12):**

| Condition | Verdict |
|---|---|
| Candidate PF < median of its own random-entry PFs | **PIVOT** — do not proceed to Phase 2 |
| Candidate PF > 90th percentile of its own random-entry PFs | **PROCEED to Phase 2** |
| Candidate PF in 50th–90th percentile | **AMBIGUOUS = TREATED AS FAIL = PIVOT** |

---

## 4. Phase 2 — Prospective OOS (only if Phase 1 passes)

- **No historical holdout file is reused.** `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`
  and the 2026-03-01→05-19 window have already informed the S25 live decision rule and the
  gap-ceiling prereg; reusing either here for a *different, untested* hypothesis is exactly
  the kind of restrict-to-favorable-subset-adjacent move this project's own retro rules out
  (see `feedback_iteration_loop_pattern`). Instead: a **fresh prospective window, starting
  from this document's seal commit date forward**, collected live/replayed as it accrues —
  same convention as the S26 KZ subgroup analysis and the `ml_proba` ordinal prereg.
- **Threshold: reuse Phase 1's own p90-of-null bar**, not a new hand-picked PF constant —
  same number that gated entry into Phase 2, applied again to the prospective data. This is
  the project's "derive, don't assert" rule applied to the decision threshold itself.
- **Minimum N before any verdict: 30 trades**, matching this project's standing floor for a
  prospective read (MIM-NB Track A, `ml_proba` prereg, S26 all used N≥30 as the trade-count
  gate before treating a result as more than noise).
- If Phase 1 fails, Phase 2 does not run. No exceptions, no "close enough."

---

## 5. Stopping rule

- Phase 1: N=100 fixed, no re-running with adjusted seeds or calibration after seeing results.
- No cherry-picking which compressed rung to test after seeing partial results — the M15
  sweep / M5 CHoCH mapping in §1 is fixed before any backtest runs.
- No subsetting the 2025 training data to a favorable stretch.
- If Phase 2's prospective window disagrees with Phase 1, Phase 2 wins — same precedence
  the project has used every other time a historical read and a live read conflicted.

---

## 6. Cost disclosure

- Phase 1 is cheap: reuses existing data, no new collection, one new isolated resample
  function, generalization of an existing script. Order of a day, not weeks.
- Phase 2 is **not fast** — it is a genuinely fresh prospective clock, N≥30 trades at
  YANK's current (post-gap-ceiling-fix) trade frequency. Recent live cadence has been thin
  enough (structural silence through most of June/July) that Mary flags this could run
  8–12+ weeks depending on how much the 0.426 fix actually restored frequency. This is not
  a "quick shorter-timeframe check" — it is a multi-month commitment if Phase 1 passes.
- If Phase 1 fails (PIVOT), total cost is the day spent on Phase 1 and nothing further.

---

## Freeze SHA

*(To be filled in by dev agent after `git commit` of this file)*

Commit SHA: `__FILL_AFTER_COMMIT__`
