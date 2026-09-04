# Pre-registration — Option 1b: YANK exit-horizon extension

**Date:** 2026-09-04
**Parent plan:** `_bmad-output/research_plan_post_r3_options_20260904.md`, Option 1.
**Sealed before running:** committed before the sweep is executed against any P&L number.

## Background

YANK's live config (`strategy_config.yaml`, loaded by `src/research/yank_streaming_working.py::_build_strategy_config`; engine `src/research/strategy_core.py`, harness `backtest_tier2_1year_validation.py`) uses `max_hold_bars: 60` — a fixed 60-bar time exit from fill, sealed in `preregistration_yank_sl2tp8_ml050.md`. This is Option 1a's same hypothesis (MNQ headroom scales with hold time) applied to a different strategy with a different exit mechanism (bar-count time-stop, not a wall-clock RTH cutoff) — hence its own seal, per one-knob-at-a-time.

## Data boundary — stays out of the sealed holdout

`HOLDOUT_CUTOFF = 2026-03-01` (`backtest_tier2_1year_validation.py`). This is an **exploratory dev-window test**, not a holdout-gated validation — it deliberately does **not** touch data on/after the cutoff, and needs no `--preregistration`-gated holdout access. Date range: full `mnq_1min_2025.csv` (≥ 2025-05-19) through `mnq_1min_2026_ytd.csv` truncated at `end=2026-03-01`. If this clears Gate 0, a *separate* seal governs any holdout look.

## Frozen for this seal

- Entry logic, `sl_multiplier=2.0`, `tp_multiplier=8.0`, `ml_threshold=0.50` (YANK's sealed values) — unchanged from `strategy_config.yaml`.
- **The one swept knob: `max_hold_bars`**, applied via `run_backtest(..., config_overrides={"max_hold_bars": N})` — same engine (`Tier2StreamingTrader`/`strategy_core.py`), same mocked-broker replay path `run_backtest` already uses for every other YANK backtest in this project.
- Grid, fixed here: **{60 (baseline/current), 120, 180, 240}** — 1×/2×/3×/4× the sealed value.
- Null design (necessarily different from Option 1a's per-trade random draw: `max_hold_bars` is a single config value for the whole run, not a per-trade choice, so it cannot be randomized trade-by-trade without altering the trader's internals). Null: **N_null full independent backtest runs**, each with `max_hold_bars` drawn once (uniformly, integer) from **[30, 300]** — a wider, still-plausible range bracketing the grid — before that run's PF is computed. `N_null` and its exact value are fixed by a timing pilot run *before* this seal is finalized (recorded below) so the null is neither so small it's uninformative nor an unbounded compute commitment; whatever N_null is set to here is not increased after seeing any null-distribution result.

## Gate 0 — sealed before any run

1. Best-grid-cell PF > baseline (60-bar) PF, same underlying signal set.
2. Best-grid-cell PF > the null distribution's 95th percentile.
3. N ≥ 60 (inherited GAP-1/YANK floor convention) at the best cell.
4. Ex-top-3-days PF at the best cell > ex-top-3-days PF at baseline.
5. Reported, not a hard gate alone: monotonicity across the 4-cell grid — a lone interior spike is flagged, not trusted, even if it clears 1–4 (Option 1a precedent).

**Verdict rule:** PASS only if 1–4 all hold.

## Stopping rule

One run. FAIL closes this seal, no re-sweep. PASS on the dev window only authorizes a *separately pre-registered* single-shot holdout check before any live config change — YANK is live capital, so nothing here changes `strategy_config.yaml` regardless of outcome.

---

## Timing pilot (recorded before finalizing N_null)

A single full-sample run (`max_hold_bars=60`, `ml_threshold=0.50`, dev window) was still executing after **20+ minutes of CPU time** on this box (single-threaded) — `Tier2StreamingTrader`'s bar-by-bar async replay (H1/M15 resample, CHoCH state machine, liquidity-sweep detection every bar) is far more expensive per run than GAP-1's vectorized pandas backtest. At that rate, the originally-planned 200-draw null (Option 1a's design) would cost 50+ CPU-hours — not proceeding with that design.

**Revised, sealed before any grid/null result is observed (this is a compute-budget decision, not an outcome-informed one):**

- **N_null = 16** (down from 200), each draw a fully independent backtest run with `max_hold_bars` drawn uniformly from the integer range **[30, 300]**.
- Parallelized across the box's 4 cores (`ProcessPoolExecutor(max_workers=4)`), each worker loading its own bars and running its own `run_backtest` — 4 grid cells (60/120/180/240) + 16 null draws = 20 runs total, ≈5 sequential batches of 4.
- **Disclosed limitation:** N=16 gives a much coarser null-percentile estimate than Option 1a's N=200 — the p95 threshold below is a rougher line, not a precise one. Reported as such in the verdict; if the result is close to that line, treat it as inconclusive rather than a clean PASS/FAIL, don't round in either direction.

