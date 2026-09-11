# Pre-registration — Option 1: GAP-1 time-stop horizon sweep

**Date:** 2026-09-04
**Parent plan:** `_bmad-output/research_plan_post_r3_options_20260904.md`, Option 1.
**Sealed before running:** this document is committed before the sweep script is executed against any P&L number.

## Background

`backtest_gap_fade.py` (frozen params, `preregistration_gap_fade_panic_open.md`) fades large overnight MNQ gaps at RTH open, target = prior RTH close, stop = 2× gap, and forces exit at market on the **13:00 ET time-stop** if neither is hit first — a hard ~3.5-hour hold cap regardless of whether the reversion is still developing. GAP-1 backtested STRONG_EDGE (PF 1.761, N=117 per prior result) but is not live (`project_gap_fade_projectx_promotion.md`: NO-GO pending more live data) — so there is no live P&L to compare against. The baseline for this test is the frozen backtest itself, re-run identically inside this sweep, isolating exactly one variable.

## Hypothesis

The 13:00 ET cutoff is an arbitrary bound, not a discovered optimum (it was never swept — chosen for the original pre-registration's own reasons). Per the project's edge-headroom finding, MNQ headroom scales steeply with hold time; if GAP-1's true reversion often isn't finished by 13:00, a later time-stop should raise PF without changing anything else about the strategy (same entry, same gap filter, same stop, same day-of-week exclusion).

## Frozen for this seal — do not change mid-run

- Entry, gap filter (`GAP_MIN_PCT=0.005`), stop (`STOP_MULT=2.0`), Friday exclusion, data source (`mnq_1min_2025.csv` + `mnq_1min_2026_ytd.csv`), all **unchanged** from `backtest_gap_fade.py`.
- **The one swept knob: `TIME_STOP_HOUR`.**
- Grid, fixed here, never extended after seeing results: **{13 (baseline/current), 14, 15, 16 (= no time-stop; runs to RTH close)}**.
- Random-hour null: for each qualifying trade, independently draw `TIME_STOP_HOUR` uniformly from the same 4-value grid (not from the trade's real day) and re-simulate. 200 draws of the full-sample null distribution.

## Gate 0 — sealed before any run

1. **Primary bar:** PF at the winning cell > PF at hour=13 (the baseline cell), on identical N (same trade set — the time-stop only changes *when* a trade exits, not *whether* one is taken, so N is invariant across the grid by construction).
2. **Beats the null:** the winning cell's PF must exceed the 95th percentile of the 200-draw random-hour null distribution. This is the direct analog of the TSMOM-1 mechanism check — guards against "later exits just harvest MNQ's average intraday drift," not a genuine reversion-completion effect.
3. **N floor:** inherited from the base strategy's own gate, `GATE_N_MIN = 60`.
4. **Fat-day robustness:** ex-top-3-days PF at the winning cell must still exceed ex-top-3-days PF at the baseline cell (a later exit that only wins because of 3 outlier days is not a robust exit change).
5. **Monotonicity / no lone spike:** PF across {13,14,15,16} should move in a consistent direction, not spike at one interior value with the neighbors flat or reversed — a lone spike among 4 tested cells at this N is exactly the pattern the order-flow ballpark check just caught and rejected.

**Verdict rule:** PASS only if 1–4 all hold. Rule 5 is reported, not a hard gate by itself, but a lone-spike winner that clears 1–4 gets flagged, not trusted, per the project's "one knob, not one lucky cell" discipline.

## Stopping rule

One run. If the grid PASSES, a *separately pre-registered* second cycle may test a trailing-stop variant (a different mechanism, not an extension of this seal). If it FAILS, record and close — same as TSC-1/TSMOM-1/R3, no re-sweep on this seal.

## Known pre-existing issue disclosed, not related to this seal

`backtest_gap_fade.py`'s `_REPO = Path(__file__).resolve().parents[3]` assumes the script runs from inside a 3-level-deep worktree path; it raises `IndexError` when run from the repo root (where it currently lives, both on `main` and in this worktree). This sweep imports the module's pure functions (`load_data`, `build_session_map`, `run`, `simulate_day`, `report`) and overrides its path constants directly rather than fixing the file in place — out of scope for this seal, flagged separately.
