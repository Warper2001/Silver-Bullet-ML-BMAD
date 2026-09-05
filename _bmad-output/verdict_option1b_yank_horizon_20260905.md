# Verdict — Option 1b: YANK exit-horizon extension

**Date:** 2026-09-05
**Pre-registration:** `_bmad-output/preregistration_option1b_yank_horizon.md`
**Result: INCONCLUSIVE / ABANDONED — compute cost proved unpredictable, not the hypothesis.**

## What's actually known

One confirmed data point: baseline (`max_hold_bars=60`, the live sealed config, full ~14-month dev window) — **N=78, PF=2.125, PF_ex_top3=1.567, gross $10,293**, run time 48.8 minutes uncontended. That number is real and usable on its own (it's simply YANK's dev-window backtest at its current live config, not a new finding), but it's not enough alone to answer the horizon question — no comparison cell was completed.

## What went wrong, honestly

Every attempt to extend the exit horizon (or the null draws at other `max_hold_bars` values) ran far longer than the confirmed baseline suggested it should — a relaunch under supposedly-identical conditions (4 dedicated cores, same engine, same window, the only difference being `max_hold_bars`) ran 3+ hours with zero of 9 remaining tasks completing, nearly 4x the ~49 min/run the baseline confirmed. This happened **twice** at different scopes (the original 20-run design, then the reduced 10-run design), which rules out a one-off fluke. The most likely explanation, not confirmed: `Tier2StreamingTrader`'s replay cost may scale with how much time a position is actually held open (more bars spent in an active-trade state doing per-bar exit checks), not just total bars processed — a longer `max_hold_bars` config would then be **inherently** more expensive to replay, on top of whatever baseline cost the engine already has. This is a real, disclosed possibility, not a confirmed diagnosis; nobody profiled the engine to confirm it.

## Disposition

- **Closed as inconclusive**, not as FAIL — no comparison result was ever obtained, so there is nothing to accept or reject about the hypothesis itself.
- **Do not re-attempt with this engine without first profiling `Tier2StreamingTrader`'s replay loop** to understand its actual cost structure (bar count vs. hold-time-dependent cost) — guessing at run counts a fifth time is not worth the (real, project-relevant, already-flagged) uncertainty about *why* this is so slow.
- The one usable output — YANK's current-config dev-window PF (2.125) — is filed here for reference but changes nothing operationally; it's the config already live.
- Time and compute spent on this sub-task materially exceeded its value as an *exploratory* check; if the horizon question is worth revisiting, it should wait for either a profiled/fixed engine or a much smaller, tightly-scoped single comparison (one alternate cell, not a grid+null), sized only after a confirmed same-conditions timing.
