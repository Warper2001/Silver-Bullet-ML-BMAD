# YANK Compressed-Cascade Holdout Screening Verdict — 20260819

## Status
Screening read, not the sealed Phase 2 verdict. See Amendment 2 in
`_bmad-output/preregistration_yank_compressed_cascade.md` for why this exists and what it
does and doesn't authorize. Logged per protocol in
`data/sealed_holdout/ACCESS_LOG.md`.

## What was run
The same candidate as Phase 1 (M15 sweep / M5 CHoCH / M1 FVG, Amendment-1 baseline config),
against `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv` (2026-03-01 → 2026-05-19,
75,081 bars) instead of a fresh prospective window, at Alex's explicit direction.

## Result

| Metric | Value |
|---|---|
| N trades | 61 |
| PF | **0.7797** |
| Win rate | 41.0% |
| Total P&L | −$3,389.00 |
| Avg win / avg loss | +$479.90 / −$427.40 |
| Exits | 20 SL / 18 TP / 23 TIME_STOP |

Compare to Phase 1 (2025 training data): PF=1.397, N=199, 100th percentile of a 100-seed
random-null distribution (median 0.912, p90 1.218).

## Reading this correctly

- **0.78 is below the null median (0.912) from Phase 1's own distribution**, not just below
  the p90 pass bar (1.218). A random-entry control under the same structure gates would have
  been expected to do *better* than this on average. That is a strong negative signal, not a
  marginal miss.
- **This holdout is not fresh.** 32 prior accesses, several for the same strategy family
  (S25 deployment, YANK's own ml_threshold sweeps). A candidate that fails on a well-worn
  dataset is meaningfully damning; a candidate that *passed* on this same dataset would have
  needed much more scrutiny before being trusted, for the identical reason.
- **199 trades (2025) vs. 61 trades (2026 holdout, ~2.5 months)** — the trade rate roughly
  tripled per unit time relative to 2025's full-year rate. Consistent with a real regime
  change in how often M15/M5 structure aligns, not just noise in a small sample.

## What this does NOT mean

- It does not retroactively invalidate Phase 1 — Phase 1 answered a narrower question
  (does the compressed cascade beat random entry under its own gates, in-sample) and the
  answer to that question is unchanged.
- It does not stop the sealed Phase 2 (prospective, 2026-08-19 forward). Per the seal's
  stopping rule, no result triggers re-tuning or subgroup rescue — Phase 2 continues
  exactly as designed, and its eventual verdict is still the one that governs any live
  deployment decision.

## Honest read

Taken together — strong in-sample, weak on a well-worn 2026 holdout — this looks like the
pattern this project's own methodology exists to catch: a structural change (compressing
the cascade) that fit 2025's regime and did not generalize to 2026's. It is not yet a
closure verdict (Phase 2 is the actual gate, and 61 holdout trades is a real but imperfect
substitute for 30 genuinely fresh ones), but it substantially lowers the prior that Phase 2
will pass. Recommend treating this as a caution flag, not a stop-work order — the systemd
timer keeps running either way.

---
_Ad hoc read via `yank_compressed_cascade_phase1.py`'s shared `_run_cascade` against the
sealed holdout; no CLI flag was added for this one-off invocation._
_Prospective Phase 2 remains the sealed gate: `_bmad-output/preregistration_yank_compressed_cascade.md` §4._
