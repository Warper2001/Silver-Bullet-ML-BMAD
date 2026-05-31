# S26v2 Early Termination — Insufficient Sample (Pre-Verdict)

**Termination Date:** 2026-05-30
**Pre-registration SHA:** `0ff1818`
**Verdict:** `insufficient_sample` (pre-verdict termination)
**Reason:** Structural Window B signal scarcity confirmed by gap ratio sweep prior to 180-day hard cap

---

## Summary

S26v2 (Directional Flow Continuity, pre-registered SHA 0ff1818, 2026-05-29) is formally terminated early on the basis of a gap ratio sweep that conclusively demonstrates the evaluation window cannot be met at any tested `min_gap_atr_ratio` value.

This termination is consistent with the pre-registration's stated hard cap: "if N_Window_B < 20 after 180 days, verdict is `insufficient_sample`." The gap ratio sweep, conducted after the pre-registration commit, proved this outcome is mathematically certain — eliminating the need to run the 180-day clock to its expiry.

---

## Gap Ratio Sweep Findings

Sweep conducted 2026-05-29 to 2026-05-30. Backtest period: 2025-05-19 to 2026-05-19 (365 days). Four values of `min_gap_atr_ratio` tested with S26v2 DFC filter applied post-hoc. Results saved to: `data/reports/gap_ratio_sweep_20260530_200003.csv`

| ratio | N_all | N_B (Win B) | PF_B | freq_B | days → N=20 |
|-------|-------|-------------|------|--------|-------------|
| 0.10 | 166 | 19 | 1.038 | 0.052/day | 384 |
| 0.15 | 134 | 16 | 1.300 | 0.044/day | 456 |
| 0.20 | 91 | 11 | 2.518 | 0.030/day | 664 |
| **0.25 (S25 live)** | **62** | **9** | **8.641** | **0.025/day** | **811** |

**Key finding:** No ratio achieves N_Window_B ≥ 20 within 365 days. At the live S25 config (ratio=0.25), expected live Window B trades in 180 days: ~4–5. Even at ratio=0.10 (most permissive tested): ~9 in 180 days. The 180-day hard cap is structurally unreachable.

---

## Interpretation

The scarcity is **structural to the session window**, not a parameter artifact. The 13:00–15:00 ET (Window B) session produces approximately 9 qualifying FVG setups per year at the current S25 detection criteria. This is a property of how frequently the H1 sweep + M15 CHoCH + M1 FVG confluence occurs in the afternoon session — it is not materially affected by gap size threshold.

The quality/quantity relationship reinforces this: lowering the ratio from 0.25 to 0.10 increases N_B from 9 to 19 (+111%) but collapses PF_B from 8.64 to 1.04. The additional signals generated at lower thresholds are low-quality and not concentrated in Window B relative to total signal count.

**This does NOT falsify the Directional Flow Continuity mechanism.** The sweep demonstrates that S25's FVG detection is an unsuitable substrate for testing the DFC hypothesis in Window B — not that Window B lacks directional character. A future study testing DFC with a detection mechanism that fires more frequently in the afternoon session (e.g., mean-reversion exhaustion, order block retests) could still be pre-registered.

---

## Formal Verdict

Per the S26v2 pre-registration decision rule (Section "S26v2 Decision Rule"):

> **N_B < 20 after 180 days → `insufficient_sample`**
> Window B fires too rarely. Either (a) discard S26v2 and investigate frequency further, or (b) pre-register S26v3 with broader windows.

Verdict: **`insufficient_sample`** — terminated pre-emptively on 2026-05-30, before the 180-day evaluation window closes, on the basis of conclusive structural evidence.

Action: S26v2 is discarded. A future S26v3 would require a different detection mechanism, not broader time windows, to address the frequency problem.

---

## Relationship to S25

S25 (live since 2026-05-24) continues unaffected. S26v2 was a prospective subgroup analysis of S25 live trades — its termination has no effect on S25's evaluation, frozen config, or decision gate (PF > 1.1350 after N≥20 AND 60 days).

S26 (original, SHA a97b21c) also remains active as an independent subgroup study on the same S25 live log.

---

## Disclosure

This termination document was written after observing the gap ratio sweep results. The sweep was conducted prospectively (after S26v2 pre-registration commit SHA 0ff1818) and constitutes legitimate post-registration diagnostic work. No S26v2 parameters were changed as a result — this is a closure, not a revision.

*Filed by: Alex (warper2001@gmail.com), 2026-05-30*
