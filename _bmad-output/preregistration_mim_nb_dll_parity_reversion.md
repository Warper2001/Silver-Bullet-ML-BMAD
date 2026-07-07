# Pre-Registration: MIM-NB DLL Guard Parity Reversion (−500 → −1000)

**Generated:** 2026-07-07
**Experiment ID:** mim-nb-dll-parity-reversion
**Type:** Parity reversion to the Monte-Carlo-authorized spec — NO new experiment, NO
holdout access. One constant, one restart.

**Status:** DRAFT — seals on merge to main; deployment only after Alex's explicit go.

---

## 0. Motivation — restoring the system that was actually validated

The 2026-07-07 halt-and-review (`halt_review_mim_nb_parity_20260707.md`) found the live
bot is not the system the evidence authorized:

- The authorizing combine Monte Carlo (`study_mim_nb_catstop.py`, seal 6957daa — the run
  that produced 54% pass / 33% blow and justified deployment) cuts a trading day only at
  **−$1,000** and therefore **credits post-cat-stop re-entries** (the sealed spec:
  "after stop-out: flat; re-entry permitted at any subsequent HH:00/HH:30 check").
- Prereg 30bc6a8 (2026-06-25) set `DLL_GUARD_USD = −500` ("DLL tracks cat-stop") — an
  asserted pairing. At 250pt cat-stop this halts the session after ONE stop, so live can
  never take the re-entries the MC credited. **The deployed combination (cat-stop 250 +
  guard −500) was never backtested.**

Deterministic counterfactual (recorded live bands, actual path, all four cat-stop days):
the −500 guard suppressed exactly one re-entry that would have fired — 2026-07-02 11:00
SHORT, +301.5 pts (+$603) to EOD; zero days produced a second stop signal. Cited as
context only, NOT as justification (that would be outcome-peeking). The justification is
parity: **−1000 is what the authorizing MC simulated.**

## 1. Change spec (the only diff)

| Constant (`src/research/mim_nb_live.py:52`) | Before | After |
|---|---|---|
| `DLL_GUARD_USD` | `-500.0` | `-1000.0` |

Frozen and unchanged: `CAT_STOP_PTS = 250`, `CONTRACTS = 1`, signal/band logic, entry
timing, EOD flatten, the **dynamic buffer clamp**
(`dynamic_dll = -min(abs(DLL_GUARD_USD), max(0, buffer + cat_cost))`, line ~665) — the
guard still tightens automatically to the remaining shared-floor buffer, so floor
protection is independent of this constant — and the per-entry BUFFER_GATE
(blocks entries when shared buffer ≤ $500 cat cost).

## 2. Disclosures

- MIM worst-case single day returns to **−$1,000** (two cat-stops or stop + losing
  re-entry). Joint worst-day with YANK worsens accordingly; the companion YANK
  daily-breaker rescale (−750 → −300, this PR) partially offsets, and the floor monitor
  halt (floor + $750) remains the account-level backstop.
- At the current ≈$104 room-to-halt this change has near-zero immediate effect: the
  dynamic clamp binds long before −1000 does. The reversion matters if/when buffer
  recovers — and for the integrity of citing the sealed MC numbers at all.
- The 30bc6a8 assertion ("max single-trade loss ≤ 25% of DD budget") governed
  CAT_STOP_PTS, which is NOT changed here; only the day-guard pairing is reverted.
  CAT_STOP 250-vs-500 remains governed by the shadow-ledger prereg (this PR).

## 3. Deployment (after merge + explicit go)

1. Edit `src/research/mim_nb_live.py:52` → `-1000.0`.
2. `systemctl restart trader-mim-nb` while flat (outside 10:00–16:00 ET, or when no
   position is open).
3. Verify next DLL log line reads "≤ -1000.00 (dynamic)" (or the buffer-clamped value).

Rollback: revert the constant, restart.

## 4. Integrity hashes

| Hash | Value |
|---|---|
| (a) `src/research/mim_nb_live.py` SHA-256 (pre-change) | `e2a9b70399684a48176dd61be438d0bf8f90a11f8d9525fe6142c2cdb7e200d2` |
| (b) `study_mim_nb_catstop.py` MC day-cut | `if day_pnl <= -1000.0: break` (line ~... verified 2026-07-07) |
| (c) Git HEAD at draft | `22a9ea4cfefef9bac25c5781156dad9a48a85232` |
