# Pre-Registration: MIM-NB Cat-Stop Reduction 500 → 250 pts

**Generated:** 2026-06-25
**Experiment ID:** mim-nb-catstop-250
**Base commit:** e12ee36 (branch `feat/yank-ml-canary`)
**Supersedes:** `preregistration_mim_nb_honest_expectations.md` §1 (frozen strategy),
specifically the `CAT_STOP_PTS = 500` and `DLL_GUARD_USD = -1000` constants.

**Status:** SEALED — config change only, no signal-logic change. The live bot
(`trader-mim-nb.service`, acct 23884932, 1× MNQ) will restart with the new constants
after this commit is deployed.

---

## 0. Motivation

Post-mortem of the Jun 25, 2026 combine failure identified a structural risk-sizing
defect: MIM-NB's max single-trade loss ($1,000 at 500pt cat-stop, 1ct MNQ) equals 50%
of the Topstep 50K trailing drawdown budget ($2,000). Portfolio theory and combine math
require max single-trade loss ≤ 25% of DD budget to avoid path-dependent ruin even with
positive-EV strategies running concurrently. MIM-NB violated this gate by 2×.

The fix is mechanical: halve the catastrophe stop. No entry/exit logic changes.

---

## 1. Config change (the only diff)

| Constant | Before | After |
|---|---|---|
| `CAT_STOP_PTS` | `500.0` | `250.0` |
| `DLL_GUARD_USD` | `-1000.0` | `-500.0` |

**DLL_GUARD_USD tracks CAT_STOP:** at 1ct MNQ ($2/pt), one cat-stop = 250 × 2 = $500.
DLL guard set to −$500 means the session halts after one triggered cat-stop per day,
consistent with the prior design intention.

All other constants (CONTRACTS=1, LOOKBACK_DAYS=14, PT_VAL=2.0, signal bands, entry
timing) are **frozen and unchanged**.

---

## 2. Budget math after change

| Item | Before | After |
|---|---|---|
| Cat-stop distance | 500 pts | 250 pts |
| Max loss per cat-stop (1ct, $2/pt) | $1,000 | **$500** |
| As % of $2,000 trailing DD budget | 50% ❌ | **25% ✓** |
| DLL guard / max daily loss | $1,000 | **$500** |
| Two-bot combined max single bad day | ~$1,750 | **~$1,250** |

---

## 3. Expected performance impact

Reducing the stop from 500 to 250 pts means:
- Trades that previously survived to EOD may now hit the tighter stop intraday.
- OOS STOP exit rate was 61% (50/82) at 500pt. At 250pt the stop-out rate will be
  higher — offset by smaller loss per stop.
- Net expectancy changes direction is **unknown prospectively**. The tighter stop
  catches more noise; the larger EOD winners (the fat-tail days) are unaffected if
  price runs away from entry.
- This is a **risk management change**, not an alpha change. We are accepting lower
  expected P&L per trade in exchange for structural viability on the combine.

No prospective decision rule is registered for this change. Performance will be
evaluated as part of the ongoing MIM-NB live data collection period. The OOS PF 1.30
benchmark cited in `preregistration_mim_nb_honest_expectations.md` no longer applies
to the modified strategy; a new OOS benchmark requires a fresh backtest with
CAT_STOP_PTS=250 (not performed here — this is a risk-management deployment, not an
alpha study).

---

## 4. Combine floor monitor update (companion change)

The combine floor monitor halt buffer should also be raised from the flat $500 to
`max_single_trade_loss × 1.5 = $750` for the next combine reset. That change is
**not** included in this pre-reg (floor monitor is infrastructure, not strategy code)
and will be deployed at Sunday combine reset.

---

## 5. Integrity

Sealed by committing this document before touching `src/research/mim_nb_live.py`.
The SHA of this pre-reg commit will be recorded in the code change commit message.
