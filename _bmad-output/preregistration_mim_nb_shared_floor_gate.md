# Pre-Registration Amendment: MIM-NB Buffer Gate → Shared Floor State

**Status:** OPERATIONAL RISK-GATE FIX — SEALED 2026-06-26
**Amends:** `preregistration_mim_nb_live_deployment.md` (deployment, 7939eed)
and `preregistration_mim_nb_catstop_250.md` (catstop-250, 30bc6a8)
**Strategy:** MIM-NB (mim-nb-v2-catstop)
**Scope:** Risk-gate plumbing only. **No strategy parameter changes.**

---

## 1. The bug (why this is needed)

MIM-NB's pre-entry `BUFFER_GATE` blocks new entries when the remaining Topstep MLL
buffer is `≤ cat_cost` (one catastrophe-stop's dollar cost, $500 = 250pt × $2 × 1ct).
The buffer was computed from **MIM-NB's own trade ledger only**:

```
balance = COMBINE_START_BALANCE + own_realized_pnl + day_pnl
mll_floor = own_eod_hwm − MLL_DD          # own_eod_hwm only ratchets up, never resets
buffer = balance − mll_floor
```

After the two cat-stops (06-24 −$1000, 06-25 −$500), this produced:

| | value |
|---|---|
| own-ledger balance | $49,287.50 |
| own internal HWM | $50,787.50 (peak after 06-11/06-12 wins; never reset) |
| own internal floor | $48,787.50 |
| **buffer** | **$500.00** = cat_cost → **entries blocked** |

Two defects:

1. **Deadlock.** `buffer` only recovers if MIM trades and profits, but it will not
   trade while `buffer ≤ cat_cost`. Left alone MIM-NB stays parked indefinitely.
2. **Floor $1,500 too tight.** MIM's self-computed floor ($48,787.50) is stricter
   than the *real* recalibrated combine floor ($47,287.50, reset by the floor
   monitor on 06-26 because the prior HWM had been overstated by unrealized UPL).
   MIM also ignores YANK's contribution to the **shared** combine account
   (acct 23884932). Against the real account there are ~$2,000 of room (≈4 cat-stops),
   not zero.

The floor monitor would happily let MIM trade; MIM was stopping itself on stale,
own-ledger-only numbers.

## 2. What changes

**Single source of truth.** `combine_floor_monitor` already polls the real combined
account every 30s and maintains the authoritative recalibrated trailing floor. It now
also publishes the real combined `balance` / `equity` and a UTC timestamp into
`data/combine_joint/floor_state.json`.

MIM-NB's `_remaining_mll_buffer()` now reads that shared state:

```
buffer = shared_equity − shared_floor        # real combined account incl. YANK
```

with a **staleness fallback**: if `floor_state.json` is missing, malformed, or older
than `FLOOR_STATE_MAX_AGE_S` (300s ≈ 10 monitor ticks), MIM reverts to its previous
own-ledger computation. MIM never trades on stale risk data — if the monitor is down,
the conservative gate re-engages.

The gate threshold is **unchanged**: still block when `buffer ≤ cat_cost ($500)`.
Because the monitor halts the combine at `equity ≤ floor + $500`, MIM's entry gate and
the monitor's halt now share the same boundary — MIM self-limits exactly where the
monitor would halt, so the monitor should rarely need to fire.

## 3. What does NOT change

- The MIM-NB signal, bands, entry/exit logic, cat-stop (250pt), DLL guard, EOD flatten.
- `CONTRACTS`, `CAT_STOP_PTS`, `PT_VAL`, `MLL_DD`, `COMBINE_START_BALANCE`.
- The floor monitor's halt triggers (distance-to-floor $500, combined PF<0.70@30),
  HWM/floor ratchet logic, and chain-hash integrity (new fields are written outside
  the hashed payload).
- YANK is unaffected (it has no buffer gate and was never blocked).

## 4. Expected effect

With the current shared state (equity ≈ $49,287.50, floor $47,287.50) MIM's buffer
becomes ≈ $2,000 > $500, so MIM resumes taking trades. If the combined account is
drawn down to within $500 of the floor, MIM stops entering (and the monitor halts) —
the intended protection, now driven by the real account rather than a stale self-model.

## 5. Files

- `src/research/combine_floor_monitor.py` — publish `balance`/`equity`/`ts_utc` to state.
- `src/research/mim_nb_live.py` — `_shared_floor_buffer()` + `_remaining_mll_buffer()`
  prefers shared state with staleness fallback; gate log tags the source.

## 6. Validation

Unit-tested the buffer source selection: fresh shared state → uses real combined
buffer (unblocks); stale / malformed / missing → conservative own-ledger fallback
(stays blocked); fresh-but-near-floor → shared buffer, blocked (conservative). All pass.
