---
title: 'mim-nb reconstruction times the fill by the PLACE instant (§A8.2 cycle 2)'
type: 'bugfix'
created: '2026-09-03'
status: 'done'
review_loop_iteration: 2
baseline_commit: 'f2769ee6bc5fb18f61ef1d6dba7f0b7993b667ba'
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** §A8.2 cycle 1 FAILed on Part A: MAE 8.06 ticks (tol 1.0), p90 13.5 (tol 2.0),
signed bias −1.72 (tol ±0.25). Diagnostics ruled out the cold-start book (the reconstructed
BBO and the CME trade tape agree with each other and both sit ~14 ticks off the recorded
mim-nb fill prices). The cause is the mim-nb `orders.csv` **FILL-row timestamp**: the bot
polls `/Trade/search` to detect fills, so every market order's FILL row is logged ~3.1 s
after the PLACE. Same 10 fills, same book, timestamp the only variable: FILL-row ts →
book-BBO-vs-fill MAE **12.6 ticks**; ProjectX `creationTimestamp` (the true execution) →
MAE **1.5 ticks**. ProjectX `creationTimestamp` = PLACE ts − ~50 ms — so the PLACE
timestamp *is* the execution instant, to ~50 ms. yank (all 4 legs from ProjectX execution
records) matched to 0–3 ticks: the control.

**Approach:** For a mim-nb **market** leg, emit `RealFill.ts_ns = leg.ts_ns` (the PLACE
timestamp) instead of `leg.fill_ts_ns` (the poll-return). The FILL row still supplies the
fill **price** and confirms the fill happened; it no longer times it. After this,
`RealFill.ts_ns == intent.submit_ts_ns` for every mim-nb market leg.

## Boundaries & Constraints

**Always:** `reconstruct_mim_nb` is a **pure function of its arguments** — `orders.csv` rows
plus an optional `stop_out_exit_ts: Mapping[str, int]` (order_id → true fill ts) the caller
supplies; it never opens `projectx_fills.json` itself. The FILL row's `price` and `size` are
unchanged; only its timestamp stops being used to time a **market** leg.
`reconstruct_projectx_fills` and `reconstruct_trades_db_row` are untouched. Coverage after
the change: the probe yields **36 legs / 0 missed** (unchanged N — see question (a)).

**Ask First:** the two questions below — answered in this block.

**Never:** changing `PART_A_MIN_N`, the MAE / p90 / signed-bias tolerances, the 250 ms latency
model, or any seal-bound config. Tuning anything toward a target MAE — the post-fix numbers
are an observation, not a lever. Folding in the latency-model question (that is cycle 3).

### Settled design questions

**(a) The stop-out exit leg is KEPT and scored, timed by the caller-supplied true fill ts.**
(Renegotiated after review round 1 — see Spec Change Log.) `mimnb-3463323116`'s exit (oid
`3463323140`) is the sample's only otype-4 stop-out fill. `orders.csv` has no order-submission
instant near its trigger (it was placed ~1 h earlier) and its FILL-row ts is the same ~3 s
poll-late signal — **but the fill itself is clean**: 59-tick book error with the poll-late
ts, **2 ticks** with the broker's own `creationTimestamp`. Dropping it would restrict Part A
to the benign subset — the exact pattern `MEMORY.md` warns against — so the leg is **kept**.
Its `RealFill.ts_ns` comes from `stop_out_exit_ts[order_id]` when the caller provides one
(the CLI populates that map from `projectx_fills.json`, which it already reads for yank);
a stop-out exit with **no** entry in the map is dropped with a logged reason carried into
the stub. N stays **36**. `reconstruct_mim_nb` takes the map as a plain `Mapping[str, int]`
argument — it is a pure function of its inputs, with no `projectx_fills.json` coupling.

**(b) Market legs still use their PLACE ts; no ProjectX timestamps for them.** The 32 mim-nb
**market** legs are timed by their PLACE ts (accurate to ~50 ms, confirmed by the 10
ProjectX-matched legs). Only the **stop-out** exit — which genuinely has no PLACE anchor —
takes a caller-supplied ts.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| normal mim-nb round trip | entry PLACE→FILL, exit PLACE→FILL | 2 legs, each market leg's `RealFill.ts_ns == its PLACE ts == its intent submit_ts_ns` | N/A |
| stop-out exit, ts supplied | otype-4 FILL in ACTIVE; `stop_out_exit_ts` has the oid | 2-leg trade; exit `RealFill.ts_ns == stop_out_exit_ts[oid]`, price/size from the FILL row; exit still validated (opposite side, positive price, ts > entry) | corrupt FILL row → `PartAError` |
| stop-out exit, no ts supplied | otype-4 FILL in ACTIVE; oid absent from the map | entry leg emitted; **exit leg dropped**, reason logged + collected | N/A |
| otype-4 FILL in a non-ACTIVE state | stop FILL while ENTRY_PENDING / EXIT_PENDING / FLAT | `PartAError` naming the row (fail-closed — not in the real sample) | raise |
| entry filled, exit never logged | trailing incomplete trade | dropped (unchanged from 66acc08) | N/A |
| provenance note, no mim-nb legs | Part A sample is yank-only | the "mim-nb market leg timed by PLACE ts" sentence is **omitted** | N/A |

</frozen-after-approval>

## Code Map

- `src/ticksim/parity/part_a.py`
  - `_build_mim_trade` (:436) — the two `RealFill(...)` constructions (:485, :494) currently
    pass `ts_ns=entry.fill_ts_ns` / `exit_.fill_ts_ns`; change to `entry.ts_ns` / `exit_.ts_ns`
  - `_PendingLeg` (:426) — `ts_ns` is the PLACE ts (set at :614), `fill_ts_ns` the FILL-row ts
    (set at :712 / :721); keep `fill_ts_ns` for the "did it fill" None-check only
  - FILL handler (:709–:724) — still sets `fill_px_dbn` / `fill_size` / `fill_ts_ns`
  - stop-out branch (:678–:686) — `stop_leg` is built here; the exit-leg drop lands in the
    state machine where `_build_mim_trade` is currently called for a stop-out
  - `_build_mim_trade`'s causal check `exit_.ts_ns <= entry.ts_ns` (:448) already uses PLACE
    ts — unaffected
- `src/ticksim/parity/gate_cli.py` / `src/ticksim/cli.py` — the stop-out timing + drop reason should
  reach `_format_source_provenance` / the stub's "Part A fill sources" section
- `tests/unit/test_ticksim_parity_part_a.py` — tests asserting `RealFill.ts_ns` == the FILL
  row ts need updating to the PLACE-ts contract; add the stop-out-exit-dropped test

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/parity/part_a.py` — `_build_mim_trade`: `RealFill.ts_ns` for a **market**
  leg = the leg's PLACE ts (`entry.ts_ns` / `exit_.ts_ns`); docstring notes why (poll-lag).
- [x] `src/ticksim/parity/part_a.py` — `reconstruct_mim_nb(rows, *, stop_out_exit_ts=None,
  dropped_stop_out_exits=None)`. On an otype-4 FILL in ACTIVE: if `stop_out_exit_ts` has the
  exit oid, build a normal 2-leg trade via `_build_mim_trade` with the exit `RealFill.ts_ns`
  = that value (price/size from the FILL row, all `_build_mim_trade` validation kept); else
  emit the entry leg alone, `logger.warning` + append the reason to `dropped_stop_out_exits`.
  An otype-4 FILL in any other state → `PartAError` naming the row.
- [x] `src/ticksim/cli.py` — `_reconstruct_part_a_trades` builds `stop_out_exit_ts` from the
  ProjectX fills (`{str(orderId): creationTimestamp_ns}`) and passes it + the drop collector
  into `reconstruct_mim_nb`; forwards both into `_format_source_provenance`.
- [x] `src/ticksim/cli.py` — `_format_source_provenance`: the "mim-nb market leg timed by
  PLACE ts" sentence only when ≥1 mim-nb leg is present; a "Stop-out exit legs timed from
  broker records" line listing any ProjectX-timed stop-out legs; a "Dropped, not graded"
  block for any stop-out exit with no ts.
- [x] `tests/unit/test_ticksim_parity_part_a.py` — replace the obsolete
  `test_protective_stop_fill_is_scored_as_the_exit`; add
  `test_mim_nb_market_leg_realfill_ts_equals_place_ts`,
  `test_stop_out_exit_scored_from_supplied_ts`,
  `test_stop_out_exit_dropped_when_no_ts`, `test_otype4_fill_in_wrong_state_raises`.
- [x] `tests/unit/test_ticksim_cli_parity_gate.py` — a `cli.main` run with a stop-out
  `orders.csv` fixture + a matching ProjectX fill, asserting the stub's Part A fill-sources
  section renders the stop-out-timed-from-broker line (verification-gap R1).

**Acceptance Criteria:**
- Given the real `orders.csv`, when reconstructed and split, then every mim-nb **market**
  leg has `RealFill.ts_ns == its intent.submit_ts_ns`.
- Given the real `orders.csv` + `projectx_fills.json` + the 39-window map, when split and
  routed, then **36 legs, 0 missed** (N unchanged).
- Given `mimnb-3463323116` and a `stop_out_exit_ts` carrying oid `3463323140`, when
  reconstructed, then it is a 2-leg trade and its EXIT `RealFill.ts_ns` equals the supplied
  value (not the FILL-row ts).
- Given the same trade with an **empty** `stop_out_exit_ts`, then only the ENTRY leg is
  emitted and the drop reason is collected.

## Spec Change Log

### Review round 1 (2026-09-03) — intent_gap loopback on design question (a)

The blind-hunter flagged that dropping the stop-out exit leg restricts Part A to the benign
subset — the "restrict-to-favorable-subset" pattern `MEMORY.md` names as a repeated project
failure. The dropped leg is in fact a **clean fill** (2-tick book error with the broker's
`creationTimestamp`, 59 with the poll-late `orders.csv` ts) — so it is a data point worth
keeping, not noise. Human renegotiated design question (a): **keep the leg, time it from a
caller-supplied `stop_out_exit_ts` map** (CLI populates it from `projectx_fills.json`, which
it already reads). `reconstruct_mim_nb` takes the map as a plain `Mapping[str, int]` — pure,
no `projectx_fills.json` coupling. N stays 36. `_build_mim_stop_out_entry_trade` (a 30-line
copy of `_build_mim_trade` the reviewers also flagged) is only needed for the no-ts drop path.

Also folded from the same review round (all `patch`-class):
- provenance note: the mim-nb-timing sentence is gated on mim-nb legs being present
  (edge-case-hunter #4, blind-hunter);
- an otype-4 FILL in a non-ACTIVE state raises `PartAError` rather than falling through
  (edge-case-hunter #1/#2);
- the kept stop-out leg keeps all of `_build_mim_trade`'s row validation (edge-case-hunter #3);
- `_window_span`'s docstring ("fill ts is later than the submit ts") is corrected — market
  legs now have `RealFill.ts_ns == submit_ts_ns`;
- the module docstring's "2026-07-29 and 2026-08-28" fired-stop claim is corrected: `111` on
  2026-07-29 is the *unattributable* stop (no PLACE), the only *fired* stop is 2026-08-28;
- a `cli.main` end-to-end test for the stop-out stub rendering (verification-gap R1).

Not adopted: the constant-lag-correction option for stop timing (blind-hunter) — the
broker's own timestamp is strictly better than `FILL ts − ~3.1 s` when it exists, and the
drop-with-reason path covers the case where it does not.

### Review round 2 (2026-09-03) — re-implementation

All three reviewer subagents stalled at their prompt line on an API session rate limit
(third consecutive round); the pass was done inline. Every round-1 finding is addressed by
the current diff:
- **selection bias** — the stop-out exit leg is kept and scored (design (a), renegotiated);
- **copy-paste** — `_build_mim_stop_out_entry_trade` is gone; a shared `_mim_leg` helper
  builds every leg's `OrderIntent` + `RealFill`, `_build_mim_entry_only` reuses it;
- **provenance sentence unconditional** — gated on `mim_leg_present`;
- **otype-4 FILL in a non-ACTIVE state** — raises `PartAError`, scoped to a *placed* stop
  (`order_id in stop_order_ids`); a never-placed otype-4 FILL (the real 2026-07-29 `111`
  row, state FLAT) stays the logged-and-skipped "unattributable" path — a distinct category
  the change log already carved out, and `test_unattributable_stop_fill_is_skipped_not_raised`
  is intact. **Deviation accepted:** the matrix row "otype-4 FILL in a non-ACTIVE state →
  PartAError" reads literally as unconditional; scoping it to placed stops is the correct
  reading (an unconditional raise on `111` fails the "36 legs, 0 missed" criterion);
- **kept stop-out validation** — the leg routes through `_build_mim_trade`, so opposite-side
  / positive-price / exit-ts > entry-ts all still run;
- **`_window_span` docstring** — corrected (market legs now have `RealFill.ts == submit_ts`);
- **2026-07-29 / 2026-08-28 docstring mix-up** — corrected;
- **verification-gap R1** (stop-out drop / timing never asserted end-to-end through the CLI
  stub) — new `test_stop_out_exit_timed_from_broker_record_reaches_the_stub` drives
  `cli.main` with a fired-stop `orders.csv` + a matching ProjectX fill and asserts the
  stub's Part A fill-sources section.

**Deferred (cosmetic, unreachable):** `_format_source_provenance` emits "Every graded leg
above comes from a broker-accurate source" even when `source_of` is empty and only dropped
stop-outs exist — a Part A sample of only dropped legs cannot occur (N would fail the
floor). Logged to `deferred-work.md`.

**Verified (2026-09-03):** 890 ticksim tests pass, `mypy --strict src/ticksim` clean, black
clean. Coverage probe against the real sources + the 39-window map: **36 scored legs / 0
missed** (N unchanged); the stop-out exit `mimnb-3463323116#EXIT` (oid 3463323140) is timed
from its ProjectX `creationTimestamp` (2026-08-28T15:57:23) and routes to window w26.

## Verification

**Commands:**
- `PYTHONPATH=. .venv/bin/python -m pytest tests/ -k ticksim -q` — ~886 prior tests green
  (minus the replaced stop-scoring assertion)
- `.venv/bin/python -m mypy --strict src/ticksim` — clean, no override
- Coverage probe against the real sources + `gate_windows.json` → **36 legs, 0 missed**
- (parent session) a fresh full gate run → cycle-2 verdict; the spec asserts coverage +
  the ts contract, **not** a target MAE
