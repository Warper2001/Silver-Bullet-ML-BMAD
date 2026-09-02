---
title: 'Cold-start book tolerance for the §A8.2 Part A replay path'
type: 'bugfix'
created: '2026-09-02'
status: 'draft'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `run_parity_gate` aborts — `sim.simulate` raises `BookInconsistency` on a
crossed market in a Part A window (w03, 2026-06-25 18:43: bid 29569.75 ≥ ask 29540.75,
persisted 51.9 ms ≥ `config.MAX_TRANSIENT_CROSS_NS`). The ±90-min windows (prereg Mode C)
carry no UTC-midnight book snapshot, so every window's book is reconstructed cold: pre-window
resting orders are never `ADD`ed, their `CANCEL`/`MODIFY` arrive as unseen-ref no-ops, stale
price levels sit in the book. `integrity.preflight_integrity` already tolerates this (flags,
never raises); the Part A sim does not.

**Approach:** Give the Part A replay path a cold-start tolerance compatible with
`integrity.py`'s model. Load-bearing mechanism: `book._check_cross` treats a cross wider than
`config.STALE_CROSS_MAX_TICKS` as a stale-book artifact — count it (`Book.stale_cross_count`),
do not run the persistence timer, do not `_fail`. A cross within the bound keeps the existing
50 ms timer and stays fatal. Optional warmup grace for the unseen-ref path if planning shows
it is needed. Every flagged window + its counts surfaced in the amendment stub.

## Boundaries & Constraints

**Always:** the seal's 50 ms `MAX_TRANSIENT_CROSS_NS` for *real* (bounded-width) crosses is
unchanged; ts-regression stays fatal on every path; every tolerance applied is counted and
rendered in the stub — never a silent pass; `STALE_CROSS_MAX_TICKS` derived from the observed
cross-depth distribution across the purchased windows, not hand-set (memory:
derive-don't-assert).

**Ask First:** whether mechanism (a) applies universally (book.py, all paths) or is Part-A-gated
via a `strict` flag threaded sim→book; whether a warmup grace is needed at all once (a) lands;
the exact value of `STALE_CROSS_MAX_TICKS`; reordering integrity-before-Part-A in the frozen
`run_parity_gate`.

**Never:** changing `MAX_TRANSIENT_CROSS_NS`; making Part B or the standalone `simulate` CLI
tolerant; auto-excluding a flagged window from Part A (record, never drop — AD-13).

</frozen-after-approval>

## Code Map

- `src/ticksim/book.py:594` — `_check_cross`; `:125` `_fail`; `Book` fields `stale_cross_count` (new)
- `src/ticksim/config.py:161` — `MAX_TRANSIENT_CROSS_NS`; add `STALE_CROSS_MAX_TICKS`
- `src/ticksim/sim.py:242` — `Manifest`; surface `stale_cross_count`
- `src/ticksim/parity/integrity.py:70` — `_WARMUP_NS` / `warmup_unknown_ref` reference model
- `src/ticksim/parity/gate_cli.py` — `run_parity_gate` step order; `GateRun` per-window summary

## Tasks & Acceptance

**Execution:**
- [ ] PARKED — see Spec Change Log 2026-09-02.

**Acceptance Criteria:**
- Given a cold-reconstructed Part A window with a stale ghost cross, when `run_parity_gate`
  runs, then it completes with the window FLAGGED and its stale-cross count in the stub,
  instead of aborting.

## Spec Change Log

### 2026-09-02 — PARKED before CHECKPOINT 1

Planning investigation found the cold-start abort is **not the blocking issue**. A
price-basis probe (`~/.claude/jobs/.../basis_probe.py`) over every covered Part A fill showed:

- `data/mim_nb/orders.csv` fills align with CME MBO to ±5.7 ticks stdev, ~0 mean (n=22) —
  **usable, and the only usable source.**
- `data/trades.db` fills (mim-nb *and* yank) are bar-reconstructed / backfilled: stdev
  78–236 ticks, outliers to ±587. Not real broker fills.
- `data/mim_nb/trades.csv`, `data/gap_fade/fills.csv` — coarser still (minute / date only).

So Part A must be rebuilt on `orders.csv` with **per-leg** window routing (22 legs land in
purchased windows; the trade-atomic `_window_of` only fits 4), yank is excluded (no accurate
fill log), and N is 22 vs the pre-registered floor of 28 — pending a decision (small window
top-up, floor amendment, or a cycle-1 finding). This slice resumes once the Part A source
question is settled; the cold-start tolerance is still wanted as a safety net for legs near a
window edge.

## Verification

**Commands:**
- `mypy --strict src/ticksim` — expected: clean, no override
- `pytest tests/unit/ -k ticksim -q` — expected: all green (existing strict-raise tests intact)
