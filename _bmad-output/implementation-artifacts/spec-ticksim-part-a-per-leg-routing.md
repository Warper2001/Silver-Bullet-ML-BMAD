---
title: 'Part A per-leg window routing + broker-accurate fill sources'
type: 'bugfix'
created: '2026-09-02'
status: 'done'
review_loop_iteration: 1
baseline_commit: '1b0a918090ad0faee0d50a38ea5839060975dcf0'
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Part A cannot score its sample. `gate_cli._window_of` requires a trade's whole
stamp span inside one `[lo_ns, hi_ns)`, but mim-nb enters ~14:00 UTC and holds to the 20:00
EOD stop — a ~6 h span against ±90-min windows — so only **4 of 16** trades fit. Separately
the CLI feeds Part A from `data/trades.db`, whose timestamps are **not fill times** (its
2026-06-25 mim-nb row says 19:34 @ 29318.50; `orders.csv` shows that trade filled 14:00 @
29359.5), so the simulator prices the wrong minute — observed MAE ~1157 ticks.

**Approach:** Part A grades **fills**, and each fill is an independent marketable order hours
from its sibling. So split every reconstructed trade into **one pseudo-trade per leg** (1
intent + 1 `RealFill`) before routing. Each leg's span is then a point, `_window_of` routes it
unchanged, and `run_part_a` / `compare_fills` / `aggregate` need no restructuring. Feed Part A
from the two broker-accurate sources only: `data/mim_nb/orders.csv` (mim-nb) and
`data/mim_nb/projectx_fills.json` (yank).

## Boundaries & Constraints

**Always:** a split leg keeps its parent's `order_id`, `leg`, `side`, `size` and `RealFill`
verbatim — splitting changes routing, never the graded values. `trade_id` becomes
`f"{parent_trade_id}#{leg.name}"`, which must stay unique and must keep its existing prefix so
`_trader_of` still classifies it. `aggregate` is still called **once**; N is the count of
scored legs. Every scored leg's source (`orders.csv` / `projectx`) is recorded in the stub.
Verified coverage to preserve: 32 `orders.csv` legs + 4 ProjectX legs = **N 36**, 0 missed.

**Ask First:** the three design questions below — they are answered in this block and are the
human-owned part of the intent.

**Never:** feeding Part A from `trades.db` (`reconstruct_trades_db_row` stays in the module for
other callers and provenance, but the `parity-gate` Part A path must not use it). Changing
`PART_A_MIN_N`, the MAE / p90 / signed-bias tolerances, or any seal-bound `config` constant.
Folding in the cold-start book tolerance — that is the separate parked slice.

### Settled design questions

**(a) Mechanism — per-leg pseudo-trades.** Chosen over merging windows via
`events.merge_streams`, which would fold two disjoint ±90-min windows separated by hours and
strain the AD-20 canonical order and the `valid_intervals` session mask. **Seal-relevant
consequence, stated explicitly:** a one-leg pseudo-trade carries no sibling, so the AD-25
leg-aware OCO cascade never fires during Part A. This is sound here and only here — Part A's
reconstruction emits exactly two *marketable* legs hours apart with no queue interaction, so
the cascade is already inert for them (AD-17 keeps the replay orders-only; no outcome feeds
back). It is **not** a licence to weaken AD-25 anywhere else.

**(b) N counts scored legs.** Prereg §A8.2 reads "N ≥ 28 real broker **fills**". A leg is a
fill. N = 36.

**(c) A leg with no covering window stays fail-closed** — `GateCliError` naming the leg. Zero
such legs today; silently dropping one later would shrink N without a trace.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| mim-nb 2-leg trade, legs in different windows | `orders.csv` trade w/ entry 14:00, exit 20:00 | 2 pseudo-trades, each routed to its own window, both scored | N/A |
| yank ProjectX round trip | open fill (`profitAndLoss` null) + close fill (non-null), same contract | 1 trade → 2 pseudo-trades, `fidelity="broker_fill"` | N/A |
| ProjectX open fill with no close | trailing unpaired opening fill | dropped, counted, logged — nothing to pair | N/A |
| ProjectX close with no preceding open | `profitAndLoss` set, no open pending | `PartAError` naming the fill | raise |
| leg outside every window | fill ts in no `[lo_ns, hi_ns)` | `GateCliError` naming leg + ts | raise |
| both sources empty | no CSV, no ProjectX file | Part A FAIL on the N floor (`n == 0`) | N/A |

</frozen-after-approval>

## Code Map

- `src/ticksim/parity/part_a.py` — `reconstruct_mim_nb` (:482, done); add
  `reconstruct_projectx_fills`; `ReconstructedTrade` (:166) `__post_init__` requires ≥1 intent
  and ≥1 `real_fill` — a 1-leg pseudo-trade satisfies it; `_build_mim_trade` (:~700) is the
  shape to mirror; `compare_fills` (:~800) filters outcomes by `trade_id` then joins
  `(order_id, leg)` — unchanged by splitting
- `src/ticksim/parity/part_a_runner.py` — `run_part_a` (:73) and `_window_span` (:149)
  unchanged; `_window_span` on a 1-leg trade yields a point span
- `src/ticksim/parity/gate_cli.py` — `_window_of` / `_trade_span` unchanged; add the split
  before routing; `_trader_of` (:156) prefix match must survive the `#LEG` suffix; drop
  `skip_uncovered` if unused
- `src/ticksim/cli.py` — `_reconstruct_part_a_trades`, `_db_rows`; add `--projectx-fills`
- `_bmad-output/parity/gate_windows.json` — 39 verified windows
- `data/mim_nb/projectx_fills.json` — 14 fills; 4 are yank (`orderId` not in `orders.csv`)

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/parity/part_a.py` — add `reconstruct_projectx_fills(fills)`: sort by
  `creationTimestamp`, pair open (`profitAndLoss is None`) → close per `contractId`, emit two
  marketable legs, `trade_id=f"yank-{open.orderId}"`, `fidelity="broker_fill"`.
- [x] `src/ticksim/parity/part_a.py` — add `split_legs(trade) -> list[ReconstructedTrade]`:
  one pseudo-trade per `RealFill`, carrying only that leg's intent, `trade_id` suffixed
  `#{leg.name}`, all graded values verbatim.
- [x] `src/ticksim/parity/gate_cli.py` — apply `split_legs` to every Part A trade before
  `_window_of`; keep fail-closed routing; retire `skip_uncovered` if per-leg routing makes it
  dead.
- [x] `src/ticksim/cli.py` — `--projectx-fills PATH`; Part A sources = `orders.csv` +
  ProjectX; remove the `trades.db` Part A feed; record each leg's source for the stub.
- [x] `tests/unit/test_ticksim_parity_part_a.py` — cover the I/O matrix rows.
- [x] `tests/unit/test_ticksim_cli_parity_gate.py` — CLI wiring: both sources, `--projectx-fills`.

**Acceptance Criteria:**
- Given the real `orders.csv` + `projectx_fills.json` and the 39-window map, when the Part A
  trades are split and routed, then every leg resolves to a window and `n == 36`.
- Given a split leg, when it is graded, then its `order_id`, `leg`, `side`, `size` and real
  fill price/ts equal the parent's.
- Given a ProjectX opening fill with no closing fill, when reconstructed, then it is dropped
  with a logged reason and does not raise.

## Spec Change Log

### Review round 1 (2026-09-02)

The three reviewer subagents (blind-hunter, edge-case-hunter, verification-gap) all
terminated on an API session rate limit before reporting. The pass was done inline.
One `patch`-class finding, no `intent_gap`, no `bad_spec` — the frozen block stands.

**R1-1 (patch) — `--projectx-fills` without `--orders-csv` silently mislabelled fills.**
A ProjectX fill carries no trader field: both bots trade the same `contractId` on the
same `accountId`, so `orders.csv` is the only thing separating them. The implementation
made an absent CSV a non-error, so `csv_order_ids` was empty, the mim-nb filter passed
everything through, and the open->close pairing (which walks per `contractId`) let the
two bots' fills interleave. Reproduced on the real 14-fill export: **no error, 7 "yank"
trades, 5 of them actually mim-nb** — the sealed stub's per-trader breakdown would simply
be wrong.

The shipped test `test_missing_orders_csv_leaves_only_yank_legs` asserted this behaviour
as correct; it passed only because its fixture is yank-only, so it could never catch the
real case. That test is replaced.

Resolution: fail closed when `--projectx-fills` is given without `--orders-csv`, with an
explicit `--projectx-yank-only` opt-out for a genuinely yank-only export (a plain hard
failure would have broken nine unrelated tests that legitimately use yank-only fixtures).
New tests: `test_projectx_without_orders_csv_fails_closed` (mutation-verified — disabling
the guard makes it fail) and `test_mixed_projectx_export_is_split_by_orders_csv` (a
both-bots export is attributed correctly and mim-nb is not double-scored as yank).

**Checked and sound (no change):** `_record_source` calls `split_legs` itself rather than
duplicating the `#LEG` id scheme, so the provenance note cannot drift from what is scored;
`_read_projectx_fills` rejects non-UTF-8 / non-JSON / non-array / non-object elements;
`split_legs` copies the intent with `model_copy(update=...)` so the intent's `trade_id`
matches the pseudo-trade's, which `compare_fills` joins on; the `trades.db` Part A feed
and the `skip_uncovered` crutch are fully removed (no leftover references).


**Implementation notes (2026-09-02)** — decisions the tasks implied but did not name:

1. **`--trades-db` removed from `parity-gate`.** With the DB Part A feed gone the flag had no
   consumer; leaving it would advertise a source the Never clause forbids. `_db_rows` /
   `_reconstruct_db_row` / `_PART_A_DB_SINCE` / the DB-fallback provenance note went with it,
   as did the `sqlite3` import. `part_a.reconstruct_trades_db_row` itself stays (other callers
   + its own tests), now documented as "not a `parity-gate` Part A source".
2. **`skip_uncovered` retired**, per settled question (c): `run_parity_gate`'s
   `skip_uncovered` parameter, `GateRun.skipped_uncovered` and the `--skip-uncovered` flag are
   gone. Per-leg routing covers all 36 legs, and a future uncovered leg must fail closed.
3. **ProjectX ↔ `orders.csv` de-dup lives in `cli.py`**, not in `reconstruct_projectx_fills`:
   the export carries mim-nb's fills too (10 of 14), and an `orderId` already in `orders.csv`
   is a mim-nb order that the richer lifecycle ledger already scored. Filtering there keeps
   the reconstructor a pure function of the fills it is handed.
4. **The source-provenance note is appended, not prepended.** It is now written on every run
   (it records every leg's source), so prepending would displace the `# Amendment N` heading
   the analyst appends to the seal. It renders as a trailing `## Part A fill sources` section
   listing each source's leg ids.
5. **Two unlisted ProjectX shapes fail closed** (module convention: "any unexpected shape
   raises `PartAError` naming the row"): a `voided` fill, and a second opening fill on a
   contract that already has one pending (add-to-position). Neither occurs in the real export.

## Verification

**Result (2026-09-02):** 868 ticksim tests pass, `mypy --strict src/ticksim` clean with no
override, black clean on every in-scope file. Real-data coverage probe: 18 parent trades ->
**36 legs, 36 routed, 0 missed** (32 `trader-mim-nb` + 4 `trader-yank`), 29 of 38 windows
used. All six I/O-matrix rows covered by passing tests.

**Commands:**
- `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/ -k ticksim -q` — 837 prior tests stay green
  → **PASS**: 867 passed / 13 skipped (`tests/ -k ticksim`), stable under pytest-randomly.
- `.venv/bin/python -m mypy --strict src/ticksim` — clean, no override → **PASS** (20 files).
- `.venv/bin/python -m black --check src/ticksim tests/unit` — clean → **PASS for the touched
  files** (`src/ticksim` + the 3 edited test modules). The unscoped `tests/unit` sweep reports
  139 pre-existing unformatted modules unrelated to this slice; none were touched.
- Coverage probe: split+route the real sources against `gate_windows.json` → expect 36 legs, 0
  missed → **PASS**: 18 parent trades → **36 legs, 36 routed, 0 missed**; 32 `trader-mim-nb` +
  4 `trader-yank`; 29 of the 39 windows used.
