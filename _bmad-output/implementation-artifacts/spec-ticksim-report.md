---
title: 'ticksim report.py — the AD-14 three-way P&L report'
type: 'feature'
created: '2026-08-30'
status: 'done'
review_loop_iteration: 1
baseline_commit: '1e00409'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/project-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `src/ticksim/` produces `OrderOutcome` logs but no P&L. The prereg (§2.3) requires every strategy P&L reported **three ways** — (a) primary, (b) primary + a 1-tick adverse slip on every entry and exit, (c) optimistic — and the §6 decision rule is evaluated on (a) and must hold under (b).

**Approach:** add `src/ticksim/report.py` — `build_report(primary_outcomes, primary_manifest, optimistic_outcomes, optimistic_manifest) -> ThreeWayReport`. Pairs `OrderOutcome`s into round trips by `trade_id`, computes per-trade net P&L in **int USD cents** for the three models, and aggregates. It does **not** evaluate the §6 PASS/FAIL verdict (study-level: walk-forward, regime split, permutation, deflated Sharpe). Money is report-layer only (AD-24).

## Boundaries & Constraints

**Always:**
- `build_report(...)` is pure and never mutates its inputs. `primary_manifest` / `optimistic_manifest` are `Mapping[str, Any]` (the `Manifest.to_dict()` shape — report.py must **not** import from `sim`). Fees come from `manifest["config"]["exch_reg_fee_usd_cents"]` + `["commission_usd_cents"]` (int cents, per-contract round turn, §2.1). `DOLLARS_PER_INDEX_POINT` / `MNQ_TICK_DBN` imported from `config`.
- **Input validation → `ReportError`** (never a bare `KeyError`/`TypeError`): a manifest missing `["config"]` or either fee key; a fee that is not a non-negative `int`; the two manifests are not a **`PRIMARY` + `OPTIMISTIC` pair** (`primary_manifest["config"]["queue_model"] == "back_of_queue"` and `optimistic_manifest["config"]["queue_model"] == "time_priority"` — guards against a swap or two of the same); a duplicate `order_id` across the outcomes of one run; the primary and optimistic `trade_id` sets differ.
- **Round-trip pairing (AD-12), per run.** Group outcomes by `trade_id`. Per `trade_id`: over the fills of every `leg == ENTRY` outcome sum `entry_size` and `entry_notional_dbn = Σ(f.px_dbn*f.size)`; likewise `leg == EXIT`. `entry_ts_ns = min(f.ts_ns)` over entry fills, `exit_ts_ns = max(f.ts_ns)` over exit fills. **Side rules:** the set of `side` over *filled* entry legs must be a singleton → else `ReportError`; that side is `entry_side`; every *filled* exit leg's `side` must be the opposite of `entry_side` → else `ReportError`. `direction = +1` if `entry_side == BUY` else `−1`. A cancelled / zero-fill leg contributes nothing to any sum or side set.
- **Classification.** `entry_size == 0 and exit_size == 0` → skipped silently. `entry_size == 0 and exit_size > 0` → `ReportError`. `entry_size > 0 and exit_size == 0` → **open position**: `OpenPosition(trade_id, open_size=entry_size, avg_entry_px_dbn, entry_ts_ns)`. `exit_size > entry_size` → `ReportError`. Else a **round trip**: `matched_size = min(entry_size, exit_size)`; `entry_size > exit_size` → also `partially_closed` `(trade_id, entry_size − exit_size)`.
- **P&L (int cents), model (a) primary — exact for a full close:**
  when `entry_size == exit_size`: `gross_dbn = direction * (exit_notional_dbn − entry_notional_dbn)` (no averaging, exact).
  when `entry_size > exit_size` (partial): `entry_notional_matched = entry_notional_dbn * matched_size // entry_size`; `gross_dbn = direction * (exit_notional_dbn − entry_notional_matched)`.
  `gross_cents = _to_cents(gross_dbn)` where `_to_cents(x) = sign(x) * (abs(x) * DOLLARS_PER_INDEX_POINT * 100 // 1_000_000_000)` — **symmetric** truncation toward zero (exact for any tick-aligned `x`: a tick = 50¢).
  `fees_cents = (exch_reg_fee_usd_cents + commission_usd_cents) * matched_size`.
  `net_primary_cents = gross_cents − fees_cents`. `net_cents` is **net of exchange+commission fees only** — the §6 decision friction ($4 RT) is the downstream evaluator's, not report.py's.
- **Model (b) STRESSED** — a pure transform on (a): `net_stressed_cents = net_primary_cents − 2 * TICK_VALUE_CENTS * matched_size` (`TICK_VALUE_CENTS = _to_cents(MNQ_TICK_DBN) = 50`).
- **Model (c) OPTIMISTIC** — the same pairing + math on the **optimistic** run + its manifest's fees, but **only over trade_ids that completed a round trip in *both* runs**. A `RoundTrip.net_optimistic_cents` is `None` when the primary-completed trade did not complete in the optimistic run (rare — optimistic fills are ≥ primary). `optimistic_only_completed` lists trade_ids the optimistic run closed that primary left open.
- **Output.** `ThreeWayReport`: `round_trips: tuple[RoundTrip, ...]` ordered by `(entry_ts_ns, trade_id)` (**chronological, not `trade_id`** — downstream buckets by time). `RoundTrip = (trade_id, entry_ts_ns, exit_ts_ns, matched_size, direction, net_primary_cents, net_stressed_cents, net_optimistic_cents: int | None, adverse: bool)` — `adverse` iff any *filled* leg of the primary group had `adverse_selection`. `primary` / `stressed` / `optimistic` `ModelPnL`s whose `net_cents` tuples are in that same chronological order (`optimistic` over the both-completed subset). `incomplete: tuple[OpenPosition, ...]`, `partially_closed: tuple[tuple[str,int], ...]`, `optimistic_only_completed: tuple[str, ...]` — all sorted. `to_dict()` also carries `config_primary` / `config_optimistic` (the two `manifest["config"]` dicts) as provenance.
- **`ModelPnL`.** `mean_net_cents -> float | None` (`None` only when `n == 0`). `profit_factor -> float | None` = `gross_profit / -gross_loss`; `float("inf")` when `n > 0` and there are no losing trades; `None` only when `n == 0`.
- **Determinism (AD-11):** no `set`/`dict` iteration order reaches output; every output sequence has a stated total order.
- `PERMITTED_INTERNAL_EDGES["report"]` widens `{"orders"}` → `{"orders", "config"}` (this slice — spine AD-7 note). `mypy --strict src/ticksim` clean, no override; `black`-88; relative imports.

**Ask First:**
- The fee model is **per-contract round turn** (`fee * matched_size`) — confirmed against §2.1 "round turn" (CME exch+reg fees are per-contract). HALT only if a later seal amendment says flat-per-trade.

**Never:**
- Importing `sim` / `book` / `events` / `fills` / `parity` / `cli` / `databento`.
- Evaluating the §6 decision rule / emitting a PASS/FAIL verdict.
- A monetary field entering `OrderOutcome`, or re-running any simulation.
- Mutating the input outcomes/manifests.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Clean long round trip | entry BUY 5 @ Pe, exit SELL 5 @ Px | one `RoundTrip`; `net_primary_cents = _to_cents(5*(Px−Pe)) − (72+58)*5`; `direction == 1` | N/A |
| Short round trip | entry SELL 5 @ Px, exit BUY 5 @ Pe | mirror of the long, `direction == −1` | N/A |
| Bracket, TP fills, SL cancelled | entry FILLED, tp EXIT FILLED, sl EXIT CANCELLED | one round trip from entry + tp; sl's empty fills ignored | N/A |
| Model (b) | any round trip, matched `N` | `stressed.net_cents[i] == primary.net_cents[i] − 100*N` | N/A |
| Chronological order | two trades, T2 entry-ts < T1 entry-ts | `round_trips` and every `net_cents` tuple are `[T2, T1]` | N/A |
| Optimistic completes a primary-open trade | primary exit EXPIRED, optimistic exit FILLED for the same `trade_id` | `trade_id` in `incomplete` **and** in `optimistic_only_completed`; **not** counted in any `ModelPnL` (c is the both-completed subset); no `RoundTrip` emitted | N/A |
| Optimistic leaves a primary round trip open | primary FILLED, optimistic exit EXPIRED (rare) | `RoundTrip.net_optimistic_cents is None`; `optimistic.n` excludes it | N/A |
| Partially closed | entry filled 5, exit filled 3 | `RoundTrip` on 3 (partial entry notional); `("T", 2)` in `partially_closed` | N/A |
| Incomplete (open) trade | entry FILLED 5, exit EXPIRED (no fills) | `OpenPosition("T", 5, avg_entry_px_dbn, entry_ts_ns)` in `incomplete`; not in any `ModelPnL` | N/A |
| exit_size > entry_size | entry filled 3, exit filled 5 | — | `ReportError` |
| Exit no entry | exit fills, entry never | — | `ReportError` |
| Mixed entry sides | two filled entry legs, BUY and SELL, same `trade_id` | — | `ReportError` |
| Exit same side as entry | entry BUY filled, exit BUY filled | — | `ReportError` |
| Mismatched trade_id sets | primary `{T1,T2}`, optimistic `{T1,T3}` | — | `ReportError` |
| Swapped / wrong manifests | both manifests `queue_model == back_of_queue` | — | `ReportError` |
| Manifest missing a fee key | `manifest["config"]` has no `commission_usd_cents` | — | `ReportError` |
| Duplicate order_id | same `order_id` twice in one run's outcomes | — | `ReportError` |
| Adverse round trip | a *filled* leg of the primary group has `adverse_selection == True` | that `RoundTrip.adverse is True` | N/A |
| PF with no losers | every round trip `net > 0` | `primary.profit_factor == float("inf")`, not `None` | N/A |
| Empty study | no outcomes | `round_trips == ()`; every `ModelPnL` `n == 0`, `mean` / `profit_factor` `None` | N/A |
| `to_dict()` populated | a report with a partial close + an open position | `json.dumps` succeeds; `d["partially_closed"] == [["T", 2]]`; `d["config_primary"]["queue_model"] == "back_of_queue"` | N/A |

</frozen-after-approval>

## Code Map

- `src/ticksim/orders.py` — `OrderOutcome` (`trade_id, leg, order_id, kind, side, submit_ts_ns, arrival_ts_ns, terminal_state, fills: tuple[Fill,...], adverse_selection, …`), `Fill` (`px_dbn, size, ts_ns`), `Leg` (ENTRY/EXIT), `Side` (BUY/SELL), `TerminalState` (FILLED/…).
- `src/ticksim/config.py` — `DOLLARS_PER_INDEX_POINT` (= 2), `MNQ_TICK_DBN` (= 250_000_000). Both module-level; fee fields (`exch_reg_fee_usd_cents`, `commission_usd_cents`) reach report.py **only via the manifest dict**, per AD-24.
- `_bmad-output/planning-artifacts/architecture/…/ARCHITECTURE-SPINE.md` — AD-14 (2 runs / 3 reports), AD-24 (money is report-layer, from the manifest), AD-10 (`float` only in report.py), AD-12 (`trade_id` pairs legs), AD-7 (edge list — widened here).
- `tests/unit/test_ticksim_imports.py:46` — `"report": {"orders"}` → `{"orders", "config"}`.

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/report.py` — `ReportError`, `RoundTrip` (frozen), `OpenPosition` (frozen), `ModelPnL` (frozen; `net_cents`, `n`, sums, `wins`, `losses`; `mean_net_cents` / `profit_factor` computed), `ThreeWayReport` (frozen; `round_trips`, `primary`/`stressed`/`optimistic`, `incomplete`, `partially_closed`, `optimistic_only_completed`, `to_dict()`), `build_report(...)`, `TICK_VALUE_CENTS`, `_to_cents`. `__all__`.
- [x] `tests/unit/test_ticksim_imports.py` — `"report"` edge → `{"orders", "config"}`.
- [x] `_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md` — AD-7 rule note `report → orders, config` + `report --> config` graph edge; AD-14 entry-point name note (`build_report`, not `three_way_report`).
- [x] `tests/unit/test_ticksim_report.py` — a test per matrix row + the ACs + the review-1 set.

**Acceptance Criteria:**
- Given a clean long round trip (entry BUY N @ Pe, exit SELL N @ Px), then one `RoundTrip` with `net_primary_cents == _to_cents(N*(Px−Pe)) − (72+58)*N`, `net_stressed_cents == net_primary_cents − 100*N`, `direction == 1`.
- Given a bracket (entry + TP fill, SL cancelled), then exactly one round trip; the SL outcome contributes nothing.
- Given two round trips whose entry timestamps are out of `trade_id` order, then `round_trips` and every `ModelPnL.net_cents` are in ascending `entry_ts_ns` order.
- Given a `trade_id` that is an open position in primary but a completed round trip in optimistic, then it is in `incomplete` and `optimistic_only_completed`, counted in `optimistic.n` only.
- Given `mypy --strict src/ticksim`, then zero errors, no override; `report.py` imports only `orders` + `config`; the widened import-graph test passes.
- Given `ThreeWayReport.to_dict()` on a populated report, then `json.dumps` succeeds and `partially_closed` / `config_primary` round-trip.

## Spec Change Log

### 2026-08-30 review-1 (blind / edge-case / verification-gap) — `bad_spec` on the output shape

**Defect:** all three reviewers flagged that `ModelPnL.net_cents` as a bare `trade_id`-ordered tuple is unusable for the report's own downstream purpose — §6 walk-forward, §7 regime / per-year split, deflated Sharpe and the permutation test all bucket trades by time or key, and `trade_id` is opaque (AD-12). A consumer would have to re-pair the `OrderOutcome` logs, duplicating `build_report`'s core. **Fix (frozen block reworked):** `ThreeWayReport.round_trips: tuple[RoundTrip, ...]` — a per-trade record `(trade_id, entry_ts_ns, exit_ts_ns, matched_size, direction, net_primary_cents, net_stressed_cents, net_optimistic_cents, adverse)` ordered **chronologically** by `(entry_ts_ns, trade_id)`; every `ModelPnL.net_cents` is in that same order.

Also reworked (reviewer-driven, all within intent):
- **(c) population.** A `trade_id` can be an open position under `PRIMARY` but a completed round trip under `OPTIMISTIC` (optimistic fills ≥ primary). Model (c) is now over the **both-completed** subset; `RoundTrip.net_optimistic_cents` is `None` for a primary-completed trade the optimistic run left open; new `optimistic_only_completed` diagnostic.
- **Input validation.** A malformed manifest (missing `["config"]` / a fee key), a non-`int`/negative fee, two manifests that are not a `PRIMARY`+`OPTIMISTIC` pair (swap guard), and a duplicate `order_id` all raise `ReportError`, not a bare `KeyError`/`TypeError`.
- **Side rules.** Mixed `side` across filled entry legs, or an exit leg on the same side as the entry, raise `ReportError` (was: silently pick `min(order_id)`'s side).
- **P&L exactness.** Full close = `direction*(exit_notional − entry_notional)` (no pre-averaging, exact). Partial close uses `entry_notional * matched // entry_size`. `_to_cents` truncates **symmetrically toward zero** (was floor toward −∞ — an asymmetric loss/win bias).
- **`incomplete`** is now `tuple[OpenPosition, ...]` carrying `open_size` / `avg_entry_px_dbn` / `entry_ts_ns` (§2.2 exposure detail), not a bare id list.
- **`profit_factor`** returns `float("inf")` for a profitable strategy with no losers (was `None`, conflated with "no data"); `None` only when `n == 0`.
- `adverse` is read from *filled* legs only; `to_dict()` carries `config_primary` / `config_optimistic` provenance; docstring notes `net_cents` is net of fees only, not the §6 friction. Docstring line-break rendering fixed.

KEEP: the fee model (per-contract round turn, §2.1-confirmed); fees from the manifest not an import (AD-24); the `_pair_run` group-by-`trade_id` structure; `ReportError` on `exit_size > entry_size` (a real impossibility, not a quirk — hard stop for both runs).

## Design Notes

**AD-7 edge widening.** AD-7's rule text lists `report → orders`, but AD-24 (finalized later) makes report.py the sole consumer of `config.DOLLARS_PER_INDEX_POINT`. Widening the edge to `{orders, config}` is the reconciliation; the alternative (a `dollars_per_index_point` field on `Manifest`) is a `sim.py` change for one constant. Fees still come from the manifest (AD-24).

**`_to_cents`.** `sign(x) * (abs(x) * 200 // 1_000_000_000)`. `200 / 1e9 = 1/5_000_000`; `MNQ_TICK_DBN // 5_000_000 = 50`, so a tick is exactly 50¢ and any tick-aligned delta converts exactly. A non-tick-aligned delta (only synthetic test data) truncates toward zero, symmetric for wins and losses.

**Sub-cent VWAP.** For a partial close, `entry_notional * matched // entry_size` truncates below the true weighted average by `< entry_size` dbn ≈ `< 1e-9 * 5` index points — invisible at cent resolution (`// 5_000_000`). Documented, not corrected.

**Multiple filled exit legs** of one `trade_id` (a scale-out, or a bracket bug) are VWAP'd together — correct for scale-out; an upstream bug that fills both TP and SL is the sim's to prevent, not report.py's.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_report.py tests/unit/test_ticksim_imports.py -q` — expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` — expected: full ticksim suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` — expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim/report.py tests/unit/test_ticksim_report.py` — expected: clean.

Result: **384 ticksim tests pass** (30 in `test_ticksim_report.py`), `mypy --strict src/ticksim` clean (8 files), `black` clean.

## Suggested Review Order

**Pairing (`_pair_run`)** — group by `trade_id`; dup-`order_id` guard; sum entry/exit fill size + notional + ts; **side rules** (singleton over filled entry legs; exits opposite); classify → skip / `ReportError` / `OpenPosition` / round trip / partially-closed.

**P&L (`_to_cents`, in `_pair_run`)** — full close exact `direction*(exit_notional − entry_notional)`; partial `entry_notional*matched//entry_size`; `_to_cents` symmetric truncate-toward-zero; `net = gross_cents − fee_per_contract*matched_size`.

**`build_report`** — `_check_manifest_pair` (PRIMARY=back_of_queue + OPTIMISTIC=time_priority); `_fee_per_contract` (KeyError/TypeError → `ReportError`, non-int/negative → `ReportError`); trade_id-set equality; **chronological** `(entry_ts_ns, trade_id)` ordering of `round_trips` and every `ModelPnL`; `net_optimistic_cents = None` for a primary-completed trade the optimistic run left open; `optimistic` `ModelPnL` over the both-completed subset; `optimistic_only_completed`.

**Output** — `RoundTrip` / `OpenPosition` / `ModelPnL` (`profit_factor` → `inf` no-losers, `None` only n==0) / `ThreeWayReport` (+ `config_primary`/`config_optimistic` provenance); all frozen; `to_dict()` JSON-safe.

**Deferred** (`deferred-work.md`): a full-provenance `study_id` / manifest-SHA on the report (cli concern); `from_dict`; per-partial adverse tracking on `OpenPosition`.
