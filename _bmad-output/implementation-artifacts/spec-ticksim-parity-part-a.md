---
title: 'ticksim parity/part_a.py — real-fill calibration core (prereg §A8.2 Part A)'
type: 'feature'
created: '2026-08-30'
status: 'done'
review_loop_iteration: 1
baseline_commit: '770ca86'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/preregistration_tick_data_infrastructure.md'
---

<!-- SPLIT (2026-08-30, planning token check): the MBO-window runner — DbnMboSource
     loading, per-window sim.simulate calls, the AD-17 unfilled-exit touch lookup,
     the OPTIMISTIC pass, AD-7 `parity → events` widening, and the integration
     test against the test window — is deferred to part_a.py slice 2 (`run_part_a`).
     This slice is the pure core: reconstruction + fill comparison + aggregate. -->

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The parity gate's Part A (prereg §A8.2) must replay the **orders** the live bots (`trader-mim-nb`, `trader-yank` ≥2026-06-17) actually placed and check simulated fill prices against real broker fills — mean abs error ≤1.0 tick, p90 ≤2.0 ticks, mean signed error within ±0.25 tick, N≥28. Spine AD-17 forbids feeding outcomes back in: the intent log is *reconstructed orders*; the real fill price is only the comparison target.

**Approach:** add `src/ticksim/parity/part_a.py` with the pure core — (1) reconstruct a per-trade `OrderIntent` log + the list of real broker fills from the real order records, (2) given the `OrderOutcome`s a sim run produced for that trade, join each real fill to its outcome and compute the signed tick error, (3) aggregate all per-fill errors into MAE / p90 / signed-bias and a PASS/FAIL verdict. Each trade carries a **fidelity tier** (`broker_fill` vs `bar_reconstructed`) so a pass hinging on low-fidelity rows is visible (AD-17: recorded and counted, never silently excluded). Calling `sim.simulate` over real MBO windows is the next slice (`run_part_a`).

## Boundaries & Constraints

**Always:**
- **Reconstruction is orders, never outcomes (AD-17).** Real `entry_price` / `exit_price` / `exit_reason` fix *when* an order is submitted and are the comparison target — never a sim input.
- **mim-nb reconstruction — minimal 2-leg replay** (human decision, loopback 1). The `data/mim_nb/orders.csv` lifecycle (`ts_utc,event,order_id,otype,side,size,price,outcome,chain`; `otype 2` = market, `otype 4` = protective stop; `event ∈ PLACE|CANCEL|FILL|REJECTED`; `side` `0`=buy/`1`=sell) is walked to identify each trade's **entry market order** (`otype 2` PLACE→FILL) and its **exit market order** (a later `otype 2` PLACE→FILL, placed when the bot's cat-stop / EOD logic fires). Emit exactly two `OrderIntent`s per trade: `marketable` entry at the entry `PLACE` ts, `marketable` `exit` at the exit `PLACE` ts, sharing one `oco_group_id`. The `otype 4` protective stop is parsed to follow the lifecycle but **never emitted** — in the real ledger it is always cancelled before the exit and never fills, so it produces no fill to grade; Part A does not exercise resting-stop queue behaviour (that is Part B). The entry FILL and exit FILL become `RealFill`s tagged `fidelity="broker_fill"` (price + ts from the FILL rows).
- **mim-nb unexpected shapes → `PartAError` naming the row** (fail-closed, all verified absent from the ledger 2026-08-30): a non-market entry (`otype != "2"` on the entry PLACE); an `otype 4` FILL (the protective stop actually fired); a `REPLACE` / `MODIFY` event. Rows with `event`/`outcome` `REJECTED` or `order_id` `FAIL` are dropped (not orders). A trade left mid-lifecycle at ledger end (entry seen, exit not) → `PartAError` — a truncated ledger is a data fault, never a silently-dropped or 0-fill trade.
- **yank reconstruction — strictly 2 marketable legs** (human decision, loopback 1). From a `trades.db` `trades` row (used when there is no order log; yank ≥2026-06-17 is almost all `metadata.backfill = true`): `timestamp` (entry ts), `direction` (`S`/`L`/`SHORT`/`LONG`), `entry_price` / `exit_price` (index points), optional `exit_timestamp`, `metadata` JSON (`contracts`, optional `bars_held`). Emit a `marketable` entry at `timestamp` and a `marketable` `exit` at the exit ts (`exit_timestamp` if present, else `timestamp + max(bars_held,1)` minutes, `bars_held` default 60), sharing one `oco_group_id`. **No TP/SL limit legs** — the real sample carries none. Both `RealFill`s tagged `fidelity="bar_reconstructed"`.
- All legs of one trade share an `oco_group_id` (AD-25). `OrderIntent` records are schema-valid (AD-23): non-decreasing `submit_ts_ns`, `passive_limit`/`marketable_limit` ⟹ `limit_px_dbn` set (n/a here — both legs are `marketable`). An exit intent whose ts is ≤ its entry intent's ts → `PartAError` (causally corrupt bracket).
- **Price conversion.** MNQ index price → DBN 1e-9 fixed-point as `round(px * 1e9)` (exact nano-units, no tick snap) for **both** the reconstructed order context and every `RealFill` — the comparison is done in ticks (`/ MNQ_TICK_DBN`) and must reveal a real fill that sits off a tick boundary, not hide it. A non-finite or non-positive price → `PartAError`.
- **Fill comparison** (`compare_fills(outcomes, trade)`): for each `RealFill`, find the `OrderOutcome` with the matching `(order_id, leg)`; a duplicate `(order_id, leg)` outcome or a missing one → `PartAError`. If `outcome.side != real.side` → `PartAError` (join/reconstruction mismatch). `signed_error_ticks = (sim_vwap_dbn − real_fill_dbn) / MNQ_TICK_DBN`, sign oriented so **positive = simulator worse for the trader** (paid more on a buy, received less on a sell). Multiple sim fills for one leg → size-weighted mean sim price. A single `RealFill` per `(order_id, leg)` is assumed (real sample is 1–2 lot, single fill); a leg's summed sim fill size not matching `RealFill.size` is allowed (the sim may split) but is not itself an error. An outcome with `terminal_state != FILLED` for a leg that really filled → `FillError` with `miss_reason="leg_unfilled"` and `signed_error_ticks=None` (magnitude supplied by the runner slice); it still counts toward `n` and blocks PASS.
- **Aggregate + verdict** (`aggregate(errors)`): over all `FillError`s with a non-`None` error — `mae_ticks = mean(|e|)`, `p90_ticks` = 90th percentile of `|e|` (linear interpolation, numpy default), `signed_bias_ticks = mean(e)`; `n` = count of all `FillError`s (including unresolved misses). `PASS` iff `n >= PART_A_MIN_N and mae_ticks <= PARITY_MAE_MAX_TICKS and p90_ticks <= PARITY_P90_MAX_TICKS and abs(signed_bias_ticks) <= PARITY_SIGNED_BIAS_MAX_TICKS and unresolved_miss_count == 0`. The result carries the same three stats over the `broker_fill`-only subset. A `warning` is set when **either** the `broker_fill` subset is empty (the verdict rests entirely on `bar_reconstructed` rows) **or** the subset's quality verdict (same bounds, no N floor) disagrees with the full verdict.
- `mypy --strict src/ticksim` clean, no override; `black`-88; relative imports (`from ..orders import …`, `from ..config import …`). `PERMITTED_INTERNAL_EDGES["part_a"] = {"orders", "config"}`.

**Ask First:**
- (resolved, loopback 1) Sample fidelity + the mim-nb/yank reconstruction shape were renegotiated with the human — see the Spec Change Log.

**Never:**
- Feeding `entry_price` / `exit_price` into the sim as an order price or a fill (AD-17).
- Emitting the `otype 4` protective stop as an `OrderIntent`, or reconstructing a TP/SL limit leg for yank — Part A is pure fill-price replay; resting-order queue realism is Part B's ≥1000-synthetic battery.
- Importing `sim`, `events`, `report`, `book`, `databento`, or calling `simulate` — this slice is pure and takes `OrderOutcome`s as input. Running Part B, evaluating the §6 H1 verdict, writing the frozen SHA, or any network / Tranche-1 pull.
- Silently excluding any trade or fill from `n`; producing a `ReconstructedTrade` with zero `real_fills`.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| mim-nb market entry + cat-stop/EOD market exit | `orders.csv`: entry PLACE(otype2)→FILL, stop PLACE(otype4)→CANCEL, exit PLACE(otype2)→FILL | one `ReconstructedTrade`: **2-leg** intent log (`marketable` entry + `marketable` exit), one `oco_group_id`, 2 `RealFill`s `fidelity="broker_fill"` | N/A |
| mim-nb entry never fills (cancelled) | entry PLACE then CANCEL, no FILL | trade skipped only if no exit either; if entry FILL missing at ledger end mid-trade → error | `PartAError` (truncated ledger) |
| mim-nb non-market entry | entry PLACE `otype` ≠ `"2"` | — | `PartAError` naming the row |
| mim-nb protective stop fires | `otype 4` FILL row present | — | `PartAError` naming the row (never seen; fail-closed) |
| yank backfilled row | `trades.db` row, `metadata={"contracts":2,...,"backfill":true}` | 2-leg `marketable` bracket; `fidelity="bar_reconstructed"`; 2 `RealFill`s | N/A |
| yank row, exit ts ≤ entry ts | bad `exit_timestamp` | — | `PartAError` (causally corrupt) |
| sim buy fill one tick above real | outcome entry fill `dbn` = real + `MNQ_TICK_DBN` | `FillError` `signed_error_ticks == +1.0` | N/A |
| sim sell fill one tick below real | short entry | `signed_error_ticks == +1.0` (sim worse) | N/A |
| leg really filled, sim outcome unfilled | outcome leg `terminal_state != FILLED` | `FillError` `miss_reason="leg_unfilled"`, `signed_error_ticks=None`; counts toward `n`; blocks PASS | N/A |
| N below minimum | fewer than `PART_A_MIN_N` `FillError`s | `verdict="FAIL"`, `reason` names the N shortfall | N/A |
| MAE / p90 / bias over tolerance | aggregate exceeds any bound | `verdict="FAIL"` naming each failed bound | N/A |
| all bounds met, N≥28, no misses | — | `verdict="PASS"` | N/A |
| verdict rests entirely on bar-reconstructed | every `FillError.fidelity == "bar_reconstructed"` | `warning` set (broker_fill subset empty); `verdict` unchanged | N/A |
| `broker_fill` subset verdict ≠ full verdict | full PASS, subset FAIL (or vice-versa) | `warning` set; `verdict` still from the full N≥28 set | N/A |
| real fill, no matching / duplicate / side-mismatched outcome | reconstructed `(order_id, leg)` absent, doubled, or wrong side in `outcomes` | — | `PartAError` |
| malformed record | `orders.csv` FILL with no `order_id`; unparseable / non-positive price; bad `direction` | — | `PartAError` naming the row |

</frozen-after-approval>

## Code Map

- `src/ticksim/parity/part_a.py` — NEW. Frozen dataclasses: `RealFill(order_id, leg, side, size, price_dbn, ts_ns, fidelity)`; `ReconstructedTrade(trade_id, intents: tuple[OrderIntent,...], real_fills: tuple[RealFill,...], fidelity)` with a `__post_init__` asserting ≥1 real fill, non-decreasing `intents` `submit_ts_ns`, and a single shared `oco_group_id`; `FillError(trade_id, order_id, leg, real_dbn, sim_vwap_dbn|None, signed_error_ticks|None, miss_reason|None, fidelity)` — the 8th field `fidelity` is required so `aggregate` can compute the subset stats from `errors` alone; `PartAStats(n, mae_ticks, p90_ticks, signed_bias_ticks)`; `PartAResult(stats, broker_fill_stats, verdict, reason, warning, errors)`. `Fidelity` / `Verdict` `Literal` aliases (both in `__all__`). Functions `reconstruct_mim_nb(rows) -> list[ReconstructedTrade]`, `reconstruct_trades_db_row(row) -> ReconstructedTrade`, `compare_fills(outcomes, trade) -> list[FillError]`, `aggregate(errors) -> PartAResult`, `PartAError`.
- `src/ticksim/orders.py:124` — `OrderIntent` (`schema_version, action, order_id, trade_id, leg, kind, side, size, limit_px_dbn, submit_ts_ns, replaces_order_id, oco_group_id`); `OrderOutcome` (`order_id, leg, kind, side, terminal_state, fills: tuple[Fill(px_dbn,size,ts_ns),...]`); `IntentAction`, `Leg`, `OrderKind`, `Side`, `TerminalState`. `OrderIntent` rejects `submit_ts_ns < 0` (pydantic) — `reconstruct_*` must raise `PartAError` first for a pre-epoch / negative ts.
- `src/ticksim/config.py` — `MNQ_TICK_DBN=250_000_000`, `PART_A_MIN_N=28`, `PARITY_MAE_MAX_TICKS=1.0`, `PARITY_P90_MAX_TICKS=2.0`, `PARITY_SIGNED_BIAS_MAX_TICKS=0.25`. (`ticks_to_dbn` converts *ticks*, not index prices — not the right helper here; a local `round(px * 1_000_000_000)` is correct and asserts nothing about the tick size.)
- `data/mim_nb/orders.csv` — real lifecycle; every entry verified `otype 2`, every `otype 4` PLACE→CANCEL (never FILL), no `REPLACE`. `chain` = a hash-chain integrity column (not validated here — deferred to the loader).
- `data/trades.db` `trades` — `trader_id, timestamp (+00:00 UTC-aware), symbol, direction, entry_price, exit_price, exit_reason, metadata`. yank ≥2026-06-17 metadata: `{"contracts", "gap_size", "backfill"}` — no `tp_price`/`sl_price`.
- `src/ticksim/parity/__init__.py` — extend the allowed-edges docstring for `part_a` (`{orders, config}` only).
- `tests/unit/test_ticksim_imports.py:39` — add `"part_a": {"orders", "config"}`.
- `_bmad-output/…/ARCHITECTURE-SPINE.md` — AD-17, AD-23, AD-25 (no AD-7 change — pure core imports only `orders`/`config`).

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/parity/part_a.py` — the dataclasses + `reconstruct_mim_nb` + `reconstruct_trades_db_row` + `compare_fills` + `aggregate` + `PartAError`, `__all__`, relative imports.
- [x] `src/ticksim/parity/__init__.py` — extend the allowed-edges docstring note for `part_a`.
- [x] `tests/unit/test_ticksim_imports.py` — add the `"part_a"` edge row.
- [x] `tests/unit/test_ticksim_parity_part_a.py` — 69 tests: reconstruction from hand-built `orders.csv` rows (2-leg output, kinds, shared `oco_group_id`, `RealFill` capture, fidelity; non-market entry / `otype 4` FILL / trailing-mid-trade / malformed-row / `MODIFY` / `FILL`+REJECTED / FLAT-state-desync → `PartAError`; `reconstruct_mim_nb([]) == []`) and a `trades.db` row (2-leg, `bars_held` fallback, empty-`exit_timestamp` fallback, `direction` numeric tokens, exit-ts-≤-entry-ts error, bad direction, `metadata` None / non-mapping, sqlite `id` in trade_id); the `compare_fills` sign convention both sides, size-weighted VWAP, `leg_unfilled` miss, foreign-`trade_id` filter, no-outcome / duplicate / side-mismatch `PartAError`; `aggregate` MAE/p90/bias on a known vector, every verdict branch, the `broker_fill`-empty warning, subset-disagreement warning, N-floor-only no-spurious-warning, no-warning-on-agreement; `ReconstructedTrade.__post_init__` rejections; `_parse_ts_ns` `Z`/space/naive/fractional; no-tick-snap exact conversion.

**Acceptance Criteria:**
- Given a hand-built mim-nb `orders.csv` lifecycle (entry PLACE→FILL @ X, stop PLACE→CANCEL, exit PLACE→FILL @ Y), when `reconstruct_mim_nb` runs, then one `ReconstructedTrade` with a **2-leg** `OrderIntent` log (`marketable` entry, `marketable` exit) sharing an `oco_group_id`, and two `RealFill`s tagged `fidelity="broker_fill"`; the `otype 4` order appears in no intent.
- Given `compare_fills` on outcomes whose fills exactly equal the real fills, then every `FillError.signed_error_ticks == 0.0`; and `aggregate` → `mae_ticks == 0`, `signed_bias_ticks == 0`, `verdict == "PASS"` iff `n >= 28`, with `warning` set (broker_fill subset empty) when every error is `bar_reconstructed`.
- Given one sim buy fill one tick above the real fill, then that `FillError.signed_error_ticks == +1.0`.
- Given `mypy --strict src/ticksim`, then zero errors, no override; `part_a.py` imports only `orders` + `config` from `src.ticksim`; the import-graph test passes with the new row.

## Spec Change Log

**Loopback 1 — 2026-08-30 — `intent_gap` (reviewer trio: blind-hunter + edge-case-hunter, verification-gap clean).** The frozen reconstruction shape was underspecified; renegotiated with the human.

| # | Was (iteration 0) | Now | Why |
|---|---|---|---|
| 1 | mim-nb = "a `marketable` entry, the protective stop (`otype 4`) as an OCO-sibling `exit`-leg intent, and the real exit as a `marketable` `exit` intent" (3 legs); CANCEL emission unspecified | **2-leg minimal replay**: `marketable` entry + `marketable` exit only; the `otype 4` stop is parsed but never emitted (always cancelled, never fills in the real ledger) | The 3-leg form leaves the stop `WORKING` in the sim for the whole window — it can fill intraday when reality had cancelled it, silently diverging the replay. Part A grades fill *prices*; resting-order queue realism is Part B's job. Human chose "minimal replay". |
| 2 | "the AD-23 replace convention is preserved where the order log shows a replace" | a `REPLACE`/`MODIFY` row → `PartAError` naming it (fail-closed) | The mim-nb ledger is `PLACE`/`CANCEL`/`FILL`/`REJECTED` only (verified). Implementing speculative replace handling for a format that has none is dead, untested code. |
| 3 | yank: "reconstruct the generic bracket … `metadata` (`tp_price`/`sl_price`, …)" — implementer added `marketable_limit` TP/SL legs | **strictly 2 `marketable` legs** (entry + exit); no TP/SL legs | The real yank ≥2026-06-17 rows carry no `tp_price`/`sl_price`. The extra legs were dead for this sample and created a brittle `exit_reason`-guessed `(order_id, leg)` join → spurious `leg_unfilled` auto-FAILs. Human confirmed. |
| 4 | `aggregate` `warning` "when the two verdicts disagree" | `warning` when the `broker_fill` subset is **empty** OR the two verdicts disagree | The frozen Intent promises "a pass hinging on low-fidelity rows is visible". A PASS built from 28 `bar_reconstructed` rows and zero broker fills was completely silent under the disagreement-only rule. Human confirmed. |
| 5 | (unspecified) trailing mid-lifecycle mim-nb trade | → `PartAError` | The implementation produced a 0-fill `ReconstructedTrade` that silently dropped out of `n` — violates the frozen "Never silently exclude". Human confirmed. |
| 6 | "MNQ index prices → DBN 1e-9 units, rounded to `MNQ_TICK_DBN`" | `round(px * 1e9)` exact nano-units, no tick snap, for reconstruction context **and** every `RealFill` | Snapping the real-fill comparison target to the tick grid before the comparison hides a real fill that sits off a tick boundary — exactly the anomaly Part A should catch. |

**KEEP (survived re-derivation):** the sign convention (`positive = sim worse for the trader`, side-driven not leg-driven); the `fidelity` field on `FillError` (justified — `aggregate` needs it); the `_percentile` linear-interpolation helper matching numpy default (verification-gap confirmed `[0,1,-2,3,-4] → MAE 2.0 / p90 3.6 / bias −0.4`); the fail-closed `PartAError` posture throughout; `reconstruct_mim_nb` as a timestamp-sorted lifecycle walk.

## Design Notes

**Part A does not exercise the resting protective stop.** mim-nb's `otype 4` disaster stop is always cancelled before the real exit and never fills, so it grades nothing; yank has no resting limit legs. Part A is a pure *fill-price* replay of the two market orders per trade. Resting-order queue realism — back-of-queue position, partial fills against real trade volume, non-through-limit fills — is what Part B's ≥1000 synthetic orders test. A `deferred-work.md` entry records this so the gate's coverage is honest.

**`leg_unfilled` split of responsibility.** This pure slice can detect a leg the bot really filled coming back unfilled from the sim, but cannot price the miss (needs the window book at `exit_ts`). It emits `FillError(signed_error_ticks=None, miss_reason="leg_unfilled")`; the `run_part_a` slice fills in the magnitude (touch @ exit_ts + 1 tick slip, AD-17) before the final aggregate. `aggregate` here treats any unresolved miss as an automatic FAIL so a half-run cannot report PASS.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_parity_part_a.py tests/unit/test_ticksim_imports.py -q` — expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` — expected: full ticksim unit suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` — expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim/parity tests/unit/test_ticksim_parity_part_a.py` — expected: clean.

## Suggested Review Order

**Reconstruction — orders never outcomes (AD-17)**

- Entry point: the frozen 2-leg replay contract and why the `otype 4` stop is never emitted.
  [`part_a.py:475`](../../src/ticksim/parity/part_a.py#L475)

- mim-nb lifecycle state machine: PLACE/FILL/CANCEL walk, fail-closed on every unexpected shape.
  [`part_a.py:520`](../../src/ticksim/parity/part_a.py#L520)

- `_build_mim_trade`: the two `marketable` intents, shared `oco_group_id`, opposite-side + causal-order guards.
  [`part_a.py:399`](../../src/ticksim/parity/part_a.py#L399)

- yank generic bracket from a `trades.db` row: strictly 2 marketable legs, `bars_held` fallback, sqlite `id` in the trade_id.
  [`part_a.py:667`](../../src/ticksim/parity/part_a.py#L667)

**Comparison + verdict**

- `compare_fills`: trade-scoped `(order_id, leg)` join, size-weighted VWAP, sign convention (positive = sim worse for the trader), `leg_unfilled` miss.
  [`part_a.py:787`](../../src/ticksim/parity/part_a.py#L787)

- `aggregate`: MAE / p90 (numpy-default interp) / signed-bias, PASS rule, `broker_fill`-subset warning (empty OR quality-disagreement, N floor excluded).
  [`part_a.py:933`](../../src/ticksim/parity/part_a.py#L933)

**Value types + parsing**

- `ReconstructedTrade.__post_init__` invariants; `FillError` carries `real_ts_ns` / `sim_terminal_state` for the runner slice.
  [`part_a.py:156`](../../src/ticksim/parity/part_a.py#L156)

- `_parse_ts_ns` (Z/space/fractional normalization) and `_px_to_dbn` (exact `Decimal` nano-units, no tick snap).
  [`part_a.py:273`](../../src/ticksim/parity/part_a.py#L273)

**Peripherals**

- Import-graph edge: `part_a → {orders, config}` only.
  [`test_ticksim_imports.py:47`](../../tests/unit/test_ticksim_imports.py#L47)

- 69 unit tests.
  [`test_ticksim_parity_part_a.py:1`](../../tests/unit/test_ticksim_parity_part_a.py#L1)
