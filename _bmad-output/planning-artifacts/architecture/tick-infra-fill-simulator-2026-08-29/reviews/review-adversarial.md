# Review — adversarial (attack the spine as a builder-vs-builder divergence hunt)

**Target:** `ARCHITECTURE-SPINE.md` (TICK-INFRA Fill Simulator, `src/ticksim/`)
**Inputs obeyed, not reviewed:** `preregistration_tick_data_infrastructure.md` §2 / §2.1–2.3 / §3, Amendment 8 §A8.2, Amendment 9 §A9.3–A9.4.
**Method:** for each seam, construct two units one level down (two modules, or two stories a builder picks up) that each obey **every** AD to the letter and still build incompatibly. Every such pair is a hole to close with a new or tightened AD.

**Verdict: NEEDS-WORK.** The spine is well-formed at the top (18 ADs, single-mutator discipline, enforced import graph, one output contract). But it has several genuine multi-builder divergence holes — not nitpicks — where two conforming builders produce parts that will not compose: `FillEvent` is named but never shaped; the whole per-order→per-trade gap that every §6 decision criterion depends on is unbridged; "our orders in the L3 book or not" is a true architectural fork left open; the passive-fill counter has no owner and crosses a seam AD-9 forbids; and the fee numbers have no legal path to `report.py`. Close the findings below (mostly one new AD each, or a tightening of an existing one) and it is SOLID.

---

## F1 — The merged event stream has no total order; within-tick sequence is nobody's rule (probes: AD-1 merge, AD-11 determinism, AD-18)

**The paradigm line** promises "one merged, strictly time-ordered event stream (MBO book deltas ∪ latency-delayed order arrivals ∪ fills)." **The module decomposition does not build that.** AD-6/AD-18 put `merge_streams` in `events.py` and scope it to `BookEventSource`s only ("`DbnMboSource` … is the only impl now"). Order-arrival interleaving is therefore implicitly `sim.py`'s job, and a "fill event" stream exists nowhere. So the "3-way merge" in the paradigm is actually a book-only k-way merge + ad-hoc arrival handling + no fill-event channel.

**Divergent pair (both obey AD-1 "non-decreasing `ts`", AD-11 "sort explicitly where order matters", AD-5, AD-18):**

- `sim.py` builder interleaves order-arrival-at-`T` **after** all book events at `T` ("the world updates, then we act"). A trade at `T` at our resting order's price is applied to the book before our order is even considered to have arrived.
- `fills.py` builder writes the passive-fill rule (seal §2.1: "cumulative trade volume at our price, **after our arrival**") with a **strict** `trade_ts > arrival_ts`. A trade whose `ts` equals our arrival is neither "queue ahead" (we weren't resting) nor "volume after arrival" (not strictly after) — it silently vanishes.
- Flip either choice and the same trade is now **double-counted** instead.

Both readings satisfy every AD. Neither is designated correct. The same ambiguity poisons AD-16 invariant 1 ("no price-improvement vs the touch at arrival+latency" — the touch *before* or *after* same-`ts` book events?) and invariant 6 (a trade at exactly `ts = clock` is legal to use — but only if `apply_event` has already folded it before `fills.decide` runs, which is unstated).

**Also unspecified:** *which* timestamp is the clock. DBN MBO carries `ts_event` (matching engine) and `ts_recv` (gateway). AD-1/AD-10 say "monotonic int64 ns" without naming the field. `fill_ts ≥ submit_ts + latency` (invariant 3) and "arrival + latency" (invariant 1) are latency arithmetic that implies a gateway-stamped reference; queue causality implies matching-engine order; and `OrderIntent.submit_ts` is in some third epoch never stated to be GLBX-comparable. Two builders will pick different fields and the invariant battery will flake by microseconds. Within a single ns, multiple records tie; DBN preserves file order and carries `sequence`, but the spine never says file order / `sequence` is the tie-break, so a non-stable sort in `merge_streams` permutes an `A` ahead of its own `C` at the same ns and corrupts the book non-deterministically.

**Fix — new AD "canonical event order":**
1. The clock is `ts_event` (state it). All latency arithmetic and every invariant check use `ts_event`; `OrderIntent.submit_ts` is declared to be in the same GLBX `ts_event` ns epoch.
2. `DbnMboSource` yields records in file order; `merge_streams` is a **stable** k-way merge keyed `(ts_event, sequence)`.
3. A single **total event order** over the merged stream: `(ts_event, class_rank, sequence)` where `class_rank` is fixed as `book_delta(0) < order_arrival(1) < deferred_fill_apply(2)`.
4. The per-tick sequence is pinned: at each distinct `ts_event` `T` — fold **all** book deltas at `T` via `apply_event`; **then** inject arrivals at `T`; **then** call `fills.decide` once; **then** `sim` applies returned `FillEvent`s.
5. The passive-rule "volume after arrival" is defined against this order (a same-`T` trade that sorted before the arrival is *not* after arrival; one that sorted after *is*) — pick one and write the inequality.

---

## F2 — `FillEvent` is named but never shaped (probe: AD-5)

AD-5: `fills.decide(book, tracker, clock, config) → list[FillEvent]` and "`sim.py` applies the returned `FillEvent`s through the tracker." **`FillEvent` is defined in no module and has no fields.** AD-12 defines `Fill(px_dbn, size, ts)` (nested inside `OrderOutcome`, owned by `orders.py`). AD-6 gives `orders.py` "`OrderIntent, OrderTracker, OrderOutcome`" and `fills.py` "`decide` + the two queue models" — neither is assigned `FillEvent`.

**Divergent pair:**

- `fills.py` builder defines `FillEvent` locally as `{order_id, px_dbn, size, ts_ns, queue_rank_consumed, adverse_flag}` — carrying the diagnostics the decision computed.
- `sim.py`/`orders.py` builder expects `fills.decide` to return the AD-12 `Fill` (or `(order_id, Fill)` tuples) and calls `tracker.apply_fill(order_id, fill)`; it computes queue rank and adverse-selection elsewhere.

They disagree on the name, the owning module, whether `order_id` is carried, and whether queue/adverse diagnostics ride on the event. **Worse — incremental vs cumulative:** `fills.decide` is stateless (AD-5) and called every tick. One builder returns "all fills that should exist for this order so far"; another returns "the new fill delta this tick." If `sim` applies additively, interpretation 1 double-counts every partial.

**Fix — new AD (or extend AD-5/AD-12):** `FillEvent` is a frozen model in `orders.py` next to `Fill`: `{order_id, px_dbn, size_ns:int, ts_ns}`. It is **strictly this-tick incremental**. It carries **no** queue-rank or adverse-selection field (those are set by defined mechanisms in F4/F5). `sim` applies each `FillEvent` via one named tracker method that is the sole path to a partial/terminal fill transition (AD-8).

---

## F3 — The passive-fill counter ("cumulative trade volume at price since arrival") has no owner and crosses a seam AD-9 forbids (probe: AD-5 vs AD-9 vs AD-8)

AD-5 says this per-order state "lives on the order in `OrderTracker`." AD-9 says `book.apply_event` is the **sole** authority on MBO→book semantics and consumes `T`. AD-7 says `orders.py` (hence `OrderTracker`) imports nothing from ticksim and knows nothing of `book.py`. AD-3 says `fills.py` is **read-only**. So when a trade `T` arrives:

- `apply_event` mutates the book, but **cannot** touch the tracker (AD-7) — ruled out.
- `OrderTracker` holds the counter but never sees a book event unless `sim` feeds it one — it needs a method `sim` calls.
- `fills.decide` is read-only and stateless and gets **no trade record** in its signature (`book, tracker, clock, config`) — it cannot observe the trade at all.
- `sim.py` is the only actor with both the trade record and the tracker.

**Divergent pair:**

- `sim.py` builder: after `apply_event` for a `T` record, also calls `tracker.on_trade(price_dbn, size, ts)`, which walks working passive orders and bumps counters for those at/through price with `arrival_ts < ts`.
- `orders.py` builder: `OrderTracker` has no `on_trade`; the builder assumes `fills.decide` is where trades are observed. `fills.decide` has no trade in its signature → **the passive-fill rule is literally uncomputable** and the two stories deadlock.

**Compounding gap — what consumes "queue ahead":** the back-of-queue rule needs `queue_ahead` to *decrease*. AD-16 invariant 4 only says "non-increasing," so it permits decrease but never says **by what**. Trades ahead of us, yes. But do `C` (cancel) and size-reducing `M` of orders *ahead of us* also decrement our `queue_ahead`? In a real FIFO book they must (if everyone ahead cancels, the next trade fills us). If the model only nets trade volume against a frozen `queue_ahead_size_at_submit`, cancels ahead never help us — a large, unstated, arguably-wrong pessimism. `BackOfQueueModel` and `TimePriorityModel` builders will each pick, and `invariants.py`'s author may assume a third thing.

**Fix — new AD "book-event → order-state seam":** `sim.py` is the sole driver of the seam. After each `apply_event`, `sim` calls one defined method — `queue_model.observe_book_event(tracker, record)` (or `tracker.observe_book_event`) — that updates per-order counters. The AD **enumerates** which MBO actions move `queue_ahead`: `T` at/through price decrements it; `C` and size-reducing `M` of resting orders **ahead of** our order (by `add_ts_ns`/`sequence`) decrement it; everything else does not. `fills.decide` stays a pure function of the resulting tracker state. State the ordering: `apply_event` → `observe_book_event` → (arrivals) → `fills.decide`.

---

## F4 — "Our orders in the L3 book" is an unresolved architectural fork (probes: AD-3, AD-9, queue-rank ownership)

AD-3: "exactly one live book … holding **every resting order** … `sim.py` is the only mutator." Our working passive limit *is* resting. AD-9: `apply_event` governs only MBO **records**; our order is not an MBO record. These are jointly consistent with two incompatible builds:

- `book.py` builder: the L3 book is a **venue-faithful reconstruction only** — real orders from the stream. Our orders live in `OrderTracker`; `fills.decide` overlays tracker + book to reason about queue position.
- `sim.py` builder: `sim` **injects** our working orders into the L3 book as synthetic resting orders (reserved `order_id` range, `arrival_seq` from `sim`'s fold counter) so that `book.query_*` "just works" for the fill engine.

Both satisfy AD-3 ("every resting order"), AD-9 (no MBO record involved), AD-5 (state on the tracker — the injecting builder keeps a book copy *too*). They are different architectures. Queue-rank computation forks the same way: `queue_rank_at_submit` / `queue_ahead_size_at_submit` (AD-12) — computed by `sim` at the arrival transition? by `fills.decide` on its first call for that order? Both? AD-8 ("every field derived from a tracker transition") says the tracker must own the value, but AD-7 forbids the tracker to query `book.py`, so `sim` or `fills` must compute and hand it in — unstated which.

**Also:** `TimePriorityModel` needs to rank our order against real resting orders by time. AD-3's stored per-order tuple is `(instrument_id, side, price_dbn, size, arrival_seq)` — **no `ts_event`**. A `sequence`/`arrival_seq` is a venue-global counter our synthetic order has no natural value in. As stored, **time-priority is not reconstructible** — either the tuple is missing `add_ts_ns` or `TimePriorityModel` is unbuildable.

**Fix — new AD:** (a) the L3 book is venue-faithful only; our orders are **never** inserted. (b) AD-3's per-resting-order tuple gains `add_ts_ns` (the `A` record's `ts_event`). (c) One shared interface both queue models implement identically: `queue_ahead_size(book, side, price_dbn, our_arrival_ts_ns) -> int`. (d) `queue_rank_at_submit` / `queue_ahead_size_at_submit` are computed **exactly once**, by `fills.decide` on the arrival tick (or by `sim` — pick one), written to the tracker order; `report`/`parity` read them only from `OrderOutcome`. (e) The single formula for back-of-queue fill: `filled_qty = max(0, cum_trade_vol_at_price_since_arrival − queue_ahead_size_at_submit + Σ(queue_ahead_decrements))`, capped at order size — write it so both models and `invariants.py` share it.

---

## F5 — `OrderOutcome` is per-order; every §6 decision criterion is per-trade; nothing pairs the legs (probes: AD-14(b), AD-17, AD-12 `kind`)

`report.py` and `parity/` read **only** `OrderOutcome` (AD-12), which is **per order**: `kind, side, submit_ts, arrival_ts, terminal_state, fills[], queue_*`. No position, no trade, no link between the entry order and the exit order of one round trip. But:

- §6 decision rule: "per-trade **net** P&L", "N ≥ 200 **trades**", "both regime partitions same-sign" (a per-trade split), permutation test on trades.
- AD-14 (b): "post-hoc 1-tick adverse slip **on entry and exit**." `report.py` cannot tell which `OrderOutcome` is the entry and which is the exit, nor the position direction to know which way "adverse" points.
- AD-17: part_a replays real **bracket** orders ("limit entry, TP-limit / SL-stop exit"). To compute fill-price error it must pair the sim's exit fill with the real exit fill — but there is no key.

**`kind` is an undefined enum.** One builder: `kind ∈ {marketable, passive_limit}` (matches §3 "by order type limit vs market"). Another: `kind ∈ {entry, tp_exit, sl_exit}`. Both "obey" AD-12.

**Divergent pair:** `orders.py` builder ships `kind = execution-type` and no linkage. `report.py` builder needs `kind = leg-role` + a `trade_id`. Result: `report.py` **cannot produce figure (b), the per-trade net, the regime split, or the trade count** — i.e. cannot evaluate the decision rule at all.

**Fix:** AD-12 adds `trade_id` (opaque, assigned by the intent-log producer) and `leg ∈ {entry, exit}`; `kind` is frozen with explicit values. The **`OrderIntent` schema must be frozen too** (it is leaned on by AD-2 and AD-17 but never enumerated) and must carry `trade_id`/`leg` plus a fixed **replace convention** (one record reusing `order_id`, vs cancel+new) — because AD-17 parity silently breaks if the strategy's intent-log producer and part_a's reconstruction encode replaces or legs differently. Add an AD parallel to AD-12 that freezes `OrderIntent` as a versioned model.

---

## F6 — Fees live in `SimConfig` but `report.py` has no legal path to them (probes: AD-15 vs AD-10 vs AD-12)

AD-15 puts `$0.72 exch+reg + configurable commission` in `config.PRIMARY`. AD-10: "**No float between event ingestion and the `OrderOutcome` log** … Float appears only in `report.py`." AD-12: "`report.py` … read **only** this schema [`OrderOutcome`]." `OrderOutcome` has no fee field. So: fees are dollars (float), must be applied in `report.py`, and `report.py` may only read `OrderOutcome`, which doesn't carry them.

**Divergent pair (three, actually):**

- `report.py` builder A reads the **run manifest** ("`SimConfig` dump", per Consistency Conventions) to recover fees.
- `report.py` builder B takes fees as an explicit function argument from `cli.py`, hardcoded.
- `sim.py` builder C bakes a `fees_charged` float into each `OrderOutcome` — **violating AD-10**.

Compounding: AD-15 calls `PRIMARY` a "module-level `SimConfig` constant" and *also* says commission is "configurable / user sets at seal time" — so either `PRIMARY` is not fully frozen (contradiction) or commission is not in `PRIMARY` and "run … `config = PRIMARY`" (AD-14) is underspecified. And the **`$/index-point` multiplier** for MNQ ($2/pt; tick = $0.50) that `report.py` needs to turn `px_dbn` deltas into the §6 dollar figures is defined nowhere (AD-10 gives only the DBN tick size `250_000_000`).

**Fix — new AD "monetary values are report-layer only":** (a) no monetary field ever enters `OrderOutcome` (ratify AD-10). (b) `report.py` reads fees/commission and the point multiplier from the manifest's `SimConfig` dump — explicitly permitted (the manifest is a run artifact, not "simulator internals"); amend AD-12's "only this schema" to "`OrderOutcome` for fills + the manifest for config/fees." (c) `PRIMARY`/`OPTIMISTIC` bake the **seal-default** commission ($0.58); a study overriding it builds a derived config, recorded in the manifest, and its output is §6-secondary only. (d) `config.py` holds `DOLLARS_PER_INDEX_POINT = 2` as a named, seal-cited constant; `report.py` is its only consumer.

---

## F7 — AD-13 mask hard-fail checks the wrong thing / conflates two failures (probe: AD-13)

AD-13: "Any `OrderIntent` submit-ts, **or any fill ts**, outside the union of intervals ⇒ **abort the run**." Intents are validated at submit, so how does a *fill* land outside? Easily: a passive limit (or a real bracket TP-limit / SL-stop, which part_a replays) submitted near the RTH close and still working at 16:00, then filled by a legitimate trade at 16:00:30 in the post-close session that is present in the MBO stream. Nothing in the seal or AD-8 forces a working order to expire at a session boundary.

**Divergent pair:**

- `sim.py` builder A treats `valid_intervals` as an **input filter**: at each interval's end, force-expire every working order (`working → expired`), so no fill can occur outside. AD-13's fill-check becomes dead defensive code.
- `sim.py` builder B treats it as a **post-hoc assertion**: run the whole stream, let orders fill whenever, then check all fills and abort if any is outside. This build **aborts on a completely normal "limit rode through the close"** — and the parity gate (part_a, real bracket orders held across boundaries) aborts with it.

Both obey AD-13's letter; one completes runs the other kills. Also: `valid_intervals: list[(start_ns, end_ns)]` — inclusive both ends? half-open? unspecified → off-by-one at every one of the 28 window boundaries (Amendment 8/9).

**Fix:** split AD-13 into (a) **intent-log validation** — submit/cancel/replace `ts` outside the mask ⇒ `IntentLogError` (analyst's bad input); (b) **an explicit tracker rule** — a working order is force-transitioned to `expired` at the end of its containing interval (new trigger on the AD-8 state machine), yielding a normal `expired` `OrderOutcome`; (c) **a fill outside the mask is then a true `InvariantViolation`** (must be impossible), not an analyst-facing abort. State intervals are `[start_ns, end_ns)` (or closed — pick one).

---

## F8 — `parity/gate.py` "emits … the seal amendment stub" is undefined (probe: structural seed, §4)

Structural seed: `gate.py # runs A + B, emits PASS/FAIL + the seal amendment stub`. §4 requires the passing simulator **commit SHA** be recorded in the amendment and frozen; Amendment 8 says the parity-gate result is "the next appended" amendment. The stub's format, sections, and content are unspecified.

**Divergent pair:** builder A emits `PASS`/`FAIL` + a numbers table. Builder B emits a full append-only amendment with the frozen SHA, the three Part-A figures vs tolerances, per-trader breakdown, and the Part-B per-invariant violation log. The seal needs B; A silently under-delivers and the "frozen SHA" requirement is unmet.

**Sub-holes:** (i) does `gate.py` shell out to `git rev-parse HEAD` for the SHA? AD-11 bans wall-clock but is silent on `git`/subprocess; AD-4 (isolation) doesn't mention it. (ii) "**3 revision cycles**" — what is a cycle, and does `gate.py` track the count across runs? Builder A: stateless, analyst counts. Builder B: reads/writes a `cycle_count` file in the run dir — persistent state nobody owns. (iii) Part B on first `InvariantViolation` — abort the gate, or run all ≥1000 and report the full violation set? A8.2 "any violation = FAIL" reads fail-fast; "≥1000 synthetic orders … 100%" reads run-all. Two builders, two behaviours, very different debug value.

**Fix:** an AD or Consistency-Conventions entry that (a) gives the exact amendment-stub template (sample `N`; the three Part-A numbers vs the three tolerances; Part-B pass/fail with per-invariant violation counts; simulator SHA obtained via an explicitly-permitted `git rev-parse` call; verdict; cycle number); (b) states cycle-count / kill-criterion bookkeeping is the analyst's (out of code) or a named manifest field — pick one; (c) states Part B runs **all** synthetic orders, collects every violation, and FAILs if the set is non-empty.

---

## Smaller holes (each still a two-builder divergence)

- **F9 — `adverse_selection` timing.** AD-12 field; seal §2.1 needs book state **1 s after the fill**. `fills.decide` can't look ahead; the order is already terminal (AD-8) when the 1 s elapses. Builder A: a second replay pass (but AD-14 says "exactly two `SimRun`s", not two + a post-pass). Builder B: `sim` keeps a deferred-check queue and mutates the terminal outcome before serialization. **Fix:** AD — `sim` computes it via a bounded 1 s-deferred check queue; the tracker order stays mutable for this one field only until `fill_ts + 1 s` or run end; it is **not** a second replay.

- **F10 — our-orders `arrival_seq` provenance.** Is it the DBN `sequence` (global, our synthetic order has none) or a `book.py` fold counter (then our order gets one only if inserted — see F4)? **Fix:** folded into F4's `add_ts_ns` decision — real orders keyed by `(add_ts_ns, sequence)`, our order by `(arrival_ts_ns, +∞)` i.e. always last at its price at submit.

- **F11 — run-id / run-dir.** "`data/ticksim/runs/<run-id>/`" + AD-14 "exactly two `SimRun`s per study." One `<run-id>` with two outcome logs, or two ids? `<run-id>` = deterministic input hash (good, AD-11) or timestamp (wall-clock-ish, new dir every rerun)? **Fix:** `run-id = sha256(source-bytes-manifest ∥ intent-log-sha ∥ config-hash)`; the PRIMARY/OPTIMISTIC pair lives under one `study-id/` with `primary/` and `optimistic/` subdirs; each manifest names its sibling.

- **F12 — AD-18 Protocol doesn't pin L3.** "`BookEventSource` … yielding **book events**." The deferred compact-cache impl could yield L2 deltas; `TimePriorityModel` needs L3 per-order events. The seam is *not* actually kept open for the primary model. **Fix:** AD-18 states the Protocol yields **L3 MBO-equivalent per-order events** (`A/C/M/T/F` with `order_id`), not merely "book events."

- **F13 — AD-9 "a few ms" is not a number.** `book.apply_event`'s transient-cross tolerance and `parity/gate.py`'s preflight cross-check will pick different thresholds; the gate then fails runs `apply_event` accepted. **Fix:** `config.py` names `MAX_TRANSIENT_CROSS_NS`; both consume it. (Amendment 9 §A9.3 already refined the seal's "100.000%" check to "no *persistent* cross" and "must consume `A/C/M/T/F`" — put the millisecond number in code.)

- **F14 — unfilled-exit in parity.** part_a: every real trade is *closed* (both legs have real fills). If the sim's book says no liquidity and the sim leaves the exit unfilled, part_a has a real exit fill and no sim fill to diff. **Fix:** define the handling — a defined-magnitude parity miss, or excluded-with-logged-reason — in AD-17.

- **F15 — Part B over 28 disjoint windows.** "≥1000 synthetic orders … across the Tranche 1 data" (Amendment 8/9: 28 windows, 28 disjoint mask intervals). One stream (fake `ts` discontinuities at seams trip invariant 6) or per-window pooled? Synthetic timestamps outside a window's interval trip AD-13's abort. **Fix:** Part B runs **per-window**, pools results; timestamps drawn only within each window's interval, offset ≥ warm-up from the window start.

---

## ADs whose Rule is not actually enforceable

- **AD-2** — "No … feedback from fills to new intents" is enforceable by construction, but "a non-deterministic or non-causal strategy contaminating the parity gate" is **not** something the sim can detect: an intent log with look-ahead-priced orders runs fine. The Rule should say the sim *validates* the intent log is causally orderable (monotonic `submit_ts`, no replace-before-submit, no cancel of an unknown id) and that deeper non-causality is the producer's responsibility, explicitly out of enforcement scope.
- **AD-10** — "No float between ingestion and the outcome log" is enforced only by review. Add a test that asserts every numeric in the `OrderOutcome` JSONL is integer except in a documented allow-list of zero fields (there should be none).
- **AD-11** — "byte-identical `OrderOutcome` log" is testable (run twice, diff) **only** if the log contains no run-id / path / timestamp. The manifest carries "tool versions" and "input paths" — say explicitly that AD-11's byte-identity is scoped to the `OrderOutcome` log, **not** the manifest.
- **AD-16 invariant 1** — "no price-improvement vs the touch at arrival+latency" is not checkable until F1 pins whether "the touch at arrival" is snapshotted before or after same-`ts` book events.

## Deferred items that let two units diverge

- **"H1 grid-cache … AD-18 keeps the seam open"** — it does not, for the L3 model (F12).
- **"Multi-instrument … book is already per-`instrument_id` so no rework"** — false: `OrderTracker`, `fills.decide`, the merged stream, `valid_intervals`, and `report` P&L are all single-instrument as specified. Not a today problem; the "no rework" claim should be softened to "book is per-id; tracker/report/mask would need per-id keys."
- **"`pyproject.toml` `databento` entry … a build task, not an architecture decision"** — Amendment 9 §A9.5 explicitly makes it a **required** pre-build step ("must be added to `pyproject.toml` before the simulator build"). Move it out of Deferred into the structural seed / first story's acceptance.

---

## What the spine gets right (so the rebuild is cheap)

Single-mutator rule (AD-3 / AD-8 / "only `SimRun` mutates"), the enforced import graph (AD-7), one versioned output contract (AD-12), integer hot path (AD-10), seal-bound presets with a violation tripwire (AD-15), invariants defined once and shared by gate + unit tests (AD-16), and the pure-function core (AD-2/AD-5) are all correct and load-bearing. The holes above are almost all *underspecification at a named seam*, not wrong decisions — each closes with one new AD or one tightened sentence.
