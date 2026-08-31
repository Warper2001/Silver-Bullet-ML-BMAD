"""Parity gate for the ``src/ticksim`` fill simulator (spine AD-6, AD-16, AD-26).

Pre-registration §A8.2 defines a two-part parity gate: Part A replays the real
orders the live bots placed and compares fill prices; Part B runs >=1000
synthetic orders through the simulator and requires all six invariants to hold
100%. The six invariants are defined once here, in :mod:`invariants`, as pure
assertion functions consumed by both ``part_b.py`` and ``tests/unit/``
(spine AD-16).

This subpackage may import from ``sim``, ``report``, ``book``, ``config`` and
``events`` (spine AD-7), plus ``orders`` (AD-7 inline note, 2026-08-30 —
``invariants.py`` needs the ``OrderIntent`` / ``OrderOutcome`` types); it imports
nothing from any other ``src.*`` package (spine AD-4). Within the subpackage a
module may import a sibling (``part_a_runner`` imports ``part_a``); those edges
are tracked in ``tests/unit/test_ticksim_imports.py`` under the sibling stem.

``part_a.py`` (the prereg §A8.2 Part A pure core — order reconstruction + fill
comparison + aggregate) is the tightest case: it imports only ``orders`` and
``config`` from ``src.ticksim`` — it never touches ``sim`` / ``events`` /
``report`` / ``book`` / ``databento`` and never calls ``simulate`` (it takes
``OrderOutcome``\\ s as input). ``PERMITTED_INTERNAL_EDGES["part_a"] =
{"orders", "config"}``.

``part_a_runner.py`` (the AD-17 Part A MBO-window runner, ``run_part_a``) is the
slice that *does* run the simulation: for each ``ReconstructedTrade`` it calls
``sim.simulate`` over an injected window ``events.BookEventSource``, grades the
outcomes with ``part_a.compare_fills``, resolves each ``leg_unfilled`` miss from
a bounded read-only re-walk of that source (the window book's touch at the real
fill ts + one tick of adverse slip, AD-17), and folds the whole error set
through ``part_a.aggregate``. The bounded re-walk is ``_bookwalk``'s job now, so
``part_a_runner`` no longer imports ``book`` directly.
``PERMITTED_INTERNAL_EDGES["part_a_runner"] = {"sim", "events", "orders",
"config", "part_a", "_bookwalk"}``.

``_bookwalk.py`` (``BookReplay`` / ``BookWalkError``) is the shared bounded,
read-only forward book replay -- one ``book.Book`` folded over one
``events.BookEventSource`` up to a non-decreasing ``ts_ns`` cutoff, with
fail-closed guards (source ``ts_event`` regression, cutoff regression across
calls, a second ``instrument_id``, a non-``StopIteration`` iterator error).
``part_a_runner._touch_at`` is a thin wrapper over it; ``part_b`` does **not**
use it (loopback 1 -- no book replay in Part B). The slice-2 synthetic-order
generator will need it for BBO sampling.
``PERMITTED_INTERNAL_EDGES["_bookwalk"] = {"book", "events"}``.

``part_b.py`` (the prereg §A8.2 Part B battery runner, ``run_part_b``) does one
``sim.simulate`` over the >=1000 synthetic orders, joins each ``OrderOutcome``
to its ``OrderIntent`` on ``order_id``, runs ``invariants.check_order`` per pair
and collects every ``Violation`` into a ``PartBResult``. It does **no** book
replay -- invariant 5's book-liquidity half is a ``fills.py`` construction
guarantee, treated exactly as ``invariants.py`` already treats invariant 4's
queue time-series and invariant 6's merge ordering (loopback 1, 2026-08-31);
``PART_B_COVERAGE_NOTE`` records this verbatim. ``events`` is imported only for
the ``BookEventSource`` type annotation. ``PERMITTED_INTERNAL_EDGES["part_b"] =
{"sim", "orders", "config", "invariants", "events"}``.

``gate.py`` (the prereg §A8.2 / spine AD-26 output contract) folds a
``part_a.PartAResult`` + a ``part_b.PartBResult`` into the two-part verdict
(``evaluate`` -- PASS iff **both** parts pass), resolves the frozen simulator
commit via the one sanctioned ``subprocess`` call (``frozen_sha`` --
``git rev-parse HEAD``; AD-4 / AD-11 otherwise stand), and renders the
fixed-template append-only amendment stub (``build_amendment_stub`` -- returns
text, never writes a file). It imports only ``config`` and its two siblings and
makes no ``sim`` / ``events`` / ``book`` call -- it consumes the two part
results as values. ``PERMITTED_INTERNAL_EDGES["gate"] = {"config", "part_a",
"part_b"}``.
"""
