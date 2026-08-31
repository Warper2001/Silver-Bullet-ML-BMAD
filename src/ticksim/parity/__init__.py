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
through ``part_a.aggregate``. ``PERMITTED_INTERNAL_EDGES["part_a_runner"] =
{"sim", "events", "book", "orders", "config", "part_a"}``.
"""
