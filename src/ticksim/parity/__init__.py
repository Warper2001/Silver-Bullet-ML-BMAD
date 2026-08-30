"""Parity gate for the ``src/ticksim`` fill simulator (spine AD-6, AD-16, AD-26).

Pre-registration §A8.2 defines a two-part parity gate: Part A replays the real
orders the live bots placed and compares fill prices; Part B runs >=1000
synthetic orders through the simulator and requires all six invariants to hold
100%. The six invariants are defined once here, in :mod:`invariants`, as pure
assertion functions consumed by both ``part_b.py`` and ``tests/unit/``
(spine AD-16).

This subpackage may import from ``sim``, ``report``, ``book`` and ``config``
(spine AD-7), plus ``orders`` (AD-7 inline note, 2026-08-30 — ``invariants.py``
needs the ``OrderIntent`` / ``OrderOutcome`` types); it imports nothing from any
other ``src.*`` package (spine AD-4).
"""
