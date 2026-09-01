"""The §A8.2 parity-gate orchestrator -- ``run_parity_gate`` (spine AD-26).

Every §A8.2 building block is built and merged:

* ``part_a`` / ``part_a_runner.run_part_a`` -- real-fill calibration (Part A),
* ``synthetic.generate_synthetic_orders`` + ``part_b.run_part_b`` -- the
  >=1000-order synthetic invariant battery (Part B),
* ``integrity.preflight_integrity`` / ``format_integrity`` -- the seal-§5 /
  prereg §1 window preflight,
* ``gate.evaluate`` / ``gate.build_amendment_stub`` -- the AD-26 two-part
  verdict + the append-only amendment stub.

Nothing wired them into the one call an analyst makes to get a verdict + stub.
:func:`run_parity_gate` is that wiring -- a **pure orchestrator**: it opens no
``.dbn.zst`` path, touches no ``data/trades.db`` / ``data/mim_nb/`` file and
imports no ``databento``. The CLI (``cli.py parity-gate``) does all file / DB I/O
and injects ``source_for`` (a ``WindowKey -> BookEventSource`` callable that
already did the front-month filter + the ``.dbn.zst`` open + the ts-clip) and the
reconstructed trades.

Steps, in order (spec Always):

1. **Part A** -- ``part_a_runner.run_part_a`` over the reconstructed trades, each
   trade routed to its window's source by ``_window_of`` (the trade's full stamp
   span -- every intent ``submit_ts_ns`` and every ``RealFill.ts_ns`` -- must lie
   inside one window's ``[lo_ns, hi_ns)``; CHECKPOINT 1d).
2. **Part B** -- ``generate_synthetic_orders`` over the one dense
   ``synthetic_window`` (CHECKPOINT 1b), then ``run_part_b`` over the same
   source.
3. **Integrity** -- ``preflight_integrity`` for every distinct window touched
   (Part A windows union ``synthetic_window``); each ``format_integrity`` result
   is flattened onto one ``window <key>: ...`` line and joined into the ``integrity``
   string (``build_amendment_stub``'s ``_reject_template_break`` bars newlines and
   ``## `` -- round-1). Window identity + OK/FLAGGED status preserved.
4. **Verdict + stub** -- ``gate.evaluate`` then ``gate.build_amendment_stub``.

A ``FLAGGED`` integrity report does **not** change the verdict (CHECKPOINT 1a --
AD-26's rule "Part A PASS AND Part B PASS" stands verbatim). :class:`GateRun`
carries ``integrity_flagged=True``, the stub's integrity section names the flagged
window(s), and the CLI exits ``3`` on a flagged-but-PASS run.

Frozen-signature reconciliation (Spec Change Log, round 1): ``run_parity_gate``
gains a ``windows: Mapping[WindowKey, WindowSpec]`` positional (2nd) -- a
``WindowKey`` is a ``str`` and cannot carry the ``lo_ns`` / ``hi_ns`` the
trade->window routing and Part B need. ``config`` is ``SimConfig | None`` (``None``
forwards each runner's own ``PRIMARY`` default -- the ``gate_cli`` edge excludes
``config`` so ``PRIMARY`` cannot be imported at runtime; ``SimConfig`` is a
``TYPE_CHECKING``-only reference). ``amendment_number <= 0`` is rejected at the top
of ``run_parity_gate``, before any compute.

Dependencies (spine AD-7, ``PERMITTED_INTERNAL_EDGES["gate_cli"] = {"part_a",
"part_a_runner", "synthetic", "part_b", "integrity", "gate", "events"}``):
relative form only. No runtime ``config`` / ``orders`` / ``sim`` / ``book`` /
``databento`` import, no ``subprocess`` (only ``gate.frozen_sha`` may), no
``datetime.now``, no network.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..events import BookEventSource
from .gate import GateError, GateVerdict, build_amendment_stub, evaluate
from .integrity import IntegrityReport, format_integrity, preflight_integrity
from .part_a import PartAError, PartAResult, ReconstructedTrade
from .part_a_runner import run_part_a
from .part_b import PartBError, PartBResult, run_part_b
from .synthetic import SyntheticError, generate_synthetic_orders

if TYPE_CHECKING:  # type-only -- not a runtime dependency edge (imports-test carve-out)
    from ..config import SimConfig

__all__ = [
    "WindowKey",
    "WindowSpec",
    "GateRun",
    "GateCliError",
    "run_parity_gate",
    # re-exported so ``cli.py`` catches every handled fault via
    # ``parity.gate_cli`` alone (spec Design Notes: cli imports gate_cli + part_a)
    "GateError",
    "PartAError",
    "PartBError",
    "SyntheticError",
]

WindowKey = str
"""A window's stable string key -- the key of a ``--windows`` JSON entry and the
label under which its integrity report is joined into the stub."""


class GateCliError(Exception):
    """The gate orchestrator cannot route a trade or a window.

    Raised for: ``amendment_number <= 0``; ``synthetic_window`` absent from
    ``windows``; a reconstructed trade whose full stamp span (every intent
    ``submit_ts_ns`` and every ``RealFill.ts_ns``) is not fully inside exactly one
    ``windows`` entry's ``[lo_ns, hi_ns)`` (CHECKPOINT 1d -- there is no second
    mapping file). Distinct from a :class:`~src.ticksim.parity.gate.GateError` (a
    bad SHA / template break) or a :class:`~src.ticksim.parity.part_a.PartAError`
    (a window-book data fault) -- those propagate from the sequenced modules
    unchanged.
    """


@dataclass(frozen=True)
class WindowSpec:
    """One window's timestamp bounds + degraded tag (spec ``--windows`` schema).

    ``[lo_ns, hi_ns)`` is half-open: it routes a trade (whose full stamp span must
    lie inside it) and bounds the ``synthetic_window`` order draw.
    ``degraded_days`` is carried verbatim into ``preflight_integrity`` (recorded,
    never a flag reason -- §A9.3 / AD-13). The ``.dbn.zst`` path + front-month
    ``instrument_id`` live only in the CLI's ``source_for`` closure, never here.
    """

    lo_ns: int
    hi_ns: int
    degraded_days: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateRun:
    """The full parity-gate result (spec Always).

    ``verdict`` is ``gate.evaluate``'s two-part :class:`GateVerdict` -- a
    ``FLAGGED`` integrity report never changes it (CHECKPOINT 1a).
    ``integrity_flagged`` is ``any(r.verdict == "FLAGGED")`` over
    ``integrity_reports`` (each ``(window key, report)``, keyed-sorted).
    ``stub`` is the ``gate.build_amendment_stub`` text -- returned, never written
    (the analyst appends; the CLI writes a standalone ``.md``).
    """

    part_a: PartAResult
    part_b: PartBResult
    integrity_reports: tuple[tuple[WindowKey, IntegrityReport], ...]
    verdict: GateVerdict
    stub: str
    integrity_flagged: bool


_MIM_NB_TRADE_ID_PREFIXES = ("mimnb-", "trader-mim-nb-")


def _trader_of(trade: ReconstructedTrade) -> str:
    """``"trader-mim-nb"`` when ``trade_id`` starts ``"mimnb-"`` (CSV
    ``reconstruct_mim_nb``) or ``"trader-mim-nb-"`` (DB-fallback
    ``reconstruct_trades_db_row``), else ``"trader-yank"`` (spec Always). Feeds
    ``build_amendment_stub``'s per-trader breakdown -- without the second prefix a
    CSV-absent run labels every mim-nb trade ``trader-yank``.
    """
    return (
        "trader-mim-nb"
        if trade.trade_id.startswith(_MIM_NB_TRADE_ID_PREFIXES)
        else "trader-yank"
    )


def _trade_span(trade: ReconstructedTrade) -> tuple[int, int]:
    """``(min, max)`` over every intent ``submit_ts_ns`` and every
    ``RealFill.ts_ns`` of ``trade`` -- the full window the runner needs folded
    (the exit leg / real fill ts is later than the entry submit for a
    mim-nb-reconstructed trade)."""
    stamps = [intent.submit_ts_ns for intent in trade.intents]
    stamps.extend(fill.ts_ns for fill in trade.real_fills)
    return min(stamps), max(stamps)


def _window_of(
    trade: ReconstructedTrade, windows: Mapping[WindowKey, WindowSpec]
) -> WindowKey:
    """The window key whose ``[lo_ns, hi_ns)`` fully contains ``trade``'s span.

    Matches on the **full** stamp span (:func:`_trade_span`), not just the entry
    ts: a trade whose exit leg / real fill ts pokes past ``hi_ns`` would have its
    book truncated by the CLI's ``_ClippedSource`` under ``run_part_a``'s exit-leg
    pricing -> a spurious miss. Zero or more than one match ->
    :class:`GateCliError` naming the trade (CHECKPOINT 1d: fail closed, never
    guess).
    """
    span_lo, span_hi = _trade_span(trade)
    matches = [
        key
        for key in sorted(windows)
        if windows[key].lo_ns <= span_lo and span_hi < windows[key].hi_ns
    ]
    if len(matches) != 1:
        raise GateCliError(
            f"trade {trade.trade_id!r} stamp span [{span_lo}, {span_hi}] is "
            f"contained by {len(matches)} of {len(windows)} windows "
            f"({matches!r}) -- expected exactly 1 [lo_ns, hi_ns) range in "
            f"--windows"
        )
    return matches[0]


def _flatten(text: str) -> str:
    """Neutralise the two markers ``build_amendment_stub._reject_template_break``
    bars in its ``integrity`` field -- a newline and a ``## `` heading marker."""
    return text.replace("\n", " / ").replace("## ", "")


def _join_integrity(
    reports: Sequence[tuple[WindowKey, IntegrityReport]], *, flagged: bool
) -> str:
    """Join every ``format_integrity`` block into one ``integrity`` string for
    ``gate.build_amendment_stub``.

    Each per-window block is flattened (:func:`_flatten`) and labelled
    ``"window <key>: "``. On a flagged run a lead segment names the flagged
    window(s) and states that the §A8.2 verdict is unchanged (AD-26) -- it does
    **not** narrate a CLI exit code (this is a library artifact; on a FAIL run the
    same flag rides an exit ``1``, not ``3``).
    """
    segments: list[str] = []
    if flagged:
        flagged_keys = [key for key, report in reports if report.verdict == "FLAGGED"]
        segments.append(
            f"integrity FLAGGED on window(s) {', '.join(flagged_keys)} -- the "
            f"parity verdict is unchanged per AD-26 (a FLAGGED preflight does "
            f"not fail the gate); review this run's data before relying on it."
        )
    for key, report in reports:
        segments.append(f"window {key}: {_flatten(format_integrity(report))}")
    return "  ||  ".join(segments)


def run_parity_gate(
    part_a_trades: Sequence[ReconstructedTrade],
    windows: Mapping[WindowKey, WindowSpec],
    synthetic_window: WindowKey,
    source_for: Callable[[WindowKey], BookEventSource],
    *,
    synthetic_seed: int,
    synthetic_n: int,
    amendment_number: int,
    cycle_number: int,
    config: SimConfig | None = None,
    sha: str | None = None,
    date: str | None = None,
) -> GateRun:
    """Run the §A8.2 parity gate end to end and return the :class:`GateRun`.

    Args:
        part_a_trades: the reconstructed live-bot trades to calibrate (Part A).
            May be empty (Part A then FAILs on its N floor).
        windows: ``window key -> WindowSpec``. Routes each trade to a source and
            bounds the ``synthetic_window`` draw. ``synthetic_window`` must be a
            key here.
        synthetic_window: the one dense RTH window Part B's orders are generated
            over (CHECKPOINT 1b).
        source_for: ``WindowKey -> BookEventSource`` -- a single-instrument,
            re-iterable, ts-clipped source (front-month filtering + the
            ``.dbn.zst`` open are the CLI's job). It is called many times --
            ``run_part_a`` calls it once per trade and again per unfilled leg,
            Part B once, integrity once per distinct window -- so the CLI
            memoises it per key.
        synthetic_seed: the sole Part B entropy source (spine AD-11).
        synthetic_n: how many synthetic orders to draw. Below
            ``PART_B_MIN_ORDERS`` Part B FAILs on the order-count floor.
        amendment_number: the amendment's number in the seal (analyst-supplied,
            AD-26 -- never derived here). ``<= 0`` -> :class:`GateCliError`.
        cycle_number: the revision-cycle number (analyst-supplied, AD-26).
        config: the ``SimConfig`` both runners simulate under. ``None`` (default)
            forwards each runner's own ``PRIMARY`` default -- the decision-bearing
            model.
        sha: the frozen simulator SHA. ``None`` -> ``gate.build_amendment_stub``
            calls ``gate.frozen_sha`` once; a failure raises
            :class:`~src.ticksim.parity.gate.GateError` (never a SHA-less stub).
        date: the append date. ``None`` -> a ``"TBD (fill on append)"``
            placeholder (no wall-clock -- spine AD-1).

    Returns:
        The :class:`GateRun`.

    Raises:
        GateCliError: ``amendment_number <= 0``; ``synthetic_window`` absent from
            ``windows``; a trade whose span is not inside exactly one window.
        PartAError / SyntheticError / PartBError / GateError: propagated verbatim
            from the sequenced modules (a window-book fault, a too-sparse
            synthetic window, a structural Part B fault, a bad SHA / template
            break). ``sim`` faults (``IntentLogError`` / ``InvariantViolation`` /
            ``BookInconsistency`` / ``ValueError``) propagate too.
    """
    if amendment_number <= 0:
        raise GateCliError(
            f"amendment_number must be > 0, got {amendment_number} "
            f"(analyst-owned; AD-26)"
        )
    if synthetic_window not in windows:
        raise GateCliError(
            f"synthetic_window {synthetic_window!r} is not a key in windows "
            f"({sorted(windows)!r})"
        )

    # --- step 1: Part A --------------------------------------------------- #
    trade_window: dict[str, WindowKey] = {
        trade.trade_id: _window_of(trade, windows) for trade in part_a_trades
    }

    def _trade_source(trade: ReconstructedTrade) -> BookEventSource:
        return source_for(trade_window[trade.trade_id])

    if config is None:
        part_a_result = run_part_a(part_a_trades, _trade_source)
    else:
        part_a_result = run_part_a(part_a_trades, _trade_source, config=config)

    # --- step 2: Part B ------------------------------------------------- #
    synth_spec = windows[synthetic_window]
    synth_source = source_for(synthetic_window)
    intents = generate_synthetic_orders(
        synth_source,
        synth_spec.lo_ns,
        synth_spec.hi_ns,
        n=synthetic_n,
        seed=synthetic_seed,
    )
    if config is None:
        part_b_result = run_part_b(intents, synth_source)
    else:
        part_b_result = run_part_b(intents, synth_source, config=config)

    # --- step 3: integrity, every distinct window touched --------------- #
    all_windows: list[WindowKey] = sorted(
        set(trade_window.values()) | {synthetic_window}
    )
    integrity_reports: list[tuple[WindowKey, IntegrityReport]] = []
    for key in all_windows:
        report = preflight_integrity(
            source_for(key), degraded_days=windows[key].degraded_days
        )
        integrity_reports.append((key, report))

    integrity_flagged = any(
        report.verdict == "FLAGGED" for _key, report in integrity_reports
    )
    integrity_str = _join_integrity(integrity_reports, flagged=integrity_flagged)

    # --- step 4: verdict + stub --------------------------------------- #
    verdict = evaluate(part_a_result, part_b_result)
    trader_by_trade_id = {trade.trade_id: _trader_of(trade) for trade in part_a_trades}
    stub = build_amendment_stub(
        part_a_result,
        part_b_result,
        amendment_number=amendment_number,
        cycle_number=cycle_number,
        sha=sha,
        integrity=integrity_str,
        date=date,
        trader_by_trade_id=trader_by_trade_id,
    )

    return GateRun(
        part_a=part_a_result,
        part_b=part_b_result,
        integrity_reports=tuple(integrity_reports),
        verdict=verdict,
        stub=stub,
        integrity_flagged=integrity_flagged,
    )
