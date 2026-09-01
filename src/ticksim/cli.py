"""``ticksim`` command-line entry point (spine AD-6).

An ``argparse`` sub-command dispatcher. It ships three sub-commands,
``simulate``, ``report`` and ``parity-gate``.

``simulate`` reads a JSONL :class:`~src.ticksim.orders.OrderIntent` log, opens a
Databento ``.dbn.zst`` MBO window, filters it to a single front-month
``instrument_id``, runs :func:`~src.ticksim.sim.simulate` under one seal-bound
config (``PRIMARY`` or ``OPTIMISTIC``), and writes the
:class:`~src.ticksim.orders.OrderOutcome` JSONL plus the run
:class:`~src.ticksim.sim.Manifest` JSON (both written atomically -- spine AD-11).

Also lands the shared, re-iterable :class:`FrontMonthSource` wrapper and
:func:`detect_front_month` -- the ``MNQ.FUT`` parent stream is ~96% front month,
~4% calendar spread (pre-registration Amendment 9), so every consumer that feeds
``sim`` a single-instrument stream needs this filter.

``report`` reads the :class:`~src.ticksim.orders.OrderOutcome` JSONL logs plus
run :class:`~src.ticksim.sim.Manifest` JSON files from a prior ``PRIMARY`` and
``OPTIMISTIC`` ``simulate`` pair, calls
:func:`~src.ticksim.report.build_report`, and writes the §2.3 / AD-14 three-way
P&L (:meth:`~src.ticksim.report.ThreeWayReport.to_dict`) as pretty JSON.

``parity-gate`` reconstructs the live bots' orders (mim-nb from
``data/mim_nb/orders.csv`` via ``csv.DictReader``, yank from ``data/trades.db``
rows on/after 2026-06-17 via ``sqlite3``), reads a ``--windows`` JSON, builds the
ts-clipped front-month ``source_for`` callable, and hands all of it to
:func:`~src.ticksim.parity.gate_cli.run_parity_gate` (the pure §A8.2
orchestrator). It writes the ``gate.build_amendment_stub`` text to ``--out`` as a
**standalone** ``.md`` (the analyst appends it to the seal -- AD-26). Exit ``0``
PASS / ``3`` PASS with an integrity flag / ``1`` FAIL-or-handled-error / ``2``
usage.

Dependencies (spine AD-7): ``.sim`` / ``.events`` / ``.orders`` / ``.config`` /
``.book`` (for :class:`~src.ticksim.book.BookInconsistency` -- an analyst-facing
simulator fault this boundary catches) / ``.report`` (the AD-14 money layer the
``report`` sub-command wraps) / ``.parity`` (``parity.gate_cli`` +
``parity.part_a`` for the reconstruction helpers) + stdlib (``csv`` / ``sqlite3``
for the ``parity-gate`` reconstruction). No ``datetime`` -- a ``--degraded-day``
is carried through to the manifest as the exact ``str`` token the analyst typed
(spine AD-13). Relative imports only (``mypy --strict`` duplicate-module-errors
on the absolute form).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .book import BookInconsistency
from .config import OPTIMISTIC, PART_B_MIN_ORDERS, PRIMARY, SimConfig
from .events import BookEvent, BookEventSource, DbnMboSource
from .orders import OrderIntent, OrderOutcome, OrderStateError
from .parity.gate_cli import (
    GateCliError,
    GateError,
    GateRun,
    PartBError,
    SyntheticError,
    WindowSpec,
    run_parity_gate,
)
from .parity.part_a import (
    PartAError,
    ReconstructedTrade,
    reconstruct_mim_nb,
    reconstruct_trades_db_row,
)
from .report import ModelPnL, ReportError, ThreeWayReport, build_report
from .sim import IntentLogError, InvariantViolation, Manifest, simulate

__all__ = ["FrontMonthSource", "detect_front_month", "main"]

logger = logging.getLogger(__name__)

_PROG = "ticksim"
_NS_PER_MINUTE = 60 * 1_000_000_000
_DEFAULT_PAD_MINUTES = 5.0
_DEGRADED_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_CLI_LOG_HANDLER: logging.Handler | None = None


class _CliError(Exception):
    """A handled, analyst-facing error: printed to stderr, exit code ``1``.

    Never a traceback -- a malformed intent log, an unwritable output path, or a
    declared simulator fault (:class:`~src.ticksim.sim.IntentLogError`,
    :class:`~src.ticksim.sim.InvariantViolation`,
    :class:`~src.ticksim.book.BookInconsistency`, ``ValueError``) is a user
    condition here, not a bug (spec Design Notes). An
    :class:`~src.ticksim.orders.OrderStateError` -- and any ``TypeError`` /
    ``AttributeError`` / ``KeyError`` from a cli<->sim wiring bug -- *is* a bug
    and is left to propagate.
    """


# --------------------------------------------------------------------------- #
# FrontMonthSource -- the shared re-iterable single-instrument filter
# --------------------------------------------------------------------------- #


def _require_reiterable(source: BookEventSource) -> None:
    """Reject a one-shot source (spine AD-18: sources are replayed).

    ``iter(gen)`` on a bare generator returns the *same* object twice; a
    properly re-iterable source hands out a fresh iterator each call. Calling
    ``iter`` does not consume a ``DbnMboSource`` / list-backed source (``__iter__``
    only *builds* the generator/iterator, it does not advance it).
    """
    if iter(source) is iter(source):
        raise TypeError(
            f"{type(source).__name__} is not re-iterable -- a BookEventSource "
            f"must yield the same events on every pass (spine AD-18)"
        )


class FrontMonthSource:
    """A :class:`~src.ticksim.events.BookEventSource` yielding one instrument.

    Wraps an inner source and yields only the events whose ``instrument_id``
    matches (spine AD-18: the merge is pull-based and downstream studies replay
    sources, so this must be a **class**, not a one-shot generator -- each
    :meth:`__iter__` rebuilds the filtered iterator over a fresh pass of
    ``inner``). ``class_rank`` is forwarded from ``inner``. No other filtering,
    no mutation, no reordering.
    """

    def __init__(self, inner: BookEventSource, instrument_id: int) -> None:
        _require_reiterable(inner)
        self._inner = inner
        self._instrument_id = instrument_id
        self.class_rank: int = inner.class_rank

    def __iter__(self) -> Iterator[BookEvent]:
        iid = self._instrument_id
        return (ev for ev in self._inner if ev.instrument_id == iid)

    def __repr__(self) -> str:
        return (
            f"FrontMonthSource({self._inner!r}, "
            f"instrument_id={self._instrument_id})"
        )


def detect_front_month(source: BookEventSource) -> int:
    """Return the modal ``instrument_id`` in ``source`` (one pass).

    The ``MNQ.FUT`` parent is dominated (~96%) by the front-month outright; the
    remainder is the calendar spread (pre-registration Amendment 9). One pass
    counting ``ev.instrument_id`` and the most common id wins.

    Raises:
        TypeError: ``source`` is not re-iterable (spine AD-18).
        ValueError: the stream is empty (nothing to detect), or two ids are
            equally the most common -- an ambiguous front month the analyst must
            resolve with ``--instrument-id``.
    """
    iid, _count, _total = _detect_front_month_with_stats(source)
    return iid


def _detect_front_month_with_stats(source: BookEventSource) -> tuple[int, int, int]:
    """``(modal_id, count_for_modal_id, total_events)`` in one pass over ``source``."""
    _require_reiterable(source)
    counts: Counter[int] = Counter(ev.instrument_id for ev in source)
    if not counts:
        raise ValueError(
            "DBN stream is empty -- no book events to detect an instrument"
        )
    ranked = counts.most_common()
    if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
        raise ValueError(
            f"front-month instrument_id is ambiguous -- {ranked[0][0]} and "
            f"{ranked[1][0]} both have {ranked[0][1]} events; pass --instrument-id"
        )
    iid, count = ranked[0]
    return iid, count, sum(counts.values())


class _ClippedSource:
    """A re-iterable :class:`~src.ticksim.events.BookEventSource` view of
    ``inner`` clipped to a half-open ``[lo_ns, hi_ns)`` ``ts_event`` window.

    A parity window's ``.dbn.zst`` is the +/-90-min neighbourhood of a cluster of
    live fills; clipping keeps :func:`~src.ticksim.parity.integrity.preflight_integrity`
    (no interval argument -- it surveys the whole source) and the Part B book
    replay bounded to the analyst-declared window. ``inner`` is already
    front-month-filtered (:class:`FrontMonthSource`); ``class_rank`` is forwarded,
    no reordering, no mutation. Each :meth:`__iter__` rebuilds the filtered
    iterator over a fresh pass of ``inner`` (spine AD-18).
    """

    def __init__(self, inner: BookEventSource, lo_ns: int, hi_ns: int) -> None:
        _require_reiterable(inner)
        self._inner = inner
        self._lo_ns = lo_ns
        self._hi_ns = hi_ns
        self.class_rank: int = inner.class_rank

    def __iter__(self) -> Iterator[BookEvent]:
        lo, hi = self._lo_ns, self._hi_ns
        return (ev for ev in self._inner if lo <= ev.ts_event < hi)

    def __repr__(self) -> str:
        return (
            f"_ClippedSource({self._inner!r}, lo_ns={self._lo_ns}, "
            f"hi_ns={self._hi_ns})"
        )


# --------------------------------------------------------------------------- #
# intent log + interval helpers
# --------------------------------------------------------------------------- #


_JsonlModel = TypeVar("_JsonlModel", bound=BaseModel)


def _read_jsonl(
    path: Path, model_cls: type[_JsonlModel], label: str, empty_noun: str
) -> list[_JsonlModel]:
    """Parse a UTF-8 JSONL log of ``model_cls`` records, one per non-blank line.

    The shared reader behind :func:`_read_intents` and :func:`_read_outcomes`:
    same read / decode guard, same per-line ``model_validate_json`` loop that
    names the offending line, same empty-file guard.

    Args:
        path: the ``.jsonl`` file.
        model_cls: the Pydantic model each line must validate against.
        label: human name for ``path`` in read / line error messages
            (e.g. ``"intent log"``).
        empty_noun: plural noun for the empty-file error (e.g. ``"intents"``).

    Raises:
        _CliError: the file is unreadable or not UTF-8, a non-blank line is not
            a valid ``model_cls`` (the line number is named), or the file has no
            records.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _CliError(f"cannot read {label} {path}: {exc}") from exc
    records: list[_JsonlModel] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(model_cls.model_validate_json(line))
        except ValueError as exc:
            raise _CliError(
                f"{label} {path} line {lineno}: not a valid "
                f"{model_cls.__name__} ({exc})"
            ) from exc
    if not records:
        raise _CliError(f"{label} {path}: no {empty_noun}")
    return records


def _read_intents(path: Path) -> list[OrderIntent]:
    """Parse a UTF-8 JSONL :class:`~src.ticksim.orders.OrderIntent` log.

    Thin wrapper over :func:`_read_jsonl`. The list is returned as-is -- ``sim``
    validates causal replayability (spine AD-2) and raises
    :class:`~src.ticksim.sim.IntentLogError`.
    """
    return _read_jsonl(path, OrderIntent, "intent log", "intents")


def _span_interval(intents: Sequence[OrderIntent], pad_ns: int) -> tuple[int, int]:
    """One half-open ``[start, end)`` interval covering the log's ``submit_ts_ns``
    range, padded ``pad_ns`` each side (mirrors ``parity.part_a_runner`` -- a
    boundary order must not be expired by the AD-13 mask before it can fill).

    ``start`` is clamped at 0 (spine AD-1). ``end`` gets a ``+1`` ns so the last
    intent is *strictly* inside the half-open window even with ``pad_ns == 0``
    (otherwise a single-intent log with ``--pad-minutes 0`` yields
    ``start == end`` -> ``ValueError`` in ``sim``, and every last intent sits
    exactly on the excluded ``end`` edge).
    """
    stamps = [intent.submit_ts_ns for intent in intents]
    lo = min(stamps) - pad_ns
    hi = max(stamps) + pad_ns + 1
    return (lo if lo > 0 else 0, hi)


# --------------------------------------------------------------------------- #
# the `simulate` sub-command
# --------------------------------------------------------------------------- #


def _cmd_simulate(args: argparse.Namespace) -> int:
    try:
        return _run_simulate(args)
    except _CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        # A downstream reader (`ticksim simulate ... | head`) closed the pipe.
        # The output files are already written; redirect stdout to devnull so
        # the interpreter's shutdown flush does not re-raise, and exit clean.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0


def _run_simulate(args: argparse.Namespace) -> int:
    dbn_path = Path(args.dbn)
    intents_path = Path(args.intents)
    out_outcomes = Path(args.out_outcomes)
    out_manifest = Path(args.out_manifest)
    config_name: str = args.config
    cfg: SimConfig = PRIMARY if config_name == "primary" else OPTIMISTIC
    degraded_days = _degraded_days(args.degraded_day)
    pad_ns = int(float(args.pad_minutes) * _NS_PER_MINUTE)

    intents = _read_intents(intents_path)

    valid_intervals: list[tuple[int, int]]
    if args.interval:
        valid_intervals = [(int(start), int(end)) for start, end in args.interval]
    else:
        valid_intervals = [_span_interval(intents, pad_ns)]

    if not dbn_path.is_file():
        raise _CliError(f"no such DBN file: {dbn_path}")

    if args.instrument_id is None:
        try:
            iid, count, total = _detect_front_month_with_stats(
                DbnMboSource(str(dbn_path))
            )
        except ValueError as exc:
            raise _CliError(str(exc)) from exc
        pct = (100.0 * count / total) if total else 0.0
        line = (
            f"detected front-month instrument_id={iid} "
            f"({pct:.1f}% of {total} events)"
        )
        print(line, file=sys.stderr)
        logger.info("%s -- this was a full extra pass over %s", line, dbn_path)
    else:
        iid = int(args.instrument_id)

    logger.info(
        "simulate: config=%s instrument_id=%d intents=%d intervals=%s",
        config_name,
        iid,
        len(intents),
        valid_intervals,
    )

    source = FrontMonthSource(DbnMboSource(str(dbn_path)), iid)

    try:
        outcomes, manifest = simulate(
            source, intents, cfg, valid_intervals, degraded_days=degraded_days
        )
    except OrderStateError:
        raise  # an illegal tracker transition is a bug -- propagate, never mask
    except (IntentLogError, InvariantViolation, BookInconsistency, ValueError) as exc:
        # The four declared analyst-facing simulator faults (spec Run). Anything
        # else -- TypeError / AttributeError / KeyError from a wiring bug -- is a
        # real bug and is deliberately NOT caught here.
        raise _CliError(f"simulator fault: {type(exc).__name__}: {exc}") from exc

    if manifest.event_count == 0:
        raise _CliError(f"DBN window yielded no book events for instrument_id={iid}")

    _atomic_write_pair(
        out_outcomes,
        "".join(outcome.model_dump_json() + "\n" for outcome in outcomes),
        out_manifest,
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
    )
    logger.info("wrote %d outcomes -> %s", len(outcomes), out_outcomes)

    _print_summary(config_name, intents, outcomes, manifest, out_outcomes, out_manifest)
    return 0


def _degraded_days(tokens: list[str] | None) -> tuple[str, ...]:
    """Carry ``--degraded-day`` tokens through verbatim (spine AD-13: recorded,
    never excluded). A non ``YYYY-MM-DD`` token gets a stderr warning but is
    still passed on -- ``sim`` sorts + de-dups the list for the manifest."""
    out: list[str] = []
    for token in tokens or []:
        if not _DEGRADED_DAY_RE.fullmatch(token):
            print(
                f"warning: --degraded-day {token!r} is not YYYY-MM-DD; "
                f"recorded as-is",
                file=sys.stderr,
            )
        out.append(token)
    return tuple(out)


def _atomic_write_pair(
    outcomes_path: Path,
    outcomes_text: str,
    manifest_path: Path,
    manifest_text: str,
) -> None:
    """Write both outputs or neither (spine AD-11: a re-run is idempotent).

    Each file is written to a sibling ``*.tmp`` first; only once **both** ``.tmp``
    writes succeed are they ``os.replace``\\ d onto the final paths. Any failure
    unlinks whatever ``.tmp`` files exist and raises :class:`_CliError` -- there
    is never an orphan outcome log with no matching manifest.
    """
    tmp_outcomes = outcomes_path.with_name(outcomes_path.name + ".tmp")
    tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    cleanup: list[Path] = []
    try:
        outcomes_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        # Both .tmp writes (the failure-prone step) BEFORE either replace.
        _write_tmp(tmp_outcomes, outcomes_text)
        cleanup.append(tmp_outcomes)
        _write_tmp(tmp_manifest, manifest_text)
        cleanup.append(tmp_manifest)
        # Same-dir os.replace is atomic; the second failing after the first
        # succeeded is near-impossible, but if it does the just-replaced
        # outcome log is rolled back so there is never an orphan.
        os.replace(tmp_outcomes, outcomes_path)
        cleanup = [tmp_manifest, outcomes_path]
        os.replace(tmp_manifest, manifest_path)
        cleanup = []
    except OSError as exc:
        for leftover in cleanup:
            try:
                leftover.unlink()
            except OSError:
                pass
        raise _CliError(f"cannot write outputs: {exc}") from exc


def _write_tmp(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` as UTF-8 (its own function so a test can inject
    a failure on one of the two atomic-write legs)."""
    path.write_text(text, encoding="utf-8")


def _print_summary(
    config_name: str,
    intents: Sequence[OrderIntent],
    outcomes: Sequence[OrderOutcome],
    manifest: Manifest,
    out_outcomes: Path,
    out_manifest: Path,
) -> None:
    """A short, ``grep``-able human block to stdout (not JSON)."""
    fill_events = sum(len(outcome.fills) for outcome in outcomes)
    terminal = Counter(outcome.terminal_state.value for outcome in outcomes)
    terminal_str = " ".join(f"{state}={terminal[state]}" for state in sorted(terminal))
    print(f"config:     {config_name}")
    print(f"intents:    {len(intents)}")
    print(f"outcomes:   {len(outcomes)}")
    print(f"fills:      {fill_events} fill events")
    print(f"terminal:   {terminal_str or '(none)'}")
    print(
        f"manifest:   seed={manifest.seed} "
        f"oco_cascade_cancels={manifest.oco_cascade_cancel_count} "
        f"adverse_fills={manifest.adverse_fill_count} "
        f"events={manifest.event_count}"
    )
    if manifest.degraded_days:
        # sim sorted + de-duped the CLI list for the manifest
        print(f"degraded:   {' '.join(manifest.degraded_days)}")
    print(f"outcomes -> {out_outcomes}")
    print(f"manifest -> {out_manifest}")


# --------------------------------------------------------------------------- #
# the `report` sub-command
# --------------------------------------------------------------------------- #


def _cmd_report(args: argparse.Namespace) -> int:
    try:
        return _run_report(args)
    except _CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        # A downstream reader (`ticksim report ... | head`) closed the pipe. The
        # `--out` file is already written; redirect stdout to devnull so the
        # interpreter's shutdown flush does not re-raise, and exit clean.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0


def _run_report(args: argparse.Namespace) -> int:
    primary_outcomes = _read_outcomes(Path(args.primary_outcomes))
    primary_manifest = _read_manifest(Path(args.primary_manifest))
    optimistic_outcomes = _read_outcomes(Path(args.optimistic_outcomes))
    optimistic_manifest = _read_manifest(Path(args.optimistic_manifest))
    out_path = Path(args.out)

    logger.info(
        "report: primary=%d outcomes (queue_model=%s), "
        "optimistic=%d outcomes (queue_model=%s)",
        len(primary_outcomes),
        _manifest_queue_model(primary_manifest),
        len(optimistic_outcomes),
        _manifest_queue_model(optimistic_manifest),
    )

    try:
        report = build_report(
            primary_outcomes,
            primary_manifest,
            optimistic_outcomes,
            optimistic_manifest,
        )
    except ReportError as exc:
        # Every malformed-input case (a non-PRIMARY/OPTIMISTIC manifest pair,
        # mixed entry sides, a duplicate order_id, differing trade_id sets, a
        # bad fee field, ...) funnels through build_report's ReportError
        # taxonomy -- an analyst condition, exit 1, never a traceback.
        raise _CliError(str(exc)) from exc

    logger.info(
        "built report: %d round trips, %d incomplete, %d partially closed",
        len(report.round_trips),
        len(report.incomplete),
        len(report.partially_closed),
    )

    # `ModelPnL.profit_factor` can be `float('inf')`; `json.dumps` would write a
    # bare `Infinity` token, which is not valid JSON for a non-Python parser.
    # Sanitize the on-disk dict only -- the stdout summary still renders "inf".
    _atomic_write_one(
        out_path,
        json.dumps(_json_safe(report.to_dict()), indent=2, sort_keys=True) + "\n",
    )
    logger.info("wrote three-way P&L report -> %s", out_path)

    _print_report_summary(report, out_path)
    return 0


def _json_safe(obj: object) -> object:
    """Recursively replace non-finite ``float``s (``inf`` / ``-inf`` / ``nan``)
    with ``None`` so ``json.dumps`` emits strict, portable JSON. Everything else
    passes through untouched.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(item) for item in obj]
    return obj


def _manifest_queue_model(manifest: dict[str, object]) -> str:
    """The ``config.queue_model`` token in ``manifest``, or ``"?"`` if absent --
    a best-effort read for the ``-v`` log line (``build_report`` owns the real
    validation)."""
    config = manifest.get("config")
    if isinstance(config, dict):
        return str(config.get("queue_model", "?"))
    return "?"


def _read_outcomes(path: Path) -> list[OrderOutcome]:
    """Parse a UTF-8 JSONL :class:`~src.ticksim.orders.OrderOutcome` log.

    Thin wrapper over :func:`_read_jsonl`. ``build_report`` validates the
    round-trip structure and raises :class:`~src.ticksim.report.ReportError`.
    """
    return _read_jsonl(path, OrderOutcome, "outcome log", "outcomes")


def _read_manifest(path: Path) -> dict[str, object]:
    """Read a run :class:`~src.ticksim.sim.Manifest` JSON file.

    ``json.loads(path.read_text(...))`` -- the parsed value must be a JSON
    object (``dict``); it is handed straight to ``build_report`` as the
    ``Manifest.to_dict()``-shaped ``Mapping`` (spine AD-24).

    Raises:
        _CliError: the file is unreadable / not UTF-8, not valid JSON, or not a
            JSON object.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _CliError(f"cannot read manifest {path}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise _CliError(f"manifest {path}: not valid JSON ({exc})") from exc
    if not isinstance(parsed, dict):
        raise _CliError(
            f"manifest {path}: expected a JSON object, got {type(parsed).__name__}"
        )
    return parsed


def _atomic_write_one(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (spine AD-11): a sibling ``*.tmp``
    then ``os.replace``. Parent dirs are created; an existing file is
    overwritten. Any failure unlinks the ``.tmp`` and raises :class:`_CliError`.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_tmp(tmp, text)
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise _CliError(f"cannot write output {path}: {exc}") from exc


def _fmt_ratio(value: float | None) -> str:
    """Render a profit factor: ``None`` -> ``"n/a"``, ``inf`` -> ``"inf"``."""
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.3f}"


def _fmt_win_rate(model: ModelPnL) -> str:
    """``wins / n`` as a 3-dp string, or ``"n/a"`` for an empty model."""
    if model.n == 0:
        return "n/a"
    return f"{model.wins / model.n:.3f}"


def _print_report_summary(report: ThreeWayReport, out_path: Path) -> None:
    """A short, ``grep``-able human block to stdout (not JSON)."""
    n_open = len(report.incomplete)
    print(f"round trips: {len(report.round_trips)}")
    print(f"incomplete: {n_open} open position" + ("" if n_open == 1 else "s"))
    print(f"partially closed: {len(report.partially_closed)}")
    print(f"optimistic-only completed: {len(report.optimistic_only_completed)}")
    for name, model in (
        ("primary", report.primary),
        ("stressed", report.stressed),
        ("optimistic", report.optimistic),
    ):
        print(
            f"{name:<11} n={model.n} net_cents={model.sum_net_cents} "
            f"win_rate={_fmt_win_rate(model)} "
            f"profit_factor={_fmt_ratio(model.profit_factor)}"
        )
    print(f"optimistic n={report.optimistic.n} (primary n={report.primary.n})")
    if report.optimistic.n != report.primary.n:
        print(
            "note: optimistic P&L is over the trades that completed under BOTH "
            "models"
        )
    print(f"report -> {out_path}")


# --------------------------------------------------------------------------- #
# the `parity-gate` sub-command
# --------------------------------------------------------------------------- #

_PART_A_DB_SINCE = "2026-06-17"

_MIM_NB_DB_FALLBACK_PROVENANCE = (
    "<!-- parity-gate provenance -->\n"
    "> **Provenance note (mim-nb):** `data/mim_nb/orders.csv` was absent -- the "
    "mim-nb trades in this run were reconstructed from `data/trades.db` rows "
    "(2-leg market reconstruction, no protective-stop leg, `bar_reconstructed` "
    "fidelity), not from the authoritative order-lifecycle CSV.\n\n"
)


@dataclass(frozen=True)
class _WindowEntry:
    """One parsed + validated ``--windows`` JSON entry."""

    dbn: str
    instrument_id: int
    lo_ns: int
    hi_ns: int
    degraded_days: tuple[str, ...]


def _cmd_parity_gate(args: argparse.Namespace) -> int:
    try:
        rc, run, out_path = _run_parity_gate(args)
    except _CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # ``rc`` is fully determined (and the stub is already written). The summary
    # is the only pipe-fragile step, and a broken pipe must NOT downgrade a
    # FAIL / flagged-PASS to a clean ``0`` -- here the exit code IS the verdict.
    try:
        if run.verdict.verdict != "PASS":
            print(f"reason: {run.verdict.reason}", file=sys.stderr)
        print(f"verdict:           {run.verdict.verdict}")
        print(f"integrity_flagged: {run.integrity_flagged}")
        print(f"stub ->            {out_path}")
    except BrokenPipeError:
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (ValueError, OSError):
            pass
    return rc


def _run_parity_gate(args: argparse.Namespace) -> tuple[int, GateRun, Path]:
    orders_csv = Path(args.orders_csv)
    trades_db = Path(args.trades_db)
    out_path = Path(args.out)
    config_name: str = args.config
    cfg: SimConfig = PRIMARY if config_name == "primary" else OPTIMISTIC

    windows_raw: dict[str, _WindowEntry] = getattr(
        args, "windows_parsed", None
    ) or _read_windows(Path(args.windows))
    windows = {
        key: WindowSpec(
            lo_ns=entry.lo_ns,
            hi_ns=entry.hi_ns,
            degraded_days=entry.degraded_days,
        )
        for key, entry in windows_raw.items()
    }
    trades, provenance = _reconstruct_part_a_trades(orders_csv, trades_db)
    source_for = _source_for_factory(windows_raw)

    logger.info(
        "parity-gate: %d Part A trades, %d windows, synthetic_window=%s, config=%s",
        len(trades),
        len(windows),
        args.synthetic_window,
        config_name,
    )

    try:
        run = run_parity_gate(
            trades,
            windows,
            args.synthetic_window,
            source_for,
            synthetic_seed=int(args.synthetic_seed),
            synthetic_n=int(args.synthetic_n),
            amendment_number=int(args.amendment_number),
            cycle_number=int(args.cycle_number),
            skip_uncovered=bool(args.skip_uncovered),
            config=cfg,
            sha=args.sha,
            date=args.date,
        )
    except OrderStateError:
        raise  # an illegal tracker transition is a bug -- propagate, never mask
    except (
        PartAError,
        SyntheticError,
        PartBError,
        GateError,
        GateCliError,
        BookInconsistency,
        IntentLogError,
        InvariantViolation,
        ValueError,
        FileNotFoundError,
    ) as exc:
        # Every declared analyst-facing fault of the sequenced modules (a
        # window-book data fault, a too-sparse synthetic window, a structural
        # Part B fault, a bad SHA / template break, a mis-mapped trade or
        # window, a sim fault). A TypeError / AttributeError / KeyError from a
        # cli<->gate_cli wiring bug is NOT caught here.
        raise _CliError(
            f"parity gate could not complete: {type(exc).__name__}: {exc}"
        ) from exc

    document = run.stub if run.stub.endswith("\n") else run.stub + "\n"
    if provenance is not None:
        document = provenance + document
    _atomic_write_one(out_path, document)

    rc = 1 if run.verdict.verdict != "PASS" else (3 if run.integrity_flagged else 0)
    return rc, run, out_path


def _read_windows(path: Path) -> dict[str, _WindowEntry]:
    """Read + shallow-validate the ``--windows`` JSON.

    Schema: ``{"<key>": {"dbn": <path str>, "instrument_id": <int>, "lo_ns":
    <int>, "hi_ns": <int>, "degraded_days": [<"YYYY-MM-DD">, ...]}}`` (the last
    optional).

    Raises:
        _CliError: unreadable / not UTF-8 / not JSON / not a JSON object / no
            entries; an entry that is not an object, is missing a required key,
            has a non-integer / negative bound (``instrument_id`` / ``lo_ns`` /
            ``hi_ns``), a ``hi_ns <= lo_ns``, an empty ``dbn``, a ``degraded_days``
            that is not a list of strings, or a window key / ``degraded_days``
            entry carrying a newline or ``"## "`` (both land in the amendment
            template and trip ``build_amendment_stub``'s guard).
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _CliError(f"cannot read windows file {path}: {exc}") from exc
    try:
        parsed: object = json.loads(raw)
    except ValueError as exc:
        raise _CliError(f"windows file {path}: not valid JSON ({exc})") from exc
    if not isinstance(parsed, dict):
        raise _CliError(
            f"windows file {path}: expected a JSON object, got "
            f"{type(parsed).__name__}"
        )
    if not parsed:
        raise _CliError(f"windows file {path}: no window entries")

    out: dict[str, _WindowEntry] = {}
    for key, entry in parsed.items():
        label = f"windows file {path}, entry {key!r}"
        key_str = str(key)
        if "\n" in key_str or "## " in key_str:
            raise _CliError(
                f"{label}: window key contains a newline or '## ' -- it lands "
                f"verbatim in the amendment template"
            )
        if not isinstance(entry, dict):
            raise _CliError(f"{label}: not a JSON object")
        for field in ("dbn", "instrument_id", "lo_ns", "hi_ns"):
            if field not in entry:
                raise _CliError(f"{label}: missing required key {field!r}")
        dbn = entry["dbn"]
        if not isinstance(dbn, str) or not dbn:
            raise _CliError(f"{label}: 'dbn' must be a non-empty string")
        instrument_id = _windows_int(entry["instrument_id"], "instrument_id", label)
        if instrument_id < 0:
            raise _CliError(f"{label}: instrument_id ({instrument_id}) must be >= 0")
        lo_ns = _windows_int(entry["lo_ns"], "lo_ns", label)
        hi_ns = _windows_int(entry["hi_ns"], "hi_ns", label)
        if lo_ns < 0:
            raise _CliError(f"{label}: lo_ns ({lo_ns}) must be >= 0 (spine AD-1)")
        if hi_ns <= lo_ns:
            raise _CliError(f"{label}: hi_ns ({hi_ns}) must exceed lo_ns ({lo_ns})")
        degraded_raw = entry.get("degraded_days", [])
        if not isinstance(degraded_raw, list) or not all(
            isinstance(day, str) for day in degraded_raw
        ):
            raise _CliError(f"{label}: 'degraded_days' must be a list of date strings")
        for day in degraded_raw:
            if "\n" in day or "## " in day:
                raise _CliError(
                    f"{label}: degraded_days entry {day!r} contains a newline "
                    f"or '## ' -- it lands verbatim in the amendment template"
                )
        out[str(key)] = _WindowEntry(
            dbn=dbn,
            instrument_id=instrument_id,
            lo_ns=lo_ns,
            hi_ns=hi_ns,
            degraded_days=tuple(str(day) for day in degraded_raw),
        )
    return out


def _windows_int(value: object, field: str, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _CliError(f"{label}: {field!r} must be an integer, got {value!r}")
    return value


def _source_for_factory(
    windows: dict[str, _WindowEntry],
) -> Callable[[str], BookEventSource]:
    """The ``WindowKey -> BookEventSource`` callable ``run_parity_gate`` injects.

    For a key: a ts-clipped :class:`_ClippedSource` over a :class:`FrontMonthSource`
    over a :class:`~src.ticksim.events.DbnMboSource` of that window's ``.dbn.zst``.
    **Memoised per key** -- ``run_parity_gate`` calls ``source_for`` once per Part A
    trade (+ once per unfilled leg) + once for Part B + once per integrity window;
    without the cache N trades in one window would each spin up a fresh
    ``DBNStore.from_file`` decompression pass. Each cached source is still
    re-iterable, so a re-fold just re-opens the handle. A missing ``.dbn.zst`` ->
    :class:`_CliError` (raised on the first ``source_for`` call for that window).
    """
    cache: dict[str, BookEventSource] = {}

    def source_for(key: str) -> BookEventSource:
        cached = cache.get(key)
        if cached is not None:
            return cached
        entry = windows[key]
        dbn_path = Path(entry.dbn)
        if not dbn_path.is_file():
            raise _CliError(f"window {key!r}: no such DBN file: {dbn_path}")
        source: BookEventSource = _ClippedSource(
            FrontMonthSource(DbnMboSource(str(dbn_path)), entry.instrument_id),
            entry.lo_ns,
            entry.hi_ns,
        )
        cache[key] = source
        return source

    return source_for


def _reconstruct_part_a_trades(
    orders_csv: Path, trades_db: Path
) -> tuple[list[ReconstructedTrade], str | None]:
    """The Part A trade set: mim-nb from ``orders.csv`` (CHECKPOINT 1c -- the CSV
    is authoritative; the DB's mim-nb rows are a fallback only when the CSV is
    absent), yank always from ``trades.db`` rows on/after 2026-06-17.

    Returns ``(trades, provenance)`` where ``provenance`` is a one-line Markdown
    note (else ``None``) that the CLI prepends to the written stub when >= 1
    mim-nb trade came from the lower-fidelity DB fallback.

    Raises:
        _CliError: an unreadable CSV / DB, or a reconstruction ``PartAError``
            (naming the offending row).
    """
    trades: list[ReconstructedTrade] = []
    provenance: str | None = None

    if orders_csv.is_file():
        try:
            with orders_csv.open(encoding="utf-8-sig", newline="") as handle:
                rows: list[dict[str, object]] = [
                    dict(row) for row in csv.DictReader(handle)
                ]
        except (OSError, UnicodeDecodeError) as exc:
            raise _CliError(f"cannot read orders CSV {orders_csv}: {exc}") from exc
        try:
            trades.extend(reconstruct_mim_nb(rows))
        except PartAError as exc:
            raise _CliError(
                f"mim-nb reconstruction from {orders_csv} failed: {exc}"
            ) from exc
    else:
        logger.warning(
            "orders CSV %s absent -- falling back to trades.db mim-nb rows "
            "(no bracket legs, bar_reconstructed fidelity)",
            orders_csv,
        )
        mim_db_trades = [
            _reconstruct_db_row(row, trades_db)
            for row in _db_rows(trades_db, "trader-mim-nb", since=_PART_A_DB_SINCE)
        ]
        if mim_db_trades:
            provenance = _MIM_NB_DB_FALLBACK_PROVENANCE
        trades.extend(mim_db_trades)

    for row in _db_rows(trades_db, "trader-yank", since=_PART_A_DB_SINCE):
        trades.append(_reconstruct_db_row(row, trades_db))

    return trades, provenance


def _reconstruct_db_row(row: dict[str, object], trades_db: Path) -> ReconstructedTrade:
    try:
        return reconstruct_trades_db_row(row)
    except PartAError as exc:
        raise _CliError(
            f"trades.db reconstruction ({trades_db}) failed: {exc}"
        ) from exc


def _db_rows(trades_db: Path, trader_id: str, *, since: str) -> list[dict[str, object]]:
    """Rows of ``trades_db``'s ``trades`` table for ``trader_id`` with
    ``timestamp >= since`` (NULL timestamps excluded, not mis-compared), ordered
    ``timestamp, id`` so the reconstructed stub is byte-deterministic across runs.
    Each row is a plain ``dict`` via a :class:`sqlite3.Row` row factory.

    Raises:
        _CliError: the DB file is missing, or ``sqlite3`` raised (unreadable, no
            ``trades`` table).
    """
    if not trades_db.is_file():
        raise _CliError(f"no such trades DB: {trades_db}")
    query = (
        "SELECT * FROM trades WHERE trader_id = ? AND timestamp IS NOT NULL "
        "AND timestamp >= ? ORDER BY timestamp, id"
    )
    params: list[object] = [trader_id, since]
    try:
        connection = sqlite3.connect(str(trades_db))
    except sqlite3.Error as exc:
        raise _CliError(f"cannot open trades DB {trades_db}: {exc}") from exc
    try:
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(query, params)
        return [{name: row[name] for name in row.keys()} for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        raise _CliError(f"trades DB {trades_db} query failed: {exc}") from exc
    finally:
        connection.close()


def _validate_parity_gate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Usage-level checks (every failure -> ``parser.error`` -> exit 2).

    ``--windows`` is read + fully validated **here, once** -- the parsed
    ``dict[str, _WindowEntry]`` is stashed on ``args.windows_parsed`` so
    ``_run_parity_gate`` never re-reads it (a second parse at a different
    strictness was the root of the ``--synthetic-window`` exit-code
    inconsistency -- round 1). Checks: ``--amendment-number`` / ``--cycle-number``
    ``> 0``; ``--out`` names a file, does not collide with any input
    (``--orders-csv`` / ``--trades-db`` / ``--windows`` / any window ``.dbn.zst``),
    and does not already exist unless ``--force``; ``--synthetic-window`` is a key
    in ``--windows``.
    """
    if int(args.amendment_number) <= 0:
        parser.error("--amendment-number must be > 0 (analyst-owned; AD-26)")
    if int(args.cycle_number) <= 0:
        parser.error("--cycle-number must be > 0 (analyst-owned; AD-26)")

    if not Path(args.out).name:
        parser.error("--out must name a file")

    out_resolved = Path(args.out).resolve()
    for label, value in (
        ("--orders-csv", args.orders_csv),
        ("--trades-db", args.trades_db),
        ("--windows", args.windows),
    ):
        if Path(value).resolve() == out_resolved:
            parser.error(f"--out must not be the same file as {label} ({out_resolved})")

    try:
        windows = _read_windows(Path(args.windows))
    except _CliError as exc:
        parser.error(str(exc))

    for key, entry in windows.items():
        if Path(entry.dbn).resolve() == out_resolved:
            parser.error(
                f"--out must not be the same file as the .dbn.zst for window {key!r}"
            )

    if args.synthetic_window not in windows:
        parser.error(
            f"--synthetic-window {args.synthetic_window!r} is not a key in "
            f"--windows ({sorted(windows)!r})"
        )

    if Path(args.out).exists() and not args.force:
        parser.error(
            f"--out {args.out} already exists -- pass --force to overwrite (each "
            f"kill-clock cycle keeps its own amendment stub)"
        )

    args.windows_parsed = windows


# --------------------------------------------------------------------------- #
# argparse dispatcher
# --------------------------------------------------------------------------- #


def _nonneg_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{raw!r} is not an integer")
    if value < 0:
        raise argparse.ArgumentTypeError(f"{value} must be >= 0 (spine AD-1)")
    return value


def _pad_minutes(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{raw!r} is not a number")
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{raw!r} must be finite and >= 0")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="TICK-INFRA queue-aware fill simulator (research R3).",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sp = sub.add_parser(
        "simulate",
        help="run the fill simulator over a DBN window + a JSONL intent log",
        description=(
            "Filter a Databento .dbn.zst MBO window to one front-month "
            "instrument_id and run sim.simulate over a JSONL OrderIntent log, "
            "writing an OrderOutcome JSONL + a run Manifest JSON (both atomic)."
        ),
    )
    sp.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="INFO-level logging to stderr (sim progress every 1M events + warnings)",
    )
    sp.add_argument("--dbn", required=True, metavar="PATH", help=".dbn.zst MBO window")
    sp.add_argument(
        "--intents", required=True, metavar="PATH", help="JSONL OrderIntent log"
    )
    sp.add_argument(
        "--config",
        choices=("primary", "optimistic"),
        default="primary",
        help="seal-bound sim config (default: primary)",
    )
    sp.add_argument(
        "--instrument-id",
        type=_nonneg_int,
        default=None,
        metavar="INT",
        help=(
            "front-month instrument_id; omitted -> auto-detect the modal id, "
            "which costs a SECOND full pass over the .dbn.zst -- pass this "
            "explicitly for large windows (Amendment 9 pins the MNQ id)"
        ),
    )
    sp.add_argument(
        "--interval",
        action="append",
        nargs=2,
        type=_nonneg_int,
        metavar=("START_NS", "END_NS"),
        help=(
            "half-open [start, end) valid interval in ns; repeatable; omitted "
            "-> one interval spanning the intent log +/- --pad-minutes"
        ),
    )
    sp.add_argument(
        "--pad-minutes",
        type=_pad_minutes,
        default=_DEFAULT_PAD_MINUTES,
        metavar="MIN",
        help=(
            "pad each side of the auto-spanned interval "
            f"(default: {_DEFAULT_PAD_MINUTES:g}); ignored when --interval is given"
        ),
    )
    sp.add_argument(
        "--out-outcomes",
        required=True,
        metavar="PATH",
        help="OrderOutcome JSONL output (atomically overwritten if it exists)",
    )
    sp.add_argument(
        "--out-manifest",
        required=True,
        metavar="PATH",
        help="run Manifest JSON output (atomically overwritten if it exists)",
    )
    sp.add_argument(
        "--degraded-day",
        action="append",
        metavar="YYYY-MM-DD",
        help=(
            "Databento-degraded day, recorded in the manifest; never excluded "
            "(spine AD-13). Repeatable; sim sorts + de-dups the list"
        ),
    )

    rp = sub.add_parser(
        "report",
        help="build the AD-14 three-way P&L from a PRIMARY + OPTIMISTIC run pair",
        description=(
            "Read the OrderOutcome JSONL logs + run Manifest JSON files from a "
            "prior `simulate --config primary` and `simulate --config "
            "optimistic` pair, call report.build_report, and write the "
            "ThreeWayReport (prereg §2.3 / spine AD-14) as pretty JSON."
        ),
    )
    rp.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="INFO-level logging to stderr",
    )
    rp.add_argument(
        "--primary-outcomes",
        required=True,
        metavar="PATH",
        help="PRIMARY run OrderOutcome JSONL (from `simulate --config primary`)",
    )
    rp.add_argument(
        "--primary-manifest",
        required=True,
        metavar="PATH",
        help="PRIMARY run Manifest JSON",
    )
    rp.add_argument(
        "--optimistic-outcomes",
        required=True,
        metavar="PATH",
        help="OPTIMISTIC run OrderOutcome JSONL (from `simulate --config optimistic`)",
    )
    rp.add_argument(
        "--optimistic-manifest",
        required=True,
        metavar="PATH",
        help="OPTIMISTIC run Manifest JSON",
    )
    rp.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="ThreeWayReport JSON output (atomically overwritten if it exists)",
    )

    gp = sub.add_parser(
        "parity-gate",
        help="run the §A8.2 two-part parity gate and write the amendment stub",
        description=(
            "Reconstruct the live bots' orders (mim-nb from "
            "data/mim_nb/orders.csv, yank from data/trades.db rows on/after "
            "2026-06-17), run Part A real-fill calibration over their MBO "
            "windows, a >=1000-order Part B invariant battery over one "
            "synthetic window, the seal-5 integrity preflight per window, then "
            "gate.evaluate + build_amendment_stub. Writes a STANDALONE "
            "amendment .md (the analyst appends it to the seal -- AD-26). Exit "
            "0 PASS / 3 PASS+integrity-flagged / 1 FAIL-or-error / 2 usage."
        ),
    )
    gp.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="INFO-level logging to stderr",
    )
    gp.add_argument(
        "--orders-csv",
        default="data/mim_nb/orders.csv",
        metavar="PATH",
        help=(
            "mim-nb order-lifecycle CSV (default: data/mim_nb/orders.csv); "
            "authoritative for mim-nb -- the DB's mim-nb rows are used only if "
            "this file is absent"
        ),
    )
    gp.add_argument(
        "--trades-db",
        default="data/trades.db",
        metavar="PATH",
        help=(
            "trades.db (default: data/trades.db); yank rows on/after "
            "2026-06-17 always, mim-nb rows only when --orders-csv is absent"
        ),
    )
    gp.add_argument(
        "--windows",
        required=True,
        metavar="PATH",
        help=(
            "window JSON: {key: {dbn, instrument_id, lo_ns, hi_ns, "
            "degraded_days?}}; a trade is routed to the entry whose "
            "[lo_ns, hi_ns) fully contains its stamp span"
        ),
    )
    gp.add_argument(
        "--synthetic-window",
        required=True,
        metavar="KEY",
        help="the --windows key Part B's synthetic orders are generated over",
    )
    gp.add_argument(
        "--synthetic-seed",
        type=_nonneg_int,
        default=0,
        metavar="INT",
        help="Part B RNG seed -- the sole entropy source (default: 0)",
    )
    gp.add_argument(
        "--synthetic-n",
        type=_nonneg_int,
        default=PART_B_MIN_ORDERS,
        metavar="INT",
        help=f"synthetic order count (default: PART_B_MIN_ORDERS = {PART_B_MIN_ORDERS})",
    )
    gp.add_argument(
        "--amendment-number",
        type=int,
        required=True,
        metavar="INT",
        help="the amendment's number in the seal (analyst-owned; AD-26)",
    )
    gp.add_argument(
        "--cycle-number",
        type=int,
        required=True,
        metavar="INT",
        help="the revision-cycle number, rendered `cycle N of 3` (analyst-owned)",
    )
    gp.add_argument(
        "--config",
        choices=("primary", "optimistic"),
        default="primary",
        help="seal-bound sim config (default: primary)",
    )
    gp.add_argument(
        "--sha",
        default=None,
        metavar="SHA",
        help=(
            "frozen simulator SHA (40-char hex); omitted -> gate.frozen_sha() "
            "(`git rev-parse HEAD`)"
        ),
    )
    gp.add_argument(
        "--date",
        default=None,
        metavar="STR",
        help="append date for the stub; omitted -> 'TBD (fill on append)'",
    )
    gp.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="amendment stub .md output -- a NEW standalone file (atomic write)",
    )
    gp.add_argument(
        "--skip-uncovered",
        action="store_true",
        help="drop Part A trades whose full span is not inside any --windows "
        "entry (their MBO window was never purchased) instead of aborting; the "
        "dropped ids are logged and recorded in the stub",
    )
    gp.add_argument(
        "--force",
        action="store_true",
        help="overwrite --out if it already exists (default: refuse -- each "
        "kill-clock cycle keeps its own stub)",
    )
    return parser


def _validate_simulate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Usage-level checks that argparse ``type=`` callables cannot do (they see
    one value at a time). Every failure -> ``parser.error`` -> exit 2."""
    for start, end in args.interval or []:
        if end <= start:
            parser.error(
                f"--interval: {start} {end} is not a valid [start, end) window"
            )

    dbn = Path(args.dbn).resolve()
    intents = Path(args.intents).resolve()
    out_outcomes = Path(args.out_outcomes).resolve()
    out_manifest = Path(args.out_manifest).resolve()
    if out_outcomes == out_manifest:
        parser.error("--out-outcomes and --out-manifest must be different paths")
    inputs = {"--dbn": dbn, "--intents": intents}
    for out_label, out_path in (
        ("--out-outcomes", out_outcomes),
        ("--out-manifest", out_manifest),
    ):
        for in_label, in_path in inputs.items():
            if out_path == in_path:
                parser.error(
                    f"{out_label} must not be the same file as {in_label} "
                    f"({out_path})"
                )


def _validate_report_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """All five ``report`` paths must resolve to distinct files (a collision --
    e.g. ``--out`` equal to an input -- is a usage error), and ``--out`` must
    name a file (not ``.`` / ``/`` / ``""``, which would make the atomic write's
    ``Path.with_name`` raise past its ``except OSError``). ``parser.error`` ->
    exit 2."""
    if not Path(args.out).name:
        parser.error("--out must name a file")
    labeled = (
        ("--primary-outcomes", Path(args.primary_outcomes).resolve()),
        ("--primary-manifest", Path(args.primary_manifest).resolve()),
        ("--optimistic-outcomes", Path(args.optimistic_outcomes).resolve()),
        ("--optimistic-manifest", Path(args.optimistic_manifest).resolve()),
        ("--out", Path(args.out).resolve()),
    )
    seen: dict[Path, str] = {}
    for label, resolved in labeled:
        if resolved in seen:
            parser.error(
                f"{label} and {seen[resolved]} must be different paths " f"({resolved})"
            )
        seen[resolved] = label


def _configure_logging(verbose: bool) -> None:
    """Route ``src.ticksim`` logs to stderr at INFO (``-v``) or WARNING.

    A dedicated handler on the package logger (not ``logging.basicConfig`` on
    the root) so a re-invocation in the same process rebinds to the current
    ``sys.stderr`` and nothing clobbers a caller's root logging / pytest's
    capture.
    """
    global _CLI_LOG_HANDLER
    level = logging.INFO if verbose else logging.WARNING
    pkg_logger = logging.getLogger("src.ticksim")
    pkg_logger.setLevel(level)
    if _CLI_LOG_HANDLER is not None:
        pkg_logger.removeHandler(_CLI_LOG_HANDLER)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    pkg_logger.addHandler(handler)
    pkg_logger.propagate = False
    _CLI_LOG_HANDLER = handler


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a sub-command; return its exit code (``0`` ok, ``1`` handled
    error, ``2`` usage error)."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        if getattr(args, "command", None) == "simulate":
            _validate_simulate_args(parser, args)
        elif getattr(args, "command", None) == "report":
            _validate_report_args(parser, args)
        elif getattr(args, "command", None) == "parity-gate":
            _validate_parity_gate_args(parser, args)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(str(code), file=sys.stderr)
        return 2

    if args.command is None:
        parser.print_usage(sys.stderr)
        return 2
    if args.command == "simulate":
        _configure_logging(bool(args.verbose))
        return _cmd_simulate(args)
    if args.command == "report":
        _configure_logging(bool(args.verbose))
        return _cmd_report(args)
    if args.command == "parity-gate":
        _configure_logging(bool(args.verbose))
        return _cmd_parity_gate(args)
    # defensive: argparse already rejects an unknown sub-command with exit 2
    parser.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
