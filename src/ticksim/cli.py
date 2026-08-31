"""``ticksim`` command-line entry point (spine AD-6).

An ``argparse`` sub-command dispatcher. This slice ships **one** sub-command,
``simulate`` -- read a JSONL :class:`~src.ticksim.orders.OrderIntent` log, open a
Databento ``.dbn.zst`` MBO window, filter it to a single front-month
``instrument_id``, run :func:`~src.ticksim.sim.simulate` under one seal-bound
config (``PRIMARY`` or ``OPTIMISTIC``), and write the
:class:`~src.ticksim.orders.OrderOutcome` JSONL plus the run
:class:`~src.ticksim.sim.Manifest` JSON (both written atomically -- spine AD-11).

Also lands the shared, re-iterable :class:`FrontMonthSource` wrapper and
:func:`detect_front_month` -- the ``MNQ.FUT`` parent stream is ~96% front month,
~4% calendar spread (pre-registration Amendment 9), so every consumer that feeds
``sim`` a single-instrument stream needs this filter.

The ``report`` (3-way P&L) and ``parity-gate`` sub-commands are later slices.

Dependencies (spine AD-7): ``.sim`` / ``.events`` / ``.orders`` / ``.config`` /
``.book`` (for :class:`~src.ticksim.book.BookInconsistency` -- an analyst-facing
simulator fault this boundary catches) + stdlib. No ``datetime`` -- a
``--degraded-day`` is carried through to the manifest as the exact ``str`` token
the analyst typed (spine AD-13). Relative imports only (``mypy --strict``
duplicate-module-errors on the absolute form).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path

from .book import BookInconsistency
from .config import OPTIMISTIC, PRIMARY, SimConfig
from .events import BookEvent, BookEventSource, DbnMboSource
from .orders import OrderIntent, OrderOutcome, OrderStateError
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


# --------------------------------------------------------------------------- #
# intent log + interval helpers
# --------------------------------------------------------------------------- #


def _read_intents(path: Path) -> list[OrderIntent]:
    """Parse a UTF-8 JSONL :class:`~src.ticksim.orders.OrderIntent` log.

    One record per non-blank line via ``OrderIntent.model_validate_json``. The
    list is returned as-is -- ``sim`` validates causal replayability (spine
    AD-2) and raises :class:`~src.ticksim.sim.IntentLogError`.

    Raises:
        _CliError: the file is unreadable or not UTF-8, a non-blank line is not
            a valid ``OrderIntent`` (the line number is named), or the file has
            no intents.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _CliError(f"cannot read intent log {path}: {exc}") from exc
    intents: list[OrderIntent] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            intents.append(OrderIntent.model_validate_json(line))
        except ValueError as exc:
            raise _CliError(
                f"intent log {path} line {lineno}: not a valid OrderIntent ({exc})"
            ) from exc
    if not intents:
        raise _CliError(f"intent log {path}: no intents")
    return intents


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
    # defensive: argparse already rejects an unknown sub-command with exit 2
    parser.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
