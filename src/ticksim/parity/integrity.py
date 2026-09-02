"""The seal-§5 MBO-window integrity preflight (prereg §1 / §A9.3, spine AD-26).

Prereg §1 requires an **integrity check before use** on every contract-window
file: monotonic non-decreasing ``ts_event``; no *persistent* crossed market
(a cross living ``> config.MAX_TRANSIENT_CROSS_NS`` -- momentary CME locks are
not errors, §A9.3); the exact BBO-cross pass rate; trade prices inside the
session's own low/high; the reconstruction must consume ``A/C/M/T/F``; no
unrecorded halt / missing-data hole; the ``degraded`` days (2026-05-24 and
2026-07-30) are noted, never dropped (§A8.3 / AD-13). Spine AD-26 says
``gate.build_amendment_stub`` carries this report in its ``integrity:`` slot,
which is a ``pending`` placeholder until this module fills it.

:func:`preflight_integrity` makes **one** read-only pass over a window
:class:`~src.ticksim.events.BookEventSource` (re-iterable per AD-18; iterated
once here). Every per-event body runs inside ``try/except Exception`` -- a
mid-stream vendor ``ValueError``, a malformed record, anything, is counted in
``malformed_events`` and the scan continues; the function only ever raises if
``iter(source)`` itself fails. It folds a :class:`~src.ticksim.book.Book` (a
foreign second ``instrument_id`` is counted and **skipped**, never folded) to
watch the BBO after every event, and counts: ``ts_event`` regressions (vs the
running maximum, matching ``book.apply_event``'s own guard) and over-threshold
inter-event gaps; transient vs persistent crossed markets and a cross left open
at EOF; trades printing outside the book BBO; caught
:class:`~src.ticksim.book.BookInconsistency`\\ s (bucketed -- the cross's own
raises and the regression's own raise are not double-counted); unknown-order
``C`` / ``M`` references (a warm-up bucket that never flags vs a real
post-warm-up ``book_inconsistencies`` bucket, §A9.2); missing action classes;
and a single end-of-stream ``book.check_invariants()``.

It **never raises on a data problem** -- it always returns an
:class:`IntegrityReport` whose ``verdict`` is ``"OK"`` or ``"FLAGGED"``.
**Verdict-reporting, not verdict-bearing**: whether a ``FLAGGED`` report blocks
the parity gate is that CLI slice's decision.

:func:`format_integrity` renders the report as a deterministic, ASCII-only
Markdown block (no ``#`` / ``##`` headings -- it nests inside the gate stub's
own section) for the ``integrity:`` slot.

Dependencies (spine AD-7, ``PERMITTED_INTERNAL_EDGES["integrity"]``): ``events``,
``book``, ``config`` -- relative form only (``mypy --strict`` duplicate-module-
errors on the absolute form). Standard library: ``dataclasses``, ``typing`` only
-- no ``logging``-gated behaviour, no wall-clock, no second source pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from ..book import Book, BookInconsistency, apply_event
from ..config import MAX_TRANSIENT_CROSS_NS, MNQ_TICK_DBN
from ..events import BookEventSource, MboAction

__all__ = [
    "IntegrityReport",
    "preflight_integrity",
    "format_integrity",
]

# The first few examples of each anomaly kept for the amendment stub. Every
# recorded-example list is capped at this length.
_MAX_EXAMPLES: int = 5

# An intraday-start window has no opening snapshot: a ``C`` / ``M`` for an order
# the book never ``A``-dded is expected in roughly the first minute (§A9.2:
# ~0.3 % of C/M, concentrated in the first ~60 s). Within this window of the
# first event such an unknown-order reference is counted in ``warmup_unknown_ref``
# (never a flag); past it, it is a real problem and lands in
# ``book_inconsistencies``.
_WARMUP_NS: int = 60_000_000_000

# A trade printing more than this many ticks outside a two-sided book BBO is an
# off-book print (``trades_off_book``). A one-sided or empty book at that instant
# has nothing to compare against and is not counted.
_TRADE_BBO_TOLERANCE_TICKS: int = 2

# A forward jump larger than this between two consecutive primary events is an
# unrecorded halt / missing-data hole (§1) -- ``gaps_over_threshold`` and a flag.
# 5 minutes.
_MAX_GAP_NS: int = 300_000_000_000

# DBN's "undefined price" sentinel (INT64_MAX). Pinned as a literal so this
# module stays free of any ``databento`` import (spine AD-18). ``events._normalize``
# rejects it on an ``A`` / ``M`` but lets it through on a ``T`` -- so a trade can
# carry it, and the session-range update must skip such a print.
_UNDEF_PRICE_DBN: int = 9_223_372_036_854_775_807

_ACTION_CLASSES: frozenset[str] = frozenset({"A", "C", "M", "T", "F"})


@dataclass(frozen=True)
class IntegrityReport:
    """The result of one :func:`preflight_integrity` pass (prereg §1 / §A9.3).

    ``verdict == "FLAGGED"`` iff any of: ``ts_regressions > 0``;
    ``persistent_cross_count > 0``; ``unresolved_cross_at_end``;
    ``foreign_instrument_events > 0``; ``missing_actions != ()``;
    ``book_inconsistencies > 0``; ``trades_off_book > 0``;
    ``malformed_events > 0``; ``gaps_over_threshold > 0``;
    ``check_invariants_failed``; ``n_events == 0``. Transient crosses,
    ``degraded_days``, ``warmup_unknown_ref`` and ``stale_cross_count`` never
    flag (all expected -- §A9.2 / §A9.3).

    ``stale_cross_count`` is ``Book.stale_cross_count`` after the fold: the
    number of crossed-market *episodes* wider than
    ``config.STALE_CROSS_MAX_TICKS`` that ``book._check_cross`` tolerated as
    cold-start ghosts (this window's book has no opening snapshot, so a
    pre-window resting order is never ``A``-dded and its stale level can sit on
    one side of the book). Reported so a reader can judge the tolerance; it is
    never a flag reason, and a ghost cross that *does* outlive
    ``MAX_TRANSIENT_CROSS_NS`` still lands in ``persistent_cross_count`` via
    this module's own (unchanged) BBO state machine.

    ``flags`` is a fixed-order human list of the reasons and is
    empty iff ``verdict == "OK"``.
    """

    n_events: int
    malformed_events: int
    instrument_id: int | None
    foreign_instrument_events: int
    first_ts_ns: int | None
    last_ts_ns: int | None
    duration_ns: int
    largest_gap_ns: int
    gaps_over_threshold: int
    ts_regressions: int
    ts_regression_examples: tuple[tuple[int, int], ...]
    transient_cross_count: int
    persistent_cross_count: int
    persistent_crosses: tuple[tuple[int, int, int], ...]
    bbo_cross_rate: float
    unresolved_cross_at_end: bool
    n_trades: int
    session_low_dbn: int | None
    session_high_dbn: int | None
    trades_off_book: int
    actions_seen: frozenset[str]
    missing_actions: tuple[str, ...]
    book_inconsistencies: int
    warmup_unknown_ref: int
    stale_cross_count: int
    check_invariants_failed: bool
    degraded_days: tuple[str, ...]
    verdict: Literal["OK", "FLAGGED"]
    flags: tuple[str, ...]


def preflight_integrity(
    source: BookEventSource,
    *,
    degraded_days: Sequence[str] = (),
) -> IntegrityReport:
    """Survey one window ``source`` for the seal-§5 integrity properties.

    Makes a single forward pass, folding a :class:`~src.ticksim.book.Book` and
    inspecting ``book.snapshot_bbo`` after every event. Never raises on a data
    problem -- a malformed record / vendor exception is counted in
    ``malformed_events`` and the scan continues, and an :class:`IntegrityReport`
    is always returned (the only way this raises is ``iter(source)`` failing).

    Args:
        source: a **single-instrument** window
            :class:`~src.ticksim.events.BookEventSource` (front-month filtering
            is the caller's job -- a second ``instrument_id`` in the stream is
            counted in ``foreign_instrument_events``, skipped, and flags the
            report).
        degraded_days: date strings Databento marked ``degraded`` for this
            window (e.g. ``("2026-07-30",)``). Recorded in the report and the
            ``format_integrity`` note; **never** a flag reason (§A9.3).

    Returns:
        An :class:`IntegrityReport` with ``verdict`` ``"OK"`` or ``"FLAGGED"``.
    """
    iterator = iter(source)  # the one place this function may propagate

    book = Book()
    instrument_id: int | None = None
    first_ts_ns: int | None = None
    last_ts_ns: int | None = None
    prev_ts: int | None = None
    max_ts_seen: int | None = None

    n_events = 0
    malformed_events = 0
    foreign_instrument_events = 0
    largest_gap_ns = 0
    gaps_over_threshold = 0
    ts_regressions = 0
    ts_regression_examples: list[tuple[int, int]] = []
    transient_cross_count = 0
    persistent_cross_count = 0
    persistent_crosses: list[tuple[int, int, int]] = []
    n_trades = 0
    session_low_dbn: int | None = None
    session_high_dbn: int | None = None
    trades_off_book = 0
    actions_seen: set[str] = set()
    book_inconsistencies = 0
    warmup_unknown_ref = 0

    cross_open_ts: int | None = None
    unresolved_cross_at_end = False
    tolerance_dbn = _TRADE_BBO_TOLERANCE_TICKS * MNQ_TICK_DBN

    while True:
        try:
            ev = next(iterator)
        except StopIteration:
            break
        except Exception:  # vendor _normalize ValueError, decode failure, ...
            malformed_events += 1
            continue

        try:
            n_events += 1

            if instrument_id is None:
                instrument_id = ev.instrument_id
                first_ts_ns = ev.ts_event

            if ev.instrument_id != instrument_id:
                foreign_instrument_events += 1
                continue  # skipped: not folded, not in actions_seen

            assert first_ts_ns is not None

            regressed = max_ts_seen is not None and ev.ts_event < max_ts_seen
            if regressed:
                assert max_ts_seen is not None
                ts_regressions += 1
                if len(ts_regression_examples) < _MAX_EXAMPLES:
                    ts_regression_examples.append((max_ts_seen, ev.ts_event))
            elif prev_ts is not None:
                gap = ev.ts_event - prev_ts
                if gap > largest_gap_ns:
                    largest_gap_ns = gap
                if gap > _MAX_GAP_NS:
                    gaps_over_threshold += 1

            prev_ts = ev.ts_event
            last_ts_ns = ev.ts_event
            max_ts_seen = (
                ev.ts_event if max_ts_seen is None else max(max_ts_seen, ev.ts_event)
            )

            cross_was_open = cross_open_ts is not None
            unseen_before = book.unseen_cm_count
            overcancel_before = book.overcancel_count
            caught_inconsistency = False
            try:
                apply_event(book, ev)
            except BookInconsistency:
                caught_inconsistency = True
            unseen_delta = book.unseen_cm_count - unseen_before
            overcancel_delta = book.overcancel_count - overcancel_before

            within_warmup = 0 <= ev.ts_event - first_ts_ns <= _WARMUP_NS

            # Unknown-order C/M reference: warm-up bucket (never flags) vs a
            # real post-warm-up book inconsistency.
            if unseen_delta > 0:
                if within_warmup:
                    warmup_unknown_ref += unseen_delta
                else:
                    book_inconsistencies += unseen_delta

            # Over-cancel of a *known* order is real damage regardless of when.
            if overcancel_delta > 0:
                book_inconsistencies += overcancel_delta

            # A caught structural inconsistency -- but not the regression's own
            # raise (already flagged) and not the cross's own raise
            # (book.apply_event _fail()s on every event while a cross has
            # persisted >= MAX_TRANSIENT_CROSS_NS; the BBO-watch already flags
            # it once).
            if caught_inconsistency and not regressed and not cross_was_open:
                book_inconsistencies += 1

            action = str(ev.action)
            actions_seen.add(action)

            bid, ask = book.snapshot_bbo(instrument_id)
            crossed = bid is not None and ask is not None and bid >= ask
            if crossed:
                if cross_open_ts is None:
                    cross_open_ts = ev.ts_event
            elif cross_open_ts is not None:
                duration = ev.ts_event - cross_open_ts
                if duration < 0 or duration > MAX_TRANSIENT_CROSS_NS:
                    persistent_cross_count += 1
                    if len(persistent_crosses) < _MAX_EXAMPLES:
                        persistent_crosses.append(
                            (cross_open_ts, ev.ts_event, duration)
                        )
                else:
                    transient_cross_count += 1
                cross_open_ts = None

            if action == str(MboAction.TRADE):
                n_trades += 1
                price = ev.price_dbn
                if price != _UNDEF_PRICE_DBN and price > 0:
                    if session_low_dbn is None or price < session_low_dbn:
                        session_low_dbn = price
                    if session_high_dbn is None or price > session_high_dbn:
                        session_high_dbn = price
                    if bid is not None and ask is not None:
                        if price > ask + tolerance_dbn or price < bid - tolerance_dbn:
                            trades_off_book += 1
        except Exception:  # a malformed record surfaced inside the body
            malformed_events += 1
            continue

    if cross_open_ts is not None:
        unresolved_cross_at_end = True

    check_invariants_failed = False
    try:
        book.check_invariants()
    except Exception:
        check_invariants_failed = True

    duration_ns = (
        last_ts_ns - first_ts_ns
        if first_ts_ns is not None and last_ts_ns is not None
        else 0
    )
    total_crosses = transient_cross_count + persistent_cross_count
    bbo_cross_rate = total_crosses / n_events if n_events else 0.0
    missing_actions = tuple(sorted(_ACTION_CLASSES - actions_seen))

    flags: list[str] = []
    if n_events == 0:
        flags.append("no events")
    else:
        if ts_regressions > 0:
            flags.append("ts regressions")
        if persistent_cross_count > 0:
            flags.append("persistent cross")
        if unresolved_cross_at_end:
            flags.append("unresolved cross at end")
        if foreign_instrument_events > 0:
            flags.append("foreign instrument_id")
        if missing_actions:
            flags.append("missing actions: " + ",".join(missing_actions))
        if book_inconsistencies > 0:
            flags.append("book inconsistencies")
        if trades_off_book > 0:
            flags.append("off-book trades")
        if malformed_events > 0:
            flags.append("malformed events")
        if gaps_over_threshold > 0:
            flags.append("inter-event gap over threshold")
        if check_invariants_failed:
            flags.append("check_invariants failed")
    verdict: Literal["OK", "FLAGGED"] = "FLAGGED" if flags else "OK"

    return IntegrityReport(
        n_events=n_events,
        malformed_events=malformed_events,
        instrument_id=instrument_id,
        foreign_instrument_events=foreign_instrument_events,
        first_ts_ns=first_ts_ns,
        last_ts_ns=last_ts_ns,
        duration_ns=duration_ns,
        largest_gap_ns=largest_gap_ns,
        gaps_over_threshold=gaps_over_threshold,
        ts_regressions=ts_regressions,
        ts_regression_examples=tuple(ts_regression_examples),
        transient_cross_count=transient_cross_count,
        persistent_cross_count=persistent_cross_count,
        persistent_crosses=tuple(persistent_crosses),
        bbo_cross_rate=bbo_cross_rate,
        unresolved_cross_at_end=unresolved_cross_at_end,
        n_trades=n_trades,
        session_low_dbn=session_low_dbn,
        session_high_dbn=session_high_dbn,
        trades_off_book=trades_off_book,
        actions_seen=frozenset(actions_seen),
        missing_actions=missing_actions,
        book_inconsistencies=book_inconsistencies,
        warmup_unknown_ref=warmup_unknown_ref,
        stale_cross_count=book.stale_cross_count,
        check_invariants_failed=check_invariants_failed,
        degraded_days=tuple(str(d) for d in degraded_days),
        verdict=verdict,
        flags=tuple(flags),
    )


def _ts(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def format_integrity(report: IntegrityReport) -> str:
    """Render ``report`` as the deterministic ASCII Markdown ``integrity:`` block.

    No ``#`` / ``##`` headings -- the block nests inside
    ``gate.build_amendment_stub``'s own ``## Integrity`` section. The first line
    is ``integrity: OK`` or ``integrity: FLAGGED (<reasons joined by ", ">)``,
    derived from ``report.flags`` (a ``FLAGGED`` report always has >= 1 flag, an
    ``OK`` report none, so ``"FLAGGED ()"`` is unrepresentable). Then the window
    span, one bullet per counter (``bbo_cross_rate`` as a percentage,
    ``session_low`` / ``session_high`` labelled ``(informational)``), the
    degraded-day note, and -- on ``FLAGGED`` -- the capped regression /
    persistent-cross examples. Calling this twice on the same report yields
    byte-identical, ASCII-only text.
    """
    if report.flags:
        lines = ["integrity: FLAGGED (" + ", ".join(report.flags) + ")"]
    else:
        lines = ["integrity: OK"]

    actions = ",".join(sorted(report.actions_seen)) if report.actions_seen else "(none)"
    missing = ",".join(report.missing_actions) if report.missing_actions else "(none)"
    degraded = ",".join(str(d) for d in report.degraded_days) or "(none)"

    lines.extend(
        [
            f"- window: {_ts(report.first_ts_ns)} .. {_ts(report.last_ts_ns)} "
            f"({report.duration_ns} ns)",
            f"- events: {report.n_events}",
            f"- malformed events: {report.malformed_events}",
            f"- instrument_id: {_ts(report.instrument_id)}",
            f"- foreign instrument events: {report.foreign_instrument_events}",
            f"- largest inter-event gap (ns): {report.largest_gap_ns}",
            f"- gaps over threshold: {report.gaps_over_threshold}",
            f"- ts regressions: {report.ts_regressions}",
            f"- transient crosses: {report.transient_cross_count}",
            f"- persistent crosses: {report.persistent_cross_count}",
            f"- unresolved cross at end: {report.unresolved_cross_at_end}",
            f"- bbo cross rate: {report.bbo_cross_rate * 100:.4f}%",
            f"- trades: {report.n_trades}",
            f"- trades off book: {report.trades_off_book}",
            f"- session low (dbn, informational): {_ts(report.session_low_dbn)}",
            f"- session high (dbn, informational): {_ts(report.session_high_dbn)}",
            f"- actions seen: {actions}",
            f"- missing actions: {missing}",
            f"- book inconsistencies: {report.book_inconsistencies}",
            f"- warmup unknown refs: {report.warmup_unknown_ref}",
            f"- stale (cold-start) cross episodes tolerated: "
            f"{report.stale_cross_count}",
            f"- check_invariants failed: {report.check_invariants_failed}",
            f"- degraded days: {degraded}",
        ]
    )

    if report.verdict == "FLAGGED":
        if report.ts_regression_examples:
            lines.append("- ts regression examples:")
            for prev, current in report.ts_regression_examples[:_MAX_EXAMPLES]:
                lines.append(f"  - prev_max={prev} ts={current}")
        if report.persistent_crosses:
            lines.append("- persistent cross examples:")
            for start_ts, resolve_ts, duration in report.persistent_crosses[
                :_MAX_EXAMPLES
            ]:
                lines.append(
                    f"  - start={start_ts} resolve={resolve_ts} duration={duration}"
                )

    return "\n".join(lines)
