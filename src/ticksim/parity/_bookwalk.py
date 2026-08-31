"""Bounded, read-only L3 book replay shared by the parity runners (spine AD-17).

:class:`BookReplay` lifts the bounded book walk that used to live inside
``part_a_runner._touch_at`` into one reusable object. It holds a single
:class:`~src.ticksim.book.Book` and one iterator over a
:class:`~src.ticksim.events.BookEventSource`, and folds that source forward on
demand with :meth:`advance_to` -- never past the requested ``ts_ns``, and never
backwards. It runs no fills, touches no tracker, and mutates nothing outside its
own ``Book`` (spine AD-3 / AD-5: everything but ``sim`` is read-only).

One consumer today:

* :func:`~src.ticksim.parity.part_a_runner._touch_at` -- a fresh ``BookReplay``
  per unfilled-leg miss, asked for the window BBO at the real fill ts.

``part_b.run_part_b`` deliberately does **not** use it (loopback 1, 2026-08-31):
invariant 5's book-liquidity half is a ``fills.py`` construction guarantee, not
a ``part_b`` book cross-check. The extraction still stands -- it removes a
duplicated fail-closed guard and the slice-2 synthetic-order generator will need
BBO sampling.

Fail-closed (spine Consistency Conventions): a ``ts_event`` regression inside the
source, an :meth:`advance_to` call whose ``ts_ns`` is below a previous call's, a
second ``instrument_id`` appearing in the stream, or the source iterator raising
anything other than ``StopIteration`` each raise :class:`BookWalkError` -- a
mis-ordered, unfiltered, or broken window stream is a loader bug, never
silently truncated state.

Dependencies (spine AD-7, parity edge): ``book``, ``events``. Relative imports
only (``mypy --strict`` duplicate-module-errors on the absolute form).
"""

from __future__ import annotations

from collections.abc import Iterator

from ..book import Book, apply_event
from ..events import BookEvent, BookEventSource

__all__ = ["BookReplay", "BookWalkError"]


class BookWalkError(Exception):
    """The event stream driving a :class:`BookReplay` cannot be replayed.

    Raised for a ``ts_event`` regression within the source, a call to
    :meth:`BookReplay.advance_to` with a ``ts_ns`` below a previous call's, a
    second ``instrument_id`` in the stream (front-month filtering is the
    caller's job -- ``sim``'s own multi-instrument guard is lazy and can miss an
    id that only appears after the walk's cutoff), or the source iterator
    raising anything other than ``StopIteration``.
    """


class BookReplay:
    """A bounded, read-only forward replay of one L3 :class:`BookEventSource`.

    Construct with the source; call :meth:`advance_to` with a non-decreasing
    ``ts_ns`` to fold every event with ``ts_event <= ts_ns`` into :attr:`book`.
    The first event past the cutoff is held and folded by the next
    :meth:`advance_to` that reaches it.

    An instance must **not** be reused after any :meth:`advance_to` raised a
    :class:`BookWalkError` -- the replay is left half-folded and every further
    :meth:`advance_to` re-raises. Build a fresh :class:`BookReplay` (the slice-2
    generator calls :meth:`advance_to` repeatedly and must abandon a broken one).
    """

    def __init__(self, source: BookEventSource) -> None:
        self._iter: Iterator[BookEvent] = iter(source)
        self._book = Book()
        self._instrument_id: int | None = None
        self._prev_ts: int | None = None
        self._pending: BookEvent | None = None
        self._exhausted = False
        self._last_advance_ts: int | None = None
        self._broken = False

    @property
    def book(self) -> Book:
        """The book folded up to the last :meth:`advance_to` cutoff."""
        return self._book

    @property
    def instrument_id(self) -> int | None:
        """The single ``instrument_id`` folded so far, or ``None`` if no event
        has been folded yet (an empty book cannot be priced)."""
        return self._instrument_id

    def advance_to(self, ts_ns: int) -> None:
        """Fold every not-yet-folded source event with ``ts_event <= ts_ns``.

        ``ts_ns`` must be ``>=`` every previous call's -- a regression is a
        caller bug and raises :class:`BookWalkError`. Calling again with a
        ``ts_ns`` that folds nothing new is a no-op.

        Raises:
            BookWalkError: this replay already failed once; ``ts_ns`` regressed
                across calls; the source's ``ts_event`` regressed; a second
                ``instrument_id`` appeared; or the source iterator raised a
                non-``StopIteration`` exception.
            BookInconsistency: a structural book check failed (from
                ``book.apply_event``).
        """
        if self._broken:
            raise BookWalkError("BookReplay is unusable after a prior failure")
        if self._last_advance_ts is not None and ts_ns < self._last_advance_ts:
            self._broken = True
            raise BookWalkError(
                f"advance_to ts_ns regressed ({ts_ns} < {self._last_advance_ts}) "
                f"-- BookReplay requires non-decreasing cutoffs"
            )
        self._last_advance_ts = ts_ns

        if self._pending is not None:
            if self._pending.ts_event > ts_ns:
                return
            self._fold(self._pending)
            self._pending = None

        while not self._exhausted:
            try:
                event = next(self._iter)
            except StopIteration:
                self._exhausted = True
                break
            except Exception as exc:  # source iterator failure (I/O, decode, ...)
                self._exhausted = True
                self._broken = True
                raise BookWalkError(
                    f"book event source raised {type(exc).__name__}: {exc} "
                    f"-- the window stream cannot be replayed"
                ) from exc
            if self._prev_ts is not None and event.ts_event < self._prev_ts:
                self._broken = True
                raise BookWalkError(
                    f"source ts_event regressed ({event.ts_event} < "
                    f"{self._prev_ts}) -- a mis-ordered stream cannot be replayed"
                )
            self._prev_ts = event.ts_event
            if event.ts_event > ts_ns:
                self._pending = event
                break
            self._fold(event)

    def _fold(self, event: BookEvent) -> None:
        if self._instrument_id is None:
            self._instrument_id = event.instrument_id
        elif event.instrument_id != self._instrument_id:
            self._broken = True
            raise BookWalkError(
                f"source is multi-instrument ({event.instrument_id} != "
                f"{self._instrument_id}) -- front-month filtering is the "
                f"caller's job"
            )
        apply_event(self._book, event)
