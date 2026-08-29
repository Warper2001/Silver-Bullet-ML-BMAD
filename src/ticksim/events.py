"""Vendor-agnostic L3 event plumbing: ``BookEventSource``, ``DbnMboSource``, ``merge_streams``.

This module is the **entire vendor boundary** (spine AD-18). It turns a
Databento ``.dbn.zst`` MBO stream into a stream of normalized, frozen
:class:`BookEvent` records and provides the one canonical k-way merge
(:func:`merge_streams`, spine AD-20) that ``sim.py`` consumes. Nothing
downstream of this module ever sees a ``databento`` / ``databento_dbn`` type --
:class:`BookEvent`, :class:`BookEventSource`, :func:`merge_streams` and
:class:`MboAction` / :class:`MboSide` carry no vendor type on any public name.

Dependencies (spine AD-7): ``.book`` (the :class:`~src.ticksim.book.MboRecord`
Protocol a :class:`BookEvent` must satisfy), stdlib (``contextlib``,
``dataclasses``, ``enum``, ``heapq``, ``logging``, ``pathlib``, ``typing``), and
``databento`` (record types + ``DBNStore`` + the ``UNDEF_*`` sentinels, used
only inside :class:`DbnMboSource` / :func:`_normalize`). Relative imports only --
``mypy --strict src/ticksim`` duplicate-module-errors on the absolute form.

``BookEvent`` does **not** drive a fold: :func:`merge_streams` yields, ``sim.py``
consumes and calls ``book.apply_event`` (spine AD-20). ``DbnMboSource`` yields
*every* MBO record in file order -- no filtering, deduping, or reordering
(``R`` / ``T`` / ``F`` / ``N`` pass straight through; ``book.py`` decides what
they do). It never materializes the file: ``DBNStore`` iteration is already lazy
and transparently handles ``.zst``. Each ``__iter__`` opens a fresh
``DBNStore``, so a source is **re-iterable** (the future H1 grid re-reads
sources many times).

Event classes and their ``class_rank`` (spine AD-20): ``book_delta`` (0) <
``order_arrival`` (1) < ``deferred_fill_apply`` (2). This module only *produces*
rank 0 (:class:`DbnMboSource`); :func:`merge_streams` merely *accepts*
heterogeneous ranks -- ranks 1 / 2 are ``sim.py``'s.

Integer-only (spine AD-10): every :class:`BookEvent` field is ``int``. The clock
is ``ts_event`` (spine AD-1); ``ts_recv`` and ``flags`` are never read.
"""

from __future__ import annotations

import heapq
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Protocol, runtime_checkable

from databento import DBNRecord, DBNStore, MBOMsg, UNDEF_PRICE

from .book import MboRecord

__all__ = [
    "BookEvent",
    "BookEventSource",
    "DbnMboSource",
    "MboAction",
    "MboSide",
    "merge_streams",
]

logger = logging.getLogger(__name__)


class MboAction(StrEnum):
    """MBO action code as a ``str`` enum (spine AD-18).

    ``str(MboAction.MODIFY) == "M"`` -- the single-char MBO code -- so a
    :class:`BookEvent` carrying an ``MboAction`` folds through
    ``book.apply_event`` unchanged (which normalizes with ``str(record.action)``).
    Values match ``databento_dbn.Action`` at runtime but the vendor enum never
    escapes :func:`_normalize`.
    """

    ADD = "A"
    CANCEL = "C"
    MODIFY = "M"
    TRADE = "T"
    FILL = "F"
    CLEAR = "R"
    NONE = "N"


class MboSide(StrEnum):
    """MBO side code as a ``str`` enum (spine AD-18).

    ``str(MboSide.BID) == "B"``. Mirrors ``book.py``'s ``_SIDE_BID`` / ``_SIDE_ASK``
    string constants so ``book.apply_event`` folds a :class:`BookEvent` unchanged.
    """

    BID = "B"
    ASK = "A"
    NONE = "N"


@dataclass(frozen=True, slots=True)
class BookEvent:
    """One normalized L3 MBO event (spine AD-1, AD-10, AD-18).

    Frozen + ``slots`` (millions of allocations on the hot path), all-``int``
    (``MboAction`` / ``MboSide`` are ``str`` enums), matching ``book.RestingOrder``
    style. The price field is ``price_dbn`` -- DBN 1e-9 fixed-point, the same
    name ``book.MboRecord`` / ``book.RestingOrder`` use -- so a :class:`BookEvent`
    structurally satisfies :class:`~src.ticksim.book.MboRecord` and is a drop-in
    for ``book.apply_event``.
    """

    action: MboAction
    side: MboSide
    order_id: int
    price_dbn: int
    size: int
    ts_event: int
    sequence: int
    instrument_id: int


if TYPE_CHECKING:
    # Static guarantee that a BookEvent is a drop-in for book.apply_event
    # (spine AD-18). No runtime cost.
    _book_event_is_mbo_record: type[MboRecord] = BookEvent


@runtime_checkable
class BookEventSource(Protocol):
    """A stream of :class:`BookEvent`s in non-decreasing ``ts_event`` (spine AD-18).

    ``class_rank`` is a property of the *stream*, not the record (Design Notes):
    it fixes where this source's events sort relative to other event classes at
    an equal ``(ts_event, sequence)`` -- ``book_delta`` (0) < ``order_arrival``
    (1) < ``deferred_fill_apply`` (2) (spine AD-20). :class:`DbnMboSource` is
    rank 0.

    Iterating a source more than once must reproduce the same events -- the
    merge is pull-based and downstream studies replay sources repeatedly.
    """

    class_rank: int

    def __iter__(self) -> Iterator[BookEvent]: ...


class DbnMboSource:
    """Streams a Databento ``.dbn.zst`` MBO file as :class:`BookEvent`s (spine AD-18).

    The only :class:`BookEventSource` implementation today; a compact L3 cache is
    a future one. Construction does a cheap ``Path.is_file()`` check so a missing
    path fails fast; the actual ``DBNStore.from_file`` is deferred to
    :meth:`__iter__`, which opens a **fresh** handle every call -- the source is
    re-iterable. ``DBNStore`` is a lazy generator that handles ``.zst``
    transparently and never fully decompresses; one record is in flight at a time.

    Yields **every** MBO record in file order (spine AD-18 "Never: Filtering,
    deduping, or reordering within a single source"). Non-MBO administrative
    records (symbol mappings, system messages) are not book events; they are
    skipped and counted in :attr:`skipped_record_count` (updated when an
    iteration ends). An unrecognized action / side code, or an undefined price
    on an ``A`` / ``M``, raises ``ValueError``.

    Raises:
        FileNotFoundError: at construction, if ``path`` is not an existing file.
        ValueError: on first iteration, if the file is not a readable DBN stream.
    """

    class_rank: int = 0

    def __init__(self, path: str | Path) -> None:
        self._path: Path | None = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"no such DBN file: {self._path}")
        self._override: Iterable[object] | None = None
        self.skipped_record_count: int = 0

    @classmethod
    def _from_iterable(
        cls, records: Iterable[object], *, class_rank: int = 0
    ) -> "DbnMboSource":
        """Test seam: a source backed by an in-memory record iterable.

        Records flow through the exact :meth:`__iter__` normalize / skip path a
        real file would, so a future ``__init__`` change cannot silently make
        the test double diverge from production. The iterable is stored as-is
        (not copied): pass a ``list`` for a re-iterable double, a one-shot
        generator to pin pull timing.
        """
        self = cls.__new__(cls)
        self._path = None
        self._override = records
        self.class_rank = class_rank
        self.skipped_record_count = 0
        return self

    def _raw_records(self) -> Iterator[DBNRecord] | Iterator[object]:
        if self._override is not None:
            return iter(self._override)
        assert self._path is not None  # narrowed: _override is None => file-backed
        try:
            store = DBNStore.from_file(str(self._path))
        except FileNotFoundError:
            raise
        except Exception as exc:  # opaque vendor decode failure -> clear message
            raise ValueError(f"not a readable DBN file: {self._path}") from exc
        return iter(store)

    def __iter__(self) -> Iterator[BookEvent]:
        skipped = 0
        try:
            for record in self._raw_records():
                if not isinstance(record, MBOMsg):
                    skipped += 1
                    continue
                yield _normalize(record)
        finally:
            self.skipped_record_count = skipped
            logger.debug(
                "DbnMboSource(%s) iteration ended: %d non-MBO records skipped",
                self._path if self._path is not None else "<in-memory>",
                skipped,
            )

    def __repr__(self) -> str:
        target = self._path if self._path is not None else "<in-memory>"
        return f"DbnMboSource({target!r})"


def _normalize(msg: MBOMsg) -> BookEvent:
    """Map one ``databento_dbn.MBOMsg`` to a vendor-free :class:`BookEvent`.

    ``str(msg.action)`` / ``str(msg.side)`` yield the single-char MBO codes at
    runtime; the :class:`MboAction` / :class:`MboSide` lookup raises
    ``ValueError`` on an unknown code, and an ``A`` / ``M`` whose price is the
    DBN ``UNDEF_PRICE`` sentinel is rejected the same way (mirrors ``book.py``'s
    ``UNDEF_ORDER_SIZE`` hardening) -- a deterministic failure, never a silent
    mis-fold (spine AD-9 spirit).
    """
    action_code = str(msg.action)
    side_code = str(msg.side)
    try:
        action = MboAction(action_code)
    except ValueError as exc:
        raise ValueError(f"unknown MBO action code {action_code!r}") from exc
    try:
        side = MboSide(side_code)
    except ValueError as exc:
        raise ValueError(f"unknown MBO side code {side_code!r}") from exc

    price_dbn = int(msg.price)
    if action in (MboAction.ADD, MboAction.MODIFY) and price_dbn == UNDEF_PRICE:
        raise ValueError(
            f"{action_code} record for order_id {int(msg.order_id)} carries an "
            f"undefined price (UNDEF_PRICE sentinel)"
        )

    return BookEvent(
        action=action,
        side=side,
        order_id=int(msg.order_id),
        price_dbn=price_dbn,
        size=int(msg.size),
        ts_event=int(msg.ts_event),
        sequence=int(msg.sequence),
        instrument_id=int(msg.instrument_id),
    )


# Heap item: (order_key, source_index, event, iterator). ``order_key`` is a
# 4-int tuple that is globally unique (``source_index`` is its last element and
# is unique per source; a source contributes at most one heap entry at a time),
# so heapq never compares ``event`` -- ``source_index`` is a redundant but cheap
# guard.
_HeapItem = tuple[tuple[int, int, int, int], int, BookEvent, Iterator[BookEvent]]


def merge_streams(*sources: BookEventSource) -> Iterator[BookEvent]:
    """Stable k-way merge of :class:`BookEventSource`s into the canonical order.

    Order key ``(ev.ts_event, src.class_rank, ev.sequence, source_index)`` where
    ``source_index`` is the argument position (spine AD-20). Equal
    ``(ts_event, sequence)`` across sources therefore preserves argument order --
    a **stable** merge. Heap-based, lazy, O(total * log k); one record per source
    held at a time.

    Accepts heterogeneous ``class_rank``s (this slice only feeds rank 0, but
    ``sim.py`` interleaves rank 1 / 2 sources). ``merge_streams()`` with no
    arguments yields nothing.

    Raises:
        TypeError: if any argument is not a :class:`BookEventSource`.
        ValueError: if the same source object is passed more than once (it would
            be pulled twice and its events double-counted), or if any source
            yields an event whose ``(ts_event, sequence)`` is below the previous
            one it yielded -- a source contract violation (spine AD-18 requires
            non-decreasing ``ts_event``; ``sequence`` must not regress within a
            tick either). On this (or any) mid-iteration failure every other
            live source iterator is closed before the error propagates.
    """
    seen_ids: set[int] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, BookEventSource):
            raise TypeError(
                f"merge_streams argument {index} is not a BookEventSource: "
                f"{type(source).__name__}"
            )
        if id(source) in seen_ids:
            raise ValueError(
                f"merge_streams argument {index} is a source already passed "
                f"more than once -- give each stream its own source object"
            )
        seen_ids.add(id(source))
    return _merge(sources)


def _merge(sources: tuple[BookEventSource, ...]) -> Iterator[BookEvent]:
    ranks = [source.class_rank for source in sources]
    last_key: list[tuple[int, int] | None] = [None] * len(sources)
    heap: list[_HeapItem] = []

    with ExitStack() as stack:
        iterators: list[Iterator[BookEvent]] = []
        for source in sources:
            iterator = iter(source)
            iterators.append(iterator)
            closer = getattr(iterator, "close", None)
            if callable(closer):
                stack.callback(closer)

        def pull(index: int) -> None:
            try:
                event = next(iterators[index])
            except StopIteration:
                return
            key2 = (event.ts_event, event.sequence)
            previous = last_key[index]
            if previous is not None and key2 < previous:
                raise ValueError(
                    f"source {index}: (ts_event, sequence) went backwards "
                    f"{previous} -> {key2}"
                )
            last_key[index] = key2
            heapq.heappush(
                heap,
                (
                    (event.ts_event, ranks[index], event.sequence, index),
                    index,
                    event,
                    iterators[index],
                ),
            )

        for index in range(len(sources)):
            pull(index)

        while heap:
            _key, index, event, _iterator = heapq.heappop(heap)
            yield event
            pull(index)
