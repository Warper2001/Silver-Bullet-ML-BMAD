"""Unit + integration tests for ``src.ticksim.events`` (spine AD-18 / AD-20).

Layers:
  * the vendor-confinement guard -- the module's whole reason to exist: no
    public name resolves to a ``databento`` / ``databento_dbn`` type;
  * every row of the spec's I/O & Edge-Case Matrix, routed through the *public*
    surface (``DbnMboSource._from_iterable`` / ``merge_streams``), not private
    helpers;
  * ``str(MboAction.X)`` / ``str(MboSide.X)`` are the single-char MBO codes;
  * a :class:`BookEvent` structurally satisfies ``book.MboRecord`` and folds
    through ``book.apply_event`` identically to the equivalent raw ``MBOMsg``;
  * :func:`merge_streams` -- stable k-way merge, contract guard, sibling
    cleanup, laziness;
  * a committed ~7 KB real ``.dbn.zst`` slice (``tests/fixtures/``) exercised
    through the real ``DBNStore.from_file`` path in the *normal* unit suite;
  * ``@pytest.mark.integration`` tests over the full (uncommitted) GLBX capture.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import typing
from dataclasses import dataclass
from pathlib import Path

import pytest

from databento import MBOMsg, Action, Side, UNDEF_PRICE

from src.ticksim import events as events_module
from src.ticksim.book import Book, BookInconsistency, BookSide, apply_event
from src.ticksim.events import (
    BookEvent,
    BookEventSource,
    DbnMboSource,
    MboAction,
    MboSide,
    merge_streams,
)

IID = 1
FRONT_MONTH_INSTRUMENT_ID = 42004800
_TICK_DBN = 250_000_000

TINY_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mnq_mbo_tiny.dbn.zst"
FULL_CAPTURE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "tick"
    / "_test"
    / "glbx-mdp3-20260622.mbo.dbn.zst"
)


# --- builders ----------------------------------------------------------------


def _ev(
    *,
    action: MboAction = MboAction.ADD,
    side: MboSide = MboSide.BID,
    order_id: int = 1,
    price_dbn: int = 100,
    size: int = 1,
    ts_event: int = 0,
    sequence: int = 0,
    instrument_id: int = IID,
) -> BookEvent:
    return BookEvent(
        action=action,
        side=side,
        order_id=order_id,
        price_dbn=price_dbn,
        size=size,
        ts_event=ts_event,
        sequence=sequence,
        instrument_id=instrument_id,
    )


def _mbo(
    *,
    action: Action = Action.ADD,
    side: Side = Side.BID,
    order_id: int = 1,
    price: int = 100,
    size: int = 1,
    ts_event: int = 0,
    sequence: int = 0,
    instrument_id: int = IID,
) -> MBOMsg:
    return MBOMsg(
        publisher_id=1,
        instrument_id=instrument_id,
        ts_event=ts_event,
        order_id=order_id,
        price=price,
        size=size,
        action=action,
        side=side,
        ts_recv=ts_event + 1,
        sequence=sequence,
    )


@dataclass
class _StubSource:
    """A tiny in-memory :class:`BookEventSource` (spine AD-18 Protocol)."""

    events: list[BookEvent]
    class_rank: int = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.events)


# --- vendor confinement (the module's whole purpose) ------------------------


class TestVendorConfinement:
    _VENDOR_ROOTS = {"databento", "databento_dbn"}

    def _root_module(self, obj: object) -> str:
        mod = getattr(obj, "__module__", "") or getattr(type(obj), "__module__", "")
        return (mod or "").split(".")[0]

    def test_all_exports_are_not_vendor_types(self) -> None:
        for name in events_module.__all__:
            obj = getattr(events_module, name)
            assert self._root_module(obj) not in self._VENDOR_ROOTS, name

    def test_bookevent_annotations_are_vendor_free(self) -> None:
        hints = typing.get_type_hints(BookEvent)
        assert set(hints) == {
            "action",
            "side",
            "order_id",
            "price_dbn",
            "size",
            "ts_event",
            "sequence",
            "instrument_id",
        }
        for name, tp in hints.items():
            assert "databento" not in repr(tp), (name, tp)
            assert self._root_module(tp) not in self._VENDOR_ROOTS, (name, tp)

    def test_public_signatures_name_no_vendor_type(self) -> None:
        targets = [
            merge_streams,
            DbnMboSource.__init__,
            DbnMboSource.__iter__,
            DbnMboSource._from_iterable,
            BookEventSource.__iter__,
        ]
        for target in targets:
            rendered = str(inspect.signature(target)).lower()
            assert "databento" not in rendered, (target, rendered)
            assert "mbomsg" not in rendered, (target, rendered)

    def test_dbnmbosource_public_attrs_are_plain(self) -> None:
        src = DbnMboSource._from_iterable([])
        assert isinstance(src.class_rank, int)
        assert isinstance(src.skipped_record_count, int)


# --- StrEnum -> single-char code -------------------------------------------


@pytest.mark.parametrize(
    "member, code",
    [
        (MboAction.ADD, "A"),
        (MboAction.CANCEL, "C"),
        (MboAction.MODIFY, "M"),
        (MboAction.TRADE, "T"),
        (MboAction.FILL, "F"),
        (MboAction.CLEAR, "R"),
        (MboAction.NONE, "N"),
    ],
)
def test_action_str_is_mbo_code(member: MboAction, code: str) -> None:
    assert str(member) == code
    assert MboAction(code) is member


@pytest.mark.parametrize(
    "member, code",
    [(MboSide.BID, "B"), (MboSide.ASK, "A"), (MboSide.NONE, "N")],
)
def test_side_str_is_mbo_code(member: MboSide, code: str) -> None:
    assert str(member) == code
    assert MboSide(code) is member


# --- normalize (via the public DbnMboSource.__iter__ path) -----------------


def _one(*records: object) -> list[BookEvent]:
    return list(DbnMboSource._from_iterable(list(records)))


class TestNormalize:
    def test_add_bid(self) -> None:
        (ev,) = _one(
            _mbo(
                action=Action.ADD,
                side=Side.BID,
                order_id=7,
                price=250_000,
                size=3,
                ts_event=1_000,
                sequence=5,
                instrument_id=42,
            )
        )
        assert ev == BookEvent(
            action=MboAction.ADD,
            side=MboSide.BID,
            order_id=7,
            price_dbn=250_000,
            size=3,
            ts_event=1_000,
            sequence=5,
            instrument_id=42,
        )

    def test_every_action_and_side_code_maps(self) -> None:
        for act in Action.variants():
            for sd in Side.variants():
                (ev,) = _one(_mbo(action=act, side=sd, price=100))
                assert str(ev.action) == str(act)
                assert str(ev.side) == str(sd)

    def test_r_t_f_n_pass_through(self) -> None:
        out = _one(
            _mbo(action=Action.TRADE, side=Side.ASK, ts_event=1, sequence=1),
            _mbo(action=Action.FILL, side=Side.BID, ts_event=2, sequence=2),
            _mbo(action=Action.CLEAR, side=Side.NONE, ts_event=3, sequence=3),
            _mbo(action=Action.NONE, side=Side.NONE, ts_event=4, sequence=4),
        )
        assert [str(e.action) for e in out] == ["T", "F", "R", "N"]

    def test_undef_price_on_add_raises_through_public_path(self) -> None:
        doctored = _mbo(action=Action.ADD, side=Side.BID, order_id=9)
        doctored.price = UNDEF_PRICE
        with pytest.raises(ValueError, match="undefined price"):
            list(DbnMboSource._from_iterable([doctored]))

    def test_undef_price_on_modify_raises(self) -> None:
        doctored = _mbo(action=Action.MODIFY, side=Side.BID, order_id=9)
        doctored.price = UNDEF_PRICE
        with pytest.raises(ValueError, match="undefined price"):
            list(DbnMboSource._from_iterable([doctored]))

    def test_undef_price_on_cancel_is_allowed(self) -> None:
        # book.py handles an undefined price on non-A/M; events.py must not
        # reject it (it is not an A / M).
        rec = _mbo(action=Action.CANCEL, side=Side.BID, order_id=9)
        rec.price = UNDEF_PRICE
        (ev,) = _one(rec)
        assert ev.action is MboAction.CANCEL

    def test_unknown_action_code_raises(self) -> None:
        # The vendor Action enum is closed, so an unknown code can only come
        # from a non-vendor record type -- exercise _normalize directly.
        class _Doctored:
            action = "Z"
            side = "B"
            order_id = price = size = ts_event = sequence = instrument_id = 1

        with pytest.raises(ValueError, match="unknown MBO action code 'Z'"):
            events_module._normalize(_Doctored())  # type: ignore[arg-type]

    def test_unknown_side_code_raises(self) -> None:
        class _Doctored:
            action = "A"
            side = "Q"
            order_id = price = size = ts_event = sequence = instrument_id = 1

        with pytest.raises(ValueError, match="unknown MBO side code 'Q'"):
            events_module._normalize(_Doctored())  # type: ignore[arg-type]


# --- BookEvent satisfies book.MboRecord -----------------------------------


class TestBookEventIsMboRecord:
    def test_has_every_mbo_record_attribute(self) -> None:
        ev = _ev(price_dbn=250_000)
        for attr in (
            "action",
            "side",
            "order_id",
            "price_dbn",
            "size",
            "ts_event",
            "sequence",
            "instrument_id",
        ):
            assert hasattr(ev, attr), attr
        assert not hasattr(ev, "price"), "the bridging property should be gone"
        assert isinstance(ev.action, str) and isinstance(ev.side, str)

    def test_frozen_and_slotted(self) -> None:
        ev = _ev()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.size = 9  # type: ignore[misc]
        assert not hasattr(ev, "__dict__"), "slots=True keeps allocations lean"

    def test_add_bookevent_folds_like_raw_mbomsg(self) -> None:
        """Acceptance: a hand-built BookEvent mutates the book identically."""
        evs = [
            _ev(
                action=MboAction.ADD,
                side=MboSide.BID,
                order_id=1,
                price_dbn=100,
                size=4,
                ts_event=10,
                sequence=1,
            ),
            _ev(
                action=MboAction.ADD,
                side=MboSide.ASK,
                order_id=2,
                price_dbn=102,
                size=2,
                ts_event=11,
                sequence=2,
            ),
            _ev(
                action=MboAction.CANCEL,
                side=MboSide.BID,
                order_id=1,
                price_dbn=100,
                size=4,
                ts_event=12,
                sequence=3,
            ),
        ]
        mbos = [
            _mbo(
                action=Action.ADD,
                side=Side.BID,
                order_id=1,
                price=100,
                size=4,
                ts_event=10,
                sequence=1,
            ),
            _mbo(
                action=Action.ADD,
                side=Side.ASK,
                order_id=2,
                price=102,
                size=2,
                ts_event=11,
                sequence=2,
            ),
            _mbo(
                action=Action.CANCEL,
                side=Side.BID,
                order_id=1,
                price=100,
                size=4,
                ts_event=12,
                sequence=3,
            ),
        ]

        book_ev, book_mbo = Book(), Book()
        for r in evs:
            apply_event(book_ev, r)
        for r in DbnMboSource._from_iterable(list(mbos)):
            apply_event(book_mbo, r)

        assert book_ev.snapshot_bbo(IID) == book_mbo.snapshot_bbo(IID) == (None, 102)
        assert book_ev.size_at_price(IID, BookSide.ASK, 102) == 2
        assert book_ev.order_by_id(IID, 1) is None
        assert book_ev.last_ts_ns == book_mbo.last_ts_ns == 12
        book_ev.check_invariants()

    def test_modify_bookevent_matches_raw(self) -> None:
        book_ev, book_mbo = Book(), Book()
        for r in (
            _ev(
                action=MboAction.ADD,
                side=MboSide.BID,
                order_id=1,
                price_dbn=100,
                size=5,
                ts_event=1,
                sequence=1,
            ),
            _ev(
                action=MboAction.MODIFY,
                side=MboSide.BID,
                order_id=1,
                price_dbn=100,
                size=2,
                ts_event=2,
                sequence=2,
            ),
        ):
            apply_event(book_ev, r)
        for r in DbnMboSource._from_iterable(
            [
                _mbo(
                    action=Action.ADD,
                    side=Side.BID,
                    order_id=1,
                    price=100,
                    size=5,
                    ts_event=1,
                    sequence=1,
                ),
                _mbo(
                    action=Action.MODIFY,
                    side=Side.BID,
                    order_id=1,
                    price=100,
                    size=2,
                    ts_event=2,
                    sequence=2,
                ),
            ]
        ):
            apply_event(book_mbo, r)
        assert book_ev.size_at_price(IID, BookSide.BID, 100) == 2
        assert book_ev.size_at_price(IID, BookSide.BID, 100) == book_mbo.size_at_price(
            IID, BookSide.BID, 100
        )


# --- merge_streams -------------------------------------------------------


class TestMergeStreams:
    def test_no_sources_yields_nothing(self) -> None:
        assert list(merge_streams()) == []

    def test_rejects_non_source_argument(self) -> None:
        with pytest.raises(TypeError, match="argument 1 is not a BookEventSource"):
            list(merge_streams(_StubSource([]), [1, 2, 3]))  # type: ignore[list-item]

    def test_rejects_the_same_source_object_twice(self) -> None:
        src = _StubSource([_ev(ts_event=1)])
        with pytest.raises(ValueError, match="passed more than once"):
            list(merge_streams(src, src))

    def test_single_source_preserves_file_order(self) -> None:
        evs = [
            _ev(ts_event=t, sequence=s) for t, s in [(5, 1), (5, 2), (7, 3), (20, 4)]
        ]
        assert list(merge_streams(_StubSource(list(evs)))) == evs

    def test_empty_source_mixed_in(self) -> None:
        a = _StubSource([_ev(ts_event=10, sequence=1)])
        assert list(merge_streams(_StubSource([]), a, _StubSource([]))) == a.events

    def test_two_sources_distinct_ts(self) -> None:
        a = _StubSource([_ev(order_id=1, ts_event=10), _ev(order_id=3, ts_event=30)])
        b = _StubSource([_ev(order_id=2, ts_event=20)])
        assert [e.order_id for e in merge_streams(a, b)] == [1, 2, 3]

    def test_tie_on_ts_lower_sequence_first(self) -> None:
        a = _StubSource([_ev(order_id=100, ts_event=100, sequence=5)])
        b = _StubSource([_ev(order_id=200, ts_event=100, sequence=3)])
        assert [e.order_id for e in merge_streams(a, b)] == [200, 100]

    def test_tie_on_ts_and_sequence_is_argument_order(self) -> None:
        a = _StubSource([_ev(order_id=100, ts_event=100, sequence=5)])
        b = _StubSource([_ev(order_id=200, ts_event=100, sequence=5)])
        assert [e.order_id for e in merge_streams(a, b)] == [100, 200]
        assert [e.order_id for e in merge_streams(b, a)] == [200, 100]

    def test_class_rank_orders_before_sequence_tiebreak(self) -> None:
        delta = _StubSource([_ev(order_id=1, ts_event=100, sequence=9)], class_rank=0)
        arrival = _StubSource([_ev(order_id=2, ts_event=100, sequence=1)], class_rank=1)
        assert [e.order_id for e in merge_streams(arrival, delta)] == [1, 2]

    def test_three_class_ranks_interleaved(self) -> None:
        r0 = _StubSource(
            [
                _ev(order_id=i, ts_event=t, sequence=1)
                for i, t in [(10, 1), (11, 2), (12, 3)]
            ],
            class_rank=0,
        )
        r1 = _StubSource(
            [_ev(order_id=i, ts_event=t, sequence=1) for i, t in [(20, 1), (21, 2)]],
            class_rank=1,
        )
        r2 = _StubSource(
            [_ev(order_id=i, ts_event=t, sequence=1) for i, t in [(30, 1), (31, 3)]],
            class_rank=2,
        )
        merged = list(merge_streams(r0, r1, r2))
        # sort key is (ts_event, class_rank, sequence)
        keys = [(e.ts_event, e.order_id // 10 - 1) for e in merged]
        assert keys == sorted(keys)
        assert [e.order_id for e in merged if e.ts_event == 1] == [10, 20, 30]

    def test_globally_ordered_over_many(self) -> None:
        a = _StubSource(
            [
                _ev(order_id=i, ts_event=t, sequence=t)
                for i, t in enumerate([1, 4, 4, 9, 100])
            ]
        )
        b = _StubSource(
            [
                _ev(order_id=100 + i, ts_event=t, sequence=t)
                for i, t in enumerate([2, 4, 5, 50])
            ]
        )
        merged = list(merge_streams(a, b))
        keys = [(e.ts_event, e.sequence) for e in merged]
        assert keys == sorted(keys)
        assert len(merged) == len(a.events) + len(b.events)

    def test_partial_consume_then_resume(self) -> None:
        a = _StubSource([_ev(order_id=i, ts_event=i * 10) for i in range(1, 6)])
        b = _StubSource(
            [_ev(order_id=50 + i, ts_event=i * 10 + 5) for i in range(1, 6)]
        )
        expected = [e.order_id for e in merge_streams(a, b)]
        m = merge_streams(a, b)
        head = [next(m).order_id for _ in range(3)]
        tail = [e.order_id for e in m]
        assert head + tail == expected

    def test_source_contract_guard_ts_backwards(self) -> None:
        bad = _StubSource([_ev(ts_event=100, sequence=1), _ev(ts_event=90, sequence=2)])
        with pytest.raises(ValueError, match=r"\(ts_event, sequence\) went backwards"):
            list(merge_streams(bad))

    def test_source_contract_guard_sequence_regresses_within_tick(self) -> None:
        bad = _StubSource(
            [_ev(ts_event=100, sequence=5), _ev(ts_event=100, sequence=2)]
        )
        with pytest.raises(
            ValueError, match=r"went backwards \(100, 5\) -> \(100, 2\)"
        ):
            list(merge_streams(bad))

    def test_sibling_iterators_closed_on_failure(self) -> None:
        closed: list[str] = []

        def gen(tag: str, evs: list[BookEvent]):  # type: ignore[no-untyped-def]
            try:
                for e in evs:
                    yield e
            finally:
                closed.append(tag)

        @dataclass
        class _GenSource:
            tag: str
            evs: list[BookEvent]
            class_rank: int = 0

            def __iter__(self):  # type: ignore[no-untyped-def]
                return gen(self.tag, self.evs)

        good = _GenSource("good", [_ev(ts_event=t) for t in (1, 5, 9, 13, 17)])
        bad = _GenSource("bad", [_ev(ts_event=10), _ev(ts_event=2)])  # regresses
        with pytest.raises(ValueError):
            list(merge_streams(good, bad))
        assert "good" in closed, "the healthy sibling iterator was not closed"

    def test_lazy_pull_one_at_a_time(self) -> None:
        pulled: list[int] = []

        def gen(tag: int, n: int):  # type: ignore[no-untyped-def]
            for i in range(n):
                pulled.append(tag)
                yield _ev(order_id=tag, ts_event=i, sequence=i)

        @dataclass
        class _GenSource:
            it: object
            class_rank: int = 0

            def __iter__(self):  # type: ignore[no-untyped-def]
                return iter(self.it)  # type: ignore[call-overload]

        m = merge_streams(_GenSource(gen(1, 6)), _GenSource(gen(2, 6)))
        next(m)
        assert pulled.count(1) < 6 and pulled.count(2) < 6
        assert sum(pulled.count(t) for t in (1, 2)) <= 3  # 2 initial + 1 refill
        assert len(list(m)) == 11


# --- DbnMboSource (in-memory seam) -------------------------------------------


class TestDbnMboSourceInMemory:
    def test_class_rank_is_zero(self) -> None:
        assert DbnMboSource.class_rank == 0
        assert DbnMboSource._from_iterable([]).class_rank == 0

    def test_is_a_book_event_source(self) -> None:
        assert isinstance(DbnMboSource._from_iterable([]), BookEventSource)

    def test_yields_every_mbo_record_in_order(self) -> None:
        out = list(
            DbnMboSource._from_iterable(
                [
                    _mbo(
                        action=Action.ADD,
                        side=Side.BID,
                        order_id=1,
                        ts_event=10,
                        sequence=1,
                    ),
                    _mbo(
                        action=Action.TRADE,
                        side=Side.ASK,
                        order_id=0,
                        ts_event=11,
                        sequence=2,
                    ),
                    _mbo(
                        action=Action.CANCEL,
                        side=Side.BID,
                        order_id=1,
                        ts_event=12,
                        sequence=3,
                    ),
                ]
            )
        )
        assert [str(e.action) for e in out] == ["A", "T", "C"]
        assert [e.ts_event for e in out] == [10, 11, 12]

    def test_non_mbo_records_skipped_and_counted(self) -> None:
        src = DbnMboSource._from_iterable(
            [object(), _mbo(action=Action.ADD, side=Side.BID, order_id=1), "junk"]
        )
        out = list(src)
        assert len(out) == 1 and out[0].order_id == 1
        assert src.skipped_record_count == 2

    def test_empty_source_yields_nothing(self) -> None:
        src = DbnMboSource._from_iterable([])
        assert list(src) == []
        assert src.skipped_record_count == 0

    def test_from_iterable_with_list_is_reiterable(self) -> None:
        recs = [
            _mbo(action=Action.ADD, side=Side.BID, order_id=1, ts_event=1, sequence=1),
            _mbo(action=Action.ADD, side=Side.ASK, order_id=2, ts_event=2, sequence=2),
        ]
        src = DbnMboSource._from_iterable(recs)
        assert list(src) == list(src)

    def test_iter_pulls_the_source_lazily(self) -> None:
        pulled = 0

        def gen():  # type: ignore[no-untyped-def]
            nonlocal pulled
            for i in range(8):
                pulled += 1
                yield _mbo(
                    action=Action.ADD, side=Side.BID, order_id=i, ts_event=i, sequence=i
                )

        it = iter(DbnMboSource._from_iterable(gen()))
        assert pulled == 0, "iter() alone must not consume the source"
        first = next(it)
        assert first.order_id == 0
        assert pulled == 1, "one next() pulled more than one raw record"


# --- DbnMboSource (real DBNStore path, committed tiny fixture) ---------------


class TestDbnMboSourceFile:
    def test_fixture_is_committed(self) -> None:
        assert TINY_FIXTURE.is_file(), (
            f"committed fixture missing: {TINY_FIXTURE} -- regenerate with "
            f"tests/fixtures/generate_mnq_mbo_tiny.py"
        )

    def test_missing_file_raises_at_construction(self) -> None:
        with pytest.raises(FileNotFoundError):
            DbnMboSource("/definitely/not/here.dbn.zst")

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DbnMboSource(tmp_path)

    def test_non_dbn_file_raises_clear_error(self, tmp_path: Path) -> None:
        junk = tmp_path / "not.dbn.zst"
        junk.write_bytes(b"this is not a DBN stream")
        with pytest.raises(ValueError, match="not a readable DBN file"):
            list(DbnMboSource(junk))

    def test_streams_the_fixture(self) -> None:
        src = DbnMboSource(TINY_FIXTURE)
        events = list(src)
        assert len(events) == 500
        assert src.skipped_record_count == 0
        assert all(isinstance(e, BookEvent) for e in events)
        assert all(isinstance(e.action, MboAction) for e in events)
        assert all(e.instrument_id == FRONT_MONTH_INSTRUMENT_ID for e in events)
        # file order == non-decreasing (ts_event, sequence)
        keys = [(e.ts_event, e.sequence) for e in events]
        assert keys == sorted(keys)

    def test_source_is_reiterable_fresh_handle_each_time(self) -> None:
        src = DbnMboSource(TINY_FIXTURE)
        first = list(src)
        second = list(src)
        assert first == second and len(first) == 500

    def test_iteration_is_lazy(self) -> None:
        src = DbnMboSource(TINY_FIXTURE)
        it1 = iter(src)
        head1 = [next(it1) for _ in range(3)]
        # a second, independent iterator starts over from record 0
        it2 = iter(src)
        head2 = [next(it2) for _ in range(3)]
        assert head1 == head2
        # the first iterator resumes where it left off, not restarted
        assert next(it1).ts_event >= head1[-1].ts_event

    def test_fold_fixture_through_book(self) -> None:
        """The real DBNStore.from_file + iterate + fold path, in the unit suite."""
        book = Book()
        last_key = (-1, -1)
        folded = 0
        for ev in DbnMboSource(TINY_FIXTURE):
            key = (ev.ts_event, ev.sequence)
            assert key >= last_key
            last_key = key
            apply_event(book, ev)  # raises BookInconsistency on any corruption
            folded += 1
        assert folded == 500
        book.check_invariants()
        best_bid, best_ask = book.snapshot_bbo(FRONT_MONTH_INSTRUMENT_ID)
        assert best_bid is not None and best_ask is not None
        assert best_bid < best_ask

    def test_merge_streams_over_the_fixture(self) -> None:
        merged = list(merge_streams(DbnMboSource(TINY_FIXTURE)))
        assert len(merged) == 500
        keys = [(e.ts_event, e.sequence) for e in merged]
        assert keys == sorted(keys)


# --- integration: the full (uncommitted) GLBX capture ----------------------

_WINDOW = 200_000


def _capture_or_skip() -> Path:
    if FULL_CAPTURE.is_file():
        return FULL_CAPTURE
    if os.environ.get("TICKSIM_REQUIRE_FIXTURE"):
        pytest.fail(f"TICKSIM_REQUIRE_FIXTURE set but capture missing: {FULL_CAPTURE}")
    pytest.skip(f"full GLBX MDP3 capture not present: {FULL_CAPTURE}")


@pytest.mark.integration
def test_stream_full_capture_through_book() -> None:
    """Fold ~200k front-month records via DbnMboSource -> book.apply_event."""
    _capture_or_skip()
    source = DbnMboSource(FULL_CAPTURE)
    book = Book()
    last_key = (-1, -1)
    folded = 0
    seen_front = False

    for ev in source:
        assert isinstance(ev, BookEvent)
        assert isinstance(ev.action, MboAction) and isinstance(ev.side, MboSide)
        if ev.instrument_id != FRONT_MONTH_INSTRUMENT_ID:
            continue
        seen_front = True
        key = (ev.ts_event, ev.sequence)
        assert key >= last_key, "BookEvent stream (ts_event, sequence) went backwards"
        last_key = key
        try:
            apply_event(book, ev)
        except BookInconsistency as exc:  # pragma: no cover
            pytest.fail(f"BookInconsistency after {folded} records: {exc}")
        folded += 1
        if folded >= _WINDOW:
            break

    assert seen_front, (
        f"no records for instrument_id {FRONT_MONTH_INSTRUMENT_ID} "
        f"-- front-month instrument_id changed (capture regenerated?)"
    )
    assert folded == _WINDOW, "capture prefix shorter than expected"

    book.check_invariants()
    best_bid, best_ask = book.snapshot_bbo(FRONT_MONTH_INSTRUMENT_ID)
    assert best_bid is not None and best_ask is not None
    assert best_bid < best_ask
    assert (best_ask - best_bid) / _TICK_DBN <= 4


@pytest.mark.integration
def test_merge_streams_over_full_capture_is_monotonic() -> None:
    _capture_or_skip()
    merged = merge_streams(DbnMboSource(FULL_CAPTURE))
    prev = (-1, -1)
    n = 0
    for ev in merged:
        key = (ev.ts_event, ev.sequence)
        assert key >= prev, f"merge output regressed: {prev} -> {key}"
        prev = key
        n += 1
        if n >= 50_000:
            break
    assert n == 50_000
