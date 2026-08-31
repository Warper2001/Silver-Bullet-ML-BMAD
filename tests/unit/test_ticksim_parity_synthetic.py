"""Unit tests for ``src/ticksim/parity/synthetic.py`` (prereg §A8.2 Part B).

Every case builds a hand-rolled in-memory :class:`~src.ticksim.events.BookEventSource`
(a deep static two-sided book, or a deliberately partial / crossed / moving /
empty / multi-instrument one) and runs the real generator.  The ``run_part_b``
round-trip test is the one place ``part_b`` is imported -- a test may, the
generator may not.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import src.ticksim.parity.synthetic as synthetic_mod
from src.ticksim.config import MNQ_TICK_DBN, PART_B_MIN_ORDERS
from src.ticksim.events import BookEvent, MboAction, MboSide
from src.ticksim.orders import IntentAction, Leg, OrderIntent, OrderKind, Side
from src.ticksim.parity import part_b
from src.ticksim.parity._bookwalk import BookReplay, BookWalkError
from src.ticksim.parity.part_b import run_part_b
from src.ticksim.parity.synthetic import (
    SyntheticError,
    _Candidate,
    _evenly_spaced_indices,
    _price_limit,
    generate_synthetic_orders,
)

IID = 42004800
TICK = MNQ_TICK_DBN
P = 20_000_000_000_000
BID_PX = P - TICK
ASK_PX = P + TICK
B = 1_700_000_000 * 1_000_000_000
WINDOW = 10 * 1_000_000_000  # 10 s


class ListSource:
    """Re-iterable in-memory :class:`BookEventSource` (spine AD-18)."""

    class_rank = 0

    def __init__(self, events: list[BookEvent]) -> None:
        self._events = list(events)

    def __iter__(self) -> Iterator[BookEvent]:
        return iter(self._events)


def be(
    action: MboAction,
    side: MboSide,
    order_id: int,
    price_dbn: int,
    size: int,
    ts: int,
    seq: int,
    *,
    instrument_id: int = IID,
) -> BookEvent:
    return BookEvent(
        action=action,
        side=side,
        order_id=order_id,
        price_dbn=price_dbn,
        size=size,
        ts_event=ts,
        sequence=seq,
        instrument_id=instrument_id,
    )


def deep_book(ts: int = B - 1) -> list[BookEvent]:
    """A static two-sided book -- a resting bid and ask, huge size, well before
    the candidate window so the first ``advance_to`` folds both."""
    return [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 1_000_000, ts=ts, seq=1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 1_000_000, ts=ts, seq=2),
    ]


def bid_only_book(ts: int = B - 1) -> list[BookEvent]:
    return [be(MboAction.ADD, MboSide.BID, 1, BID_PX, 1_000_000, ts=ts, seq=1)]


def partial_ask_book(gap_frac: float = 0.4) -> list[BookEvent]:
    """A bid from the start; the ask only appears ``gap_frac`` into the window,
    so the early ask-needing limit candidates (a warm-up gap) get dropped."""
    ask_ts = B + int(WINDOW * gap_frac)
    return [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 1_000_000, ts=B - 1, seq=1),
        be(MboAction.ADD, MboSide.ASK, 2, ASK_PX, 1_000_000, ts=ask_ts, seq=2),
    ]


# --------------------------------------------------------------------------- #
# happy path -- a full 1000-order run
# --------------------------------------------------------------------------- #


def test_thousand_order_run_shape() -> None:
    orders = generate_synthetic_orders(ListSource(deep_book()), B, B + WINDOW)

    assert len(orders) == PART_B_MIN_ORDERS == 1000
    assert all(o.action is IntentAction.SUBMIT for o in orders)
    assert all(o.leg is Leg.ENTRY for o in orders)
    assert all(o.oco_group_id is None and o.replaces_order_id is None for o in orders)
    assert all(o.trade_id == o.order_id for o in orders)
    assert all(B <= o.submit_ts_ns < B + WINDOW for o in orders)

    stamps = [o.submit_ts_ns for o in orders]
    assert stamps == sorted(stamps)  # submit-ts non-decreasing

    assert {o.side for o in orders} == {Side.BUY, Side.SELL}
    assert {o.size for o in orders} == {1, 2, 3, 4, 5}
    assert {o.kind for o in orders} == set(OrderKind)

    for o in orders:
        if o.kind is OrderKind.MARKETABLE:
            assert o.limit_px_dbn is None
        else:
            assert o.limit_px_dbn is not None and o.limit_px_dbn > 0

    ids = [o.order_id for o in orders]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "synthetic-0000" and ids[-1] == "synthetic-0999"

    for o in orders:
        assert OrderIntent.model_validate_json(o.model_dump_json()) == o


def test_large_run_returns_exact_n() -> None:
    orders = generate_synthetic_orders(ListSource(deep_book()), B, B + WINDOW, n=3000)
    assert len(orders) == 3000
    assert {o.side for o in orders} == {Side.BUY, Side.SELL}
    assert {o.size for o in orders} == {1, 2, 3, 4, 5}
    assert {o.kind for o in orders} == set(OrderKind)
    assert [o.submit_ts_ns for o in orders] == sorted(o.submit_ts_ns for o in orders)


# --------------------------------------------------------------------------- #
# determinism (spine AD-11)
# --------------------------------------------------------------------------- #


def test_byte_identical_across_two_calls() -> None:
    a = generate_synthetic_orders(ListSource(deep_book()), B, B + WINDOW, n=200)
    b = generate_synthetic_orders(ListSource(deep_book()), B, B + WINDOW, n=200)
    assert [o.model_dump_json() for o in a] == [o.model_dump_json() for o in b]


def test_different_seed_diverges_on_most_rows() -> None:
    a = generate_synthetic_orders(ListSource(deep_book()), B, B + WINDOW, n=300, seed=0)
    b = generate_synthetic_orders(ListSource(deep_book()), B, B + WINDOW, n=300, seed=1)
    differ = sum(
        1
        for x, y in zip(a, b, strict=True)
        if x.model_dump_json() != y.model_dump_json()
    )
    assert differ / len(a) > 0.9


def test_golden_snapshot_seed0_n6() -> None:
    """Pins the exact rng draw sequence for ``seed=0`` -- a conditional-offset
    refactor (which would change the stream) fails here."""
    orders = generate_synthetic_orders(ListSource(deep_book()), B, B + WINDOW, n=6)
    got = [
        (
            o.order_id,
            o.kind.value,
            o.side.value,
            o.size,
            o.limit_px_dbn,
            o.submit_ts_ns,
        )
        for o in orders
    ]
    assert got == _GOLDEN_SEED0_N6


# captured from a first run, then frozen (see test docstring).
_GOLDEN_SEED0_N6: list[tuple[str, str, str, int, int | None, int]] = [
    ("synthetic-0", "passive_limit", "sell", 1, 20001250000000, 1700000000060308648),
    ("synthetic-1", "passive_limit", "sell", 5, 20001500000000, 1700000003246154361),
    ("synthetic-2", "marketable_limit", "buy", 3, 20001250000000, 1700000005713153566),
    ("synthetic-3", "marketable", "sell", 4, None, 1700000006818763383),
    ("synthetic-4", "passive_limit", "buy", 1, 19999500000000, 1700000008021292842),
    ("synthetic-5", "passive_limit", "sell", 5, 20000500000000, 1700000009125992521),
]


# --------------------------------------------------------------------------- #
# limit-price formulas (priced directly through _price_limit)
# --------------------------------------------------------------------------- #


def _replay_at_deep_book() -> BookReplay:
    replay = BookReplay(ListSource(deep_book()))
    replay.advance_to(B)
    assert replay.book.snapshot_bbo(IID) == (BID_PX, ASK_PX)
    return replay


@pytest.mark.parametrize(
    "kind, side, offset, expected",
    [
        (OrderKind.MARKETABLE_LIMIT, Side.BUY, 3, ASK_PX + 3 * TICK),
        (OrderKind.MARKETABLE_LIMIT, Side.SELL, 2, BID_PX - 2 * TICK),
        (OrderKind.PASSIVE_LIMIT, Side.BUY, 1, BID_PX - 1 * TICK),
        (OrderKind.PASSIVE_LIMIT, Side.SELL, 4, ASK_PX + 4 * TICK),
        (OrderKind.MARKETABLE, Side.BUY, 5, None),
    ],
)
def test_price_limit_formulas(
    kind: OrderKind, side: Side, offset: int, expected: int | None
) -> None:
    cand = _Candidate(ts_ns=B, side=side, kind=kind, size=1, offset_ticks=offset)
    assert _price_limit(cand, _replay_at_deep_book()) == expected


def test_price_limit_drops_when_touch_side_absent() -> None:
    replay = BookReplay(ListSource(bid_only_book()))
    replay.advance_to(B)
    buy_ml = _Candidate(B, Side.BUY, OrderKind.MARKETABLE_LIMIT, 1, 2)
    assert _price_limit(buy_ml, replay) is None
    sell_pl = _Candidate(B, Side.SELL, OrderKind.PASSIVE_LIMIT, 1, 2)
    assert _price_limit(sell_pl, replay) is None
    sell_ml = _Candidate(B, Side.SELL, OrderKind.MARKETABLE_LIMIT, 1, 2)
    assert _price_limit(sell_ml, replay) == BID_PX - 2 * TICK


def test_price_limit_empty_book_returns_none() -> None:
    replay = BookReplay(ListSource([]))
    replay.advance_to(B)
    assert replay.instrument_id is None
    cand = _Candidate(B, Side.BUY, OrderKind.PASSIVE_LIMIT, 1, 1)
    assert _price_limit(cand, replay) is None


def test_price_limit_drops_on_crossed_book() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, ASK_PX, 10, ts=B - 1, seq=1),  # bid above
        be(MboAction.ADD, MboSide.ASK, 2, BID_PX, 10, ts=B - 1, seq=2),  # ask below
    ]
    replay = BookReplay(ListSource(events))
    replay.advance_to(B)
    bid, ask = replay.book.snapshot_bbo(IID)
    assert bid is not None and ask is not None and bid >= ask
    for kind in (OrderKind.MARKETABLE_LIMIT, OrderKind.PASSIVE_LIMIT):
        cand = _Candidate(B, Side.BUY, kind, 1, 0)
        assert _price_limit(cand, replay) is None


def test_price_limit_drops_on_nonpositive_price() -> None:
    # a tiny bid and a large offset drives a passive/marketable-limit SELL... no,
    # SELL adds; a BUY subtracts -> below zero.
    events = [be(MboAction.ADD, MboSide.BID, 1, TICK, 10, ts=B - 1, seq=1)]
    replay = BookReplay(ListSource(events))
    replay.advance_to(B)
    cand = _Candidate(B, Side.BUY, OrderKind.PASSIVE_LIMIT, 1, 5)  # TICK - 5*TICK < 0
    assert _price_limit(cand, replay) is None


def test_price_limit_unhandled_kind_raises() -> None:
    cand = _Candidate(B, Side.BUY, OrderKind.MARKETABLE, 1, 0)
    # force an unhandled kind past the marketable short-circuit
    bad = cand._replace(kind="not_a_kind")  # type: ignore[arg-type]
    with pytest.raises(SyntheticError, match="unhandled kind"):
        _price_limit(bad, _replay_at_deep_book())


def test_moving_book_prices_each_candidate_at_its_own_bbo() -> None:
    """A MODIFY shifts the ask mid-window; a candidate before and one after must
    price against the two different BBO regimes (pins the forward fold +
    ``_pending`` carry across ``advance_to``)."""
    shift_ts = B + WINDOW // 2
    events = [
        *deep_book(ts=B - 1),
        be(
            MboAction.MODIFY,
            MboSide.ASK,
            2,
            ASK_PX + 10 * TICK,
            1_000_000,
            ts=shift_ts,
            seq=3,
        ),
    ]
    replay = BookReplay(ListSource(events))
    before = _Candidate(B + 1_000, Side.BUY, OrderKind.MARKETABLE_LIMIT, 1, 2)
    replay.advance_to(before.ts_ns)
    px_before = _price_limit(before, replay)
    after = _Candidate(shift_ts + 1_000, Side.BUY, OrderKind.MARKETABLE_LIMIT, 1, 2)
    replay.advance_to(after.ts_ns)
    px_after = _price_limit(after, replay)
    assert px_before == ASK_PX + 2 * TICK
    assert px_after == ASK_PX + 10 * TICK + 2 * TICK


# --------------------------------------------------------------------------- #
# window span + subsampling
# --------------------------------------------------------------------------- #


def test_emitted_orders_span_the_whole_window() -> None:
    orders = generate_synthetic_orders(ListSource(deep_book()), B, B + WINDOW, n=200)
    stamps = [o.submit_ts_ns for o in orders]
    assert min(stamps) < B + WINDOW // 10  # first order in the first 10%
    assert max(stamps) > B + WINDOW - WINDOW // 10  # last order in the last 10%


def test_partial_two_sided_book_still_yields_n() -> None:
    orders = generate_synthetic_orders(
        ListSource(partial_ask_book(0.4)), B, B + WINDOW, n=300
    )
    assert len(orders) == 300
    ask_ts = B + int(WINDOW * 0.4)
    for o in orders:
        needs_ask = (o.kind is OrderKind.MARKETABLE_LIMIT and o.side is Side.BUY) or (
            o.kind is OrderKind.PASSIVE_LIMIT and o.side is Side.SELL
        )
        if needs_ask:
            assert o.submit_ts_ns >= ask_ts


@pytest.mark.parametrize(
    "length, n, expected",
    [
        (10, 1, [0]),
        (10, 10, list(range(10))),
        (10, 5, [0, 2, 4, 7, 9]),
        (7, 5, [0, 2, 3, 4, 6]),
    ],
)
def test_evenly_spaced_indices(length: int, n: int, expected: list[int]) -> None:
    assert _evenly_spaced_indices(length, n) == expected


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "lo, hi, n",
    [
        (-1, B + WINDOW, 10),  # lo < 0
        (B + WINDOW, B, 10),  # lo >= hi
        (B, B, 10),  # lo == hi
        (B, B + WINDOW, 0),  # n < 1
        (B, B + 5, 10),  # hi - lo < n
    ],
)
def test_bad_bounds_raise(lo: int, hi: int, n: int) -> None:
    with pytest.raises(SyntheticError):
        generate_synthetic_orders(ListSource(deep_book()), lo, hi, n=n)


def test_too_sparse_window_raises() -> None:
    # an empty source cannot price any limit candidate; ~2/3 of the
    # ceil(2.0 * 100) = 200 candidates are limit kinds -> < 100 resolve.
    with pytest.raises(SyntheticError, match="widen the window"):
        generate_synthetic_orders(ListSource([]), B, B + WINDOW, n=100)


def test_multi_instrument_source_raises_bookwalk_error() -> None:
    events = [
        be(MboAction.ADD, MboSide.BID, 1, BID_PX, 10, ts=B - 2, seq=1),
        be(
            MboAction.ADD,
            MboSide.ASK,
            2,
            ASK_PX,
            10,
            ts=B - 1,
            seq=2,
            instrument_id=IID + 1,
        ),
    ]
    with pytest.raises(BookWalkError):
        generate_synthetic_orders(ListSource(events), B, B + WINDOW, n=50)


# --------------------------------------------------------------------------- #
# isolation
# --------------------------------------------------------------------------- #


def test_module_has_no_vendor_import() -> None:
    """No ``databento`` / ``sim`` / ``part_b`` import (mirrors
    ``test_ticksim_events.TestVendorConfinement``).  The module *docstring*
    names ``databento`` as a thing it does not import -- so parse the AST and
    check real ``import`` statements, not the raw text."""
    import ast

    tree = ast.parse(Path(synthetic_mod.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("databento" in name for name in imported)
    assert not any(
        name.endswith(("sim", "part_b", "part_a", "report")) for name in imported
    )


# --------------------------------------------------------------------------- #
# end-to-end: the generator output feeds straight into run_part_b
# --------------------------------------------------------------------------- #


def test_output_feeds_run_part_b_to_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(part_b, "PART_B_MIN_ORDERS", 20)
    orders = generate_synthetic_orders(ListSource(deep_book()), B, B + WINDOW, n=25)

    result = run_part_b(orders, ListSource(deep_book()))

    assert result.verdict == "PASS", result.reason
    assert result.violations == ()
    assert result.n_orders == 25
