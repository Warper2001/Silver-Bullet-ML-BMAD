"""Hand-built native records for the read-only displayed-book inspection."""

import importlib.util
from pathlib import Path
import pytest

PATH = (
    Path(__file__).resolve().parents[3]
    / "docs/reports/yank-native-minute/inspect-may28-book.py"
)
spec = importlib.util.spec_from_file_location("book_inspection", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
BASE = 1748390400000000000


def record(action="A", oid=1, price=100, size=5, side="B", flags=128, recv=BASE + 1):
    return (
        14,
        160,
        1,
        42009475,
        BASE,
        oid,
        price,
        size,
        flags,
        8,
        ord(action),
        ord(side),
        recv,
        0,
        1,
    )


def ready():
    book = m.Book()
    book.apply(record("R", flags=168), 0)
    return book


def test_snapshot_must_finish_before_observation():
    book = m.Book()
    book.apply(record("R", flags=40), 0)
    assert book.observe(101, 5, 1)["status"] == "UNASSESSABLE_INCOMPLETE_EVENT"
    book.apply(record(flags=168), 1)
    assert book.observe(101, 5, 2)["best_bid_size"] == 5


def test_modify_partial_cancel_full_cancel_clear():
    book = ready()
    book.apply(record(), 1)
    book.apply(record("M", price=99, size=8), 2)
    book.apply(record("C", price=99, size=3), 3)
    assert book.orders[1] == (ord("B"), 99, 5)
    book.apply(record("C", price=99, size=5), 4)
    assert not book.orders
    book.apply(record(oid=2), 5)
    book.apply(record("R"), 6)
    assert not book.orders


@pytest.mark.parametrize("action", ["T", "F", "N"])
def test_notifications_never_reduce_book(action):
    book = ready()
    book.apply(record(), 1)
    before = book.orders.copy()
    book.apply(record(action), 2)
    assert book.orders == before


def test_event_boundary_and_displayed_size():
    book = ready()
    book.apply(record(size=3, flags=0), 1)
    assert book.observe(100, 5, 2)["status"] == "UNASSESSABLE_INCOMPLETE_EVENT"
    book.apply(record(oid=2, side="A", price=101, size=7), 2)
    view = book.observe(100, 5, 3)
    assert (
        view["short_limit_marketable"] and not view["opposing_display_covers_quantity"]
    )
    assert view["bid_size_at_or_above_limit"] == 3 and view["spread_nanos"] == 1
    passive = book.observe(101, 5, 3)
    assert not passive["short_limit_marketable"] and passive["ask_size_at_limit"] == 7


@pytest.mark.parametrize(
    "bad",
    [
        record(),
        record("M", oid=2),
        record("C", size=6),
        record("C", price=99),
        record("M", side="A"),
        record("X"),
        record(flags=64),
        record(flags=8),
        record(recv=BASE - 1),
        record(oid=2, size=0),
    ],
)
def test_invalid_update_fails_explicitly(bad):
    book = ready()
    book.apply(record(), 1)
    with pytest.raises(ValueError):
        book.apply(bad, 2)


def test_missing_snapshot_rejected():
    with pytest.raises(ValueError):
        m.Book().apply(record(), 0)


def test_completed_crossed_book_rejected():
    book = ready()
    book.apply(record(), 1)
    book.apply(record(oid=2, side="A", price=99), 2)
    with pytest.raises(ValueError):
        book.observe(100, 5, 3)
