"""Part A parity core — real-fill calibration (prereg §A8.2 Part A; spine AD-17).

Part A of the parity gate replays the **orders** the live bots (``trader-mim-nb``,
``trader-yank`` ≥ 2026-06-17) actually placed and checks a simulator's fill
prices against the real broker fills. The tolerances are the seal-bound
``config`` constants ``PARITY_MAE_MAX_TICKS`` (mean abs error),
``PARITY_P90_MAX_TICKS`` (90th percentile abs error),
``PARITY_SIGNED_BIAS_MAX_TICKS`` (mean signed error, ±) and ``PART_A_MIN_N``
(minimum sample) — this module never restates their values so a seal amendment
cannot leave the prose wrong.

Spine AD-17 forbids feeding outcomes back in: the reconstructed intent log is
*orders* only. The real ``entry_price`` / ``exit_price`` / fill timestamps fix
**when** an order was submitted and are the **comparison target** — never a
simulator input.

This module is the **pure core**:

1. :func:`reconstruct_mim_nb` / :func:`reconstruct_trades_db_row` — build a
   per-trade :class:`~src.ticksim.orders.OrderIntent` log plus the list of real
   broker fills (:class:`RealFill`) from the real order records.
2. :func:`compare_fills` — join each :class:`RealFill` to the
   :class:`~src.ticksim.orders.OrderOutcome` a sim run produced for that leg and
   compute the signed tick error (positive = simulator worse for the trader).
3. :func:`aggregate` — fold every per-fill :class:`FillError` into MAE / p90 /
   signed-bias and a PASS/FAIL verdict, plus the same three stats over the
   ``broker_fill``-only subset so a pass hinging on low-fidelity rows is visible.

Calling ``sim.simulate`` over the real MBO windows is the next slice
(``run_part_a`` in ``part_a.py`` slice 2). This module imports only ``orders``
and ``config`` from ``src.ticksim`` (spine AD-7).

Reconstruction shape (human decision, spec loopback 1):

* **mim-nb** — minimal 2-leg replay. The ``data/mim_nb/orders.csv`` lifecycle is
  walked in timestamp order to identify each trade's **entry** market order
  (``otype 2`` PLACE→FILL) and its **exit** market order (a later ``otype 2``
  PLACE→FILL). Exactly two :class:`~src.ticksim.orders.OrderIntent`\\ s are
  emitted per trade — a ``marketable`` entry and a ``marketable`` exit sharing
  one ``oco_group_id``. The ``otype 4`` protective stop is parsed to follow the
  lifecycle but **never emitted** (it is always cancelled before the exit and
  never fills in the real ledger; resting-stop queue behaviour is Part B). A
  single FILL per leg is assumed — a second FILL on one ``order_id`` raises,
  matching the single-``RealFill``-per-leg assumption in :func:`compare_fills`.
* **yank** — strictly 2 marketable legs from a ``trades.db`` ``trades`` row. A
  ``marketable`` entry at ``timestamp`` and a ``marketable`` exit at the exit ts
  (``exit_timestamp`` if present, else ``timestamp + max(bars_held, 1)``
  minutes). **No TP/SL limit legs** — the real sample carries none.

Fail-closed throughout: any unexpected shape raises :class:`PartAError` naming
the offending row. No trade or fill is ever silently excluded from ``n``.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Literal, NamedTuple

from ..config import (
    MNQ_TICK_DBN,
    PART_A_MIN_N,
    PARITY_MAE_MAX_TICKS,
    PARITY_P90_MAX_TICKS,
    PARITY_SIGNED_BIAS_MAX_TICKS,
)
from ..orders import (
    IntentAction,
    Leg,
    OrderIntent,
    OrderKind,
    OrderOutcome,
    Side,
    TerminalState,
)

__all__ = [
    "Fidelity",
    "Verdict",
    "RealFill",
    "ReconstructedTrade",
    "FillError",
    "PartAStats",
    "PartAResult",
    "PartAError",
    "reconstruct_mim_nb",
    "reconstruct_trades_db_row",
    "compare_fills",
    "aggregate",
]

Fidelity = Literal["broker_fill", "bar_reconstructed"]
"""Per-trade / per-fill fidelity tier (spine AD-17: recorded and counted, never
silently excluded). ``broker_fill`` = price + ts taken straight from a real
broker FILL row; ``bar_reconstructed`` = reconstructed from a ``trades.db`` bar
row that carries no order log."""

Verdict = Literal["PASS", "FAIL"]
"""Part A calibration verdict."""

logger = logging.getLogger(__name__)

_NS_PER_SECOND = 1_000_000_000
_NS_PER_MINUTE = 60 * _NS_PER_SECOND
_DBN_PER_INDEX_POINT = 1_000_000_000

# mim-nb orders.csv otype tokens (spec: otype 2 = market, otype 4 = protective stop)
_MIM_MARKET_OTYPE = "2"
_MIM_STOP_OTYPE = "4"
# A bracketed entry places its protective otype-4 stop within ~0.1 s on the real
# ledger; 5 s is a generous bound that still cannot reach the next trade.
_MIM_BRACKET_WINDOW_NS = 5_000_000_000
# events that must never appear in the mim-nb ledger (verified absent 2026-08-30)
_REPLACE_EVENTS = frozenset({"REPLACE", "MODIFY"})
# lifecycle sort tie-break: a same-ts PLACE must sort before its FILL, and a
# FILL before a CANCEL, so a same-timestamp pair can't invert.
_EVENT_RANK: dict[str, int] = {"PLACE": 0, "FILL": 1, "CANCEL": 2}

# yank trades.db direction tokens. Numeric tokens follow the mim-nb side
# encoding used elsewhere in this module (0 = buy/long, 1 = sell/short).
_LONG_TOKENS = frozenset({"L", "LONG", "BUY", "B", "0"})
_SHORT_TOKENS = frozenset({"S", "SHORT", "SELL", "1"})


class PartAError(Exception):
    """A real order record had an unexpected shape (fail-closed; spine AD-17).

    Raised — naming the offending row — for any lifecycle Part A does not model:
    a non-market entry, a fired protective stop, a ``REPLACE`` / ``MODIFY``
    event, a truncated ledger, a causally corrupt bracket, a same-side round
    trip, a ``FILL`` row flagged rejected, an unparseable price / timestamp /
    size, or a missing / duplicate / side-mismatched
    :class:`~src.ticksim.orders.OrderOutcome`.

    The runner (:mod:`~src.ticksim.parity.part_a_runner`) reuses it for a
    **window-book data fault**: an unfilled leg whose miss cannot be priced
    because the ±90-min MBO capture has no touch on the crossed side at the real
    fill ts, a multi-instrument or mis-ordered window stream, a non-re-iterable
    ``source_for`` result, a duplicate ``trade_id``, or a ``sim`` outcome for a
    foreign ``trade_id``.
    """


# ---------------------------------------------------------------------------
# Frozen value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealFill:
    """One real broker fill — the comparison target, never a simulator input."""

    order_id: str
    leg: Leg
    side: Side
    size: int
    price_dbn: int
    ts_ns: int
    fidelity: Fidelity


@dataclass(frozen=True)
class ReconstructedTrade:
    """One trade's reconstructed intent log + its real broker fills (spine AD-17).

    ``__post_init__`` asserts the frozen-Intent guarantees: at least one intent
    and at least one real fill (spec Never: no zero-``real_fills`` trade),
    non-decreasing ``submit_ts_ns`` across ``intents`` (AD-23), one shared
    ``trade_id``, one shared non-``None`` ``oco_group_id`` across every leg
    (AD-25), and every ``RealFill`` ``(order_id, leg)`` matching an emitted
    intent.
    """

    trade_id: str
    intents: tuple[OrderIntent, ...]
    real_fills: tuple[RealFill, ...]
    fidelity: Fidelity

    def __post_init__(self) -> None:
        if len(self.intents) < 1:
            raise PartAError(f"ReconstructedTrade {self.trade_id!r} has no intents")
        if not self.real_fills:
            raise PartAError(
                f"ReconstructedTrade {self.trade_id!r} has zero real_fills"
            )
        ts = [intent.submit_ts_ns for intent in self.intents]
        if any(b < a for a, b in zip(ts, ts[1:])):
            raise PartAError(
                f"ReconstructedTrade {self.trade_id!r} intents are not "
                f"non-decreasing in submit_ts_ns: {ts}"
            )
        trade_ids = {intent.trade_id for intent in self.intents}
        if len(trade_ids) != 1:
            raise PartAError(
                f"ReconstructedTrade {self.trade_id!r} intents span multiple "
                f"trade_ids: {trade_ids}"
            )
        groups = {intent.oco_group_id for intent in self.intents}
        if len(groups) != 1 or None in groups:
            raise PartAError(
                f"ReconstructedTrade {self.trade_id!r} intents must share one "
                f"non-None oco_group_id, got {groups}"
            )
        emitted = {(intent.order_id, intent.leg) for intent in self.intents}
        for real in self.real_fills:
            if (real.order_id, real.leg) not in emitted:
                raise PartAError(
                    f"ReconstructedTrade {self.trade_id!r} real fill "
                    f"{(real.order_id, real.leg.value)!r} has no matching "
                    f"emitted intent"
                )


@dataclass(frozen=True)
class FillError:
    """One real fill vs its simulated outcome.

    ``sim_vwap_dbn`` / ``signed_error_ticks`` are ``None`` for an unresolved
    miss (``miss_reason == "leg_unfilled"``): the sim left a leg unfilled that
    really filled. The magnitude is supplied by the runner slice using
    ``real_ts_ns`` (the real fill ts — priced as the touch at that ts).
    ``sim_terminal_state`` carries the sim's terminal state on a miss (else
    ``None``) as a diagnostic. ``aggregate`` treats any unresolved miss as an
    automatic FAIL. ``fidelity`` is required so ``aggregate`` can compute the
    ``broker_fill`` subset stats from ``errors`` alone.
    """

    trade_id: str
    order_id: str
    leg: Leg
    real_dbn: int
    real_ts_ns: int
    sim_vwap_dbn: int | None
    signed_error_ticks: float | None
    miss_reason: str | None
    sim_terminal_state: str | None
    fidelity: Fidelity


@dataclass(frozen=True)
class PartAStats:
    """MAE / p90 / signed-bias over a set of :class:`FillError`\\ s.

    ``n`` counts **every** :class:`FillError` (including unresolved misses); the
    three tick figures are over the subset with a non-``None``
    ``signed_error_ticks``.
    """

    n: int
    mae_ticks: float
    p90_ticks: float
    signed_bias_ticks: float


@dataclass(frozen=True)
class PartAResult:
    """The Part A calibration verdict + its supporting stats."""

    stats: PartAStats
    broker_fill_stats: PartAStats
    verdict: Verdict
    reason: str
    warning: str | None
    unresolved_misses: int
    errors: tuple[FillError, ...] = ()


# ---------------------------------------------------------------------------
# Scalar parsing helpers (fail-closed)
# ---------------------------------------------------------------------------


def _field(row: Mapping[str, object], key: str) -> str:
    """Row-mapping cell as a stripped string. ``None`` -> ``""``; an int ``0``
    stays ``"0"`` (never collapses to empty, unlike ``str(v or "")``)."""
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _parse_ts_ns(raw: str, *, field_name: str, row_repr: str) -> int:
    """ISO-8601 timestamp -> ns since the Unix epoch. Naive -> UTC.

    Normalizes before :func:`datetime.fromisoformat`: a trailing ``Z`` / ``z``
    becomes ``+00:00``; a single space between the date and time becomes ``T``;
    fractional seconds are truncated to 6 digits. A pre-epoch / unparseable /
    empty timestamp raises :class:`PartAError` (``OrderIntent`` rejects
    ``submit_ts_ns < 0`` in pydantic — we raise first, naming the row).
    """
    text = raw.strip()
    if not text:
        raise PartAError(f"empty {field_name} in row {row_repr}")
    if text[-1] in ("Z", "z"):
        text = text[:-1] + "+00:00"
    text = re.sub(r"^(\d{4}-\d{2}-\d{2})[ ](\d{2}:\d{2})", r"\1T\2", text)
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise PartAError(f"unparseable {field_name} {raw!r} in row {row_repr}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ns = round(dt.timestamp() * _NS_PER_SECOND)
    if ns < 0:
        raise PartAError(f"pre-epoch {field_name} {raw!r} in row {row_repr}")
    return ns


def _px_to_dbn(raw: object, *, field_name: str, row_repr: str) -> int:
    """MNQ index price -> DBN 1e-9 fixed-point.

    Exact via :class:`~decimal.Decimal` — ``round(px * 1e9)`` can misround by
    one DBN unit, which would read as a real sub-tick discrepancy. **No tick
    snap** (spec loopback 1 #6). A non-finite / non-positive price raises
    :class:`PartAError`.
    """
    if isinstance(raw, bool):
        raise PartAError(f"{field_name} is a bool in row {row_repr}")
    try:
        dec = Decimal(str(raw).strip())
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise PartAError(f"unparseable {field_name} {raw!r} in row {row_repr}") from exc
    if not dec.is_finite() or dec <= 0:
        raise PartAError(
            f"non-finite / non-positive {field_name} {raw!r} in row {row_repr}"
        )
    return int((dec * _DBN_PER_INDEX_POINT).to_integral_value())


def _to_int(raw: object, *, field_name: str, row_repr: str) -> int:
    """Coerce to ``int``, accepting an integral float (``2.0`` / ``30.0`` from
    JSON or a sqlite REAL). A non-integral float / unparseable value raises
    :class:`PartAError`. No sign check — callers add one where needed."""
    if isinstance(raw, bool):
        raise PartAError(f"{field_name} is a bool in row {row_repr}")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if raw.is_integer():
            return int(raw)
        raise PartAError(f"non-integral {field_name} {raw!r} in row {row_repr}")
    try:
        number = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise PartAError(f"unparseable {field_name} {raw!r} in row {row_repr}") from exc
    if not number.is_integer():
        raise PartAError(f"non-integral {field_name} {raw!r} in row {row_repr}")
    return int(number)


def _int_size(raw: object, *, field_name: str, row_repr: str) -> int:
    val = _to_int(raw, field_name=field_name, row_repr=row_repr)
    if val <= 0:
        raise PartAError(f"non-positive {field_name} {raw!r} in row {row_repr}")
    return val


def _mim_side(token: str, row_repr: str) -> Side:
    """mim-nb ``side``: ``0`` = buy, ``1`` = sell."""
    if token == "0":
        return Side.BUY
    if token == "1":
        return Side.SELL
    raise PartAError(f"bad side token {token!r} in mim-nb row {row_repr}")


def _yank_direction(raw: object, row_repr: str) -> Literal["long", "short"]:
    token = str(raw if raw is not None else "").strip().upper()
    if token in _LONG_TOKENS:
        return "long"
    if token in _SHORT_TOKENS:
        return "short"
    raise PartAError(f"bad direction {raw!r} in trades.db row {row_repr}")


def _opposite(side: Side) -> Side:
    return Side.SELL if side is Side.BUY else Side.BUY


# ---------------------------------------------------------------------------
# mim-nb reconstruction — timestamp-sorted lifecycle walk (spec KEEP)
# ---------------------------------------------------------------------------


class _MimRow(NamedTuple):
    ts_ns: int
    event: str
    order_id: str
    otype: str
    side: str
    size_raw: str
    price_raw: str
    row_repr: str


@dataclass
class _PendingLeg:
    order_id: str
    ts_ns: int
    side: Side
    size: int
    fill_px_dbn: int | None = None
    fill_ts_ns: int | None = None
    fill_size: int | None = None


def _build_mim_trade(entry: _PendingLeg, exit_: _PendingLeg) -> ReconstructedTrade:
    if (
        entry.fill_px_dbn is None
        or entry.fill_ts_ns is None
        or entry.fill_size is None
        or exit_.fill_px_dbn is None
        or exit_.fill_ts_ns is None
        or exit_.fill_size is None
    ):  # pragma: no cover - guarded by the caller's state machine
        raise PartAError(
            f"mim-nb trade {entry.order_id!r} reached build with an unfilled leg"
        )
    if exit_.ts_ns <= entry.ts_ns:
        raise PartAError(
            f"mim-nb exit intent ts {exit_.ts_ns} <= entry intent ts "
            f"{entry.ts_ns} (causally corrupt bracket, entry order "
            f"{entry.order_id!r})"
        )
    if exit_.side != _opposite(entry.side):
        raise PartAError(
            f"mim-nb trade {entry.order_id!r}: exit side {exit_.side.value} is "
            f"not the opposite of entry side {entry.side.value} (corrupt "
            f"same-side round trip)"
        )
    trade_id = f"mimnb-{entry.order_id}"
    oco = f"{trade_id}-oco"
    entry_intent = OrderIntent(
        action=IntentAction.SUBMIT,
        order_id=entry.order_id,
        trade_id=trade_id,
        leg=Leg.ENTRY,
        kind=OrderKind.MARKETABLE,
        side=entry.side,
        size=entry.size,
        submit_ts_ns=entry.ts_ns,
        oco_group_id=oco,
    )
    exit_intent = OrderIntent(
        action=IntentAction.SUBMIT,
        order_id=exit_.order_id,
        trade_id=trade_id,
        leg=Leg.EXIT,
        kind=OrderKind.MARKETABLE,
        side=exit_.side,
        size=exit_.size,
        submit_ts_ns=exit_.ts_ns,
        oco_group_id=oco,
    )
    real_fills = (
        RealFill(
            order_id=entry.order_id,
            leg=Leg.ENTRY,
            side=entry.side,
            size=entry.fill_size,
            price_dbn=entry.fill_px_dbn,
            ts_ns=entry.fill_ts_ns,
            fidelity="broker_fill",
        ),
        RealFill(
            order_id=exit_.order_id,
            leg=Leg.EXIT,
            side=exit_.side,
            size=exit_.fill_size,
            price_dbn=exit_.fill_px_dbn,
            ts_ns=exit_.fill_ts_ns,
            fidelity="broker_fill",
        ),
    )
    return ReconstructedTrade(
        trade_id=trade_id,
        intents=(entry_intent, exit_intent),
        real_fills=real_fills,
        fidelity="broker_fill",
    )


def reconstruct_mim_nb(
    rows: Iterable[Mapping[str, object]],
) -> list[ReconstructedTrade]:
    """Reconstruct mim-nb trades from ``data/mim_nb/orders.csv`` rows.

    Each row is a mapping with at least ``ts_utc, event, order_id, otype, side,
    size, price`` (and optionally ``outcome``); extra columns (``detail``,
    ``chain``) are ignored. ``chain`` hash-chain validation is the loader's job.

    Returns one :class:`ReconstructedTrade` per completed entry/exit round trip,
    in ledger order. ``reconstruct_mim_nb([]) == []``. A single FILL per leg is
    assumed — a second FILL on one ``order_id`` raises.

    Raises :class:`PartAError` (naming the row) for: a ``REPLACE`` / ``MODIFY``
    event; a ``FILL`` row flagged ``outcome == "REJECTED"`` (contradiction); a
    non-market PLACE (``otype`` not ``2`` or ``4``); a FILL on the ``otype 4``
    protective stop; a FILL / PLACE / CANCEL that does not fit the lifecycle; an
    exit leg not the opposite side of its entry; a row with no ``order_id``; a
    ledger that ends mid-trade (entry filled, exit not).

    Pure ``event == "REJECTED"`` rows and ``order_id == "FAIL"`` placeholder
    rows are dropped silently (not orders).
    """
    parsed: list[_MimRow] = []
    for row in rows:
        row_repr = repr(dict(row))
        event = _field(row, "event").upper()
        outcome = _field(row, "outcome").upper()
        order_id = _field(row, "order_id")
        if event in _REPLACE_EVENTS:
            raise PartAError(
                f"unsupported {event} event in mim-nb row {row_repr} "
                "(ledger is PLACE/CANCEL/FILL/REJECTED only)"
            )
        if order_id == "FAIL":
            continue  # placeholder row, not an order
        if event == "FILL" and outcome == "REJECTED":
            raise PartAError(
                f"contradictory row: event=FILL but outcome=REJECTED in "
                f"mim-nb row {row_repr}"
            )
        if event == "REJECTED" or outcome == "REJECTED":
            continue  # a rejected order never filled
        if not order_id:
            raise PartAError(f"mim-nb row has no order_id: {row_repr}")
        parsed.append(
            _MimRow(
                ts_ns=_parse_ts_ns(
                    _field(row, "ts_utc"),
                    field_name="ts_utc",
                    row_repr=row_repr,
                ),
                event=event,
                order_id=order_id,
                otype=_field(row, "otype"),
                side=_field(row, "side"),
                size_raw=_field(row, "size"),
                price_raw=_field(row, "price"),
                row_repr=row_repr,
            )
        )
    parsed.sort(key=lambda r: (r.ts_ns, _EVENT_RANK.get(r.event, 9), r.order_id))

    # Classify every market PLACE as an ENTRY or an EXIT before walking the
    # lifecycle. The live bot brackets an entry: it places the market entry and,
    # ~0.1 s later, its protective otype-4 stop. An exit (the "cat stop" flatten)
    # is placed alone, right after CANCELling that stop. So a market PLACE
    # followed by a stop PLACE within `_MIM_BRACKET_WINDOW_NS` is an entry;
    # otherwise it is an exit.
    #
    # This discriminator is load-bearing on the real ledger. Without it a trade
    # whose exit was never logged (shape 2 below) swallows the *next* day's entry
    # as its exit, silently pairing legs from different trades.
    entry_place_ids: set[str] = set()
    stop_places = [
        r for r in parsed if r.event == "PLACE" and r.otype == _MIM_STOP_OTYPE
    ]
    for r in parsed:
        if r.event != "PLACE" or r.otype != _MIM_MARKET_OTYPE:
            continue
        if any(0 <= s.ts_ns - r.ts_ns <= _MIM_BRACKET_WINDOW_NS for s in stop_places):
            entry_place_ids.add(r.order_id)

    trades: list[ReconstructedTrade] = []
    state = "FLAT"
    entry: _PendingLeg | None = None
    exit_: _PendingLeg | None = None
    stop_order_ids: set[str] = set()
    abandoned = 0  # incomplete trades dropped (no fill to compare) -- reported

    for r in parsed:
        if r.event == "PLACE":
            if r.otype == _MIM_STOP_OTYPE:
                stop_order_ids.add(r.order_id)
                continue
            if r.otype != _MIM_MARKET_OTYPE:
                raise PartAError(
                    f"non-market order (otype {r.otype!r}) in mim-nb row "
                    f"{r.row_repr}"
                )
            leg = _PendingLeg(
                order_id=r.order_id,
                ts_ns=r.ts_ns,
                side=_mim_side(r.side, r.row_repr),
                size=_int_size(r.size_raw, field_name="size", row_repr=r.row_repr),
            )
            # The bracket signal is only consulted to answer one question: while a
            # trade is still open, is this PLACE the *exit* of that trade, or a
            # *new entry* whose predecessor was never closed in the ledger? When
            # FLAT the answer is unambiguous (it is an entry) and the signal is
            # not needed -- which also keeps unbracketed fixtures working.
            is_new_entry = state != "FLAT" and r.order_id in entry_place_ids
            if state == "FLAT":
                entry, exit_, state = leg, None, "ENTRY_PENDING"
            elif is_new_entry:
                # A new bracketed entry supersedes whatever was pending. Either
                # the prior entry never filled (nothing to compare), or its exit
                # was never written to the ledger -- both are incomplete trades
                # that Part A cannot score, so they are dropped, not guessed at.
                abandoned += 1
                logger.warning(
                    "mim-nb: dropping incomplete trade (state=%s) superseded "
                    "by new bracketed entry %s -- no comparable fill pair",
                    state,
                    r.order_id,
                )
                entry, exit_, state = leg, None, "ENTRY_PENDING"
            elif state == "ACTIVE":
                exit_, state = leg, "EXIT_PENDING"
            elif state == "ENTRY_PENDING":
                # The entry never filled, so this flatten has no position to
                # close and neither leg can produce a real fill. Drop both.
                abandoned += 1
                logger.warning(
                    "mim-nb: entry %s never filled; its unbracketed flatten %s "
                    "closes nothing -- dropping the pair",
                    entry.order_id if entry else "?",
                    r.order_id,
                )
                entry, exit_, state = None, None, "FLAT"
            else:
                raise PartAError(
                    f"unexpected market PLACE for {r.order_id!r} while "
                    f"{state} in mim-nb row {r.row_repr}"
                )
        elif r.event == "FILL":
            if r.otype == _MIM_STOP_OTYPE or r.order_id in stop_order_ids:
                # The protective stop FIRED. The pre-2026-09-01 code raised here,
                # on the assumption (spec Reconstruction shape) that the stop "is
                # always cancelled before the exit and never fills in the real
                # ledger". The real ledger falsifies that: 2026-07-29 and
                # 2026-08-28 both carry an otype-4 FILL with a `pnl=` detail.
                #
                # A stop-out is a real exit fill and must be scored, not
                # discarded. It is modelled the same way the cat-stop flatten is:
                # a marketable exit submitted at the moment the stop triggered
                # (the fill ts), which is the correct comparison point -- using
                # the stop's original resting PLACE ts would have the simulator
                # fill at entry time.
                if state == "ACTIVE" and entry is not None:
                    stop_px = _px_to_dbn(
                        r.price_raw, field_name="price", row_repr=r.row_repr
                    )
                    stop_size = _int_size(
                        r.size_raw, field_name="size", row_repr=r.row_repr
                    )
                    stop_leg = _PendingLeg(
                        order_id=r.order_id,
                        ts_ns=r.ts_ns,
                        side=_mim_side(r.side, r.row_repr),
                        size=stop_size,
                    )
                    stop_leg.fill_px_dbn = stop_px
                    stop_leg.fill_ts_ns = r.ts_ns
                    stop_leg.fill_size = stop_size
                    trades.append(_build_mim_trade(entry, stop_leg))
                    entry = exit_ = None
                    state = "FLAT"
                    continue
                # An otype-4 FILL with no live position -- e.g. the 2026-07-29
                # `order_id='111'` row, which has no PLACE anywhere in the
                # ledger. It cannot be attributed to a trade, so it is reported
                # and skipped rather than guessed at.
                abandoned += 1
                logger.warning(
                    "mim-nb: unattributable stop FILL on %s (state=%s, no live "
                    "position) -- skipping row %s",
                    r.order_id,
                    state,
                    r.row_repr,
                )
                continue
            fill_px = _px_to_dbn(r.price_raw, field_name="price", row_repr=r.row_repr)
            fill_size = _int_size(r.size_raw, field_name="size", row_repr=r.row_repr)
            if (
                state == "ENTRY_PENDING"
                and entry is not None
                and r.order_id == entry.order_id
            ):
                entry.fill_px_dbn = fill_px
                entry.fill_ts_ns = r.ts_ns
                entry.fill_size = fill_size
                state = "ACTIVE"
            elif (
                state == "EXIT_PENDING"
                and exit_ is not None
                and r.order_id == exit_.order_id
            ):
                exit_.fill_px_dbn = fill_px
                exit_.fill_ts_ns = r.ts_ns
                exit_.fill_size = fill_size
                assert entry is not None
                trades.append(_build_mim_trade(entry, exit_))
                entry = exit_ = None
                state = "FLAT"
            else:
                raise PartAError(
                    f"unexpected FILL for {r.order_id!r} while {state} in "
                    f"mim-nb row {r.row_repr}"
                )
        elif r.event == "CANCEL":
            if r.order_id in stop_order_ids:
                continue  # the protective stop is always cancelled — expected
            if (
                state == "ENTRY_PENDING"
                and entry is not None
                and r.order_id == entry.order_id
            ):
                entry, state = None, "FLAT"  # entry never filled, no exit -> skip
            elif (
                state == "EXIT_PENDING"
                and exit_ is not None
                and r.order_id == exit_.order_id
            ):
                raise PartAError(
                    f"exit order {r.order_id!r} cancelled after the entry "
                    f"filled (truncated / aborted trade) in mim-nb row "
                    f"{r.row_repr}"
                )
            else:
                raise PartAError(
                    f"unexpected CANCEL for {r.order_id!r} while {state} in "
                    f"mim-nb row {r.row_repr}"
                )
        else:
            raise PartAError(f"unknown event {r.event!r} in mim-nb row {r.row_repr}")

    if state != "FLAT":
        # A ledger that ends mid-trade is the live bot still holding, or an exit
        # that was never written. Either way the trade has no comparable fill
        # pair, so it is dropped with a warning rather than failing the whole
        # reconstruction -- one open trade at the tail must not cost Part A the
        # other 20+ scoreable trades. (Before 2026-09-01 this raised.)
        abandoned += 1
        logger.warning(
            "mim-nb: ledger ends mid-trade (state=%s) -- dropping the trailing "
            "incomplete trade",
            state,
        )
    if abandoned:
        logger.warning(
            "mim-nb: %d incomplete trade(s) dropped, %d reconstructed",
            abandoned,
            len(trades),
        )
    return trades


# ---------------------------------------------------------------------------
# yank reconstruction — strictly 2 marketable legs from a trades.db row
# ---------------------------------------------------------------------------


def _load_metadata(raw_meta: object, row_repr: str) -> Mapping[str, object]:
    if isinstance(raw_meta, str):
        text = raw_meta.strip()
        if not text:
            return {}
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PartAError(
                f"unparseable metadata JSON in trades.db row {row_repr}"
            ) from exc
        if not isinstance(loaded, dict):
            raise PartAError(f"metadata is not an object in trades.db row {row_repr}")
        return loaded
    if raw_meta is None:
        return {}
    if isinstance(raw_meta, Mapping):
        return raw_meta
    raise PartAError(
        f"metadata is neither a mapping nor a JSON string in trades.db "
        f"row {row_repr}"
    )


def reconstruct_trades_db_row(row: Mapping[str, object]) -> ReconstructedTrade:
    """Reconstruct one yank trade from a ``trades.db`` ``trades`` row.

    Expected keys: ``timestamp`` (entry ts, tz-aware or naive-UTC ISO),
    ``direction`` (``S`` / ``L`` / ``SHORT`` / ``LONG`` or a numeric token),
    ``entry_price`` / ``exit_price`` (index points), optional ``exit_timestamp``,
    ``metadata`` (JSON string or mapping with ``contracts`` and optional
    ``bars_held``), optional ``trader_id``, optional ``id`` (sqlite PK — used in
    the synthesised ids when present so two rows at the same ``timestamp`` can't
    collide).

    Emits a ``marketable`` entry at ``timestamp`` and a ``marketable`` exit at
    the exit ts — ``exit_timestamp`` if present and non-empty, else
    ``timestamp + max(bars_held, 1)`` minutes (``bars_held`` default 60). Both
    :class:`RealFill`\\ s are tagged ``fidelity="bar_reconstructed"``. **No
    TP/SL limit legs** (spec loopback 1 #3).

    Raises :class:`PartAError` for a bad direction, an unparseable / non-positive
    price, an unparseable ``contracts`` / ``bars_held``, malformed ``metadata``,
    or an exit ts ≤ the entry ts (causally corrupt bracket).
    """
    row_repr = repr(dict(row))
    trader_id = _field(row, "trader_id") or "yank"
    direction = _yank_direction(row.get("direction"), row_repr)
    entry_ts = _parse_ts_ns(
        _field(row, "timestamp"), field_name="timestamp", row_repr=row_repr
    )

    meta = _load_metadata(row.get("metadata"), row_repr)
    size = _int_size(
        meta.get("contracts", 1), field_name="contracts", row_repr=row_repr
    )

    exit_ts_raw = _field(row, "exit_timestamp")
    if exit_ts_raw:
        exit_ts = _parse_ts_ns(
            exit_ts_raw, field_name="exit_timestamp", row_repr=row_repr
        )
    else:
        bars_held = _to_int(
            meta.get("bars_held", 60), field_name="bars_held", row_repr=row_repr
        )
        exit_ts = entry_ts + max(bars_held, 1) * _NS_PER_MINUTE

    if exit_ts <= entry_ts:
        raise PartAError(
            f"yank exit ts {exit_ts} <= entry ts {entry_ts} (causally corrupt "
            f"bracket) in trades.db row {row_repr}"
        )

    entry_side = Side.BUY if direction == "long" else Side.SELL
    exit_side = _opposite(entry_side)
    entry_px = _px_to_dbn(
        row.get("entry_price"), field_name="entry_price", row_repr=row_repr
    )
    exit_px = _px_to_dbn(
        row.get("exit_price"), field_name="exit_price", row_repr=row_repr
    )

    row_id = _field(row, "id")
    key = row_id if row_id else str(entry_ts)
    trade_id = f"{trader_id}-{key}"
    oco = f"{trade_id}-oco"
    entry_oid = f"{trade_id}-entry"
    exit_oid = f"{trade_id}-exit"
    entry_intent = OrderIntent(
        action=IntentAction.SUBMIT,
        order_id=entry_oid,
        trade_id=trade_id,
        leg=Leg.ENTRY,
        kind=OrderKind.MARKETABLE,
        side=entry_side,
        size=size,
        submit_ts_ns=entry_ts,
        oco_group_id=oco,
    )
    exit_intent = OrderIntent(
        action=IntentAction.SUBMIT,
        order_id=exit_oid,
        trade_id=trade_id,
        leg=Leg.EXIT,
        kind=OrderKind.MARKETABLE,
        side=exit_side,
        size=size,
        submit_ts_ns=exit_ts,
        oco_group_id=oco,
    )
    real_fills = (
        RealFill(
            order_id=entry_oid,
            leg=Leg.ENTRY,
            side=entry_side,
            size=size,
            price_dbn=entry_px,
            ts_ns=entry_ts,
            fidelity="bar_reconstructed",
        ),
        RealFill(
            order_id=exit_oid,
            leg=Leg.EXIT,
            side=exit_side,
            size=size,
            price_dbn=exit_px,
            ts_ns=exit_ts,
            fidelity="bar_reconstructed",
        ),
    )
    return ReconstructedTrade(
        trade_id=trade_id,
        intents=(entry_intent, exit_intent),
        real_fills=real_fills,
        fidelity="bar_reconstructed",
    )


# ---------------------------------------------------------------------------
# Fill comparison
# ---------------------------------------------------------------------------


def compare_fills(
    outcomes: Iterable[OrderOutcome], trade: ReconstructedTrade
) -> list[FillError]:
    """Join each real fill of ``trade`` to its :class:`OrderOutcome` and grade it.

    ``outcomes`` is first filtered to those with ``trade_id == trade.trade_id``
    (broker ``order_id``\\ s can repeat across sessions — the join must be
    trade-scoped). For each :class:`RealFill`, find the outcome with the
    matching ``(order_id, leg)``. A duplicate or missing ``(order_id, leg)``, or
    a ``side`` mismatch -> :class:`PartAError` (join / reconstruction fault). A
    leftover FILLED outcome for this trade that no :class:`RealFill` claims is
    ignored (2-leg world).

    ``signed_error_ticks = (sim_vwap_dbn - real_fill_dbn) / MNQ_TICK_DBN`` on a
    **buy**, negated on a **sell**, so **positive = simulator worse for the
    trader** (paid more on a buy, received less on a sell). Multiple sim fills
    for one leg -> size-weighted mean sim price. A single :class:`RealFill` per
    ``(order_id, leg)`` is assumed (real sample is 1–2 lot, single fill).

    An outcome whose ``terminal_state != FILLED`` for a leg that really filled
    -> :class:`FillError` with ``miss_reason="leg_unfilled"``,
    ``signed_error_ticks=None`` and ``sim_terminal_state`` set (magnitude
    supplied by the runner slice); it still counts toward ``n`` and blocks PASS.
    """
    by_key: dict[tuple[str, Leg], OrderOutcome] = {}
    for outcome in outcomes:
        if outcome.trade_id != trade.trade_id:
            continue
        key = (outcome.order_id, outcome.leg)
        if key in by_key:
            raise PartAError(
                f"duplicate OrderOutcome for {key!r} in trade {trade.trade_id!r}"
            )
        by_key[key] = outcome

    errors: list[FillError] = []
    for real in trade.real_fills:
        key = (real.order_id, real.leg)
        matched = by_key.get(key)
        if matched is None:
            raise PartAError(
                f"no OrderOutcome for real fill {key!r} in trade {trade.trade_id!r}"
            )
        if matched.side != real.side:
            raise PartAError(
                f"side mismatch for {key!r} in trade {trade.trade_id!r}: "
                f"outcome {matched.side.value} vs real {real.side.value}"
            )
        if matched.terminal_state is not TerminalState.FILLED:
            errors.append(
                FillError(
                    trade_id=trade.trade_id,
                    order_id=real.order_id,
                    leg=real.leg,
                    real_dbn=real.price_dbn,
                    real_ts_ns=real.ts_ns,
                    sim_vwap_dbn=None,
                    signed_error_ticks=None,
                    miss_reason="leg_unfilled",
                    sim_terminal_state=matched.terminal_state.value,
                    fidelity=real.fidelity,
                )
            )
            continue
        if not matched.fills:
            raise PartAError(
                f"OrderOutcome {key!r} is FILLED with no fills in trade "
                f"{trade.trade_id!r}"
            )
        total_size = sum(fill.size for fill in matched.fills)
        sim_vwap = round(
            sum(fill.px_dbn * fill.size for fill in matched.fills) / total_size
        )
        diff = sim_vwap - real.price_dbn
        signed = (diff if real.side is Side.BUY else -diff) / MNQ_TICK_DBN + 0.0
        errors.append(
            FillError(
                trade_id=trade.trade_id,
                order_id=real.order_id,
                leg=real.leg,
                real_dbn=real.price_dbn,
                real_ts_ns=real.ts_ns,
                sim_vwap_dbn=sim_vwap,
                signed_error_ticks=signed,
                miss_reason=None,
                sim_terminal_state=None,
                fidelity=real.fidelity,
            )
        )
    return errors


# ---------------------------------------------------------------------------
# Aggregate + verdict
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    """``q``-th percentile with linear interpolation (numpy ``percentile`` default).

    Verification-gap confirmed: ``[0, 1, -2, 3, -4]`` (as ``|e|``) ->
    MAE 2.0 / p90 3.6 / bias -0.4.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _stats(errors: Sequence[FillError]) -> tuple[PartAStats, int]:
    resolved = [
        e.signed_error_ticks for e in errors if e.signed_error_ticks is not None
    ]
    unresolved = sum(1 for e in errors if e.signed_error_ticks is None)
    abs_errs = [abs(x) for x in resolved]
    stats = PartAStats(
        n=len(errors),
        mae_ticks=_mean(abs_errs),
        p90_ticks=_percentile(abs_errs, 90.0),
        signed_bias_ticks=_mean(resolved),
    )
    return stats, unresolved


def _bounds_pass(stats: PartAStats, unresolved: int) -> bool:
    """Quality verdict — the three tolerance bounds + no unresolved miss, **no
    N floor**."""
    return (
        unresolved == 0
        and stats.mae_ticks <= PARITY_MAE_MAX_TICKS
        and stats.p90_ticks <= PARITY_P90_MAX_TICKS
        and abs(stats.signed_bias_ticks) <= PARITY_SIGNED_BIAS_MAX_TICKS
    )


def aggregate(errors: Sequence[FillError]) -> PartAResult:
    """Fold every :class:`FillError` into MAE / p90 / signed-bias + a verdict.

    ``PASS`` iff ``stats.n >= PART_A_MIN_N`` **and** the three tick figures are
    each within their ``config`` bound (``PARITY_MAE_MAX_TICKS`` /
    ``PARITY_P90_MAX_TICKS`` / ``PARITY_SIGNED_BIAS_MAX_TICKS``) **and** there is
    no unresolved miss. ``n`` counts every :class:`FillError`, misses included.

    The result also carries the same three stats over the ``broker_fill``-only
    subset and ``unresolved_misses`` as a structured count. ``warning`` is set
    when the ``broker_fill`` subset is empty (the verdict rests entirely on
    ``bar_reconstructed`` rows) and/or the subset's bounds-only quality verdict
    disagrees with the full sample's bounds-only quality verdict (**not** the N
    floor — a sub-``PART_A_MIN_N`` FAIL is not a quality disagreement). Both
    parts are surfaced, joined with ``"; "``.
    """
    error_tuple = tuple(errors)
    stats, unresolved = _stats(error_tuple)
    bf_errors = tuple(e for e in error_tuple if e.fidelity == "broker_fill")
    bf_stats, bf_unresolved = _stats(bf_errors)

    clauses: list[str] = []
    if stats.n < PART_A_MIN_N:
        clauses.append(f"N={stats.n} < PART_A_MIN_N={PART_A_MIN_N}")
    if unresolved:
        clauses.append(f"{unresolved} unresolved leg_unfilled miss(es)")
    if stats.mae_ticks > PARITY_MAE_MAX_TICKS:
        clauses.append(f"MAE {stats.mae_ticks:.4f} > {PARITY_MAE_MAX_TICKS} ticks")
    if stats.p90_ticks > PARITY_P90_MAX_TICKS:
        clauses.append(f"p90 {stats.p90_ticks:.4f} > {PARITY_P90_MAX_TICKS} ticks")
    if abs(stats.signed_bias_ticks) > PARITY_SIGNED_BIAS_MAX_TICKS:
        clauses.append(
            f"|signed bias| {abs(stats.signed_bias_ticks):.4f} > "
            f"{PARITY_SIGNED_BIAS_MAX_TICKS} ticks"
        )

    verdict: Verdict = "PASS" if not clauses else "FAIL"
    if verdict == "PASS":
        reason = (
            f"PASS: N={stats.n}, MAE={stats.mae_ticks:.4f}, "
            f"p90={stats.p90_ticks:.4f}, signed_bias={stats.signed_bias_ticks:.4f} "
            "— all within tolerance"
        )
    else:
        reason = "FAIL: " + "; ".join(clauses)

    full_quality = _bounds_pass(stats, unresolved)
    subset_quality = _bounds_pass(bf_stats, bf_unresolved)
    parts: list[str] = []
    if not bf_errors:
        parts.append(
            "broker_fill subset is empty — the verdict rests entirely on "
            "bar_reconstructed rows"
        )
    if bf_errors and subset_quality != full_quality:
        parts.append(
            f"broker_fill subset quality "
            f"({'PASS' if subset_quality else 'FAIL'}) disagrees with the "
            f"full-sample quality ({'PASS' if full_quality else 'FAIL'})"
        )
    warning = "; ".join(parts) if parts else None

    return PartAResult(
        stats=stats,
        broker_fill_stats=bf_stats,
        verdict=verdict,
        reason=reason,
        warning=warning,
        unresolved_misses=unresolved,
        errors=error_tuple,
    )
