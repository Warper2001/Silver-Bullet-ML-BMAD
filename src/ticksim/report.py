"""The AD-14 three-way P&L report -- the sole money layer of ``src/ticksim``.

The simulator emits :class:`~src.ticksim.orders.OrderOutcome` logs but no
dollars. Pre-registration §2.3 requires every strategy P&L reported **three
ways**:

  * **(a) primary** -- the ``config.PRIMARY`` fills, as simulated;
  * **(b) stressed** -- (a) with a post-hoc 1-tick adverse slip on every entry
    *and* every exit (a pure P&L transform on (a), never a re-simulation);
  * **(c) optimistic** -- the ``config.OPTIMISTIC`` fills, as simulated, over
    the trades that completed a round trip under *both* models.

The §6 decision rule is evaluated on (a) and must also hold under (b); (c) is
context only. This module does **not** evaluate that rule -- it computes and
aggregates the per-trade money and hands downstream a chronological
:class:`RoundTrip` list keyed by ``trade_id`` + entry/exit timestamps so the
study-level walk-forward / regime-split / per-year / deflated-Sharpe /
permutation logic never has to re-pair the outcome logs.

``build_report`` is pure and never mutates its inputs. ``net_cents`` figures are
**net of the §2.1 exchange + commission fees only** -- the §6 decision friction
($4.00 round turn) is applied by the downstream evaluator, not here.

Dependencies (spine AD-7, widened for this slice -- see the AD-24 note in the
architecture spine): ``orders`` + ``config`` + stdlib only. Fees reach this
module **only** through the run manifest dict (``Manifest.to_dict()`` shape,
spine AD-24); ``DOLLARS_PER_INDEX_POINT`` and ``MNQ_TICK_DBN`` are seal-cited
module constants imported from ``config``. ``sim`` / ``book`` / ``events`` /
``fills`` / ``parity`` are never imported.

``float`` appears here for the first time in the pipeline (spine AD-10) and only
in the derived ``mean_net_cents`` / ``profit_factor`` read-outs; every stored
figure is an ``int`` count of USD cents.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .config import DOLLARS_PER_INDEX_POINT, MNQ_TICK_DBN
from .orders import Leg, OrderOutcome, Side

__all__ = [
    "ReportError",
    "RoundTrip",
    "OpenPosition",
    "ModelPnL",
    "ThreeWayReport",
    "build_report",
    "TICK_VALUE_CENTS",
]

_DBN_TO_CENTS_NUM = DOLLARS_PER_INDEX_POINT * 100
_DBN_TO_CENTS_DEN = 1_000_000_000


def _to_cents(px_delta_dbn: int) -> int:
    """DBN 1e-9 index-point units -> USD cents, truncating **toward zero**.

    ``DOLLARS_PER_INDEX_POINT * 100 / 1e9`` is exactly ``1 / 5_000_000``; one MNQ
    tick (``MNQ_TICK_DBN`` = 250_000_000 dbn) is therefore exactly 50 cents and
    any tick-aligned delta converts without loss. A non-tick-aligned delta (only
    synthetic test data) truncates symmetrically for wins and losses.
    """
    magnitude = abs(px_delta_dbn) * _DBN_TO_CENTS_NUM // _DBN_TO_CENTS_DEN
    return -magnitude if px_delta_dbn < 0 else magnitude


TICK_VALUE_CENTS: int = _to_cents(MNQ_TICK_DBN)
"""USD-cent value of one MNQ tick (= 50). The model-(b) slip is ``2 *`` this per
contract per round trip -- one tick worse on the entry, one on the exit."""


class ReportError(Exception):
    """A round trip could not be formed, or the inputs are malformed.

    Raised for: a manifest missing ``["config"]`` or a fee key; a fee that is
    not a non-negative ``int``; the two manifests not being a ``PRIMARY`` +
    ``OPTIMISTIC`` pair; a duplicate ``order_id`` within one run; primary and
    optimistic ``trade_id`` sets differing; an exit fill with no entry fill;
    ``exit_size > entry_size``; mixed ``side`` across a trade's filled entry
    legs; an exit leg on the same side as the entry.
    """


@dataclass(frozen=True)
class RoundTrip:
    """One completed round trip (matched entry+exit), all cents ``int``.

    ``direction`` is ``+1`` for a long (entry ``side == BUY``), ``−1`` for a
    short. ``net_optimistic_cents`` is ``None`` when the trade completed under
    ``PRIMARY`` but not under ``OPTIMISTIC``. ``adverse`` is ``True`` iff any
    *filled* leg of the primary group had ``adverse_selection`` set.
    """

    trade_id: str
    entry_ts_ns: int
    exit_ts_ns: int
    matched_size: int
    direction: int
    net_primary_cents: int
    net_stressed_cents: int
    net_optimistic_cents: int | None
    adverse: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "entry_ts_ns": self.entry_ts_ns,
            "exit_ts_ns": self.exit_ts_ns,
            "matched_size": self.matched_size,
            "direction": self.direction,
            "net_primary_cents": self.net_primary_cents,
            "net_stressed_cents": self.net_stressed_cents,
            "net_optimistic_cents": self.net_optimistic_cents,
            "adverse": self.adverse,
        }


@dataclass(frozen=True)
class OpenPosition:
    """An entry that filled but whose exit never did -- a position live at run
    end (spine AD-13(b) territory; prereg §2.2). Not part of any ``ModelPnL``."""

    trade_id: str
    open_size: int
    avg_entry_px_dbn: int
    entry_ts_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "open_size": self.open_size,
            "avg_entry_px_dbn": self.avg_entry_px_dbn,
            "entry_ts_ns": self.entry_ts_ns,
        }


@dataclass(frozen=True)
class ModelPnL:
    """Aggregated per-trade net P&L for one of the three models (int USD cents).

    ``net_cents`` is one entry per round trip in the report's chronological
    (``entry_ts_ns``, ``trade_id``) order. A "win" is ``net_cents > 0``; a
    "loss" is ``< 0``; a break-even (``== 0``) counts in ``n`` / ``sum`` but is
    neither.
    """

    net_cents: tuple[int, ...]
    n: int
    sum_net_cents: int
    gross_profit_cents: int
    gross_loss_cents: int
    wins: int
    losses: int

    @property
    def mean_net_cents(self) -> float | None:
        """``sum_net_cents / n``, or ``None`` when ``n == 0``."""
        return None if self.n == 0 else self.sum_net_cents / self.n

    @property
    def profit_factor(self) -> float | None:
        """``gross_profit_cents / -gross_loss_cents``. ``float("inf")`` for a
        profitable model with no losing trades; ``None`` only when ``n == 0``."""
        if self.n == 0:
            return None
        if self.gross_loss_cents == 0:
            return float("inf")
        return self.gross_profit_cents / -self.gross_loss_cents

    def to_dict(self) -> dict[str, Any]:
        return {
            "net_cents": list(self.net_cents),
            "n": self.n,
            "sum_net_cents": self.sum_net_cents,
            "gross_profit_cents": self.gross_profit_cents,
            "gross_loss_cents": self.gross_loss_cents,
            "wins": self.wins,
            "losses": self.losses,
            "mean_net_cents": self.mean_net_cents,
            "profit_factor": self.profit_factor,
        }


@dataclass(frozen=True)
class ThreeWayReport:
    """The §2.3 three-way P&L report (spine AD-14).

    ``round_trips`` (and every ``ModelPnL.net_cents``) is chronological by
    ``(entry_ts_ns, trade_id)`` -- **not** ``trade_id`` order, which is opaque
    (AD-12). ``primary`` / ``stressed`` derive from ``PRIMARY``; ``optimistic``
    is over the trades that round-tripped under **both** models.
    ``incomplete`` / ``partially_closed`` describe the ``PRIMARY`` run;
    ``optimistic_only_completed`` are ``trade_id``s the ``OPTIMISTIC`` run
    closed that ``PRIMARY`` did not.
    """

    round_trips: tuple[RoundTrip, ...]
    primary: ModelPnL
    stressed: ModelPnL
    optimistic: ModelPnL
    incomplete: tuple[OpenPosition, ...]
    partially_closed: tuple[tuple[str, int], ...]
    optimistic_only_completed: tuple[str, ...]
    config_primary: dict[str, Any]
    config_optimistic: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (``json.dumps`` succeeds)."""
        return {
            "round_trips": [rt.to_dict() for rt in self.round_trips],
            "primary": self.primary.to_dict(),
            "stressed": self.stressed.to_dict(),
            "optimistic": self.optimistic.to_dict(),
            "incomplete": [op.to_dict() for op in self.incomplete],
            "partially_closed": [[tid, qty] for tid, qty in self.partially_closed],
            "optimistic_only_completed": list(self.optimistic_only_completed),
            "config_primary": dict(self.config_primary),
            "config_optimistic": dict(self.config_optimistic),
        }


@dataclass(frozen=True)
class _Paired:
    """A completed round trip from one run (pre-cent-conversion metadata)."""

    trade_id: str
    entry_ts_ns: int
    exit_ts_ns: int
    matched_size: int
    direction: int
    net_primary_cents: int
    adverse: bool


@dataclass(frozen=True)
class _PairedRun:
    round_trips: tuple[_Paired, ...]
    open_positions: tuple[OpenPosition, ...]
    partially_closed: tuple[tuple[str, int], ...]


def _fee_per_contract(manifest: Mapping[str, Any], label: str) -> int:
    """Per-contract round-turn fee in USD cents, from the manifest (spine AD-24).

    ``manifest["config"]`` is the ``SimConfig`` dump; both fee fields are
    non-negative ``int`` cents. The fee model is per-contract round turn.
    """
    try:
        config = manifest["config"]
        exch_reg = config["exch_reg_fee_usd_cents"]
        commission = config["commission_usd_cents"]
    except (KeyError, TypeError) as exc:
        raise ReportError(f"{label} manifest is missing a fee field: {exc}") from exc

    def _cents(name: str, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ReportError(
                f"{label} manifest {name} must be a non-negative int, got {value!r}"
            )
        return value

    return _cents("exch_reg_fee_usd_cents", exch_reg) + _cents(
        "commission_usd_cents", commission
    )


def _check_manifest_pair(
    primary_manifest: Mapping[str, Any], optimistic_manifest: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """The two manifests must be a ``PRIMARY`` + ``OPTIMISTIC`` pair (AD-14/AD-15)
    -- guards against a swap or two of the same. Returns the two ``config``
    sub-dicts."""
    try:
        cfg_p = primary_manifest["config"]
        cfg_o = optimistic_manifest["config"]
        qm_p = cfg_p["queue_model"]
        qm_o = cfg_o["queue_model"]
    except (KeyError, TypeError) as exc:
        raise ReportError(f"a manifest is missing config.queue_model: {exc}") from exc
    if qm_p != "back_of_queue" or qm_o != "time_priority":
        raise ReportError(
            "expected (primary=back_of_queue, optimistic=time_priority); got "
            f"(primary={qm_p!r}, optimistic={qm_o!r}) -- swapped or wrong configs"
        )
    return cfg_p, cfg_o


def _pair_run(outcomes: list[OrderOutcome], fee_per_contract: int) -> _PairedRun:
    """Group ``outcomes`` by ``trade_id``, classify each, compute model-(a) net
    cents per completed round trip (spine AD-12).

    Every sequence built here is in ``sorted(trade_id)`` order; ``build_report``
    re-orders the round trips chronologically.
    """
    seen_order_ids: set[str] = set()
    groups: dict[str, list[OrderOutcome]] = {}
    for outcome in outcomes:
        if outcome.order_id in seen_order_ids:
            raise ReportError(f"duplicate order_id {outcome.order_id!r} in a run")
        seen_order_ids.add(outcome.order_id)
        groups.setdefault(outcome.trade_id, []).append(outcome)

    round_trips: list[_Paired] = []
    open_positions: list[OpenPosition] = []
    partially_closed: list[tuple[str, int]] = []

    for tid in sorted(groups):
        legs = groups[tid]
        entries = [o for o in legs if o.leg is Leg.ENTRY]
        exits = [o for o in legs if o.leg is Leg.EXIT]

        entry_size = sum(f.size for o in entries for f in o.fills)
        exit_size = sum(f.size for o in exits for f in o.fills)

        if entry_size == 0 and exit_size == 0:
            continue
        if entry_size == 0:
            raise ReportError(f"trade_id {tid!r}: exit fills with no entry fill")

        entry_notional_dbn = sum(f.px_dbn * f.size for o in entries for f in o.fills)
        entry_ts_ns = min(f.ts_ns for o in entries for f in o.fills)

        entry_sides = {o.side for o in entries if o.fills}
        if len(entry_sides) != 1:
            raise ReportError(f"trade_id {tid!r}: mixed side across filled entry legs")
        (entry_side,) = entry_sides
        direction = 1 if entry_side is Side.BUY else -1

        if exit_size == 0:
            open_positions.append(
                OpenPosition(
                    trade_id=tid,
                    open_size=entry_size,
                    avg_entry_px_dbn=entry_notional_dbn // entry_size,
                    entry_ts_ns=entry_ts_ns,
                )
            )
            continue
        if exit_size > entry_size:
            raise ReportError(
                f"trade_id {tid!r}: exit_size {exit_size} > entry_size {entry_size}"
            )

        exit_sides = {o.side for o in exits if o.fills}
        if any(s is entry_side for s in exit_sides):
            raise ReportError(
                f"trade_id {tid!r}: an exit leg is on the same side as the entry"
            )

        matched_size = min(entry_size, exit_size)
        if entry_size > exit_size:
            partially_closed.append((tid, entry_size - exit_size))

        exit_notional_dbn = sum(f.px_dbn * f.size for o in exits for f in o.fills)
        exit_ts_ns = max(f.ts_ns for o in exits for f in o.fills)

        if entry_size == exit_size:  # full close -- exact, no averaging
            gross_dbn = direction * (exit_notional_dbn - entry_notional_dbn)
        else:  # partial close -- the matched fraction of the entry notional
            entry_notional_matched = entry_notional_dbn * matched_size // entry_size
            gross_dbn = direction * (exit_notional_dbn - entry_notional_matched)

        net_primary_cents = _to_cents(gross_dbn) - fee_per_contract * matched_size
        adverse = any(o.adverse_selection for o in legs if o.fills)
        round_trips.append(
            _Paired(
                trade_id=tid,
                entry_ts_ns=entry_ts_ns,
                exit_ts_ns=exit_ts_ns,
                matched_size=matched_size,
                direction=direction,
                net_primary_cents=net_primary_cents,
                adverse=adverse,
            )
        )

    return _PairedRun(
        round_trips=tuple(round_trips),
        open_positions=tuple(open_positions),
        partially_closed=tuple(partially_closed),
    )


def _model_pnl(net_cents: tuple[int, ...]) -> ModelPnL:
    return ModelPnL(
        net_cents=net_cents,
        n=len(net_cents),
        sum_net_cents=sum(net_cents),
        gross_profit_cents=sum(c for c in net_cents if c > 0),
        gross_loss_cents=sum(c for c in net_cents if c < 0),
        wins=sum(1 for c in net_cents if c > 0),
        losses=sum(1 for c in net_cents if c < 0),
    )


def build_report(
    primary_outcomes: Iterable[OrderOutcome],
    primary_manifest: Mapping[str, Any],
    optimistic_outcomes: Iterable[OrderOutcome],
    optimistic_manifest: Mapping[str, Any],
) -> ThreeWayReport:
    """Build the §2.3 three-way P&L report (spine AD-14 / AD-24).

    Args:
        primary_outcomes: the ``config.PRIMARY`` run's ``OrderOutcome`` log.
        primary_manifest: that run's ``Manifest.to_dict()`` (fees + queue_model
            under ``["config"]``).
        optimistic_outcomes: the ``config.OPTIMISTIC`` run's ``OrderOutcome``
            log.
        optimistic_manifest: that run's ``Manifest.to_dict()``.

    Returns:
        A :class:`ThreeWayReport` -- chronological ``round_trips`` plus models
        (a) primary, (b) stressed, (c) optimistic (both-completed subset) and
        the primary run's open / partially-closed diagnostics.

    Raises:
        ReportError: see the class docstring.
    """
    primary_list = list(primary_outcomes)
    optimistic_list = list(optimistic_outcomes)

    cfg_p, cfg_o = _check_manifest_pair(primary_manifest, optimistic_manifest)
    primary_fee = _fee_per_contract(primary_manifest, "primary")
    optimistic_fee = _fee_per_contract(optimistic_manifest, "optimistic")

    primary_tids = {o.trade_id for o in primary_list}
    optimistic_tids = {o.trade_id for o in optimistic_list}
    if primary_tids != optimistic_tids:
        raise ReportError(
            "primary and optimistic trade_id sets differ: "
            f"primary-only={sorted(primary_tids - optimistic_tids)}, "
            f"optimistic-only={sorted(optimistic_tids - primary_tids)}"
        )

    primary_run = _pair_run(primary_list, primary_fee)
    optimistic_run = _pair_run(optimistic_list, optimistic_fee)
    optimistic_net_by_tid = {
        p.trade_id: p.net_primary_cents for p in optimistic_run.round_trips
    }

    ordered = sorted(primary_run.round_trips, key=lambda p: (p.entry_ts_ns, p.trade_id))
    round_trips = tuple(
        RoundTrip(
            trade_id=p.trade_id,
            entry_ts_ns=p.entry_ts_ns,
            exit_ts_ns=p.exit_ts_ns,
            matched_size=p.matched_size,
            direction=p.direction,
            net_primary_cents=p.net_primary_cents,
            net_stressed_cents=p.net_primary_cents
            - 2 * TICK_VALUE_CENTS * p.matched_size,
            net_optimistic_cents=optimistic_net_by_tid.get(p.trade_id),
            adverse=p.adverse,
        )
        for p in ordered
    )

    primary_completed = {p.trade_id for p in primary_run.round_trips}
    optimistic_only_completed = tuple(
        sorted(tid for tid in optimistic_net_by_tid if tid not in primary_completed)
    )

    return ThreeWayReport(
        round_trips=round_trips,
        primary=_model_pnl(tuple(rt.net_primary_cents for rt in round_trips)),
        stressed=_model_pnl(tuple(rt.net_stressed_cents for rt in round_trips)),
        optimistic=_model_pnl(
            tuple(
                rt.net_optimistic_cents
                for rt in round_trips
                if rt.net_optimistic_cents is not None
            )
        ),
        incomplete=primary_run.open_positions,
        partially_closed=primary_run.partially_closed,
        optimistic_only_completed=optimistic_only_completed,
        config_primary=dict(cfg_p),
        config_optimistic=dict(cfg_o),
    )
