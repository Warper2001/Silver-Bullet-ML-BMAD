"""Simulator contract layer: ``SimConfig``, seal-bound presets, named constants.

This module is a leaf (spine AD-7): it imports nothing from ``src.ticksim`` and
nothing from any other ``src`` package. ``fills.py`` later maps
``QueueModel -> BackOfQueueModel()/TimePriorityModel()``; ``report.py`` is the
sole consumer of the monetary / multiplier fields (spine AD-24).

Sources of truth:
  * ``_bmad-output/preregistration_tick_data_infrastructure.md`` §2.1 (preset values)
  * architecture spine AD-15 (presets are seal-bound), AD-24 (money is report-only),
    AD-27 (parity thresholds are seal-bound).

Money is stored as an integer number of **USD cents** (spine AD-10 forbids
``float`` before the ``OrderOutcome`` log; ``report.py`` converts, spine AD-24).
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class QueueModel(str, Enum):
    """Which queue-position model a ``SimConfig`` selects (spine AD-5, AD-22).

    An enum, not a strategy object, so ``config.py`` stays a leaf (spine AD-7).
    ``fills.py`` maps each member to its model class.
    """

    BACK_OF_QUEUE = "back_of_queue"  # prereg §2.1 primary column
    TIME_PRIORITY = "time_priority"  # prereg §2.1 secondary column ("optimistic")


class SimConfig(BaseModel):
    """Frozen run configuration (spine AD-15).

    Every ``PRIMARY`` / ``OPTIMISTIC`` field value is pinned to
    pre-registration §2.1; changing a value is a pre-registration violation
    absent a new seal amendment (spine AD-15). A study that overrides a value
    builds a *derived* config, records it in the run manifest, and its output
    is §6-secondary only.
    """

    model_config = ConfigDict(frozen=True)

    queue_model: QueueModel = Field(
        ..., description="Queue-position model (prereg §2.1)."
    )
    latency_ns: int = Field(
        ...,
        strict=True,
        ge=0,
        description=(
            "Fixed order-intent -> exchange-arrival delay in nanoseconds "
            "(prereg §2.1). The seal's round-trip figure is applied as the "
            "full submit->arrival delay (conservative reading; flagged Ask-First "
            "in the spec)."
        ),
    )
    exch_reg_fee_usd_cents: int = Field(
        ...,
        strict=True,
        ge=0,
        description=(
            "Exchange + regulatory fee per round turn, in USD cents "
            "(prereg §2.1). Read only by report.py (spine AD-24)."
        ),
    )
    commission_usd_cents: int = Field(
        ...,
        strict=True,
        ge=0,
        description=(
            "Broker commission per round turn, in USD cents (prereg §2.1). "
            "Read only by report.py (spine AD-24)."
        ),
    )
    seed: int = Field(
        ...,
        strict=True,
        ge=0,
        description=(
            "Sole entropy source for the run (spine AD-11). Not seal-bound; "
            "the seal constrains the fill model, not the RNG seed."
        ),
    )
    own_impact: bool = Field(
        default=False,
        strict=True,
        description=(
            "Own-order market-impact model (prereg §2.1: 'None -- assumed "
            "negligible at 1-5 micro contracts'). The §2.3 ±1-tick stress is a "
            "report-layer P&L transform, not this flag (spine AD-14)."
        ),
    )


# --- Seal-bound presets (pre-registration §2.1; spine AD-15) ----------------
#
# Changing any *value* below is a pre-registration violation absent a new seal
# amendment (spine AD-15). Every field carries its §2.1 citation.

PRIMARY: SimConfig = SimConfig(
    queue_model=QueueModel.BACK_OF_QUEUE,  # §2.1 primary: back of queue
    latency_ns=250_000_000,  # §2.1 primary: fixed 250 ms round trip (retail, no colo)
    exch_reg_fee_usd_cents=72,  # §2.1: $0.72 exchange+regulatory round turn (seal-frozen)
    # §2.1: $0.58 RT -- the seal *default* ("a commission the user sets at seal
    # time"). User-configurable: a study overriding it builds a derived config,
    # recorded in the manifest, §6-secondary only (spine AD-15). Not frozen the
    # way queue_model / latency_ns / the exch+reg fee are.
    commission_usd_cents=58,
    seed=0,  # determinism seed (spine AD-11); not seal-bound
    own_impact=False,  # §2.1 primary: no own-order market impact
)
"""Decision-bearing model (pre-registration §2.1 primary column; spine AD-15)."""

OPTIMISTIC: SimConfig = SimConfig(
    queue_model=QueueModel.TIME_PRIORITY,  # §2.1 secondary: time-priority ("optimistic")
    latency_ns=50_000_000,  # §2.1 secondary: fixed 50 ms ("near-colo")
    exch_reg_fee_usd_cents=72,  # §2.1: fees are not varied between models
    commission_usd_cents=58,  # §2.1: fees are not varied between models
    seed=0,  # determinism seed (spine AD-11); not seal-bound
    own_impact=False,  # §2.1: own-impact not modelled in the secondary model either
)
"""Context-only model (pre-registration §2.1 secondary column; never decision-bearing)."""


# --- Named seal-cited constants (spine AD-24, AD-27) ------------------------

DOLLARS_PER_INDEX_POINT: int = 2
"""MNQ dollar value of one index point (spine AD-24; tick = 0.25 pt = $0.50).

Named, seal-cited (prereg §2.1 fee basis / §A9.4 '2 ticks (0.50 index pt)').
``report.py`` is its only consumer (spine AD-24)."""

MNQ_TICK_DBN: int = 250_000_000
"""One MNQ tick (0.25 index point) in DBN 1e-9 fixed-point units (spine AD-10).

0.25 * 1e9 = 250_000_000. All parity tolerances are expressed in ticks and
scaled by this constant (spine AD-27)."""

PARITY_MAE_MAX_TICKS: float = 1.0
"""Part A: mean absolute fill-price error tolerance, in ticks
(prereg §A8.2 Part A / §3; spine AD-27). Changing this needs a seal amendment."""

PARITY_P90_MAX_TICKS: float = 2.0
"""Part A: 90th-percentile absolute fill-price error tolerance, in ticks
(prereg §A8.2 Part A / §3; spine AD-27). Changing this needs a seal amendment."""

PARITY_SIGNED_BIAS_MAX_TICKS: float = 0.25
"""Part A: mean signed fill-price error tolerance (±), in ticks
(prereg §A8.2 Part A / §3; spine AD-27). Changing this needs a seal amendment."""

PART_A_MIN_N: int = 28
"""Part A: minimum number of real broker fills required to run the calibration
(prereg §A8.2 Part A: 'Minimum to run Part A: N >= 28'; spine AD-27)."""

PART_B_MIN_ORDERS: int = 1000
"""Part B: minimum number of synthetic orders in the invariant battery
(prereg §A8.2 Part B: 'Generate >= 1,000 synthetic orders'; spine AD-27)."""

MAX_TRANSIENT_CROSS_NS: int = 50_000_000
"""Longest tolerated transient crossed market (bid >= ask), in nanoseconds
(spine AD-9, AD-27; prereg §A9.3 'no *persistent* cross (> a few ms)'). 50 ms.
A longer-lived cross is a ``BookInconsistency``. Changing this needs a seal
amendment."""

ADVERSE_SELECTION_WINDOW_NS: int = 1_000_000_000
"""Look-ahead window for the ``adverse_selection`` marker, in nanoseconds
(prereg §2.1 / spine AD-28: 'book state 1 s after' a passive fill). 1 s.
``sim.py`` keeps the tracker order mutable for this one field until
``fill_ts_ns + ADVERSE_SELECTION_WINDOW_NS`` or run end. Changing this needs a
seal amendment."""


def ticks_to_dbn(ticks: float) -> int:
    """Convert a tick-denominated tolerance to DBN 1e-9 fixed-point units.

    The single place parity code converts a tick tolerance (``PARITY_*_TICKS``,
    spine AD-27) into the DBN price units the book and fills use. Rounded to the
    nearest integer DBN unit -- there is no sub-unit price (spine AD-10).

    Args:
        ticks: A tolerance in MNQ ticks (may be fractional, e.g. 0.25).

    Returns:
        The equivalent number of DBN 1e-9 fixed-point units.
    """
    return round(ticks * MNQ_TICK_DBN)
