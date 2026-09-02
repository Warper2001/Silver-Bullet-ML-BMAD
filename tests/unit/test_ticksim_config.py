"""Unit tests for ``src.ticksim.config`` -- seal-bound presets and constants.

Guards the architecture spine:
  * AD-15 -- ``PRIMARY`` / ``OPTIMISTIC`` encode pre-registration §2.1 exactly.
  * AD-24 / AD-27 -- named constants carry the seal's values.
  * ``SimConfig`` is frozen.
"""

import pytest
from pydantic import ValidationError

from src.ticksim import config
from src.ticksim.config import (
    MNQ_TICK_DBN,
    OPTIMISTIC,
    PRIMARY,
    QueueModel,
    SimConfig,
    ticks_to_dbn,
)


class TestPrimaryPreset:
    """``PRIMARY`` field-by-field vs pre-registration §2.1 (primary column)."""

    def test_queue_model_is_back_of_queue(self) -> None:
        assert PRIMARY.queue_model is QueueModel.BACK_OF_QUEUE

    def test_latency_is_250ms_in_ns(self) -> None:
        assert PRIMARY.latency_ns == 250_000_000

    def test_exch_reg_fee_is_72_cents(self) -> None:
        # §2.1: $0.72 exchange+regulatory round turn.
        assert PRIMARY.exch_reg_fee_usd_cents == 72

    def test_commission_is_58_cents(self) -> None:
        # §2.1: $0.58 RT seal-default commission (Tradovate base).
        assert PRIMARY.commission_usd_cents == 58

    def test_no_own_impact(self) -> None:
        assert PRIMARY.own_impact is False

    def test_seed_is_deterministic_zero(self) -> None:
        assert PRIMARY.seed == 0


class TestOptimisticPreset:
    """``OPTIMISTIC`` field-by-field vs pre-registration §2.1 (secondary column)."""

    def test_queue_model_is_time_priority(self) -> None:
        assert OPTIMISTIC.queue_model is QueueModel.TIME_PRIORITY

    def test_latency_is_50ms_in_ns(self) -> None:
        assert OPTIMISTIC.latency_ns == 50_000_000

    def test_fees_match_primary(self) -> None:
        # §2.1 does not vary fees between the two models.
        assert OPTIMISTIC.exch_reg_fee_usd_cents == 72
        assert OPTIMISTIC.commission_usd_cents == 58

    def test_no_own_impact(self) -> None:
        assert OPTIMISTIC.own_impact is False


class TestPresetDrift:
    """A single assertion capturing every seal-bound preset value (spine AD-15).

    If any preset field drifts from §2.1, this fails loudly with the full dump.
    """

    def test_primary_full_dump(self) -> None:
        assert PRIMARY.model_dump() == {
            "queue_model": QueueModel.BACK_OF_QUEUE,
            "latency_ns": 250_000_000,
            "exch_reg_fee_usd_cents": 72,
            "commission_usd_cents": 58,
            "seed": 0,
            "own_impact": False,
        }

    def test_optimistic_full_dump(self) -> None:
        assert OPTIMISTIC.model_dump() == {
            "queue_model": QueueModel.TIME_PRIORITY,
            "latency_ns": 50_000_000,
            "exch_reg_fee_usd_cents": 72,
            "commission_usd_cents": 58,
            "seed": 0,
            "own_impact": False,
        }


class TestNamedConstants:
    """Named constants vs the seal (spine AD-24, AD-27)."""

    def test_dollars_per_index_point(self) -> None:
        assert config.DOLLARS_PER_INDEX_POINT == 2

    def test_mnq_tick_dbn(self) -> None:
        # 0.25 index point in DBN 1e-9 fixed-point units.
        assert config.MNQ_TICK_DBN == 250_000_000

    def test_parity_mae_max_ticks(self) -> None:
        assert config.PARITY_MAE_MAX_TICKS == 1.0

    def test_parity_p90_max_ticks(self) -> None:
        assert config.PARITY_P90_MAX_TICKS == 2.0

    def test_parity_signed_bias_max_ticks(self) -> None:
        assert config.PARITY_SIGNED_BIAS_MAX_TICKS == 0.25

    def test_part_a_min_n(self) -> None:
        assert config.PART_A_MIN_N == 28

    def test_part_b_min_orders(self) -> None:
        assert config.PART_B_MIN_ORDERS == 1000

    def test_max_transient_cross_ns(self) -> None:
        assert config.MAX_TRANSIENT_CROSS_NS == 50_000_000

    def test_stale_cross_max_ticks(self) -> None:
        # tolerance parameter (not decision-bearing): derived from the widest
        # cross seen in a clean front-month book (~17 ticks), ~3x that.
        assert config.STALE_CROSS_MAX_TICKS == 50

    def test_adverse_selection_window_ns(self) -> None:
        # prereg §2.1 / spine AD-28: book state 1 s after a passive fill.
        assert config.ADVERSE_SELECTION_WINDOW_NS == 1_000_000_000

    def test_queue_model_wire_values(self) -> None:
        assert QueueModel.BACK_OF_QUEUE.value == "back_of_queue"
        assert QueueModel.TIME_PRIORITY.value == "time_priority"


class TestTicksToDbn:
    """``ticks_to_dbn`` -- the single tick->DBN conversion point (spine AD-27)."""

    def test_one_tick(self) -> None:
        assert ticks_to_dbn(1.0) == MNQ_TICK_DBN

    def test_seal_tolerances(self) -> None:
        assert ticks_to_dbn(config.PARITY_MAE_MAX_TICKS) == 250_000_000
        assert ticks_to_dbn(config.PARITY_P90_MAX_TICKS) == 500_000_000
        assert ticks_to_dbn(config.PARITY_SIGNED_BIAS_MAX_TICKS) == 62_500_000

    def test_returns_int(self) -> None:
        result = ticks_to_dbn(0.25)
        assert isinstance(result, int) and not isinstance(result, bool)


class TestSimConfigFrozen:
    """``SimConfig`` is immutable (spine AD-15: ``frozen=True``)."""

    def test_attribute_assignment_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PRIMARY.latency_ns = 1  # type: ignore[misc]

    def test_optimistic_attribute_assignment_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OPTIMISTIC.queue_model = QueueModel.BACK_OF_QUEUE  # type: ignore[misc]

    def test_model_copy_does_not_mutate_original(self) -> None:
        # model_copy(update=...) returns a new object; the seal-bound constant
        # is untouched.
        derived = PRIMARY.model_copy(update={"commission_usd_cents": 0})
        assert derived.commission_usd_cents == 0
        assert PRIMARY.commission_usd_cents == 58

    def test_float_into_int_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SimConfig(
                queue_model=QueueModel.BACK_OF_QUEUE,
                latency_ns=250_000_000.0,  # type: ignore[arg-type]
                exch_reg_fee_usd_cents=72,
                commission_usd_cents=58,
                seed=0,
            )

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SimConfig(
                queue_model=QueueModel.BACK_OF_QUEUE,
                latency_ns=-1,
                exch_reg_fee_usd_cents=72,
                commission_usd_cents=58,
                seed=0,
            )

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SimConfig(
                queue_model=QueueModel.BACK_OF_QUEUE,
                latency_ns=0,
                exch_reg_fee_usd_cents=0,
                commission_usd_cents=0,
                seed=-5,
            )

    def test_non_bool_own_impact_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SimConfig(
                queue_model=QueueModel.BACK_OF_QUEUE,
                latency_ns=0,
                exch_reg_fee_usd_cents=0,
                commission_usd_cents=0,
                seed=0,
                own_impact=1,  # type: ignore[arg-type]
            )
