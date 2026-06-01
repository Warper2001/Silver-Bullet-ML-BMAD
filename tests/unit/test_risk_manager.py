"""Unit tests for Story 5-1: Intraday Trailing Drawdown Floor Tracker.

Tests RiskManager.check_trailing_dd(), AccountConfig trailing-DD fields,
state-dict round-trip, and EOD mode.
"""
import pytest
from datetime import datetime, date
from unittest.mock import patch

import pytz

ET_TZ = pytz.timezone("America/New_York")


def _et(year=2026, month=1, day=6, hour=10, minute=0) -> datetime:
    return ET_TZ.localize(datetime(year, month, day, hour, minute))


def _make_account_config(**overrides):
    from src.research.tier2_streaming_working import AccountConfig
    defaults = dict(
        account_id="SIM001",
        execution_mode="sim",
        symbol="MNQM26",
        point_value=2.0,
        tick_size=0.25,
        contracts=5,
        dd_type="intraday",
        topstep_trailing_dd_amount=2000.0,
        trailing_dd_alert_pct=0.10,
        starting_equity=50000.0,
    )
    defaults.update(overrides)
    return AccountConfig(**defaults)


# ---------------------------------------------------------------------------
# Task 1: AccountConfig has trailing-DD fields
# ---------------------------------------------------------------------------

class TestAccountConfigTrailingDDFields:
    def test_default_fields_exist(self):
        cfg = _make_account_config()
        assert cfg.dd_type == "intraday"
        assert cfg.topstep_trailing_dd_amount == pytest.approx(2000.0)
        assert cfg.trailing_dd_alert_pct == pytest.approx(0.10)
        assert cfg.starting_equity == pytest.approx(50000.0)

    def test_eod_mode_accepted(self):
        cfg = _make_account_config(dd_type="eod")
        assert cfg.dd_type == "eod"


# ---------------------------------------------------------------------------
# Task 2 + 3: TrailingDDTracker via RiskManager
# ---------------------------------------------------------------------------

class TestTrailingDDTracker:
    def test_trailing_floor_initializes_from_starting_equity(self):
        """AC#1: first bar sets floor = starting_equity - dd_amount."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        cfg = _make_account_config()
        bar_et = _et()
        with patch.object(StatePersistence, "save_state"):
            rm.check_trailing_dd(cfg.starting_equity, bar_et, cfg)
        assert rm.trailing_floor == pytest.approx(48000.0)

    def test_trailing_floor_rises_on_unrealized_peak(self):
        """AC#2: equity peaks above prior high → floor advances."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        cfg = _make_account_config()
        bar_et = _et()
        with patch.object(StatePersistence, "save_state"):
            rm.check_trailing_dd(50000.0, bar_et, cfg)  # init
            rm.check_trailing_dd(50800.0, bar_et, cfg)  # new equity peak
        assert rm.trailing_floor == pytest.approx(48800.0)

    def test_trailing_floor_does_not_fall(self):
        """AC#2: floor never decreases even if equity drops after peak."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        cfg = _make_account_config()
        bar_et = _et()
        with patch.object(StatePersistence, "save_state"):
            rm.check_trailing_dd(50800.0, bar_et, cfg)  # peak
            rm.check_trailing_dd(49000.0, bar_et, cfg)  # equity drop
        assert rm.trailing_floor == pytest.approx(48800.0)

    def test_breach_halts_trading(self):
        """AC#3: equity at or below floor → is_halted True, returns True."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        cfg = _make_account_config()
        bar_et = _et()
        with patch.object(StatePersistence, "save_state"):
            rm.check_trailing_dd(50800.0, bar_et, cfg)   # floor now 48800
            breached = rm.check_trailing_dd(48700.0, bar_et, cfg)  # below floor
        assert breached is True
        assert rm.is_halted is True

    def test_no_breach_above_floor(self):
        """No breach when equity is safely above floor."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        cfg = _make_account_config()
        bar_et = _et()
        with patch.object(StatePersistence, "save_state"):
            result = rm.check_trailing_dd(50000.0, bar_et, cfg)
        assert result is False
        assert rm.is_halted is False

    def test_alert_logged_near_floor(self, caplog):
        """AC#4: equity within trailing_dd_alert_pct of floor → TRAILING_DD_ALERT logged."""
        import logging
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        cfg = _make_account_config(trailing_dd_alert_pct=0.10)
        bar_et = _et()
        # Floor = 48000. Alert zone: cushion < 10% of 2000 = < $200 above floor.
        # Set equity = 48150 → cushion = 150 < 200 → should alert.
        with patch.object(StatePersistence, "save_state"):
            with caplog.at_level(logging.WARNING):
                rm.check_trailing_dd(48150.0, bar_et, cfg)
        assert "TRAILING_DD_ALERT" in caplog.text

    def test_no_alert_far_from_floor(self, caplog):
        """No alert when comfortably above floor."""
        import logging
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        cfg = _make_account_config()
        bar_et = _et()
        with patch.object(StatePersistence, "save_state"):
            with caplog.at_level(logging.WARNING):
                rm.check_trailing_dd(50000.0, bar_et, cfg)
        assert "TRAILING_DD_ALERT" not in caplog.text

    def test_eod_mode_floor_not_updated_intrabar(self):
        """AC#5: dd_type='eod' → floor stable across intrabar equity peaks."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        cfg = _make_account_config(dd_type="eod")
        bar_et = _et()
        with patch.object(StatePersistence, "save_state"):
            rm.check_trailing_dd(50000.0, bar_et, cfg)  # init floor
            initial_floor = rm.trailing_floor
            rm.check_trailing_dd(52000.0, bar_et, cfg)  # big equity peak
        # Floor must NOT have moved (EOD mode)
        assert rm.trailing_floor == pytest.approx(initial_floor)


# ---------------------------------------------------------------------------
# Task 4: Persist and restore trailing_floor
# ---------------------------------------------------------------------------

class TestTrailingDDPersistence:
    def test_trailing_floor_survives_crash_recovery(self):
        """AC#6: to_state_dict / restore_from_state round-trip preserves floor."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm1 = RiskManager()
        cfg = _make_account_config()
        bar_et = _et()
        with patch.object(StatePersistence, "save_state"):
            rm1.check_trailing_dd(50800.0, bar_et, cfg)   # floor = 48800
        state = rm1.to_state_dict()
        assert "trailing_floor" in state

        rm2 = RiskManager()
        today = bar_et.date()
        rm2.restore_from_state(state, today)
        assert rm2.trailing_floor == pytest.approx(48800.0)

    def test_trailing_floor_reset_on_new_day(self):
        """AC#6: restored state from prior day does NOT load stale floor."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm1 = RiskManager()
        cfg = _make_account_config()
        bar_et = _et(day=6)
        with patch.object(StatePersistence, "save_state"):
            rm1.check_trailing_dd(50800.0, bar_et, cfg)
        state = rm1.to_state_dict()

        rm2 = RiskManager()
        tomorrow = _et(day=7).date()
        rm2.restore_from_state(state, tomorrow)
        # Floor not restored for a different day — remains at sentinel value
        assert rm2.trailing_floor is None

    def test_to_state_dict_contains_equity_high_water(self):
        """State dict round-trip also preserves the equity high-water mark."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        cfg = _make_account_config()
        bar_et = _et()
        with patch.object(StatePersistence, "save_state"):
            rm.check_trailing_dd(51000.0, bar_et, cfg)
        state = rm.to_state_dict()
        assert "equity_high_water" in state
        assert state["equity_high_water"] == pytest.approx(51000.0)
