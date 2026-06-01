"""Unit tests for Stories 5-1 and 5-2.

5-1: RiskManager.check_trailing_dd(), AccountConfig trailing-DD fields,
     state-dict round-trip, and EOD mode.
5-2: Consistency rule monitor and auto position-size reduction.
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


# ---------------------------------------------------------------------------
# Story 5-2: Consistency Rule Monitor
# ---------------------------------------------------------------------------

def _make_cfg_consistency(**overrides):
    from src.research.tier2_streaming_working import AccountConfig
    defaults = dict(
        account_id="SIM001",
        execution_mode="sim",
        symbol="MNQM26",
        point_value=2.0,
        tick_size=0.25,
        contracts=5,
        consistency_alert_pct=0.40,
        consistency_reduce_pct=0.45,
        consistency_hard_limit_pct=0.50,
    )
    defaults.update(overrides)
    return AccountConfig(**defaults)


class TestConsistencyAccountConfigFields:
    def test_default_fields_exist(self):
        cfg = _make_cfg_consistency()
        assert cfg.consistency_alert_pct == pytest.approx(0.40)
        assert cfg.consistency_reduce_pct == pytest.approx(0.45)
        assert cfg.consistency_hard_limit_pct == pytest.approx(0.50)


class TestConsistencyEvaluator:
    def test_alert_logged_at_threshold(self, caplog):
        """AC#2: ratio ≥ alert_pct → CONSISTENCY_ALERT logged."""
        import logging
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        cfg = _make_cfg_consistency(consistency_alert_pct=0.40)
        # Simulate two previous days: [520, 850] total=1370; today so far = 600
        # all_pnls = [520, 850, 600] → best=850, total=1970 → 43.1% ≥ 40% → alert
        rm._session_pnls = [520.0, 850.0]
        rm._daily_pnl = 600.0
        with patch.object(StatePersistence, "save_state"):
            with caplog.at_level(logging.WARNING):
                ratio, _ = rm.check_consistency(cfg)
        assert "CONSISTENCY_ALERT" in caplog.text
        assert ratio > 40.0

    def test_size_reduction_triggered_at_reduce_pct(self):
        """AC#3: ratio ≥ reduce_pct → consistency_size_reduced = True."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        cfg = _make_cfg_consistency(consistency_reduce_pct=0.45)
        # [520, 850, 800] → best=850, total=2170 → 39.2%: NOT enough
        # [520, 850, 950] → best=950, total=2320 → 40.9%: still not enough
        # [520, 850, 1100] → best=1100, total=2470 → 44.5%: not enough
        # [520, 850, 1300] → best=1300, total=2670 → 48.7% ≥ 45% → reduce
        rm._session_pnls = [520.0, 850.0]
        rm._daily_pnl = 1300.0
        with patch.object(StatePersistence, "save_state"):
            ratio, should_reduce = rm.check_consistency(cfg)
        assert should_reduce is True
        assert rm.consistency_size_reduced is True

    def test_size_reduction_does_not_reverse_intraday(self):
        """AC#3: once reduced, stays reduced even if ratio dips below threshold."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        cfg = _make_cfg_consistency(consistency_reduce_pct=0.45)
        rm._session_pnls = [520.0, 850.0]
        rm._daily_pnl = 1300.0  # triggers reduction
        with patch.object(StatePersistence, "save_state"):
            rm.check_consistency(cfg)
        assert rm.consistency_size_reduced is True
        rm._daily_pnl = 500.0  # hypothetically lower
        with patch.object(StatePersistence, "save_state"):
            rm.check_consistency(cfg)
        assert rm.consistency_size_reduced is True  # still reduced

    def test_size_reduction_resets_on_new_day(self):
        """AC#4: new calendar day → _consistency_size_reduced cleared."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        cfg = _make_cfg_consistency()
        bar_day1 = _et(day=6)
        bar_day2 = _et(day=7)
        rm._session_pnls = [1000.0]
        rm._daily_pnl = 1100.0
        rm._consistency_size_reduced = True
        rm._last_trading_date = bar_day1.date()
        # Simulate day transition inside check_and_update
        with patch.object(StatePersistence, "save_state"):
            rm.check_and_update(bar_day2, max_daily_loss=-750.0)
        assert rm.consistency_size_reduced is False

    def test_session_pnls_accumulate_across_days(self):
        """AC#4: previous day PnL is appended to session_pnls on day rollover."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        bar_day1 = _et(day=6)
        bar_day2 = _et(day=7)
        rm._daily_pnl = 400.0
        rm._last_trading_date = bar_day1.date()
        with patch.object(StatePersistence, "save_state"):
            rm.check_and_update(bar_day2, max_daily_loss=-750.0)
        assert 400.0 in rm._session_pnls

    def test_consistency_state_persists_round_trip(self):
        """AC#4/5: session_pnls and consistency_size_reduced survive state-dict round-trip."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm1 = RiskManager()
        rm1._session_pnls = [520.0, 850.0]
        rm1._consistency_size_reduced = True
        rm1._last_trading_date = _et(day=6).date()
        rm1._daily_pnl = 300.0
        state = rm1.to_state_dict()
        assert "session_pnls" in state
        assert "consistency_size_reduced" in state

        rm2 = RiskManager()
        rm2.restore_from_state(state, _et(day=6).date())
        assert rm2._session_pnls == [520.0, 850.0]
        assert rm2.consistency_size_reduced is True
