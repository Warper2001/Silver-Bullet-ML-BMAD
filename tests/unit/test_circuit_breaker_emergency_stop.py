"""Unit tests for RiskManager (Story 4-3, AC#1-#3, #6) and emergency stop CLI (AC#4, #5)."""
import sys
import pytest
import asyncio
from datetime import datetime, date, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path

import pytz

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ET_TZ = pytz.timezone("America/New_York")

def _et(year=2026, month=1, day=6, hour=10, minute=0) -> datetime:
    """Return a timezone-aware datetime in ET."""
    return ET_TZ.localize(datetime(year, month, day, hour, minute))


# ---------------------------------------------------------------------------
# TestRiskManager
# ---------------------------------------------------------------------------

class TestRiskManager:
    def test_circuit_breaker_trips_when_pnl_below_threshold(self):
        """AC#1: daily_pnl below max_daily_loss → is_halted True, check returns True."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        rm.register_close(-800.0)
        with patch.object(StatePersistence, "save_state"):
            result = rm.check_and_update(_et(), max_daily_loss=-750.0)
        assert result is True
        assert rm.is_halted is True
        assert rm.daily_pnl == pytest.approx(-800.0)

    def test_circuit_breaker_does_not_trip_above_threshold(self):
        """AC#1: daily_pnl above threshold → is_halted False, check returns False."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        rm.register_close(-300.0)
        with patch.object(StatePersistence, "save_state"):
            result = rm.check_and_update(_et(), max_daily_loss=-750.0)
        assert result is False
        assert rm.is_halted is False

    def test_circuit_breaker_stays_halted_once_set(self):
        """AC#1: once halted, subsequent bars keep returning True without re-evaluating threshold."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        rm.register_close(-800.0)
        with patch.object(StatePersistence, "save_state"):
            rm.check_and_update(_et(hour=10), max_daily_loss=-750.0)  # trips
        with patch.object(StatePersistence, "save_state") as mock_save:
            result = rm.check_and_update(_et(hour=11), max_daily_loss=-750.0)
        assert result is True
        mock_save.assert_not_called()  # no extra persist on subsequent halted-check calls

    def test_day_reset_clears_halted_flag(self):
        """AC#2: halted on day D, check_and_update with day D+1 → is_halted=False, pnl=0.0."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        rm.register_close(-800.0)
        with patch.object(StatePersistence, "save_state"):
            rm.check_and_update(_et(day=6, hour=10), max_daily_loss=-750.0)  # trip on day 6
        assert rm.is_halted is True

        with patch.object(StatePersistence, "save_state"):
            result = rm.check_and_update(_et(day=7, hour=9), max_daily_loss=-750.0)  # day 7 = new day
        assert result is False
        assert rm.is_halted is False
        assert rm.daily_pnl == pytest.approx(0.0)

    def test_restore_from_state_same_day_restores_risk(self):
        """AC#3: state has today's date → pnl and halted restored."""
        from src.research.tier2_streaming_working import RiskManager
        rm = RiskManager()
        today = date(2026, 1, 6)
        state = {
            "daily_pnl": -650.0,
            "daily_halted": True,
            "last_trading_date": "2026-01-06",
        }
        rm.restore_from_state(state, today)
        assert rm.daily_pnl == pytest.approx(-650.0)
        assert rm.is_halted is True

    def test_restore_from_state_new_day_does_not_restore(self):
        """AC#3: state has yesterday's date → no restore, defaults remain."""
        from src.research.tier2_streaming_working import RiskManager
        rm = RiskManager()
        today = date(2026, 1, 7)
        state = {
            "daily_pnl": -800.0,
            "daily_halted": True,
            "last_trading_date": "2026-01-06",
        }
        rm.restore_from_state(state, today)
        assert rm.daily_pnl == pytest.approx(0.0)
        assert rm.is_halted is False

    def test_restore_from_state_no_date_is_noop(self):
        """restore_from_state with no last_trading_date is a safe no-op."""
        from src.research.tier2_streaming_working import RiskManager
        rm = RiskManager()
        rm.restore_from_state({}, date.today())
        assert rm.daily_pnl == pytest.approx(0.0)
        assert rm.is_halted is False

    def test_persist_called_when_circuit_breaker_trips(self):
        """AC#3: save_state called immediately when circuit breaker trips."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        rm.register_close(-800.0)
        with patch.object(StatePersistence, "save_state") as mock_save:
            rm.check_and_update(_et(), max_daily_loss=-750.0)
        mock_save.assert_called_once()
        saved = mock_save.call_args[0][0]
        assert saved["daily_halted"] is True
        assert saved["daily_pnl"] == pytest.approx(-800.0)

    def test_halt_manually_persists(self):
        """AC#4: halt_manually() sets is_halted=True and persists immediately."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        with patch.object(StatePersistence, "save_state") as mock_save:
            rm.halt_manually()
        assert rm.is_halted is True
        mock_save.assert_called_once()
        saved = mock_save.call_args[0][0]
        assert saved["daily_halted"] is True

    def test_to_state_dict_includes_all_keys(self):
        """AC#6: to_state_dict returns dict with required risk keys."""
        from src.research.tier2_streaming_working import RiskManager, StatePersistence
        rm = RiskManager()
        rm.register_close(-200.0)
        with patch.object(StatePersistence, "save_state"):
            rm.check_and_update(_et(), max_daily_loss=-750.0)
        d = rm.to_state_dict()
        assert "daily_pnl" in d
        assert "daily_halted" in d
        assert "last_trading_date" in d
        assert d["daily_pnl"] == pytest.approx(-200.0)
        assert d["daily_halted"] is False


# ---------------------------------------------------------------------------
# TestTier2RiskManagerIntegration
# ---------------------------------------------------------------------------

def _make_trader(symbol="MNQM26"):
    """Return a Tier2StreamingTrader with mocked deps, no network."""
    from src.research.tier2_streaming_working import Tier2StreamingTrader
    with patch("src.research.tier2_streaming_working.MetaLabelingFilter"), \
         patch("src.research.tier2_streaming_working.LRRegimeFilter"):
        trader = Tier2StreamingTrader(symbol=symbol)
    trader._ts_client = MagicMock()
    return trader


class TestTier2RiskManagerIntegration:
    def test_trader_has_risk_manager_field(self):
        """AC#7: Tier2StreamingTrader has _risk_manager attribute (no bare _daily_pnl etc.)."""
        from src.research.tier2_streaming_working import RiskManager
        trader = _make_trader()
        assert hasattr(trader, "_risk_manager")
        assert isinstance(trader._risk_manager, RiskManager)
        assert not hasattr(trader, "_daily_pnl"), "bare _daily_pnl should be removed"
        assert not hasattr(trader, "_daily_halted"), "bare _daily_halted should be removed"
        assert not hasattr(trader, "_last_trading_date"), "bare _last_trading_date should be removed"

    @pytest.mark.asyncio
    async def test_crash_recovery_restores_risk_via_risk_manager(self):
        """AC#7: _recover_from_state() delegates risk restoration to RiskManager."""
        from src.research.tier2_streaming_working import StatePersistence, TradeState
        trader = _make_trader()
        today_str = datetime.now(timezone.utc).astimezone(ET_TZ).date().isoformat()
        state = {
            "daily_pnl": -500.0,
            "daily_halted": True,
            "last_trading_date": today_str,
        }
        with patch.object(StatePersistence, "load_state", return_value=state):
            await trader._recover_from_state()
        assert trader._risk_manager.is_halted is True
        assert trader._risk_manager.daily_pnl == pytest.approx(-500.0)


# ---------------------------------------------------------------------------
# TestEmergencyStopCLI
# ---------------------------------------------------------------------------

class TestEmergencyStopCLI:
    @pytest.mark.asyncio
    async def test_emergency_stop_cancels_orders_and_halts(self, capsys):
        """AC#4: cancel_all_pending_orders called, halt persisted, 'EMERGENCY STOP COMPLETE' printed."""
        from src.cli.emergency_stop import _run_emergency_stop
        from src.research.tier2_streaming_working import StatePersistence, RiskManager

        mock_client = AsyncMock()
        mock_client.cancel_all_pending_orders = AsyncMock(return_value=["O1", "O2"])

        with patch("src.cli.emergency_stop.TradeStationAuthV3") as MockAuth, \
             patch("src.cli.emergency_stop.TradeStationClient", return_value=mock_client), \
             patch("src.cli.emergency_stop.StatePersistence") as MockSP, \
             patch("src.cli.emergency_stop.RiskManager") as MockRM, \
             patch("src.cli.emergency_stop.httpx.AsyncClient") as MockHTTP:
            MockAuth.from_file.return_value.authenticate = AsyncMock()
            MockSP.load_state.return_value = None  # no active trade
            MockHTTP.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            MockHTTP.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await _run_emergency_stop(force=False)

        assert result == 0
        mock_client.cancel_all_pending_orders.assert_called_once()
        MockRM.return_value.halt_manually.assert_called_once()
        out = capsys.readouterr().out
        assert "EMERGENCY STOP COMPLETE" in out

    @pytest.mark.asyncio
    async def test_emergency_stop_closes_position_when_active_trade_in_state(self, capsys):
        """AC#4: active trade in state → close_position_at_market called."""
        from src.cli.emergency_stop import _run_emergency_stop

        mock_client = AsyncMock()
        mock_client.cancel_all_pending_orders = AsyncMock(return_value=[])
        mock_client.close_position_at_market = AsyncMock(return_value="CLO1")

        active_state = {
            "direction": "SHORT",
            "entry_price": 18000.0,
            "tp_price": 17700.0,
            "sl_price": 18250.0,
            "entry_time": "2026-01-06T10:00:00+00:00",
        }

        with patch("src.cli.emergency_stop.TradeStationAuthV3") as MockAuth, \
             patch("src.cli.emergency_stop.TradeStationClient", return_value=mock_client), \
             patch("src.cli.emergency_stop.StatePersistence") as MockSP, \
             patch("src.cli.emergency_stop.RiskManager"), \
             patch("src.cli.emergency_stop.TradeLogger"), \
             patch("src.cli.emergency_stop.httpx.AsyncClient") as MockHTTP:
            MockAuth.from_file.return_value.authenticate = AsyncMock()
            MockSP.load_state.return_value = active_state
            MockHTTP.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            MockHTTP.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await _run_emergency_stop(force=False)

        assert result == 0
        mock_client.close_position_at_market.assert_called_once_with("SHORT", pytest.approx("SIM2797251F", rel=0))

    @pytest.mark.asyncio
    async def test_emergency_stop_appends_manual_exit_to_trade_logger(self):
        """AC#4: TradeLogger.append_trade called with MANUAL exit_reason when position closed."""
        from src.cli.emergency_stop import _run_emergency_stop
        from src.research.tier2_streaming_working import TradeRecord

        mock_client = AsyncMock()
        mock_client.cancel_all_pending_orders = AsyncMock(return_value=[])
        mock_client.close_position_at_market = AsyncMock(return_value="CLO1")

        active_state = {
            "direction": "SHORT",
            "entry_price": 18000.0,
            "tp_price": 17700.0,
            "sl_price": 18250.0,
            "entry_time": "2026-01-06T10:00:00+00:00",
        }

        mock_logger = MagicMock()

        with patch("src.cli.emergency_stop.TradeStationAuthV3") as MockAuth, \
             patch("src.cli.emergency_stop.TradeStationClient", return_value=mock_client), \
             patch("src.cli.emergency_stop.StatePersistence") as MockSP, \
             patch("src.cli.emergency_stop.RiskManager"), \
             patch("src.cli.emergency_stop.TradeLogger", return_value=mock_logger), \
             patch("src.cli.emergency_stop.httpx.AsyncClient") as MockHTTP:
            MockAuth.from_file.return_value.authenticate = AsyncMock()
            MockSP.load_state.return_value = active_state
            MockHTTP.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            MockHTTP.return_value.__aexit__ = AsyncMock(return_value=False)

            await _run_emergency_stop(force=False)

        mock_logger.append_trade.assert_called_once()
        record: TradeRecord = mock_logger.append_trade.call_args[0][0]
        assert record.exit_reason == "MANUAL"
        assert record.direction == "SHORT"

    @pytest.mark.asyncio
    async def test_emergency_stop_force_flag_persists_halt_on_api_failure(self, capsys):
        """AC#5: API raises → --force → halt persisted, exit code 1, warning to stderr."""
        from src.cli.emergency_stop import _run_emergency_stop

        mock_client = AsyncMock()
        mock_client.cancel_all_pending_orders = AsyncMock(side_effect=ConnectionError("unreachable"))

        with patch("src.cli.emergency_stop.TradeStationAuthV3") as MockAuth, \
             patch("src.cli.emergency_stop.TradeStationClient", return_value=mock_client), \
             patch("src.cli.emergency_stop.StatePersistence") as MockSP, \
             patch("src.cli.emergency_stop.RiskManager") as MockRM, \
             patch("src.cli.emergency_stop.httpx.AsyncClient") as MockHTTP:
            MockAuth.from_file.return_value.authenticate = AsyncMock()
            MockSP.load_state.return_value = None
            MockHTTP.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            MockHTTP.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await _run_emergency_stop(force=True)

        assert result == 1  # exit code 1 on API failure
        MockRM.return_value.halt_manually.assert_called_once()  # halt still persisted
        err = capsys.readouterr().err
        assert "WARNING" in err

    @pytest.mark.asyncio
    async def test_emergency_stop_no_active_trade_skips_close(self):
        """AC#4: risk-only state (no direction) → close_position_at_market NOT called."""
        from src.cli.emergency_stop import _run_emergency_stop

        mock_client = AsyncMock()
        mock_client.cancel_all_pending_orders = AsyncMock(return_value=[])

        risk_only_state = {
            "daily_pnl": -600.0,
            "daily_halted": False,
            "last_trading_date": "2026-01-06",
        }

        with patch("src.cli.emergency_stop.TradeStationAuthV3") as MockAuth, \
             patch("src.cli.emergency_stop.TradeStationClient", return_value=mock_client), \
             patch("src.cli.emergency_stop.StatePersistence") as MockSP, \
             patch("src.cli.emergency_stop.RiskManager"), \
             patch("src.cli.emergency_stop.httpx.AsyncClient") as MockHTTP:
            MockAuth.from_file.return_value.authenticate = AsyncMock()
            MockSP.load_state.return_value = risk_only_state
            MockHTTP.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            MockHTTP.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await _run_emergency_stop(force=False)

        assert result == 0
        mock_client.close_position_at_market.assert_not_called()
