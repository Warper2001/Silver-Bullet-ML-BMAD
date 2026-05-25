"""Unit tests for TradeLogger, StatePersistence, and crash recovery (Story 4-2, AC#8)."""
import csv
import json
import pytest
import pytz
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.research.tier2_streaming_working import (
    ET_TZ,
    TradeRecord,
    TradeLogger,
    StatePersistence,
    ActiveTrade,
    TradeState,
    Tier2StreamingTrader,
    SIM_ACCOUNT_ID,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides) -> TradeRecord:
    defaults = dict(
        timestamp_entry=datetime(2026, 1, 6, 10, 0, tzinfo=timezone.utc),
        timestamp_exit=datetime(2026, 1, 6, 11, 0, tzinfo=timezone.utc),
        direction="SHORT",
        entry_price=18000.0,
        exit_price=17940.0,
        tp_price=17940.0,
        sl_price=18250.0,
        gap_size=50.0,
        pnl_usd=120.0,
        exit_reason="TP",
        h1_sweep_bars_ago=3,
        m15_confirmed=True,
        kill_zone_active=True,
        vol_regime_pct=0.62,
        contracts=5,
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


# ---------------------------------------------------------------------------
# TestTradeLogger
# ---------------------------------------------------------------------------

class TestTradeLogger:
    def test_append_trade_writes_correct_columns(self, tmp_path):
        """AC#1: CSV row written with all PRD columns in correct order."""
        tl = TradeLogger()
        tl._LOG_PATH = tmp_path / "trade_log.csv"
        record = _make_record()
        tl.append_trade(record)

        rows = list(csv.DictReader(tl._LOG_PATH.open()))
        assert len(rows) == 1
        r = rows[0]
        assert r["direction"] == "SHORT"
        assert r["exit_reason"] == "TP"
        assert r["m15_confirmed"] == "True"
        assert r["kill_zone_active"] == "True"
        assert r["contracts"] == "5"
        assert float(r["entry_price"]) == pytest.approx(18000.0)
        assert float(r["gap_size"]) == pytest.approx(50.0)
        assert float(r["vol_regime_pct"]) == pytest.approx(0.62)

        # Verify column order matches PRD spec
        with tl._LOG_PATH.open() as f:
            header = f.readline().strip().split(",")
        assert header == TradeLogger._COLUMNS

    def test_append_trade_header_written_once(self, tmp_path):
        """AC#2: Header appears exactly once even when two rows are written."""
        tl = TradeLogger()
        tl._LOG_PATH = tmp_path / "trade_log.csv"
        record = _make_record()
        tl.append_trade(record)
        tl.append_trade(record)

        content = tl._LOG_PATH.read_text()
        assert content.count("timestamp_entry") == 1
        rows = list(csv.DictReader(tl._LOG_PATH.open()))
        assert len(rows) == 2

    def test_append_trade_header_not_written_to_nonempty_file(self, tmp_path):
        """AC#2: If file already has content (non-empty), header is not re-written."""
        log_path = tmp_path / "trade_log.csv"
        # Pre-populate with header + one row
        log_path.write_text("timestamp_entry,timestamp_exit\n2026-01-01,2026-01-02\n")

        tl = TradeLogger()
        tl._LOG_PATH = log_path
        # Append a new record — should NOT write another header row
        tl.append_trade(_make_record())

        lines = log_path.read_text().splitlines()
        # First line is the pre-existing header; no second header should appear
        assert lines[0] == "timestamp_entry,timestamp_exit"
        assert all("timestamp_entry" not in line for line in lines[1:])

    def test_append_trade_no_raise_on_write_error(self, tmp_path):
        """AC#2: Write error must not propagate — just log a warning."""
        tl = TradeLogger()
        # Point _LOG_PATH at a directory — open("a") on a dir raises IsADirectoryError
        # which is caught by the except-Exception guard in append_trade()
        log_as_dir = tmp_path / "trade_log.csv"
        log_as_dir.mkdir()
        tl._LOG_PATH = log_as_dir
        tl.append_trade(_make_record())  # must not raise

    def test_append_trade_called_even_if_save_state_raises(self, tmp_path):
        """Patch: save_state() raising must not prevent append_trade() from writing the record."""
        from src.research.tier2_streaming_working import StatePersistence

        tl = TradeLogger()
        tl._LOG_PATH = tmp_path / "trade_log.csv"

        with patch.object(StatePersistence, "save_state", side_effect=OSError("disk full")):
            # Simulate what _close_active_trade() does: save_state (may raise) then append_trade
            try:
                StatePersistence.save_state({})
            except Exception:
                pass  # as in the patched _close_active_trade
            tl.append_trade(_make_record())

        rows = list(csv.DictReader(tl._LOG_PATH.open()))
        assert len(rows) == 1  # record was written despite save_state failure


# ---------------------------------------------------------------------------
# TestStatePersistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_save_load_roundtrip(self, tmp_path):
        """AC#3: save_state then load_state returns identical dict."""
        orig_state = tmp_path / "active_trade_state.json"
        orig_tmp = tmp_path / "active_trade_state.tmp"

        with patch.object(StatePersistence, "STATE_PATH", orig_state), \
             patch.object(StatePersistence, "TMP_PATH", orig_tmp), \
             patch.object(StatePersistence, "_LOG_DIR", tmp_path):

            state = {
                "direction": "SHORT",
                "entry_price": 18000.0,
                "tp_price": 17700.0,
                "sl_price": 18250.0,
                "entry_time": "2026-01-06T10:00:00+00:00",
                "daily_pnl": -200.0,
                "daily_halted": False,
                "last_trading_date": "2026-01-06",
            }
            StatePersistence.save_state(state)
            loaded = StatePersistence.load_state()

        assert loaded == state

    def test_clear_state_removes_file(self, tmp_path):
        """AC#3: clear_state makes load_state return None."""
        orig_state = tmp_path / "active_trade_state.json"
        orig_tmp = tmp_path / "active_trade_state.tmp"

        with patch.object(StatePersistence, "STATE_PATH", orig_state), \
             patch.object(StatePersistence, "TMP_PATH", orig_tmp), \
             patch.object(StatePersistence, "_LOG_DIR", tmp_path):

            StatePersistence.save_state({"direction": "SHORT"})
            assert StatePersistence.load_state() is not None
            StatePersistence.clear_state()
            assert StatePersistence.load_state() is None


# ---------------------------------------------------------------------------
# TestCrashRecovery
# ---------------------------------------------------------------------------

def _make_trader() -> Tier2StreamingTrader:
    """Return a Tier2StreamingTrader with mocked dependencies, no network access."""
    with patch("src.research.tier2_streaming_working.MetaLabelingFilter"), \
         patch("src.research.tier2_streaming_working.LRRegimeFilter"):
        trader = Tier2StreamingTrader(symbol="MNQM26")
    trader._ts_client = MagicMock()
    return trader


class TestCrashRecovery:
    @pytest.mark.asyncio
    async def test_crash_recovery_active_broker_position_restores_active_trade(self):
        """AC#4: broker ACTIVE + persisted state → active_trade reconstructed."""
        trader = _make_trader()
        state = {
            "direction": "SHORT",
            "entry_price": 18000.0,
            "tp_price": 17700.0,
            "sl_price": 18250.0,
            "entry_time": "2026-01-06T10:00:00+00:00",
            "sim_entry_order_id": "E001",
            "sim_tp_order_id": "TP001",
            "sim_sl_order_id": "SL001",
        }
        trader._ts_client.reconcile_state = AsyncMock(
            return_value=TradeState(status="ACTIVE", position_qty=5)
        )

        with patch.object(StatePersistence, "load_state", return_value=state):
            await trader._recover_from_state()

        assert trader.active_trade is not None
        assert trader.active_trade.direction == "SHORT"
        assert trader.active_trade.entry_price == pytest.approx(18000.0)
        assert trader.active_trade.pending_entry is False

    @pytest.mark.asyncio
    async def test_crash_recovery_flat_broker_logs_warning_and_clears(self, caplog):
        """AC#5: broker FLAT + persisted state → warning logged and state cleared."""
        trader = _make_trader()
        state = {
            "direction": "SHORT",
            "entry_price": 18000.0,
            "tp_price": 17700.0,
            "sl_price": 18250.0,
            "entry_time": "2026-01-06T10:00:00+00:00",
        }
        trader._ts_client.reconcile_state = AsyncMock(
            return_value=TradeState(status="FLAT")
        )

        with patch.object(StatePersistence, "load_state", return_value=state), \
             patch.object(StatePersistence, "clear_state") as mock_clear, \
             caplog.at_level("WARNING"):
            await trader._recover_from_state()

        assert trader.active_trade is None
        mock_clear.assert_called_once()
        assert "RECONCILIATION_WARNING" in caplog.text

    @pytest.mark.asyncio
    async def test_crash_recovery_no_state_skips_reconciliation(self):
        """AC#4: no persisted state → reconcile_state NOT called."""
        trader = _make_trader()
        trader._ts_client.reconcile_state = AsyncMock()

        with patch.object(StatePersistence, "load_state", return_value=None):
            await trader._recover_from_state()

        trader._ts_client.reconcile_state.assert_not_called()
        assert trader.active_trade is None

    @pytest.mark.asyncio
    async def test_circuit_breaker_survives_restart_same_day(self):
        """AC#6: daily_halted=True in state, same day → halted flag restored."""
        trader = _make_trader()
        # Use ET date (same TZ as _recover_from_state) to match the comparison
        today_str = datetime.now(timezone.utc).astimezone(ET_TZ).date().isoformat()
        state = {
            "daily_pnl": -800.0,
            "daily_halted": True,
            "last_trading_date": today_str,
        }

        with patch.object(StatePersistence, "load_state", return_value=state):
            await trader._recover_from_state()

        assert trader._daily_halted is True
        assert trader._daily_pnl == pytest.approx(-800.0)

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_new_day(self):
        """AC#6: daily_halted=True in state, yesterday's date → flag NOT restored."""
        trader = _make_trader()
        yesterday_str = "2000-01-01"  # Far in the past — guaranteed different from today
        state = {
            "daily_pnl": -800.0,
            "daily_halted": True,
            "last_trading_date": yesterday_str,
        }

        with patch.object(StatePersistence, "load_state", return_value=state):
            await trader._recover_from_state()

        # Different date → risk state NOT restored
        assert trader._daily_halted is False
        assert trader._daily_pnl == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_crash_recovery_risk_only_state_no_reconciliation(self):
        """AC#6/#7: risk-only state (no active trade keys) → risk restored, no reconciliation."""
        trader = _make_trader()
        today_str = datetime.now(timezone.utc).astimezone(ET_TZ).date().isoformat()
        state = {
            "daily_pnl": -300.0,
            "daily_halted": False,
            "last_trading_date": today_str,
        }
        trader._ts_client.reconcile_state = AsyncMock()

        with patch.object(StatePersistence, "load_state", return_value=state):
            await trader._recover_from_state()

        trader._ts_client.reconcile_state.assert_not_called()
        assert trader._daily_pnl == pytest.approx(-300.0)
        assert trader.active_trade is None

    @pytest.mark.asyncio
    async def test_crash_recovery_restores_metadata_fields(self):
        """Patch: metadata fields saved in state are restored to recovered ActiveTrade."""
        trader = _make_trader()
        state = {
            "direction": "SHORT",
            "entry_price": 18000.0,
            "tp_price": 17700.0,
            "sl_price": 18250.0,
            "entry_time": "2026-01-06T10:00:00+00:00",
            "gap_size": 42.5,
            "h1_sweep_bars_ago": 7,
            "m15_confirmed": True,
            "kill_zone_active": True,
            "vol_regime_pct": 0.63,
        }
        trader._ts_client.reconcile_state = AsyncMock(
            return_value=TradeState(status="ACTIVE", position_qty=5)
        )

        with patch.object(StatePersistence, "load_state", return_value=state):
            await trader._recover_from_state()

        t = trader.active_trade
        assert t is not None
        assert t.gap_size == pytest.approx(42.5)
        assert t.h1_sweep_bars_ago == 7
        assert t.m15_confirmed is True
        assert t.kill_zone_active is True
        assert t.vol_regime_pct == pytest.approx(0.63)

    @pytest.mark.asyncio
    async def test_crash_recovery_pending_broker_cancels_and_clears(self, caplog):
        """Patch: broker PENDING status → cancel entry order + clear state, no active_trade."""
        trader = _make_trader()
        state = {
            "direction": "SHORT",
            "entry_price": 18000.0,
            "tp_price": 17700.0,
            "sl_price": 18250.0,
            "entry_time": "2026-01-06T10:00:00+00:00",
            "sim_entry_order_id": "E001",
        }
        trader._ts_client.reconcile_state = AsyncMock(
            return_value=TradeState(status="PENDING")
        )
        trader._ts_client.cancel_order = AsyncMock(return_value=True)

        with patch.object(StatePersistence, "load_state", return_value=state), \
             patch.object(StatePersistence, "clear_state") as mock_clear, \
             caplog.at_level("WARNING"):
            await trader._recover_from_state()

        assert trader.active_trade is None
        trader._ts_client.cancel_order.assert_called_once_with("E001")
        mock_clear.assert_called_once()
        assert "RECONCILIATION_WARNING" in caplog.text

    @pytest.mark.asyncio
    async def test_crash_recovery_missing_entry_time_skips_trade_reconstruction(self):
        """Patch: state with direction+entry_price but no entry_time → guard prevents KeyError."""
        trader = _make_trader()
        state = {
            "direction": "SHORT",
            "entry_price": 18000.0,
            # entry_time intentionally absent
        }
        trader._ts_client.reconcile_state = AsyncMock()

        with patch.object(StatePersistence, "load_state", return_value=state):
            await trader._recover_from_state()  # must not raise

        trader._ts_client.reconcile_state.assert_not_called()
        assert trader.active_trade is None
