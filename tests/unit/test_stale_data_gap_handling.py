"""Unit tests for stale data detection, DATA_GAP, and API timeout (Story 4-5)."""
import sys
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytz

ET_TZ = pytz.timezone("America/New_York")

RTH_HOUR_UTC = 15  # 15:00 UTC = 11:00 ET (mid-RTH)
RTH_HOUR_ET = 11


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trader(symbol="MNQM26"):
    from src.research.tier2_streaming_working import Tier2StreamingTrader
    with patch("src.research.tier2_streaming_working.MetaLabelingFilter"), \
         patch("src.research.tier2_streaming_working.LRRegimeFilter"):
        trader = Tier2StreamingTrader(symbol=symbol)
    trader._ts_client = MagicMock()
    return trader


def _make_bar(age_seconds: float = 0.0) -> MagicMock:
    """Return a mock DollarBar with timestamp age_seconds before now."""
    bar = MagicMock()
    bar.timestamp = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return bar


# ---------------------------------------------------------------------------
# TestStaleDataDetection
# ---------------------------------------------------------------------------

class TestStaleDataDetection:
    def test_stale_flag_set_when_bar_old_during_rth(self, caplog):
        """AC#1: bar >5 min old during RTH → _data_stale=True + STALE_DATA logged."""
        import logging
        trader = _make_trader()

        # Mocked wall-clock: Wednesday Jan 7 2026 at 11:00 ET = 15:00 UTC (mid-RTH, no DST)
        now_rth = datetime(2026, 1, 7, 15, 0, 0, tzinfo=timezone.utc)

        # Bar timestamp is 400 seconds BEFORE the mocked now
        bar = MagicMock()
        bar.timestamp = now_rth - timedelta(seconds=400)

        with patch("src.research.tier2_streaming_working.datetime") as MockDT:
            MockDT.now.return_value = now_rth
            MockDT.fromisoformat = datetime.fromisoformat

            with caplog.at_level(logging.WARNING, logger="src.research.tier2_streaming_working"):
                result = trader._check_stale(bar)

        assert result is True
        assert trader._data_stale is True
        assert "STALE_DATA" in caplog.text

    def test_stale_flag_not_set_outside_rth(self):
        """AC#6: bar >5 min old but outside RTH (01:00 UTC = 20:00 ET) → _data_stale stays False."""
        trader = _make_trader()

        # Wall-clock: 01:00 UTC = 20:00 ET (outside RTH)
        now_outside = datetime(2026, 1, 7, 1, 0, tzinfo=timezone.utc)
        bar = MagicMock()
        bar.timestamp = now_outside - timedelta(seconds=400)

        with patch("src.research.tier2_streaming_working.datetime") as MockDT:
            MockDT.now.return_value = now_outside
            MockDT.fromisoformat = datetime.fromisoformat
            result = trader._check_stale(bar)

        assert result is False
        assert trader._data_stale is False

    def test_stale_flag_clears_on_fresh_bar(self, caplog):
        """AC#2: _data_stale=True, fresh bar arrives → clears to False."""
        import logging
        trader = _make_trader()
        trader._data_stale = True

        # Mocked wall-clock: mid-RTH
        now_rth = datetime(2026, 1, 7, 15, 0, 0, tzinfo=timezone.utc)
        # Fresh bar: only 30 seconds old
        fresh_bar = MagicMock()
        fresh_bar.timestamp = now_rth - timedelta(seconds=30)

        with patch("src.research.tier2_streaming_working.datetime") as MockDT:
            MockDT.now.return_value = now_rth
            MockDT.fromisoformat = datetime.fromisoformat
            with caplog.at_level(logging.INFO, logger="src.research.tier2_streaming_working"):
                result = trader._check_stale(fresh_bar)

        assert result is False
        assert trader._data_stale is False

    def test_is_rth_returns_true_during_rth(self):
        """_is_rth returns True for 10:00 ET."""
        trader = _make_trader()
        et_10am = ET_TZ.localize(datetime(2026, 1, 7, 10, 0))
        assert trader._is_rth(et_10am) is True

    def test_is_rth_returns_false_before_rth(self):
        """_is_rth returns False for 09:00 ET (before 09:30)."""
        trader = _make_trader()
        et_9am = ET_TZ.localize(datetime(2026, 1, 7, 9, 0))
        assert trader._is_rth(et_9am) is False

    def test_is_rth_returns_false_after_rth(self):
        """_is_rth returns False for 16:01 ET (after 16:00)."""
        trader = _make_trader()
        et_after = ET_TZ.localize(datetime(2026, 1, 7, 16, 1))
        assert trader._is_rth(et_after) is False

    def test_detect_and_enter_blocked_when_stale(self):
        """AC#1/#7: _data_stale=True → _detect_and_enter returns without calling risk_manager."""
        trader = _make_trader()
        trader._data_stale = True
        trader.dollar_bars = [MagicMock()] * 30
        trader.active_trade = None

        check_calls = []
        trader._risk_manager.check_and_update = lambda *a: check_calls.append(a) or False

        bar_ts = ET_TZ.localize(datetime(2026, 1, 7, 11, 0))
        mock_bar = MagicMock()
        mock_bar.timestamp = bar_ts

        asyncio.run(trader._detect_and_enter(mock_bar, is_backfill=False))
        assert len(check_calls) == 0, "check_and_update should not be called when stale"


# ---------------------------------------------------------------------------
# TestDataGapHandling
# ---------------------------------------------------------------------------

class TestDataGapHandling:
    @pytest.mark.asyncio
    async def test_data_gap_logged_on_empty_bars(self, caplog):
        """AC#4: empty bars list → DATA_GAP logged, no detection."""
        import logging
        trader = _make_trader()
        trader.dollar_bars = []

        empty_response = MagicMock()
        empty_response.status_code = 200
        empty_response.json.return_value = {"Bars": []}

        with patch("src.research.tier2_streaming_working.datetime") as MockDT, \
             caplog.at_level(logging.WARNING, logger="src.research.tier2_streaming_working"):
            MockDT.now.return_value = datetime(2026, 1, 7, 15, 0, tzinfo=timezone.utc)
            MockDT.fromisoformat = datetime.fromisoformat
            trader.auth = AsyncMock()
            trader.auth.authenticate = AsyncMock(return_value="token")
            trader.client = AsyncMock()
            trader.client.get = AsyncMock(return_value=empty_response)
            trader._last_processed_timestamp = None
            trader._is_backfill = False

            await trader._poll_and_process()

        assert "DATA_GAP" in caplog.text

    @pytest.mark.asyncio
    async def test_api_timeout_caught_and_loop_continues(self, caplog):
        """AC#5: httpx.TimeoutException → API_TIMEOUT logged, no exception propagated."""
        import logging
        import httpx
        trader = _make_trader()

        trader.auth = AsyncMock()
        trader.auth.authenticate = AsyncMock(side_effect=httpx.TimeoutException("read timeout"))
        trader.client = AsyncMock()
        trader._last_processed_timestamp = None
        trader._is_backfill = False

        with caplog.at_level(logging.WARNING, logger="src.research.tier2_streaming_working"):
            await trader._poll_and_process()  # must not raise

        assert "API_TIMEOUT" in caplog.text

    @pytest.mark.asyncio
    async def test_asyncio_timeout_caught_and_loop_continues(self, caplog):
        """AC#5: asyncio.TimeoutError → API_TIMEOUT logged, no exception propagated."""
        import logging
        trader = _make_trader()

        trader.auth = AsyncMock()
        trader.auth.authenticate = AsyncMock(side_effect=asyncio.TimeoutError())
        trader.client = AsyncMock()
        trader._last_processed_timestamp = None
        trader._is_backfill = False

        with caplog.at_level(logging.WARNING, logger="src.research.tier2_streaming_working"):
            await trader._poll_and_process()

        assert "API_TIMEOUT" in caplog.text

    def test_data_stale_field_exists_on_init(self):
        """AC#1: Tier2StreamingTrader has _data_stale attribute initialized False."""
        trader = _make_trader()
        assert hasattr(trader, "_data_stale")
        assert trader._data_stale is False
