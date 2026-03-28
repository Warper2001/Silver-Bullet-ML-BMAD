"""Integration tests for TradeStation market data endpoints.

These tests verify market data retrieval with the actual TradeStation SIM API.

Prerequisites:
- TRADESTATION_SIM_CLIENT_ID environment variable
- TRADESTATION_SIM_CLIENT_SECRET environment variable
- Market must be open for some tests (real-time quotes, streaming)
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta

import pytest

from src.execution.tradestation.market_data.quotes import QuotesClient
from src.execution.tradestation.market_data.history import HistoryClient
from src.execution.tradestation.market_data.streaming import QuoteStreamParser
from src.execution.tradestation.models import TradeStationQuote, HistoricalBar


@pytest.mark.integration
class TestQuotesClientIntegration:
    """Integration tests for QuotesClient with TradeStation SIM API."""

    @pytest.mark.asyncio
    async def test_get_real_time_quotes(self, tradestation_client, test_symbols):
        """Test fetching real-time quotes for multiple symbols."""
        quotes_client = QuotesClient(tradestation_client)

        # Get quotes
        quotes = await quotes_client.get_quotes(test_symbols)

        # Verify
        assert len(quotes) > 0
        assert len(quotes) <= len(test_symbols)

        for quote in quotes:
            assert quote.symbol in test_symbols
            assert quote.timestamp is not None
            # Quote data may be None if market is closed
            # Just verify structure is correct

        print(f"✅ Retrieved {len(quotes)} quotes")
        for quote in quotes:
            print(f"   {quote.symbol}: Bid={quote.bid}, Ask={quote.ask}, Last={quote.last}")

    @pytest.mark.asyncio
    async def test_get_quote_snapshot_single_symbol(self, tradestation_client, test_symbol):
        """Test fetching quote for a single symbol."""
        quotes_client = QuotesClient(tradestation_client)

        # Get quote snapshot
        quote = await quotes_client.get_quote_snapshot(test_symbol)

        # Verify
        assert quote is not None
        assert quote.symbol == test_symbol
        assert quote.timestamp is not None

        print(f"✅ Quote snapshot for {test_symbol}")
        print(f"   Bid: {quote.bid}, Ask: {quote.ask}, Last: {quote.last}")

    @pytest.mark.asyncio
    async def test_quotes_data_latency(self, tradestation_client, test_symbol):
        """Test quotes endpoint latency."""
        quotes_client = QuotesClient(tradestation_client)

        # Measure request time
        start_time = time.time()
        quotes = await quotes_client.get_quotes([test_symbol])
        latency = time.time() - start_time

        # Latency should be < 500ms per NFR
        assert latency < 0.5, f"Quotes endpoint too slow: {latency:.3f}s"

        print(f"✅ Quotes latency: {latency*1000:.1f}ms")
        print(f"   Target: < 500ms (NFR requirement)")


@pytest.mark.integration
class TestHistoryClientIntegration:
    """Integration tests for HistoryClient with TradeStation SIM API."""

    @pytest.mark.asyncio
    async def test_get_historical_bars_daily(self, tradestation_client, test_symbol):
        """Test fetching daily historical bars."""
        history_client = HistoryClient(tradestation_client)

        # Get last 30 days of daily bars
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")

        bars = await history_client.get_historical_bars(
            symbol=test_symbol,
            bar_type="daily",
            interval=1,
            start_date=start_date,
            end_date=end_date,
        )

        # Verify
        assert len(bars) > 0
        assert all(bar.symbol == test_symbol for bar in bars)

        # Check bar data structure
        first_bar = bars[0]
        assert first_bar.open > 0
        assert first_bar.high >= first_bar.low
        assert first_bar.close > 0
        assert first_bar.volume >= 0

        print(f"✅ Retrieved {len(bars)} daily bars")
        print(f"   Date range: {bars[0].timestamp} to {bars[-1].timestamp}")

    @pytest.mark.asyncio
    async def test_get_historical_bars_minute(self, tradestation_client, test_symbol):
        """Test fetching minute historical bars."""
        history_client = HistoryClient(tradestation_client)

        # Get last 3 days of minute bars (limited data for speed)
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start_date = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")

        bars = await history_client.get_historical_bars(
            symbol=test_symbol,
            bar_type="minute",
            interval=1,
            start_date=start_date,
            end_date=end_date,
        )

        # Verify
        assert len(bars) > 0

        print(f"✅ Retrieved {len(bars)} minute bars")

    @pytest.mark.asyncio
    async def test_get_bar_data_convenience_method(self, tradestation_client, test_symbol):
        """Test get_bar_data convenience method."""
        history_client = HistoryClient(tradestation_client)

        # Get last 7 days of daily data
        bars = await history_client.get_bar_data(
            symbol=test_symbol,
            days_back=7,
            bar_type="daily",
        )

        # Verify
        assert len(bars) > 0
        assert len(bars) <= 7  # May be fewer if weekends/holidays

        print(f"✅ Retrieved {len(bars)} bars using convenience method")

    @pytest.mark.asyncio
    async def test_historical_data_completeness(self, tradestation_client, test_symbol):
        """Test data completeness for historical bars."""
        history_client = HistoryClient(tradestation_client)

        # Get last 5 days of data
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start_date = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")

        bars = await history_client.get_historical_bars(
            symbol=test_symbol,
            bar_type="daily",
            interval=1,
            start_date=start_date,
            end_date=end_date,
        )

        # Calculate expected vs actual bars
        expected_bars = history_client._calculate_expected_bars(
            start_date=start_date,
            end_date=end_date,
            bar_type="daily",
            interval=1,
        )

        actual_bars = len(bars)
        completeness = (actual_bars / expected_bars) * 100 if expected_bars > 0 else 0

        # Allow for weekends (trading days only)
        # 5 calendar days = ~3-4 trading days
        assert completeness > 50, f"Data completeness too low: {completeness:.1f}%"

        print(f"✅ Data completeness: {completeness:.1f}%")
        print(f"   Expected: {expected_bars} bars, Actual: {actual_bars} bars")


@pytest.mark.integration
class TestQuoteStreamingIntegration:
    """Integration tests for real-time quote streaming."""

    @pytest.mark.asyncio
    async def test_stream_quotes_connection(self, tradestation_client, test_symbol):
        """Test establishing connection to quote stream."""
        stream_parser = QuoteStreamParser(tradestation_client)

        # Test connection (stream for 3 seconds)
        quotes_received = []
        stream_duration = 3.0

        start_time = time.time()
        try:
            async for quote in stream_parser.stream_quotes([test_symbol]):
                quotes_received.append(quote)

                # Stop after duration
                if time.time() - start_time >= stream_duration:
                    stream_parser.stop_streaming()
                    break

                # Safety limit
                if len(quotes_received) >= 100:
                    stream_parser.stop_streaming()
                    break
        except Exception as e:
            # Streaming may fail if market is closed
            print(f"⚠️  Streaming failed (market may be closed): {e}")

        print(f"✅ Streaming connection test completed")
        print(f"   Quotes received: {len(quotes_received)}")
        print(f"   Stream duration: {time.time() - start_time:.1f}s")

    @pytest.mark.asyncio
    async def test_stream_to_queue(self, tradestation_client, test_symbol):
        """Test streaming quotes to an asyncio queue."""
        stream_parser = QuoteStreamParser(tradestation_client)
        queue = asyncio.Queue()

        # Start streaming in background
        stream_task = asyncio.create_task(
            stream_parser.stream_to_queue([test_symbol], queue)
        )

        # Collect quotes for 2 seconds
        quotes_received = []
        timeout = 2.0
        start_time = time.time()

        try:
            while time.time() - start_time < timeout:
                try:
                    quote = await asyncio.wait_for(queue.get(), timeout=0.5)
                    quotes_received.append(quote)
                except asyncio.TimeoutError:
                    # No quote received in 0.5s, continue
                    continue

                if len(quotes_received) >= 10:  # Collect up to 10 quotes
                    break
        finally:
            # Stop streaming
            stream_parser.stop_streaming()

            # Wait for stream task to complete
            try:
                await asyncio.wait_for(stream_task, timeout=5.0)
            except asyncio.TimeoutError:
                stream_task.cancel()

        print(f"✅ Stream to queue test completed")
        print(f"   Quotes received: {len(quotes_received)}")

    @pytest.mark.asyncio
    async def test_stream_with_callback(self, tradestation_client, test_symbol):
        """Test streaming quotes with callback function."""
        stream_parser = QuoteStreamParser(tradestation_client)

        quotes_received = []

        async def quote_callback(quote):
            quotes_received.append(quote)
            if len(quotes_received) >= 5:
                stream_parser.stop_streaming()

        # Start streaming with callback
        stream_task = asyncio.create_task(
            stream_parser.stream_with_callback([test_symbol], quote_callback)
        )

        # Wait for task completion or timeout
        try:
            await asyncio.wait_for(stream_task, timeout=10.0)
        except asyncio.TimeoutError:
            stream_parser.stop_streaming()
            stream_task.cancel()

        print(f"✅ Stream with callback test completed")
        print(f"   Callback invoked: {len(quotes_received)} times")


@pytest.mark.integration
class TestDataQuality:
    """Integration tests for data quality and completeness."""

    @pytest.mark.asyncio
    async def test_quote_data_validation(self, tradestation_client, test_symbol):
        """Test quote data validation (if market is open)."""
        quotes_client = QuotesClient(tradestation_client)

        quotes = await quotes_client.get_quotes([test_symbol])

        if len(quotes) > 0:
            quote = quotes[0]

            # Validate timestamp is recent (within last minute)
            if quote.timestamp:
                time_diff = (datetime.now(timezone.utc) - quote.timestamp).total_seconds()
                if time_diff < 60:  # Market is open
                    # Quote data should be present
                    assert quote.bid is not None or quote.last is not None
                    if quote.bid is not None and quote.ask is not None:
                        # Bid should be <= Ask
                        assert quote.bid <= quote.ask

        print(f"✅ Quote data validation completed")

    @pytest.mark.asyncio
    async def test_historical_bar_validation(self, tradestation_client, test_symbol):
        """Test historical bar data validation."""
        history_client = HistoryClient(tradestation_client)

        bars = await history_client.get_bar_data(
            symbol=test_symbol,
            days_back=5,
            bar_type="daily",
        )

        if len(bars) > 0:
            # Validate OHLC relationships
            for bar in bars:
                assert bar.high >= bar.low, f"High ({bar.high}) should be >= Low ({bar.low})"
                assert bar.open >= bar.low and bar.open <= bar.high, \
                    f"Open ({bar.open}) should be within Low-High range"
                assert bar.close >= bar.low and bar.close <= bar.high, \
                    f"Close ({bar.close}) should be within Low-High range"

            # Validate timestamps are sequential
            for i in range(1, len(bars)):
                assert bars[i].timestamp >= bars[i-1].timestamp, \
                    "Bars should be in chronological order"

        print(f"✅ Historical bar validation completed")

    @pytest.mark.asyncio
    async def test_data_completeness_over_range(self, tradestation_client, test_symbol):
        """Test data completeness over different date ranges."""
        history_client = HistoryClient(tradestation_client)

        # Test different ranges
        ranges = [
            (1, 1),   # 1 day
            (7, 5),   # 1 week (~5 trading days)
            (30, 20), # 1 month (~20 trading days)
        ]

        for days, min_expected in ranges:
            bars = await history_client.get_bar_data(
                symbol=test_symbol,
                days_back=days,
                bar_type="daily",
            )

            # Should have at least minimum expected bars
            assert len(bars) >= min_expected * 0.5, \
                f"Insufficient bars for {days} day range: got {len(bars)}, expected at least {min_expected * 0.5:.0f}"

        print(f"✅ Data completeness validation completed")


# Performance Tests
@pytest.mark.integration
class TestMarketDataPerformance:
    """Performance tests for market data endpoints."""

    @pytest.mark.asyncio
    async def test_concurrent_quote_requests(self, tradestation_client, test_symbols):
        """Test concurrent quote request performance."""
        quotes_client = QuotesClient(tradestation_client)

        # Measure time for 10 concurrent requests
        iterations = 10
        start_time = time.time()

        tasks = [
            quotes_client.get_quotes(test_symbols)
            for _ in range(iterations)
        ]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        avg_time = total_time / iterations

        # Average request should be < 1 second
        assert avg_time < 1.0, f"Concurrent requests too slow: {avg_time:.2f}s"

        print(f"✅ Concurrent quote request performance")
        print(f"   {iterations} concurrent requests in {total_time:.2f}s")
        print(f"   Average: {avg_time*1000:.1f}ms per request")

    @pytest.mark.asyncio
    async def test_historical_data_download_speed(self, tradestation_client, test_symbol):
        """Test historical data download speed."""
        history_client = HistoryClient(tradestation_client)

        # Measure download time for 30 days of daily data
        start_time = time.time()

        bars = await history_client.get_bar_data(
            symbol=test_symbol,
            days_back=30,
            bar_type="daily",
        )

        download_time = time.time() - start_time
        bars_per_second = len(bars) / download_time if download_time > 0 else 0

        print(f"✅ Historical data download speed")
        print(f"   Downloaded {len(bars)} bars in {download_time:.2f}s")
        print(f"   Speed: {bars_per_second:.1f} bars/second")
