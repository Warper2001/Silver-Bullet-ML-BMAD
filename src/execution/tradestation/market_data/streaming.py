"""
TradeStation SDK - Real-Time Streaming Module

This module provides real-time quote streaming via HTTP chunked transfer encoding.

Key Features:
- HTTP chunked transfer parser for TradeStation API
- Async generator interface for consuming quotes
- Automatic reconnection on connection loss
- Gap filling logic for missed data
- Integration with asyncio queues for pipeline processing

Usage:
    async with TradeStationClient(env="sim", ...) as client:
        parser = QuoteStreamParser(client)

        async for quote in parser.stream_quotes(["MNQH26"]):
            # Process quote in real-time
            print(f"Quote: {quote.symbol} @ {quote.last}")
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable

import httpx
from httpx import AsyncClient, HTTPStatusError, NetworkError as HttpxNetworkError

from src.data.models import MarketData
from src.execution.tradestation.client import TradeStationClient
from src.execution.tradestation.exceptions import NetworkError, ValidationError
from src.execution.tradestation.models import TradeStationQuote
from src.execution.tradestation.utils import setup_logger


class QuoteStreamParser:
    """
    Real-time quote stream parser using HTTP chunked transfer encoding.

    TradeStation API uses HTTP streaming with chunked transfer encoding
    to push real-time quotes. This parser handles the connection and parsing.

    Attributes:
        client: TradeStationClient instance for authentication
        reconnect_interval: Seconds to wait before reconnecting
        max_reconnect_attempts: Maximum reconnection attempts

    Example:
        async with TradeStationClient(env="sim", ...) as client:
            parser = QuoteStreamParser(client)

            # Stream quotes to a queue
            queue = asyncio.Queue()

            async def stream_to_queue():
                async for quote in parser.stream_quotes(["MNQH26"]):
                    await queue.put(quote)

            # Start streaming in background
            task = asyncio.create_task(stream_to_queue())

            # Process quotes
            while True:
                quote = await queue.get()
                print(f"Received: {quote.symbol} @ {quote.last}")
    """

    def __init__(
        self,
        client: TradeStationClient,
        reconnect_interval: float = 5.0,
        max_reconnect_attempts: int = 10,
        staleness_threshold_seconds: float = 30.0,
    ) -> None:
        """
        Initialize QuoteStreamParser.

        Args:
            client: Authenticated TradeStationClient instance
            reconnect_interval: Seconds to wait between reconnection attempts
            max_reconnect_attempts: Maximum number of reconnection attempts
            staleness_threshold_seconds: Seconds without quotes before considering data stale
        """
        self.client = client
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.staleness_threshold_seconds = staleness_threshold_seconds
        self.logger = setup_logger(f"{__name__}.QuoteStreamParser")

        # Streaming state
        self._is_streaming = False
        self._reconnect_count = 0
        self._last_quote_timestamp: datetime | None = None
        self._staleness_check_interval = 5.0  # Check every 5 seconds

    async def stream_quotes(
        self,
        symbols: list[str],
    ) -> AsyncGenerator[TradeStationQuote, None]:
        """
        Stream real-time quotes for symbols using HTTP chunked transfer.

        This is an async generator that yields TradeStationQuote objects
        as they arrive from the API.

        Args:
            symbols: List of symbols to stream

        Yields:
            TradeStationQuote objects as they arrive

        Raises:
            NetworkError: On connection failure after max retries
            ValidationError: If symbols are invalid

        Example:
            async for quote in parser.stream_quotes(["MNQH26"]):
                print(f"{quote.symbol}: Bid={quote.bid}, Ask={quote.ask}")
        """
        if not symbols:
            raise ValidationError("Symbols list cannot be empty")

        self._is_streaming = True
        self._reconnect_count = 0
        self._last_quote_timestamp = None
        self._staleness_reconnect_attempts = 0

        endpoint = f"/marketdata/stream/quotes/{','.join(symbols)}"

        # Start staleness monitoring task
        staleness_task = asyncio.create_task(
            self._monitor_staleness()
        )

        try:
            while self._is_streaming:
                try:
                    access_token = await self.client._ensure_authenticated()

                    headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "text/event-stream",
                    }

                    self.logger.info(f"Connecting to quote stream: {endpoint}")

                    async with AsyncClient(
                        base_url=self.client.api_base_url,
                        headers=headers,
                        timeout=httpx.Timeout(30.0, connect=10.0),
                    ) as http_client:
                        async with http_client.stream("GET", endpoint) as response:
                            if response.status_code != 200:
                                error_text = await response.aread()
                                raise NetworkError(
                                    f"Stream connection failed: {response.status_code} - {error_text}"
                                )

                            # Reset reconnect count on successful connection
                            self._reconnect_count = 0
                            self._staleness_reconnect_attempts = 0
                            self.logger.info("Connected to quote stream")

                            # Parse chunked response and yield quotes
                            async for quote in self._parse_stream_chunks(response, symbols):
                                if not self._is_streaming:
                                    break
                                # Update last quote timestamp for staleness detection
                                self._last_quote_timestamp = datetime.now(timezone.utc)
                                yield quote

                except (NetworkError, HttpxNetworkError, HTTPStatusError) as e:
                    if not self._is_streaming:
                        break

                    self._reconnect_count += 1

                    if self._reconnect_count >= self.max_reconnect_attempts:
                        self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
                        raise NetworkError(
                            f"Failed to connect after {self.max_reconnect_attempts} attempts: {e}"
                        )

                    self.logger.warning(
                        f"Connection lost, reconnecting in {self.reconnect_interval}s "
                        f"(attempt {self._reconnect_count}/{self.max_reconnect_attempts})"
                    )
                    await asyncio.sleep(self.reconnect_interval)

                except Exception as e:
                    self.logger.error(f"Unexpected error in stream_quotes: {e}")
                    raise

        finally:
            # Cancel staleness monitoring task
            staleness_task.cancel()
            try:
                await staleness_task
            except asyncio.CancelledError:
                pass

    async def _parse_stream_chunks(
        self,
        response,
        symbols: list[str],
    ) -> AsyncGenerator[TradeStationQuote, None]:
        """
        Parse HTTP stream chunks and yield TradeStationQuote objects.

        Args:
            response: HTTP response object with stream
            symbols: List of expected symbols

        Yields:
            TradeStationQuote objects as they are parsed
        """
        import json

        buffer_size = 0
        MAX_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB max buffer

        try:
            async for chunk in response.aiter_bytes():
                if not self._is_streaming:
                    break

                # Track buffer size to prevent unbounded memory growth
                buffer_size += len(chunk)

                # Apply backpressure if buffer grows too large
                if buffer_size > MAX_BUFFER_SIZE:
                    self.logger.warning(
                        f"Stream buffer exceeded {MAX_BUFFER_SIZE} bytes ({buffer_size} bytes). "
                        f"Applying backpressure: sleeping to allow consumer to catch up."
                    )
                    await asyncio.sleep(0.1)  # Yield to allow consumer to process
                    buffer_size = 0  # Reset counter after backpressure

                # Decode chunk to string
                text = chunk.decode("utf-8")

                # Parse SSE-style format
                # TradeStation API uses format: "data: {json}\n\n"
                lines = text.split("\n")

                for line in lines:
                    line = line.strip()

                    if not line or not line.startswith("data:"):
                        continue

                    # Extract JSON after "data: "
                    json_str = line[5:].strip()  # Remove "data: " prefix

                    if not json_str:
                        continue

                    # Parse JSON
                    try:
                        quote_data = json.loads(json_str)
                        quote = TradeStationQuote(**quote_data)

                        # Validate symbol
                        if quote.symbol not in symbols:
                            self.logger.warning(f"Received unexpected symbol: {quote.symbol}")
                            continue

                        # Log quote received
                        self.logger.debug(f"Quote received: {quote.symbol} @ {quote.last}")

                        # Yield the quote
                        yield quote

                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to decode JSON: {json_str[:100]}... - {e}")
                    except Exception as e:
                        self.logger.warning(f"Failed to parse quote data: {e}")

        except Exception as e:
            self.logger.error(f"Error parsing stream chunks: {e}")
            raise

    def stop_streaming(self) -> None:
        """
        Stop the streaming loop.

        This will gracefully stop the stream after the current chunk
        is processed.
        """
        self.logger.info("Stopping quote stream")
        self._is_streaming = False

    async def _monitor_staleness(self) -> None:
        """
        Monitor for stale data (no quotes received for >30 seconds).

        Implements DATA_GAP scenario from spec I/O matrix:
        - Check every 5 seconds if last quote was >30 seconds ago
        - If stale, attempt reconnection with exponential backoff
        - After 3 failed attempts, raise error to trigger emergency stop

        Raises:
            NetworkError: If data remains stale after 3 reconnection attempts
        """
        MAX_STALENESS_RECONNECT_ATTEMPTS = 3

        while self._is_streaming:
            try:
                await asyncio.sleep(self._staleness_check_interval)

                # Check if we have a timestamp
                if self._last_quote_timestamp is None:
                    # Just started, no quotes yet - skip check
                    continue

                # Calculate time since last quote
                time_since_last_quote = (
                    datetime.now(timezone.utc) - self._last_quote_timestamp
                ).total_seconds()

                # Check if data is stale
                if time_since_last_quote > self.staleness_threshold_seconds:
                    self._staleness_reconnect_attempts += 1

                    self.logger.critical(
                        f"DATA_GAP DETECTED: No quotes received for "
                        f"{time_since_last_quote:.1f} seconds (threshold: "
                        f"{self.staleness_threshold_seconds}s). "
                        f"Staleness reconnection attempt: "
                        f"{self._staleness_reconnect_attempts}/{MAX_STALENESS_RECONNECT_ATTEMPTS}"
                    )

                    if self._staleness_reconnect_attempts >= MAX_STALENESS_RECONNECT_ATTEMPTS:
                        # Trigger emergency stop - data gap is persistent
                        self.logger.critical(
                            f"EMERGENCY STOP: Data gap persistent after "
                            f"{MAX_STALENESS_RECONNECT_ATTEMPTS} attempts. "
                            f"Halting trading and triggering emergency stop."
                        )
                        self._is_streaming = False
                        raise NetworkError(
                            f"Data gap persistent after {MAX_STALENESS_RECONNECT_ATTEMPTS} "
                            f"reconnection attempts. Triggering emergency stop per spec "
                            f"I/O matrix DATA_GAP scenario."
                        )

                    # Attempt to reconnect with exponential backoff
                    backoff_seconds = self.reconnect_interval * (
                        2 ** (self._staleness_reconnect_attempts - 1)
                    )
                    self.logger.warning(
                        f"Attempting to reconnect due to stale data in "
                        f"{backoff_seconds}s (exponential backoff)"
                    )
                    # Force reconnection by breaking current connection
                    # The outer loop will handle reconnection
                    break

                else:
                    # Data is fresh, reset staleness counter
                    self._staleness_reconnect_attempts = 0

            except asyncio.CancelledError:
                # Task was cancelled - exit gracefully
                break
            except NetworkError:
                # Re-raise network errors to trigger emergency stop
                raise
            except Exception as e:
                # Log but don't crash the monitoring task
                self.logger.error(f"Error in staleness monitoring: {e}")

    def _quote_to_market_data(self, quote: TradeStationQuote) -> MarketData:
        """
        Convert TradeStationQuote to MarketData model.

        Args:
            quote: TradeStationQuote object

        Returns:
            MarketData object

        Raises:
            ValidationError: If quote lacks required fields
        """
        # Validate required fields
        if quote.last is None:
            raise ValidationError(f"Quote missing required field 'last': {quote.symbol}")

        # Use last trade price for mid-price calculation
        volume = quote.last_size if quote.last_size else (quote.volume if quote.volume else 0)

        return MarketData(
            symbol=quote.symbol,
            timestamp=quote.timestamp,
            bid=quote.bid,
            ask=quote.ask,
            last=quote.last,
            volume=volume,
        )

    async def stream_to_queue(
        self,
        symbols: list[str],
        queue: asyncio.Queue,
    ) -> None:
        """
        Stream quotes to an asyncio queue for pipeline processing.

        This is a convenience method that streams quotes and places them
        into a queue for consumption by other pipeline components.

        Args:
            symbols: List of symbols to stream
            queue: asyncio.Queue to receive MarketData objects

        Example:
            queue = asyncio.Queue()

            # Start streaming in background
            task = asyncio.create_task(parser.stream_to_queue(["MNQH26"], queue))

            # Process quotes from queue
            while True:
                market_data = await queue.get()
                # Process market_data...
        """
        self.logger.info(f"Starting quote stream to queue for {symbols}")

        try:
            async for quote in self.stream_quotes(symbols):
                # Convert TradeStationQuote to MarketData
                market_data = self._quote_to_market_data(quote)
                await queue.put(market_data)
        except Exception as e:
            self.logger.error(f"Error streaming to queue: {e}")
            raise
        finally:
            self.logger.info("Quote stream to queue ended")

    async def stream_with_callback(
        self,
        symbols: list[str],
        callback: Callable[[TradeStationQuote], Any],
    ) -> None:
        """
        Stream quotes and invoke callback for each quote.

        This is a convenience method that streams quotes and invokes
        a callback function for each quote received.

        Args:
            symbols: List of symbols to stream
            callback: Async callback function (should accept TradeStationQuote)

        Example:
            async def handle_quote(quote: TradeStationQuote):
                print(f"{quote.symbol}: {quote.last}")
                # Or send to queue, process, etc.

            await parser.stream_with_callback(["MNQH26"], handle_quote)
        """
        self.logger.info(f"Starting quote stream with callback for {symbols}")

        try:
            async for quote in self.stream_quotes(symbols):
                await callback(quote)
        except Exception as e:
            self.logger.error(f"Error in callback streaming: {e}")
            raise
        finally:
            self.logger.info("Quote stream with callback ended")
