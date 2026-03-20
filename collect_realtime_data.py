#!/usr/bin/env python3
"""Real-time data collector for MNQ futures.

This module collects real-time market data from TradeStation WebSocket
and saves it to HDF5 format for backtesting.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import h5py

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.auth_web import TradeStationAuthWeb
from src.data.websocket import TradeStationWebSocketClient
from src.data.models import MarketData
from src.data.exceptions import AuthenticationError

logger = logging.getLogger(__name__)


class RealtimeDataCollector:
    """Collect real-time MNQ futures data and save to HDF5.

    Features:
    - Connect to TradeStation WebSocket
    - Stream real-time market data
    - Aggregate into bars (5-minute default)
    - Auto-save to HDF5 files
    - Handle connection issues
    """

    def __init__(self, auth: TradeStationAuthWeb):
        """Initialize real-time collector.

        Args:
            auth: TradeStationAuth instance
        """
        self.auth = auth
        self.bars: list[dict] = []
        self.current_bar: Optional[dict] = None
        self.bar_interval = 5  # 5-minute bars
        self.last_bar_time: Optional[datetime] = None
        self.output_dir = Path("data/processed/dollar_bars")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def start_collection(self, duration_minutes: int = 60) -> None:
        """Start collecting real-time data.

        Args:
            duration_minutes: How long to collect data (default: 1 hour)
        """
        logger.info(f"Starting real-time data collection for {duration_minutes} minutes")

        try:
            # Connect to WebSocket
            ws_client = TradeStationWebSocketClient(self.auth)
            data_queue = await ws_client.subscribe()

            # Process data for specified duration
            end_time = datetime.now() + pd.Timedelta(minutes=duration_minutes)

            while datetime.now() < end_time:
                try:
                    # Get data with timeout
                    market_data = await asyncio.wait_for(
                        data_queue.get(),
                        timeout=5.0
                    )

                    # Process tick data
                    self._process_tick(market_data)

                except asyncio.TimeoutError:
                    # No data in 5 seconds, check if we need to save current bar
                    if self.current_bar:
                        self._finalize_bar()
                    continue

                except Exception as e:
                    logger.error(f"Error processing data: {e}")
                    continue

            # Save remaining data
            if self.current_bar:
                self._finalize_bar()

            # Save to HDF5
            self._save_to_hdf5()

            logger.info("Real-time data collection completed")

        except AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Collection failed: {e}")
            raise

    def _process_tick(self, market_data: MarketData) -> None:
        """Process incoming tick data.

        Args:
            market_data: MarketData object from WebSocket
        """
        timestamp = market_data.timestamp

        # Check if we need to start a new bar
        if self.last_bar_time is None:
            self.last_bar_time = timestamp
            self.current_bar = {
                "timestamp": timestamp,
                "open": market_data.last or market_data.close,
                "high": market_data.last or market_data.close,
                "low": market_data.last or market_data.close,
                "close": market_data.last or market_data.close,
                "volume": market_data.volume or 0,
            }
            return

        # Check if time interval has passed
        time_diff = (timestamp - self.last_bar_time).total_seconds() / 60
        if time_diff >= self.bar_interval:
            # Finalize current bar and start new one
            self._finalize_bar()
            self.last_bar_time = timestamp
            self.current_bar = {
                "timestamp": timestamp,
                "open": market_data.last or market_data.close,
                "high": market_data.last or market_data.close,
                "low": market_data.last or market_data.close,
                "close": market_data.last or market_data.close,
                "volume": market_data.volume or 0,
            }
        else:
            # Update current bar
            if self.current_bar:
                price = market_data.last or market_data.close
                self.current_bar["high"] = max(self.current_bar["high"], price)
                self.current_bar["low"] = min(self.current_bar["low"], price)
                self.current_bar["close"] = price
                self.current_bar["volume"] += market_data.volume or 0

    def _finalize_bar(self) -> None:
        """Finalize current bar and add to bars list."""
        if self.current_bar:
            # Calculate dollar volume
            self.current_bar["dollar_volume"] = (
                self.current_bar["volume"] * self.current_bar["close"] * 2
            )
            self.bars.append(self.current_bar)
            logger.debug(
                f"Added bar: {self.current_bar['timestamp']} - "
                f"O:{self.current_bar['open']:.2f} "
                f"H:{self.current_bar['high']:.2f} "
                f"L:{self.current_bar['low']:.2f} "
                f"C:{self.current_bar['close']:.2f}"
            )
            self.current_bar = None

    def _save_to_hdf5(self) -> None:
        """Save collected bars to HDF5 file."""
        if not self.bars:
            logger.warning("No bars to save")
            return

        df = pd.DataFrame(self.bars)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Get current month for filename
        month_str = df["timestamp"].iloc[0].strftime("%Y%m")
        filename = f"MNQ_dollar_bars_{month_str}.h5"
        filepath = self.output_dir / filename

        # Prepare data for HDF5
        data_array = []
        for _, row in df.iterrows():
            ts_ms = int(row["timestamp"].timestamp() * 1000)
            data_array.append([
                ts_ms,
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
                row["dollar_volume"]
            ])

        # Save to HDF5
        with h5py.File(filepath, "w") as f:
            f.create_dataset(
                "dollar_bars",
                data=data_array,
                compression="gzip"
            )

        logger.info(f"✅ Saved {len(df)} bars to {filepath}")
        print(f"\n📊 Data Collection Summary:")
        print(f"   Bars collected: {len(df)}")
        print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Saved to: {filepath}")


async def main():
    """Main entry point for real-time collection."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialize authentication
        logger.info("Initializing TradeStation authentication...")
        auth = TradeStationAuthWeb(port=8080)

        # Test authentication (will open browser)
        try:
            token = await auth.get_access_token()
            logger.info("✅ Authentication successful")
        except AuthenticationError as e:
            logger.error(f"❌ Authentication failed: {e}")
            logger.error("Please check your TradeStation credentials in .env file")
            logger.error("\nTo get valid credentials:")
            logger.error("1. Go to https://developers.tradestation.com/")
            logger.error("2. Create a developer account")
            logger.error("3. Generate API credentials (Client ID & Secret)")
            logger.error("4. Update your .env file with the new credentials")
            return
        finally:
            # Clean up auth resources
            await auth.cleanup()

        # Initialize collector
        collector = RealtimeDataCollector(auth)

        # Collect data for 1 hour (adjust as needed)
        await collector.start_collection(duration_minutes=60)

        logger.info("✅ Real-time data collection completed!")

    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())