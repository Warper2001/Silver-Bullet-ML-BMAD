#!/usr/bin/env python3
"""Historical data collector for MNQ futures from TradeStation API.

This module provides functionality to fetch historical market data from TradeStation
and store it in HDF5 format for backtesting.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import h5py

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.auth_web import TradeStationAuthWeb
from src.data.exceptions import AuthenticationError

logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """Collect historical MNQ futures data from TradeStation.

    Features:
    - Fetch historical bar data (daily, hourly, minute)
    - Store in HDF5 format for efficient backtesting
    - Handle rate limiting and pagination
    - Support date range queries
    """

    BASE_URL = "https://api.tradestation.com/v3/marketdata/bars"

    def __init__(self, auth: TradeStationAuthWeb):
        """Initialize data collector.

        Args:
            auth: TradeStationAuth instance for API access
        """
        self.auth = auth

    async def fetch_historical_data(
        self,
        symbol: str = "MNQH26",  # MNQ March 2026 futures
        interval: str = "1min",   # 1-minute bars
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31"
    ) -> pd.DataFrame:
        """Fetch historical bar data from TradeStation.

        Args:
            symbol: Futures symbol (default: MNQH26)
            interval: Bar interval (1min, 5min, 1hour, 1day)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        import httpx

        # Ensure we have valid access token
        try:
            access_token = await self.auth.get_access_token()
        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            raise AuthenticationError(f"Authentication failed: {e}")

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }

        # TradeStation API endpoint for historical bars
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "bar_type": " trades"  # Trade-based bars
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self.BASE_URL,
                    headers=headers,
                    params=params,
                    timeout=30.0
                )

                if response.status_code == 200:
                    data = response.json()
                    return self._parse_response(data)
                elif response.status_code == 401:
                    raise AuthenticationError("Token expired or invalid")
                else:
                    logger.error(f"API Error: {response.status_code} - {response.text}")
                    raise Exception(f"API request failed: {response.status_code}")

            except httpx.TimeoutException:
                logger.error("Request timed out")
                raise Exception("Request timed out")

    def _parse_response(self, data: dict) -> pd.DataFrame:
        """Parse TradeStation API response into DataFrame.

        Args:
            data: JSON response from TradeStation

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        bars = data.get("Bars", [])

        if not bars:
            logger.warning("No bars returned from API")
            return pd.DataFrame()

        # Parse bars
        records = []
        for bar in bars:
            # TradeStation timestamp format: "2024-01-01T00:00:00Z"
            timestamp = pd.to_datetime(bar.get("Timestamp"))

            records.append({
                "timestamp": timestamp,
                "open": float(bar.get("Open", 0)),
                "high": float(bar.get("High", 0)),
                "low": float(bar.get("Low", 0)),
                "close": float(bar.get("Close", 0)),
                "volume": int(bar.get("TotalVolume", 0)),
            })

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Calculate dollar volume (MNQ multiplier: $2 per point)
        df["dollar_volume"] = df["volume"] * df["close"] * 2

        return df

    def save_to_hdf5(
        self,
        df: pd.DataFrame,
        output_path: Path,
        month_str: str
    ) -> None:
        """Save DataFrame to HDF5 format.

        Args:
            df: DataFrame with OHLCV data
            output_path: Directory to save file
            month_str: Month identifier (YYYYMM)
        """
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"MNQ_dollar_bars_{month_str}.h5"
        filepath = output_path / filename

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
                compression="gzip"  # Compress for smaller files
            )

        logger.info(f"Saved {len(df)} bars to {filepath}")

    async def collect_and_save(
        self,
        start_date: str,
        end_date: str,
        output_dir: str = "data/processed/dollar_bars",
        interval: str = "5min"
    ) -> None:
        """Collect historical data and save by month.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Output directory for HDF5 files
            interval: Bar interval
        """
        logger.info(f"Starting data collection: {start_date} to {end_date}")

        try:
            # Fetch data
            df = await self.fetch_historical_data(
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )

            if df.empty:
                logger.warning("No data retrieved")
                return

            logger.info(f"Retrieved {len(df)} bars")

            # Group by month and save
            output_path = Path(output_dir)
            df_grouped = df.set_index("timestamp").resample("ME")

            for month_start, month_df in df_grouped:
                month_str = month_start.strftime("%Y%m")
                month_df = month_df.reset_index()

                self.save_to_hdf5(month_df, output_path, month_str)

            logger.info("Data collection completed successfully")

        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            raise


async def main():
    """Main entry point for data collection."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
            return
        finally:
            # Clean up auth resources
            await auth.cleanup()

        # Initialize collector
        collector = HistoricalDataCollector(auth)

        # Collect 3 months of data (example)
        await collector.collect_and_save(
            start_date="2024-12-19",
            end_date="2025-03-19",
            interval="5min"  # 5-minute bars
        )

        logger.info("✅ Data collection completed!")

    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())