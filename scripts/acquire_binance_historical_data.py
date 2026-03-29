#!/usr/bin/env python3
"""
Binance Historical Data Acquisition Script

This script fetches historical kline (candlestick) data from Binance API
and saves it to Parquet files for analysis and model training.

Features:
- Fetch multiple intervals (1m, 5m, 15m, 1h, 1d)
- Pagination support for large date ranges
- Data quality validation
- Gap detection and reporting
- Progress tracking

Usage:
    python scripts/acquire_binance_historical_data.py --symbol BTCUSDT --interval 5m --days 365

API Documentation: https://binance-docs.github.io/apidocs/#kline-candlestick-data
"""

import asyncio
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import httpx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class BinanceDataAcquirer:
    """
    Acquire historical kline data from Binance API.

    Attributes:
        base_url: Binance API base URL
        symbol: Trading symbol (e.g., BTCUSDT)
        interval: Kline interval (1m, 5m, 15m, 1h, 1d)
        days: Number of days of history to fetch
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "5m",
        days: int = 365,
        base_url: str = "https://api.binance.com",
    ) -> None:
        """
        Initialize data acquirer.

        Args:
            symbol: Trading symbol (default: BTCUSDT)
            interval: Kline interval (default: 5m)
            days: Number of days of history (default: 365)
            base_url: Binance API base URL (default: production)
        """
        self.symbol = symbol.upper()
        self.interval = interval
        self.days = days
        self.base_url = base_url

        # Calculate time range
        self.end_time = datetime.now(timezone.utc)
        self.start_time = self.end_time - timedelta(days=days)

        logger.info(f"Acquiring {self.symbol} data for {self.days} days ({self.interval})")
        logger.info(f"Time range: {self.start_time} to {self.end_time}")

    async def fetch_klines(
        self,
        limit: int = 1000,
    ) -> list[list]:
        """
        Fetch klines from Binance API.

        Args:
            limit: Number of klines per request (max 1000)

        Returns:
            List of kline data arrays

        API Docs: https://binance-docs.github.io/apidocs/#kline-candlestick-data
        """
        klines_data = []
        current_start_time = int(self.start_time.timestamp() * 1000)
        end_time_ms = int(self.end_time.timestamp() * 1000)

        request_count = 0
        total_klines = 0

        async with httpx.AsyncClient() as client:
            while current_start_time < end_time_ms:
                try:
                    # Build request parameters
                    params = {
                        "symbol": self.symbol,
                        "interval": self.interval,
                        "startTime": current_start_time,
                        "endTime": end_time_ms,
                        "limit": limit,
                    }

                    request_count += 1
                    logger.debug(
                        f"Request #{request_count}: "
                        f"Fetching {limit} klines from {datetime.fromtimestamp(current_start_time / 1000, tz=timezone.utc)}"
                    )

                    # Send request
                    response = await client.get(
                        f"{self.base_url}/api/v3/klines",
                        params=params,
                    )
                    response.raise_for_status()

                    # Parse response
                    klines = response.json()

                    if not klines:
                        logger.warning("No more klines available")
                        break

                    klines_data.extend(klines)
                    total_klines += len(klines)

                    # Update start time for next request
                    # Use the close time of the last kline + 1ms
                    current_start_time = klines[-1][6] + 1

                    # Progress update
                    progress = min(100, (current_start_time - int(self.start_time.timestamp() * 1000)) / (end_time_ms - int(self.start_time.timestamp() * 1000)) * 100)
                    logger.info(
                        f"Fetched {len(klines)} klines (total: {total_klines}, progress: {progress:.1f}%)"
                    )

                    # Rate limiting: Binance allows 1200 weight/minute
                    # klines endpoint weight = 1
                    await asyncio.sleep(0.1)  # Small delay to be safe

                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                    raise
                except Exception as e:
                    logger.error(f"Failed to fetch klines: {e}")
                    raise

        logger.info(f"Total klines fetched: {len(klines_data)} in {request_count} requests")

        return klines_data

    def validate_data(self, klines_data: list[list]) -> pd.DataFrame:
        """
        Validate and convert klines data to DataFrame.

        Args:
            klines_data: Raw klines data from API

        Returns:
            DataFrame with validated klines data
        """
        # Define column names (Binance API format)
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]

        # Create DataFrame
        df = pd.DataFrame(klines_data, columns=columns)

        # Convert types
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_base",
            "taker_buy_quote",
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["trades"] = df["trades"].astype(int)

        # Sort by time
        df = df.sort_values("open_time").reset_index(drop=True)

        # Data quality checks
        logger.info("Running data quality validation...")

        # Check for NaN values
        nan_counts = df[numeric_columns].isna().sum()
        if nan_counts.any():
            logger.warning(f"NaN values found:\n{nan_counts[nan_counts > 0]}")

        # Check OHLC relationships
        invalid_ohlc = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )
        if invalid_ohlc.any():
            logger.warning(f"Invalid OHLC relationships in {invalid_ohlc.sum()} rows")

        # Check for negative values
        negative_values = (df[numeric_columns] < 0).any()
        if negative_values.any():
            logger.warning(f"Negative values found:\n{negative_values[negative_values]}")

        # Check for duplicates
        duplicates = df.duplicated(subset=["open_time"]).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate timestamps")

        # Detect gaps
        df["time_diff"] = df["open_time"].diff()
        # Expected time difference based on interval
        interval_minutes = self._parse_interval()
        expected_diff = pd.Timedelta(minutes=interval_minutes)

        gaps = df[df["time_diff"] > expected_diff * 1.5]  # Allow 50% tolerance
        if len(gaps) > 0:
            logger.warning(f"Found {len(gaps)} time gaps > 1.5x expected interval")
            for _, row in gaps.head(10).iterrows():
                logger.warning(
                    f"  Gap at {row['open_time']}: "
                    f"{row['time_diff'].total_seconds() / 60:.1f} minutes "
                    f"(expected ~{interval_minutes}m)"
                )

        logger.info("Data quality validation complete")

        return df

    def _parse_interval(self) -> int:
        """
        Parse interval string to minutes.

        Returns:
            Interval in minutes
        """
        interval_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }

        return interval_map.get(self.interval.lower(), 5)  # Default to 5m

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        output_dir: str | None = None,
    ) -> Path:
        """
        Save klines data to Parquet file.

        Args:
            df: DataFrame with klines data
            output_dir: Output directory (default: data/binance/historical)

        Returns:
            Path to saved file
        """
        # Create output directory
        if output_dir is None:
            output_dir = "data/binance/historical"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"{self.symbol}_{self.interval}_{self.days}days.parquet"
        file_path = output_path / filename

        # Save to Parquet
        df.to_parquet(file_path, index=False, compression="snappy")

        logger.info(f"Saved {len(df)} klines to {file_path}")
        logger.info(f"File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")

        return file_path


async def main():
    """Main entry point for data acquisition."""
    parser = argparse.ArgumentParser(
        description="Acquire historical kline data from Binance"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Kline interval (default: 5m)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of history to fetch (default: 365)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/binance/historical)",
    )

    args = parser.parse_args()

    # Create acquirer
    acquirer = BinanceDataAcquirer(
        symbol=args.symbol,
        interval=args.interval,
        days=args.days,
    )

    # Fetch data
    klines_data = await acquirer.fetch_klines()

    # Validate data
    df = acquirer.validate_data(klines_data)

    # Save to Parquet
    output_path = acquirer.save_to_parquet(df, args.output_dir)

    logger.info("Data acquisition complete!")


if __name__ == "__main__":
    asyncio.run(main())
