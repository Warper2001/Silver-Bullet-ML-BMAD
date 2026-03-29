#!/usr/bin/env python3
"""
BYDFI Historical Data Acquisition Script

Acquires historical kline data from BYDFI for ML model training.
Based on BYDFI API documentation.

Usage:
    python scripts/acquire_bydfi_historical_data.py --symbol BTC-USDT --interval 5m --days 365

Options:
    --symbol: Trading symbol (default: BTC-USDT)
    --interval: Kline interval (default: 5m)
    --days: Number of days to acquire (default: 365)
    --output-dir: Output directory (default: data/bydfi/historical)
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from src.data.bydfi_config import load_bydfi_settings
from src.execution.bydfi.client import BYDFIClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


async def fetch_klines(
    client: BYDFIClient,
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime,
) -> list[dict]:
    """
    Fetch klines for a time range.

    Args:
        client: BYDFI client
        symbol: Trading symbol
        interval: Kline interval
        start_time: Start time
        end_time: End time

    Returns:
        list[dict]: Kline data
    """
    klines = []

    # BYDFI API returns up to 1000 klines per request
    # Implement pagination if needed
    try:
        response_klines = await client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            limit=1000,
        )

        for kline in response_klines:
            klines.append(
                {
                    "timestamp": kline.timestamp.isoformat(),
                    "open": kline.open,
                    "high": kline.high,
                    "low": kline.low,
                    "close": kline.close,
                    "volume": kline.volume,
                }
            )

        logger.info(f"Fetched {len(klines)} klines for {symbol}")

    except Exception as e:
        logger.error(f"Error fetching klines: {e}")

    return klines


async def acquire_historical_data(
    symbol: str,
    interval: str,
    days: int,
    output_dir: Path,
):
    """
    Acquire historical data from BYDFI.

    Args:
        symbol: Trading symbol
        interval: Kline interval
        days: Number of days to acquire
        output_dir: Output directory
    """
    console.print(f"[bold blue]Acquiring BYDFI historical data[/bold blue]")
    console.print(f"Symbol: {symbol}")
    console.print(f"Interval: {interval}")
    console.print(f"Days: {days}")
    console.print(f"Output: {output_dir}")
    console.print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize client
    client = BYDFIClient()

    try:
        await client.__aenter__()

        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        console.print(f"[cyan]Time range:[/cyan] {start_time.isoformat()} to {end_time.isoformat()}")
        console.print()

        # Fetch klines with progress bar
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Fetching klines...",
                total=days,
            )

            klines = []
            current_day = 0

            while current_day < days:
                # Fetch day of data
                day_start = start_time + timedelta(days=current_day)
                day_end = day_start + timedelta(days=1)

                try:
                    day_klines = await fetch_klines(
                        client=client,
                        symbol=symbol,
                        interval=interval,
                        start_time=day_start,
                        end_time=day_end,
                    )

                    klines.extend(day_klines)

                except Exception as e:
                    logger.error(f"Error fetching data for day {current_day}: {e}")

                progress.update(task, advance=1)
                current_day += 1

                # Rate limiting
                await asyncio.sleep(0.1)

        # Save to file
        output_file = output_dir / f"{symbol}_{interval}_{days}days.json"

        with open(output_file, "w") as f:
            json.dump(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "klines": klines,
                },
                f,
                indent=2,
            )

        console.print()
        console.print(f"[green]✓[/green] Saved {len(klines)} klines to {output_file}")

    finally:
        await client.__aexit__(None, None, None)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Acquire BYDFI historical data",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC-USDT",
        help="Trading symbol (default: BTC-USDT)",
    )

    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        help="Kline interval (default: 5m)",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days to acquire (default: 365)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/bydfi/historical"),
        help="Output directory (default: data/bydfi/historical)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            acquire_historical_data(
                symbol=args.symbol,
                interval=args.interval,
                days=args.days,
                output_dir=args.output_dir,
            )
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Data acquisition interrupted by user[/yellow]")
        sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.error(f"Data acquisition failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
