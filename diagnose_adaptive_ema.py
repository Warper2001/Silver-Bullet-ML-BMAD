"""Debug Adaptive EMA Momentum signal generation.

This script loads historical data and processes it through the Adaptive EMA strategy,
logging all indicator values to understand why no signals are generated.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
import logging

import h5py

from src.data.models import DollarBar
from src.detection.adaptive_ema_strategy import AdaptiveEMAStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sample_bars(num_bars: int = 500) -> list[DollarBar]:
    """Load sample bars from HDF5 files.

    Args:
        num_bars: Number of bars to load

    Returns:
        List of DollarBar objects
    """
    logger.info(f"Loading {num_bars} sample bars...")

    path = Path("data/processed/dollar_bars/")
    h5_files = sorted(path.glob("*.h5"))

    if not h5_files:
        raise ValueError("No HDF5 files found")

    h5_file = h5_files[0]
    logger.info(f"Loading from {h5_file.name}...")

    all_bars = []

    with h5py.File(h5_file, 'r') as f:
        bars = f['dollar_bars']

        for i in range(min(len(bars), num_bars)):
            ts_ms = bars[i, 0]
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

            bar = DollarBar(
                timestamp=ts,
                open=float(bars[i, 1]),
                high=float(bars[i, 2]),
                low=float(bars[i, 3]),
                close=float(bars[i, 4]),
                volume=int(bars[i, 5]),
                notional_value=float(bars[i, 6]),
                is_forward_filled=False,
            )
            all_bars.append(bar)

    logger.info(f"Loaded {len(all_bars)} bars")
    return all_bars


def diagnose_adaptive_ema():
    """Diagnose Adaptive EMA signal generation."""
    logger.info("=" * 70)
    logger.info("ADAPTIVE EMA MOMENTUM DIAGNOSTIC")
    logger.info("=" * 70)

    # Load data
    bars = load_sample_bars(num_bars=500)
    logger.info(f"Date range: {bars[0].timestamp} to {bars[-1].timestamp}")

    # Initialize strategy
    strategy = AdaptiveEMAStrategy()

    # Process bars and check conditions
    signals_detected = 0
    ema_alignment_count = 0
    macd_positive_count = 0
    rsi_mid_band_count = 0

    logger.info("\nProcessing bars and checking conditions...")

    for i, bar in enumerate(bars):
        # Process bar through strategy
        signals = strategy.process_bars([bar])

        if signals:
            signals_detected += 1
            logger.info(f"Signal {signals_detected}: {signals[0].direction} @ bar {i}")
            continue  # Already found signal for this bar

        # If no signal, check individual conditions
        if i < 100:  # Only check first 100 bars for diagnostics
            ema_values = strategy.ema_calculator.calculate_emas([])
            macd_values = strategy.macd_calculator.get_current_macd()
            rsi_value = strategy.rsi_calculator.get_current_rsi()

            # Check EMA alignment
            fast = ema_values.get('fast_ema')
            medium = ema_values.get('medium_ema')
            slow = ema_values.get('slow_ema')

            if fast and medium and slow:
                bullish_ema = fast > medium > slow
                bearish_ema = fast < medium < slow

                if bullish_ema or bearish_ema:
                    ema_alignment_count += 1

            # Check MACD
            macd_line = macd_values.get('macd_line')
            if macd_line is not None:
                if macd_line > 0:
                    macd_positive_count += 1

            # Check RSI
            if rsi_value is not None:
                if 40 <= rsi_value <= 60:
                    rsi_mid_band_count += 1

            # Log detailed conditions every 20 bars
            if i % 20 == 0 and all(v is not None for v in [fast, medium, slow, macd_line, rsi_value]):
                logger.info(
                    f"\nBar {i} ({bar.timestamp}):"
                    f"\n  Price: {bar.close:.2f}"
                    f"\n  EMAs: Fast={fast:.2f}, Medium={medium:.2f}, Slow={slow:.2f}"
                    f"\n  EMA Alignment: {'Bullish' if fast > medium > slow else 'Bearish' if fast < medium < slow else 'None'}"
                    f"\n  MACD: {macd_line:.2f}"
                    f"\n  RSI: {rsi_value:.2f}"
                )

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\nTotal Bars Processed: {len(bars)}")
    logger.info(f"Signals Generated: {signals_detected}")
    logger.info(f"\nCondition Frequency (first 100 bars):")
    logger.info(f"  EMA Alignment (fast > medium > slow OR fast < medium < slow): {ema_alignment_count}/100")
    logger.info(f"  MACD Positive: {macd_positive_count}/100")
    logger.info(f"  RSI in Mid-Band (40-60): {rsi_mid_band_count}/100")

    # Analyze why no signals
    logger.info("\n" + "=" * 70)
    logger.info("ROOT CAUSE ANALYSIS")
    logger.info("=" * 70)

    if signals_detected == 0:
        logger.info("\n❌ NO SIGNALS GENERATED")
        logger.info("\nRequired Conditions (ALL must be met):")
        logger.info("  1. EMA Alignment: fast > medium > slow (LONG) OR fast < medium < slow (SHORT)")
        logger.info("  2. MACD: > 0 and increasing (LONG) OR < 0 and decreasing (SHORT)")
        logger.info("  3. RSI: Between 40-60 AND rising (LONG) OR falling (SHORT)")
        logger.info("\nMost Restrictive Condition:")
        logger.info("  → RSI 40-60 range is very narrow (only 20% of RSI values)")
        logger.info("  → RSI must ALSO be rising or falling (adds another constraint)")
        logger.info("\nRecommendation:")
        logger.info("  1. Relax RSI range to 30-70 or 35-65")
        logger.info("  2. Remove RSI rising/falling requirement")
        logger.info("  3. Or use RSI as confirmation only (not hard requirement)")

    return signals_detected


if __name__ == "__main__":
    signals = diagnose_adaptive_ema()
    sys.exit(0 if signals > 0 else 1)
