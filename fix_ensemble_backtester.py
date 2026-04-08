"""Patch EnsembleBacktester to use real strategies instead of mock signals."""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_dollar_bar_from_series(bar_series):
    """Convert pandas Series to DollarBar object.

    Args:
        bar_series: pandas Series with timestamp, open, high, low, close, volume

    Returns:
        DollarBar object
    """
    from src.data.models import DollarBar

    return DollarBar(
        timestamp=bar_series['timestamp'].to_pydatetime(),
        open=float(bar_series['open']),
        high=float(bar_series['high']),
        low=float(bar_series['low']),
        close=float(bar_series['close']),
        volume=int(bar_series['volume']),
        notional_value=bar_series.get('notional', 0),
        is_forward_filled=False,
    )


def patch_ensemble_backtester():
    """Patch EnsembleBacktester to use real strategies.

    This creates a new version of EnsembleBacktester that:
    1. Imports and initializes the 5 real strategies
    2. Converts pandas Series to DollarBar objects
    3. Calls strategies to get real signals
    4. Processes actual ensemble signals
    """

    logger.info("Creating patched EnsembleBacktester with real strategies...")

    # Read the original file
    original_file = Path("src/research/ensemble_backtester.py")
    backup_file = Path("src/research/ensemble_backtester.py.backup")

    # Backup original
    if not backup_file.exists():
        import shutil
        shutil.copy(original_file, backup_file)
        logger.info(f"✓ Backed up original to {backup_file}")

    # Read original code
    with open(original_file, 'r') as f:
        original_code = f.read()

    # Create the patched version
    patched_code = original_code.replace(
        """        # For testing: generate mock signals periodically
        signal_counter = 0

        # Process bars chronologically
        for idx, bar in bars.iterrows():
            # Generate mock signals for testing (every 100 bars)
            signal_counter += 1
            if signal_counter % 100 == 0:
                # Create mock ensemble signal
                from src.detection.models import EnsembleSignal

                is_long = signal_counter % 200 == 0

                # Calculate stop loss and take profit based on direction
                if is_long:
                    stop_loss = bar["close"] * 0.999  # Below entry
                    take_profit = bar["close"] * 1.002  # Above entry
                else:
                    stop_loss = bar["close"] * 1.001  # Above entry for short
                    take_profit = bar["close"] * 0.998  # Below entry for short

                mock_signal = EnsembleSignal(
                    strategy_name="Mock Strategy",
                    timestamp=bar["timestamp"],
                    direction="long" if is_long else "short",
                    entry_price=bar["close"],
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=0.75,
                    metadata={},
                    bar_timestamp=bar["timestamp"],
                )

                self._aggregator.add_signal(mock_signal)""",
        """        # Initialize strategies (call real strategies, not mock signals)
        if not hasattr(self, '_strategies_initialized'):
            self._initialize_strategies()
            self._strategies_initialized = True

        # Process bars chronologically
        for idx, bar in bars.iterrows():
            # Convert pandas Series to DollarBar for strategies
            dollar_bar = create_dollar_bar_from_series(bar)

            # Call each strategy and collect signals
            self._process_bar_with_strategies(dollar_bar)"""
    )

    # Write patched version
    with open(original_file, 'w') as f:
        f.write(patched_code)

    logger.info(f"✓ Patched {original_file}")

    # Now add the new methods needed
    add_new_methods(original_file)


def add_new_methods(file_path):
    """Add new methods to EnsembleBacktester."""

    with open(file_path, 'r') as f:
        code = f.read()

    # Find the class definition and add new methods before it ends
    new_methods = '''

    def _initialize_strategies(self):
        """Initialize all 5 trading strategies."""
        from src.detection.triple_confluence_strategy import TripleConfluenceStrategy
        from src.detection.wolf_pack_strategy import WolfPackStrategy
        from src.detection.adaptive_ema_strategy import AdaptiveEMAStrategy
        from src.detection.vwap_bounce_strategy import VWAPBounceStrategy
        from src.detection.opening_range_strategy import OpeningRangeStrategy

        logger.info("Initializing real trading strategies...")

        self._strategies = []

        # Strategy 1: Triple Confluence
        try:
            tc = TripleConfluenceStrategy(config={})
            self._strategies.append(("triple_confluence", tc))
            logger.info("  ✓ Triple Confluence initialized")
        except Exception as e:
            logger.warning(f"  ✗ Triple Confluence failed: {e}")

        # Strategy 2: Wolf Pack (needs tick_size and risk_ticks)
        try:
            wp = WolfPackStrategy(tick_size=0.25, risk_ticks=20, min_confidence=0.8)
            self._strategies.append(("wolf_pack", wp))
            logger.info("  ✓ Wolf Pack initialized")
        except Exception as e:
            logger.warning(f"  ✗ Wolf Pack failed: {e}")

        # Strategy 3: Adaptive EMA
        try:
            ae = AdaptiveEMAStrategy()
            self._strategies.append(("adaptive_ema", ae))
            logger.info("  ✓ Adaptive EMA initialized")
        except Exception as e:
            logger.warning(f"  ✗ Adaptive EMA failed: {e}")

        # Strategy 4: VWAP Bounce
        try:
            vb = VWAPBounceStrategy(config={})
            self._strategies.append(("vwap_bounce", vb))
            logger.info("  ✓ VWAP Bounce initialized")
        except Exception as e:
            logger.warning(f"  ✗ VWAP Bounce failed: {e}")

        # Strategy 5: Opening Range
        try:
            ob = OpeningRangeStrategy(config={})
            self._strategies.append(("opening_range", ob))
            logger.info("  ✓ Opening Range initialized")
        except Exception as e:
            logger.warning(f"  ✗ Opening Range failed: {e}")

        logger.info(f"Initialized {len(self._strategies)}/5 strategies")

    def _process_bar_with_strategies(self, dollar_bar):
        """Process a single bar with all strategies and aggregate signals.

        Args:
            dollar_bar: DollarBar object
        """
        for strategy_name, strategy in self._strategies:
            try:
                # Call strategy based on its interface
                if hasattr(strategy, 'process_bar'):
                    signal = strategy.process_bar(dollar_bar)
                elif hasattr(strategy, 'process_bars'):
                    # Some strategies need list of bars
                    signal = strategy.process_bars([dollar_bar])
                else:
                    continue

                # If signal generated, add to aggregator
                if signal is not None:
                    self._aggregator.add_signal(signal, strategy_name=strategy_name)

            except Exception as e:
                # Log error but continue processing other strategies
                logger.debug(f"Strategy {strategy_name} error on bar: {e}")
                continue
'''

    # Insert new methods before the final closing of the class
    # Find the last "def " before end of class
    lines = code.split('\n')

    # Find where to insert (before _calculate_performance_metrics)
    insert_index = None
    for i, line in enumerate(lines):
        if 'def _calculate_performance_metrics' in line:
            insert_index = i
            break

    if insert_index:
        lines.insert(insert_index, new_methods)
        patched_code = '\n'.join(lines)

        with open(file_path, 'w') as f:
            f.write(patched_code)

        logger.info("✓ Added new methods to EnsembleBacktester")
    else:
        logger.error("Could not find insertion point for new methods")


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("PATCHING ENSEMBLE BACKTESTER")
    logger.info("=" * 80)

    try:
        patch_ensemble_backtester()

        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ PATCH COMPLETE")
        logger.info("=" * 80)
        logger.info("")
        logger.info("The EnsembleBacktester has been patched to use real strategies.")
        logger.info("")
        logger.info("Changes:")
        logger.info("  1. Added _initialize_strategies() method")
        logger.info("  2. Added _process_bar_with_strategies() method")
        logger.info("  3. Replaced mock signal generation with real strategy calls")
        logger.info("  4. Converts pandas Series to DollarBar objects")
        logger.info("")
        logger.info("Original file backed up to:")
        logger.info("  src/research/ensemble_backtester.py.backup")
        logger.info("")
        logger.info("You can now re-run the full dataset test:")
        logger.info("  .venv/bin/python run_epic2_full_dataset.py")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error patching: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
