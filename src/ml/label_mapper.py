"""Label Mapper for Meta-Labeling Training Data.

This module maps Silver Bullet signals to their trade outcomes to create
binary labels for meta-labeling model training.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def map_signals_to_outcomes(
    signals_df: pd.DataFrame,
    trades_df: pd.DataFrame
) -> pd.DataFrame:
    """Map each signal to its trade outcome.

    Creates a binary label for each signal:
    - Label = 1 if trade was profitable (return_pct > 0)
    - Label = 0 if trade was unprofitable (return_pct <= 0)

    Signals that don't result in trades are labeled as 0 (no trade taken).

    Args:
        signals_df: DataFrame with signal metadata.
            Required columns: direction (index is timestamp)
        trades_df: DataFrame with trade outcomes.
            Required columns: entry_time, return_pct

    Returns:
        DataFrame with columns:
        - direction: Signal direction (bullish/bearish)
        - confidence: Signal confidence score
        - label: Binary label (1=profitable, 0=unprofitable)
        - return_pct: Actual return percentage
        Index is signal timestamp (datetime)
    """
    logger.info(f"Mapping {len(signals_df)} signals to {len(trades_df)} trades...")

    # Create a copy of signals to avoid modifying original
    labeled_signals = signals_df.copy()

    # Initialize label and return_pct columns
    labeled_signals['label'] = 0
    labeled_signals['return_pct'] = 0.0

    # Create a mapping from entry_time to return_pct
    trade_outcomes = {}
    for _, trade in trades_df.iterrows():
        entry_time = pd.Timestamp(trade['entry_time'])
        return_pct = trade['return_pct']
        trade_outcomes[entry_time] = return_pct

    # Assign labels based on trade outcomes
    profitable_count = 0
    unprofitable_count = 0
    no_trade_count = 0

    for idx in labeled_signals.index:
        signal_time = pd.Timestamp(idx)

        # Check if this signal resulted in a trade
        if signal_time in trade_outcomes:
            return_pct = trade_outcomes[signal_time]
            labeled_signals.loc[idx, 'return_pct'] = return_pct

            # Binary label: 1 if profitable, 0 if unprofitable
            if return_pct > 0:
                labeled_signals.loc[idx, 'label'] = 1
                profitable_count += 1
            else:
                labeled_signals.loc[idx, 'label'] = 0
                unprofitable_count += 1
        else:
            # Signal didn't result in a trade
            labeled_signals.loc[idx, 'label'] = 0
            no_trade_count += 1

    logger.info(f"Label mapping complete:")
    logger.info(f"  Profitable: {profitable_count} ({profitable_count/len(labeled_signals)*100:.1f}%)")
    logger.info(f"  Unprofitable: {unprofitable_count} ({unprofitable_count/len(labeled_signals)*100:.1f}%)")
    logger.info(f"  No trade: {no_trade_count} ({no_trade_count/len(labeled_signals)*100:.1f}%)")

    return labeled_signals
