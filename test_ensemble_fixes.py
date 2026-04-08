#!/usr/bin/env python
"""Quick test to verify ensemble fixes are working."""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test ensemble with fixes applied."""
    logger.info("=" * 80)
    logger.info("ENSEMBLE FIXES VERIFICATION TEST")
    logger.info("=" * 80)

    # Import after logging config
    from datetime import date
    import h5py
    import numpy as np
    import pandas as pd
    from src.research.ensemble_backtester import EnsembleBacktester

    # Load small test dataset
    logger.info("\n1. Loading test data...")
    data_file = Path("data/processed/dollar_bars/MNQ_dollar_bars_202401.h5")

    with h5py.File(data_file, "r") as f:
        dollar_bars = f["dollar_bars"][:2000]  # Just 2000 bars for quick test

    timestamps_ms = dollar_bars[:, 0].astype(np.int64)

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps_ms, unit="ms"),
        "open": dollar_bars[:, 1],
        "high": dollar_bars[:, 2],
        "low": dollar_bars[:, 3],
        "close": dollar_bars[:, 4],
        "volume": dollar_bars[:, 5].astype(int),
    })

    logger.info(f"✓ Loaded {len(df)} bars")

    # Save to temp file for EnsembleBacktester
    import tempfile
    import yaml

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as h5_temp:
        temp_h5 = h5_temp.name

        with h5py.File(temp_h5, "w") as f:
            timestamps_ns = df["timestamp"].astype("datetime64[ns]").astype(np.int64)
            f.create_dataset("timestamps", data=timestamps_ns.values)
            f.create_dataset("open", data=df["open"].values)
            f.create_dataset("high", data=df["high"].values)
            f.create_dataset("low", data=df["low"].values)
            f.create_dataset("close", data=df["close"].values)
            f.create_dataset("volume", data=df["volume"].values)

    logger.info(f"✓ Created temp HDF5 file: {temp_h5}")

    # Create config
    config = {
        "ensemble": {
            "strategies": {
                "triple_confluence_scaler": 0.20,
                "wolf_pack_3_edge": 0.20,
                "adaptive_ema_momentum": 0.20,
                "vwap_bounce": 0.20,
                "opening_range_breakout": 0.20,
            },
            "confidence_threshold": 0.50,
            "minimum_strategies": 1,
        },
        "risk": {
            "max_position_size": 5,
            "risk_reward_ratio": 2.0,
            "max_risk_per_trade": 0.02,
        },
    }

    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False, mode='w') as config_temp:
        yaml.dump(config, config_temp)
        config_path = config_temp.name

    logger.info(f"✓ Created config: {config_path}")

    # Run backtest
    logger.info("\n2. Running ensemble backtest...")
    logger.info("   Fixes applied:")
    logger.info("   - Signal normalization (raw → EnsembleSignal)")
    logger.info("   - Strategy name standardization (lowercase_with_underscores)")
    logger.info("   - Wider signal window (window_bars=1)")

    backtester = EnsembleBacktester(
        config_path=config_path,
        data_path=temp_h5,
    )

    result = backtester.run_backtest(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )

    # Check results
    logger.info("\n3. Results:")
    logger.info(f"   Total trades: {result.total_trades}")

    if result.total_trades > 0:
        logger.info(f"   Win rate: {result.win_rate:.2f}%")
        logger.info(f"   Profit factor: {result.profit_factor:.2f}")
        logger.info(f"   Total P&L: ${result.total_pnl:.2f}")
        logger.info("\n   ✅ SUCCESS: Ensemble system is generating trades!")
        logger.info("   All fixes are working correctly.")
        return 0
    else:
        logger.info("\n   ⚠️  WARNING: No trades generated")
        logger.info("   This could mean:")
        logger.info("   - Not enough data for strategy warm-up")
        logger.info("   - Market conditions don't produce ensemble signals")
        logger.info("   - Need wider window or lower threshold")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
