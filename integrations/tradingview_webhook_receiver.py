#!/usr/bin/env python3
"""TradingView Webhook Receiver for MNQ Data.

This script receives MNQ futures data from TradingView alerts/webhooks
and automatically converts it to HDF5 format for backtesting.

Usage:
    python tradingview_webhook_receiver.py
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import h5py
import numpy as np
from flask import Flask, request, jsonify
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8080"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data/historical/mnq"))
MNQ_MULTIPLIER = 0.5

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Storage for received bars
received_bars = []
current_symbol = "MNQM26"

# Flask app
app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "bars_received": len(received_bars),
        "symbol": current_symbol
    })


@app.route('/webhook/tradingview', methods=['POST'])
def receive_tradingview_data():
    """Receive MNQ data from TradingView webhook."""
    global received_bars, current_symbol

    try:
        # Get data from request
        data = request.json

        # Handle TradingView alert format
        if not data:
            logger.warning("Received empty data")
            return jsonify({"error": "No data received"}), 400

        # Parse the bar data
        # TradingView sends: {timestamp, open, high, low, close, volume}
        bar = {
            'timestamp': data.get('timestamp'),
            'open': float(data.get('open', 0)),
            'high': float(data.get('high', 0)),
            'low': float(data.get('low', 0)),
            'close': float(data.get('close', 0)),
            'volume': int(data.get('volume', 0))
        }

        # Update symbol if provided
        if 'symbol' in data:
            current_symbol = data['symbol']

        # Validate MNQ price range (should be around 20,000-25,000)
        if bar['close'] < 1000 or bar['close'] > 100000:
            logger.warning(f"Suspicious price: ${bar['close']:.2f} - might not be MNQ")

        # Add to storage
        received_bars.append(bar)

        # Log every 100 bars
        if len(received_bars) % 100 == 0:
            logger.info(f"Received {len(received_bars)} bars total")

        # Auto-save every 500 bars
        if len(received_bars) % 500 == 0:
            save_to_hdf5()

        return jsonify({
            "status": "success",
            "bars_received": len(received_bars),
            "last_bar": bar
        })

    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/save', methods=['POST'])
def manual_save():
    """Manually trigger save to HDF5."""
    try:
        count = save_to_hdf5()
        return jsonify({
            "status": "success",
            "bars_saved": count,
            "file": f"{DATA_DIR}/{current_symbol}.h5"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset_data():
    """Clear received bars buffer."""
    global received_bars
    count = len(received_bars)
    received_bars = []
    logger.info(f"Reset buffer (cleared {count} bars)")
    return jsonify({"status": "reset", "bars_cleared": count})


def save_to_hdf5() -> int:
    """Save received bars to HDF5 file.

    Returns:
        Number of bars saved
    """
    global received_bars

    if not received_bars:
        logger.warning("No bars to save")
        return 0

    # Sort by timestamp
    received_bars.sort(key=lambda x: x['timestamp'])

    # Prepare structured numpy array
    dt = np.dtype([
        ("timestamp", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("volume", "i8"),
        ("notional_value", "f8"),
    ])

    # Convert to numpy array
    data = np.zeros(len(received_bars), dtype=dt)

    for i, bar in enumerate(received_bars):
        # Parse timestamp (could be Unix ms or ISO format)
        ts = bar['timestamp']
        if isinstance(ts, (int, float)):
            # Unix timestamp (might be ms)
            if ts > 1e12:  # Milliseconds
                ts_ns = int(ts * 1e6)
            else:  # Seconds
                ts_ns = int(ts * 1e9)
        else:
            # ISO format string
            dt_obj = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            ts_ns = int(dt_obj.timestamp() * 1e9)

        data[i] = (
            ts_ns,
            bar['open'],
            bar['high'],
            bar['low'],
            bar['close'],
            bar['volume'],
            bar['close'] * bar['volume'] * MNQ_MULTIPLIER,
        )

    # Write to HDF5
    output_path = DATA_DIR / f"{current_symbol}.h5"

    # Append to existing file or create new
    if output_path.exists():
        with h5py.File(output_path, "a") as h5file:
            if "historical_bars" in h5file:
                # Get existing data
                existing_data = h5file["historical_bars"]
                # Combine
                combined_data = np.concatenate([existing_data[:], data])
                # Remove duplicates (by timestamp)
                unique_data = np.unique(combined_data, axis=0)
                # Resize and write
                h5file["historical_bars"].resize(unique_data.shape)
                h5file["historical_bars"][:] = unique_data
                logger.info(f"Appended {len(data)} bars to existing file")
            else:
                h5file.create_dataset(
                    "historical_bars",
                    data=data,
                    compression="gzip",
                    compression_opts=1,
                    maxshape=(None,)
                )
                logger.info(f"Created new dataset with {len(data)} bars")
    else:
        with h5py.File(output_path, "w") as h5file:
            dataset = h5file.create_dataset(
                "historical_bars",
                data=data,
                compression="gzip",
                compression_opts=1,
                maxshape=(None,)  # Allow resizing
            )

            # Add metadata
            dataset.attrs["symbol"] = current_symbol
            dataset.attrs["count"] = len(data)
            dataset.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
            dataset.attrs["multiplier"] = MNQ_MULTIPLIER

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Saved {len(data)} bars to {output_path} ({file_size_mb:.2f} MB)")

    count = len(received_bars)
    received_bars = []  # Clear buffer
    return count


def main():
    """Start the webhook receiver."""
    logger.info("=" * 70)
    logger.info("TradingView Webhook Receiver for MNQ Data")
    logger.info("=" * 70)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Listening on port: {WEBHOOK_PORT}")
    logger.info(f"Webhook URL: http://your-server:{WEBHOOK_PORT}/webhook/tradingview")
    logger.info("")
    logger.info("Endpoints:")
    logger.info(f"  POST http://localhost:{WEBHOOK_PORT}/webhook/tradingview  - Receive data")
    logger.info(f"  POST http://localhost:{WEBHOOK_PORT}/save               - Manual save")
    logger.info(f"  POST http://localhost:{WEBHOOK_PORT}/reset              - Clear buffer")
    logger.info(f"  GET  http://localhost:{WEBHOOK_PORT}/health             - Status check")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Setup steps:")
    logger.info("1. Copy the Pine Script (in tradingview_pine_script.txt)")
    logger.info("2. Add script to your TradingView MNQ chart")
    logger.info("3. Create alert with webhook URL")
    logger.info("4. Data will auto-save every 500 bars")
    logger.info("")
    logger.info("Ready to receive data...")
    logger.info("=" * 70)

    # Run Flask app
    app.run(host='0.0.0.0', port=WEBHOOK_PORT, debug=False)


if __name__ == "__main__":
    main()
