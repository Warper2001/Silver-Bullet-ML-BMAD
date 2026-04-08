#!/usr/bin/env python3
"""Diagnostic tool to analyze why Triple Confluence strategy isn't generating signals."""

import h5py
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import pandas as pd

from src.data.models import DollarBar
from src.detection.level_sweep_detector import LevelSweepDetector
from src.detection.fvg_detector import SimpleFVGDetector
from src.detection.vwap_calculator import VWAPCalculator


def load_dollar_bars_from_h5(file_path: str) -> list[DollarBar]:
    """Load dollar bars from HDF5 file."""
    bars = []
    with h5py.File(file_path, 'r') as f:
        if 'dollar_bars' not in f:
            return []
        dataset = f['dollar_bars']
        for row in dataset:
            try:
                timestamp_ms = int(row[0])
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)
                bar = DollarBar(
                    timestamp=timestamp,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=int(row[5]),
                    notional_value=float(row[6]),
                    is_forward_filled=False,
                )
                bars.append(bar)
            except:
                continue
    return bars


def diagnose_indicators(bars: list[DollarBar], max_bars: int = 10000) -> dict:
    """Analyze individual indicators to see what's happening.

    Args:
        bars: List of dollar bars
        max_bars: Maximum bars to analyze (for speed)

    Returns:
        Dictionary with diagnostic results
    """
    # Limit bars for faster analysis
    bars = bars[:max_bars]

    # Initialize detectors
    sweep_detector = LevelSweepDetector(lookback_period=20)
    fvg_detector = SimpleFVGDetector(min_gap_size=4)
    vwap_calc = VWAPCalculator(session_start="09:30:00")

    # Track detections
    sweeps_detected = 0
    bullish_sweeps = 0
    bearish_sweeps = 0

    fvgs_detected = 0
    bullish_fvgs = 0
    bearish_fvgs = 0

    # Track confluence
    sweep_and_fvg = 0
    sweep_and_fvg_bullish = 0
    sweep_and_fvg_bearish = 0

    # Track VWAP bias
    vwap_bullish = 0
    vwap_bearish = 0
    vwap_neutral = 0

    # Process bars
    bars_list = []
    for bar in tqdm(bars, desc="Analyzing bars"):
        bars_list.append(bar)

        # Check for sweep
        sweep = sweep_detector.detect_sweep(bars_list)
        has_sweep = sweep is not None
        if has_sweep:
            sweeps_detected += 1
            if sweep.sweep_direction == "bullish":
                bullish_sweeps += 1
            else:
                bearish_sweeps += 1

        # Check for FVGs
        fvg_list = fvg_detector.detect_fvg(bars_list)
        has_recent_fvg = len(fvg_list) > 0

        if has_recent_fvg:
            fvgs_detected += len(fvg_list)
            for fvg in fvg_list:
                if fvg.fvg_type == "bullish":
                    bullish_fvgs += 1
                else:
                    bearish_fvgs += 1

        # Calculate VWAP and bias
        vwap = vwap_calc.calculate_vwap(bars_list)
        bias = vwap_calc.get_bias(bar.close, vwap)

        if bias == "bullish":
            vwap_bullish += 1
        elif bias == "bearish":
            vwap_bearish += 1
        else:
            vwap_neutral += 1

        # Check confluence
        if has_sweep and has_recent_fvg:
            sweep_and_fvg += 1

            # Check if directions align
            if sweep.sweep_direction == "bullish":
                # Check if any bullish FVGs
                has_bullish_fvg = any(f.fvg_type == "bullish" for f in fvg_list)
                if has_bullish_fvg:
                    sweep_and_fvg_bullish += 1
            else:
                has_bearish_fvg = any(f.fvg_type == "bearish" for f in fvg_list)
                if has_bearish_fvg:
                    sweep_and_fvg_bearish += 1

    return {
        'bars_analyzed': len(bars),
        'sweeps': {
            'total': sweeps_detected,
            'bullish': bullish_sweeps,
            'bearish': bearish_sweeps,
            'pct': sweeps_detected / len(bars) * 100 if bars else 0,
        },
        'fvgs': {
            'total': fvgs_detected,
            'bullish': bullish_fvgs,
            'bearish': bearish_fvgs,
            'pct_bars_with_fvg': fvgs_detected / len(bars) * 100 if bars else 0,
        },
        'vwap': {
            'bullish_bars': vwap_bullish,
            'bearish_bars': vwap_bearish,
            'neutral_bars': vwap_neutral,
            'bullish_pct': vwap_bullish / len(bars) * 100 if bars else 0,
            'bearish_pct': vwap_bearish / len(bars) * 100 if bars else 0,
        },
        'confluence': {
            'sweep_and_fvg': sweep_and_fvg,
            'sweep_and_fvg_bullish': sweep_and_fvg_bullish,
            'sweep_and_fvg_bearish': sweep_and_fvg_bearish,
        },
    }


def print_diagnostics(results: dict):
    """Print diagnostic results.

    Args:
        results: Diagnostic results dictionary
    """
    print("\n" + "=" * 80)
    print("TRIPLE CONFLUENCE DIAGNOSTIC REPORT")
    print("=" * 80)

    print(f"\n📊 Data Analyzed: {results['bars_analyzed']:,} bars")

    # Level Sweeps
    print(f"\n🎯 Level Sweeps:")
    print(f"   Total detected:     {results['sweeps']['total']}")
    print(f"   Bullish:            {results['sweeps']['bullish']} ({results['sweeps']['bullish']/results['sweeps']['total']*100 if results['sweeps']['total'] else 0:.1f}%)")
    print(f"   Bearish:            {results['sweeps']['bearish']} ({results['sweeps']['bearish']/results['sweeps']['total']*100 if results['sweeps']['total'] else 0:.1f}%)")
    print(f"   Frequency:          {results['sweeps']['pct']:.2f}% of bars")

    # FVGs
    print(f"\n📈 Fair Value Gaps:")
    print(f"   Total detected:     {results['fvgs']['total']}")
    print(f"   Bullish:            {results['fvgs']['bullish']}")
    print(f"   Bearish:            {results['fvgs']['bearish']}")
    print(f"   Bars with FVG:      {results['fvgs']['pct_bars_with_fvg']:.1f}% of bars")

    # VWAP Bias
    print(f"\n💹 VWAP Bias:")
    print(f"   Bullish bias:       {results['vwap']['bullish_bars']} bars ({results['vwap']['bullish_pct']:.1f}%)")
    print(f"   Bearish bias:       {results['vwap']['bearish_bars']} bars ({results['vwap']['bearish_pct']:.1f}%)")
    print(f"   Neutral:            {results['vwap']['neutral_bars']} bars")

    # Confluence
    print(f"\n🔗 Confluence Analysis:")
    print(f"   Sweep + FVG:        {results['confluence']['sweep_and_fvg']}")
    print(f"   Sweep + Bullish FVG: {results['confluence']['sweep_and_fvg_bullish']}")
    print(f"   Sweep + Bearish FVG: {results['confluence']['sweep_and_fvg_bearish']}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if results['sweeps']['total'] == 0:
        print("\n⚠️  NO LEVEL SWEEPS DETECTED")
        print("   → Consider reducing lookback_period (currently 20 bars)")
        print("   → Current lookback may be too long for this data frequency")

    if results['fvgs']['total'] == 0:
        print("\n⚠️  NO FAIR VALUE GAPS DETECTED")
        print("   → Consider reducing min_gap_size (currently 4 ticks)")
        print("   → FVGs may be rare in this market data")

    if results['confluence']['sweep_and_fvg'] == 0:
        print("\n⚠️  NO SWEEP + FVG CONFLUENCE")
        print("   → Triple confluence requires both patterns in same window")
        print("   → This is expected to be rare")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("\n1. Parameter Sensitivity Analysis:")
    print("   - Test lookback_period: [10, 15, 20, 25, 30]")
    print("   - Test min_gap_size: [2, 3, 4, 5, 6] ticks")
    print("   - Test confluence_window: [3, 5, 7, 10] bars")

    print("\n2. Alternative Approaches:")
    print("   - Relax confluence requirement to 2-of-3 factors")
    print("   - Increase event lookback window")
    print("   - Add minimum confidence threshold instead")

    print("\n3. Expected Behavior:")
    print("   - Triple confluence is intentionally rare")
    print("   - May generate 0-5 signals/month on current settings")
    print("   - High specificity but low frequency is expected")

    print("\n" + "=" * 80 + "\n")


def main():
    """Run diagnostic analysis."""
    print("\n🔍 Starting Triple Confluence Diagnostic Analysis\n")

    # Load sample of recent data (most recent file)
    data_dir = Path("data/processed/dollar_bars")
    h5_files = sorted(data_dir.glob("*.h5"))

    if not h5_files:
        print("❌ No HDF5 files found")
        return

    # Use most recent file for analysis
    latest_file = h5_files[-1]
    print(f"Loading: {latest_file.name}")

    bars = load_dollar_bars_from_h5(str(latest_file))
    print(f"Loaded {len(bars)} bars\n")

    if not bars:
        print("❌ No bars loaded")
        return

    # Run diagnostics
    results = diagnose_indicators(bars, max_bars=10000)

    # Print report
    print_diagnostics(results)


if __name__ == "__main__":
    main()
