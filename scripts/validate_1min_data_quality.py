#!/usr/bin/env python3
"""
Comprehensive 1-Minute Data Quality Validation

Validates the MNQ 2025 1-minute dollar bar data for quality issues:
- Timestamp continuity and gaps
- Price range realism (MNQ ~21,000)
- Volume and notional values
- Regime distribution sanity check
- Feature coverage and NaN analysis

This ensures data reliability before running backtests and deploying models.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def validate_timestamp_continuity(df: pd.DataFrame, regime_id: int) -> dict:
    """Validate timestamp continuity and detect gaps.

    Args:
        df: DataFrame with timestamp column
        regime_id: Regime being validated

    Returns:
        Validation results dict
    """
    print(f"\n{'=' * 80}")
    print(f"Regime {regime_id} - Timestamp Continuity Validation")
    print(f"{'=' * 80}")

    if 'timestamp' not in df.columns:
        return {'status': 'ERROR', 'message': 'No timestamp column found'}

    # Convert to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Calculate time differences
    df['time_diff'] = df['timestamp'].diff()

    # Expected 1-minute frequency (allow some slack)
    expected_freq = timedelta(minutes=1)
    tolerance = timedelta(seconds=30)  # Allow 30s tolerance

    # Identify gaps
    large_gaps = df[df['time_diff'] > expected_freq + tolerance]

    # Statistics
    total_bars = len(df)
    total_time = df['timestamp'].max() - df['timestamp'].min()
    expected_bars = total_time.total_seconds() / 60

    # Calculate coverage (accounting for market hours)
    # MNQ trades 6.5 hours/day = 390 minutes/day
    trading_days = (total_time.total_seconds() / 60) / 390
    expected_bars_trading = int(trading_days * 390)
    coverage = total_bars / expected_bars_trading if expected_bars_trading > 0 else 0

    results = {
        'status': 'PASS' if len(large_gaps) == 0 else 'WARNING',
        'total_bars': total_bars,
        'date_range': {
            'start': str(df['timestamp'].min()),
            'end': str(df['timestamp'].max()),
            'duration_days': total_time.total_seconds() / 86400
        },
        'large_gaps': len(large_gaps),
        'expected_bars': int(expected_bars_trading),
        'coverage_percent': coverage * 100,
        'details': {
            'gap_count': len(large_gaps),
            'avg_gap_minutes': large_gaps['time_diff'].dt.total_seconds().div(60).mean() if len(large_gaps) > 0 else 0,
            'max_gap_minutes': large_gaps['time_diff'].dt.total_seconds().div(60).max() if len(large_gaps) > 0 else 0
        }
    }

    print(f"Total bars: {total_bars:,}")
    print(f"Date range: {results['date_range']['start']} to {results['date_range']['end']}")
    print(f"Duration: {results['date_range']['duration_days']:.1f} days")
    print(f"Expected bars (trading hours): {results['expected_bars']:,}")
    print(f"Coverage: {coverage:.1%}")
    print(f"Large gaps detected: {len(large_gaps)}")

    if len(large_gaps) > 0:
        print(f"\n⚠️  GAPS FOUND:")
        print(f"  Average gap: {results['details']['avg_gap_minutes']:.1f} minutes")
        print(f"  Max gap: {results['details']['max_gap_minutes']:.1f} minutes")
        print(f"  First 5 gaps:")
        for idx, row in large_gaps.head(5).iterrows():
            print(f"    {row['timestamp']} - gap: {row['time_diff'].total_seconds() / 60:.1f} minutes")

    return results


def validate_price_realism(df: pd.DataFrame, regime_id: int) -> dict:
    """Validate price ranges are realistic for MNQ.

    Args:
        df: DataFrame with OHLC columns
        regime_id: Regime being validated

    Returns:
        Validation results dict
    """
    print(f"\n{'=' * 80}")
    print(f"Regime {regime_id} - Price Realism Validation")
    print(f"{'=' * 80}")

    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        return {'status': 'ERROR', 'message': f'Missing columns: {missing_cols}'}

    # Statistics
    min_price = df[['open', 'high', 'low', 'close']].min().min()
    max_price = df[['open', 'high', 'low', 'close']].max().max()
    avg_price = df[['open', 'high', 'low', 'close']].mean().mean()

    # Check for MNQ realism (Micro E-mini Nasdaq-100)
    # MNQ typically trades 18,000 - 25,000 range in 2025
    expected_min = 15000
    expected_max = 30000

    # Check for OHLC consistency
    invalid_ohlc = df[
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ]

    # Check for negative prices
    negative_prices = df[
        (df['open'] <= 0) |
        (df['high'] <= 0) |
        (df['low'] <= 0) |
        (df['close'] <= 0)
    ]

    # Check for extreme outliers (10x move in single bar)
    df['range'] = df['high'] - df['low']
    df['range_pct'] = (df['range'] / df['close']) * 100
    extreme_moves = df[df['range_pct'] > 10]  # >10% move in 1 minute

    results = {
        'status': 'PASS',
        'min_price': float(min_price),
        'max_price': float(max_price),
        'avg_price': float(avg_price),
        'price_range_valid': min_price >= expected_min and max_price <= expected_max,
        'invalid_ohlc_count': len(invalid_ohlc),
        'negative_price_count': len(negative_prices),
        'extreme_move_count': len(extreme_moves)
    }

    print(f"Price range: ${min_price:.2f} - ${max_price:.2f}")
    print(f"Average price: ${avg_price:.2f}")
    print(f"Expected range (MNQ 2025): ${expected_min:,} - ${expected_max:,}")
    print(f"Price range valid: {results['price_range_valid']}")
    print(f"Invalid OHLC bars: {len(invalid_ohlc)}")
    print(f"Negative price bars: {len(negative_prices)}")
    print(f"Extreme moves (>10% in 1 min): {len(extreme_moves)}")

    if not results['price_range_valid']:
        print(f"⚠️  WARNING: Prices outside expected MNQ range!")
        results['status'] = 'WARNING'

    if len(invalid_ohlc) > 0:
        print(f"⚠️  WARNING: {len(invalid_ohlc)} bars with invalid OHLC relationships!")
        results['status'] = 'WARNING'

    if len(negative_prices) > 0:
        print(f"❌ ERROR: {len(negative_prices)} bars with non-positive prices!")
        results['status'] = 'ERROR'

    if len(extreme_moves) > 0:
        print(f"⚠️  WARNING: {len(extreme_moves)} bars with extreme moves (>10%)")
        print(f"  Max move: {extreme_moves['range_pct'].max():.2f}%")
        results['status'] = 'WARNING'

    return results


def validate_volume_and_notional(df: pd.DataFrame, regime_id: int) -> dict:
    """Validate volume and notional values.

    Args:
        df: DataFrame with volume and notional columns
        regime_id: Regime being validated

    Returns:
        Validation results dict
    """
    print(f"\n{'=' * 80}")
    print(f"Regime {regime_id} - Volume & Notional Validation")
    print(f"{'=' * 80}")

    results = {'status': 'PASS', 'details': {}}

    # Check volume if present
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        total_volume = df['volume'].sum()
        avg_volume = df['volume'].mean()
        zero_volume = (df['volume'] == 0).sum()

        results['details']['volume'] = {
            'total_volume': float(total_volume),
            'avg_volume': float(avg_volume),
            'zero_volume_count': int(zero_volume)
        }

        print(f"Total volume: {total_volume:,.0f}")
        print(f"Average volume per bar: {avg_volume:,.0f}")
        print(f"Zero volume bars: {zero_volume}")

        if zero_volume > len(df) * 0.1:  # >10% zero volume
            print(f"⚠️  WARNING: High percentage of zero-volume bars!")
            results['status'] = 'WARNING'

    # Check notional if present
    if 'notional' in df.columns:
        df['notional'] = pd.to_numeric(df['notional'], errors='coerce')
        total_notional = df['notional'].sum()
        avg_notional = df['notional'].mean()

        # Dollar bar threshold check
        # Should be close to $50M threshold
        expected_notional = 50_000_000
        notional_std = df['notional'].std()

        results['details']['notional'] = {
            'total_notional': float(total_notional),
            'avg_notional': float(avg_notional),
            'notional_std': float(notional_std),
            'expected_notional': expected_notional
        }

        print(f"\nTotal notional: ${total_notional:,.0f}")
        print(f"Average notional per bar: ${avg_notional:,.0f}")
        print(f"Notional std dev: ${notional_std:,.0f}")
        print(f"Expected notional (dollar bar threshold): ${expected_notional:,}")

        # Check if notional is close to threshold (within 20%)
        notional_diff_pct = abs(avg_notional - expected_notional) / expected_notional
        if notional_diff_pct > 0.2:
            print(f"⚠️  WARNING: Notional differs from threshold by {notional_diff_pct:.1%}!")
            results['status'] = 'WARNING'
        else:
            print(f"✅ Notional values consistent with dollar bar threshold")

    return results


def validate_regime_distribution(df: pd.DataFrame, regime_id: int) -> dict:
    """Validate regime distribution makes sense.

    Args:
        df: DataFrame (single regime)
        regime_id: Regime being validated

    Returns:
        Validation results dict
    """
    print(f"\n{'=' * 80}")
    print(f"Regime {regime_id} - Regime Distribution Check")
    print(f"{'=' * 80}")

    results = {
        'status': 'PASS',
        'regime_id': regime_id,
        'bar_count': len(df)
    }

    print(f"Regime {regime_id} bars: {len(df):,}")

    # Expected ranges from previous analysis
    expected_ranges = {
        0: (15000, 25000),  # Regime 0: ~7% of data
        1: (200000, 300000),  # Regime 1: ~91% of data
        2: (5000, 15000)  # Regime 2: ~2% of data
    }

    if regime_id in expected_ranges:
        min_expected, max_expected = expected_ranges[regime_id]
        in_range = min_expected <= len(df) <= max_expected

        results['in_expected_range'] = in_range
        results['expected_range'] = [min_expected, max_expected]

        if not in_range:
            print(f"⚠️  WARNING: Regime {regime_id} has {len(df):,} bars")
            print(f"  Expected: {min_expected:,} - {max_expected:,} bars")
            results['status'] = 'WARNING'
        else:
            print(f"✅ Regime {regime_id} within expected range")

    return results


def validate_feature_coverage(df: pd.DataFrame, regime_id: int) -> dict:
    """Validate Tier 1 feature coverage and NaN analysis.

    Args:
        df: DataFrame with Tier 1 features
        regime_id: Regime being validated

    Returns:
        Validation results dict
    """
    print(f"\n{'=' * 80}")
    print(f"Regime {regime_id} - Feature Coverage Validation")
    print(f"{'=' * 80}")

    tier1_features = [
        'volume_imbalance_3', 'volume_imbalance_5', 'volume_imbalance_10',
        'cumulative_delta_20', 'cumulative_delta_50', 'cumulative_delta_100',
        'realized_vol_15', 'realized_vol_30', 'realized_vol_60',
        'vwap_deviation_5', 'vwap_deviation_10', 'vwap_deviation_20',
        'bid_ask_bounce',
        'noise_adj_momentum_5', 'noise_adj_momentum_10', 'noise_adj_momentum_20',
    ]

    available_features = [f for f in tier1_features if f in df.columns]
    missing_features = [f for f in tier1_features if f not in df.columns]

    print(f"Tier 1 features: {len(available_features)}/{len(tier1_features)} present")

    if len(available_features) > 0:
        # Calculate NaN statistics
        feature_nan = {}
        for feat in available_features:
            nan_count = df[feat].isna().sum()
            nan_pct = nan_count / len(df) * 100
            feature_nan[feat] = {
                'count': int(nan_count),
                'percentage': float(nan_pct)
            }

        overall_nan_pct = df[available_features].isna().sum().sum() / (len(df) * len(available_features)) * 100

        print(f"\nOverall NaN percentage: {overall_nan_pct:.2f}%")

        if overall_nan_pct > 5:
            print(f"⚠️  WARNING: High NaN percentage!")
            status = 'WARNING'
        elif overall_nan_pct > 1:
            print(f"⚠️  NOTICE: Moderate NaN percentage")
            status = 'PASS'
        else:
            print(f"✅ Excellent feature coverage")
            status = 'PASS'

        # Show features with most NaNs
        sorted_features = sorted(feature_nan.items(), key=lambda x: x[1]['percentage'], reverse=True)
        print(f"\nTop 5 features by NaN count:")
        for feat, stats in sorted_features[:5]:
            print(f"  {feat}: {stats['percentage']:.2f}% NaN ({stats['count']:,} bars)")

        results = {
            'status': status,
            'features_present': len(available_features),
            'features_missing': len(missing_features),
            'missing_features': missing_features,
            'overall_nan_pct': float(overall_nan_pct),
            'feature_nan_stats': feature_nan
        }

        if len(missing_features) > 0:
            print(f"\n⚠️  Missing features: {missing_features}")
            results['status'] = 'WARNING'
    else:
        print(f"❌ ERROR: No Tier 1 features found!")
        results = {
            'status': 'ERROR',
            'features_present': 0,
            'features_missing': len(tier1_features),
            'missing_features': tier1_features
        }

    return results


def main():
    """Main validation execution."""
    print("=" * 80)
    print("1-MINUTE DATA QUALITY VALIDATION")
    print("=" * 80)

    # Configuration
    DATA_DIR = project_root / "data" / "ml_training" / "regime_aware_1min_2025_tier1_features"

    all_results = {}
    overall_status = 'PASS'

    # Validate each regime
    for regime_id in [0, 1, 2]:
        file_path = DATA_DIR / f"regime_{regime_id}_tier1_features.csv"

        if not file_path.exists():
            print(f"\n❌ ERROR: Data file not found for Regime {regime_id}")
            all_results[f'regime_{regime_id}'] = {
                'status': 'ERROR',
                'message': f'File not found: {file_path}'
            }
            overall_status = 'ERROR'
            continue

        print(f"\n{'#' * 80}")
        print(f"# VALIDATING REGIME {regime_id}")
        print(f"{'#' * 80}")

        df = pd.read_csv(file_path)
        print(f"Loaded {len(df):,} bars from {file_path.name}")

        regime_results = {}

        # Run all validations
        regime_results['timestamp'] = validate_timestamp_continuity(df, regime_id)
        regime_results['price'] = validate_price_realism(df, regime_id)
        regime_results['volume'] = validate_volume_and_notional(df, regime_id)
        regime_results['regime_dist'] = validate_regime_distribution(df, regime_id)
        regime_results['features'] = validate_feature_coverage(df, regime_id)

        # Determine regime status
        regime_status = 'PASS'
        for check, results in regime_results.items():
            if results['status'] == 'ERROR':
                regime_status = 'ERROR'
                break
            elif results['status'] == 'WARNING' and regime_status != 'ERROR':
                regime_status = 'WARNING'

        regime_results['overall_status'] = regime_status
        all_results[f'regime_{regime_id}'] = regime_results

        # Update overall status
        if regime_status == 'ERROR':
            overall_status = 'ERROR'
        elif regime_status == 'WARNING' and overall_status != 'ERROR':
            overall_status = 'WARNING'

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for regime_key, regime_results in all_results.items():
        status_symbol = {
            'PASS': '✅',
            'WARNING': '⚠️ ',
            'ERROR': '❌'
        }.get(regime_results['overall_status'], '❓')

        print(f"{status_symbol} {regime_key.upper()}: {regime_results['overall_status']}")

        # Show key issues
        for check, results in regime_results.items():
            if check == 'overall_status':
                continue
            if results['status'] != 'PASS':
                status_msg = {
                    'WARNING': '⚠️ ',
                    'ERROR': '❌'
                }.get(results['status'], '')
                print(f"  {status_msg} {check}: {results['status']}")

    print(f"\n{'=' * 80}")
    print(f"OVERALL STATUS: {overall_status}")
    print(f"{'=' * 80}")

    if overall_status == 'PASS':
        print("✅ ALL VALIDATIONS PASSED - Data is ready for production use")
    elif overall_status == 'WARNING':
        print("⚠️  WARNINGS DETECTED - Review and address before production")
    else:
        print("❌ ERRORS FOUND - Data quality issues must be resolved")

    # Save results
    REPORTS_DIR = project_root / "data" / "reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = REPORTS_DIR / f"1min_data_quality_validation_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'overall_status': overall_status,
            'results': all_results
        }, f, indent=2, default=str)

    print(f"\n✅ Validation results saved to: {results_file}")

    return 0 if overall_status != 'ERROR' else 1


if __name__ == "__main__":
    sys.exit(main())
