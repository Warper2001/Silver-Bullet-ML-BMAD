"""Simplified backtest for testing the backtesting framework."""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.research.historical_data_loader import HistoricalDataLoader
from src.research.ml_meta_labeling_backtester import MLMetaLabelingBacktester
from src.research.performance_metrics_calculator import PerformanceMetricsCalculator
from src.research.equity_curve_visualizer import EquityCurveVisualizer
from src.research.backtest_report_generator import BacktestReportGenerator

def generate_sample_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate sample Silver Bullet signals for testing."""

    np.random.seed(42)
    n_signals = 100  # Generate 100 sample signals

    # Random timestamps within the data range
    timestamps = np.random.choice(data.index, n_signals, replace=False)
    timestamps = sorted(timestamps)

    # Random directions and confidence scores
    directions = np.random.choice(['bullish', 'bearish'], n_signals)
    confidences = np.random.randint(60, 95, n_signals)

    signals = pd.DataFrame({
        'timestamp': timestamps,
        'direction': directions,
        'confidence': confidences,
        'pattern_type': np.random.choice(['MSS', 'FVG', 'LIQUIDITY_SWEEP'], n_signals)
    })

    return signals

def main():
    """Run simplified backtest."""

    print("🚀 STARTING SIMPLIFIED BACKTEST")
    print("=" * 50)

    # Step 1: Load historical data
    print("📊 Loading historical data...")
    loader = HistoricalDataLoader(
        data_directory="data/processed/dollar_bars/",
        min_completeness=0.5  # Lower threshold for real data
    )

    try:
        # Use recent real data: 2025-01-01 to 2025-03-06
        data = loader.load_data('2025-01-01', '2025-03-06')
        print(f"✅ Loaded {len(data)} bars")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Step 2: Generate sample signals
    print("\n🎯 Generating sample signals...")
    signals = generate_sample_signals(data)
    print(f"✅ Generated {len(signals)} signals")
    print(f"   Bullish: {len(signals[signals['direction'] == 'bullish'])}")
    print(f"   Bearish: {len(signals[signals['direction'] == 'bearish'])}")

    # Step 3: Simulate trades without ML filtering (simplified)
    print("\n🤖 Simulating trades (without ML model)...")
    trades = []
    for _, signal in signals.iterrows():
        entry_time = signal['timestamp']
        entry_price = data.loc[entry_time, 'close'] if entry_time in data.index else data['close'].iloc[0]

        # Simulate trade outcome (simplified)
        direction = 1 if signal['direction'] == 'bullish' else -1
        price_change = np.random.normal(0.002, 0.001)  # 0.2% average move

        exit_price = entry_price * (1 + direction * price_change)
        exit_time = entry_time + pd.Timedelta(minutes=30)  # 30 min hold

        pnl = (exit_price - entry_price) * direction

        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': signal['direction'],
            'pnl': pnl,
            'return_pct': (pnl / entry_price) * 100
        })

    trades = pd.DataFrame(trades)
    print(f"✅ Created {len(trades)} simulated trades")

    if len(trades) == 0:
        print("❌ No trades to analyze!")
        return

    # Step 4: Calculate performance metrics
    print("\n📈 Calculating performance metrics...")
    metrics_calc = PerformanceMetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(trades)

    print(f"✅ Performance Metrics:")
    print(f"   Total Return: {metrics.get('total_return', 'N/A')}")
    print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
    print(f"   Win Rate: {metrics.get('win_rate', 'N/A')}")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Max Drawdown: {metrics.get('max_drawdown', 'N/A')}")

    # Step 5: Generate equity curve
    print("\n📊 Generating equity curve...")
    try:
        visualizer = EquityCurveVisualizer(output_directory="data/reports/")
        equity_curve_path = visualizer.visualize(trades)
        print(f"✅ Equity curve saved to {equity_curve_path}")
    except Exception as e:
        print(f"⚠️  Could not generate equity curve: {e}")

    # Step 6: Generate report
    print("\n📝 Generating backtest report...")
    try:
        report_generator = BacktestReportGenerator(output_directory="data/reports/")

        report_results = {
            'trades': trades,
            'metrics': metrics,
            'backtest_date': pd.Timestamp.now(),
            'data_range': ('2024-12-19', '2025-03-19'),
            'signal_count': len(signals),
            'configuration': {
                'model': 'data/models/xgboost_latest.pkl',
                'threshold': 0.65
            }
        }

        report_paths = report_generator.generate_backtest_report(report_results)
        print(f"✅ Report saved to {report_paths['csv_path']}")

    except Exception as e:
        print(f"⚠️  Could not generate report: {e}")

    print("\n🎉 BACKTEST COMPLETED!")
    print("=" * 50)
    print(f"📅 Summary: {len(trades)} trades executed")

    # Extract values from nested dicts
    total_return = metrics.get('total_return', {})
    if isinstance(total_return, dict):
        total_return_val = total_return.get('total_return_pct', 'N/A')
    else:
        total_return_val = total_return

    print(f"💰 Total Return: {total_return_val}%")
    print(f"📊 Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
    print(f"🎯 Win Rate: {metrics.get('win_rate', 'N/A')}%")

if __name__ == '__main__':
    main()