#!/usr/bin/env python
"""
Comprehensive performance analysis of Silver Bullet ML trading system.

Analyzes:
1. Time-of-day performance patterns
2. Win/loss patterns and characteristics
3. Monthly/weekly breakdown
4. Trade duration analysis
5. Drawdown periods
6. Streaks and runs
"""

import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run comprehensive analysis."""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE SILVER BULLET ML PERFORMANCE ANALYSIS")
    logger.info("=" * 80)

    # Load trade details
    trades_path = Path('_bmad-output/reports/trade_details.csv')
    if not trades_path.exists():
        logger.error(f"Trade details not found at {trades_path}")
        logger.info("Run full_backtest_with_threshold.py first to generate trade data")
        return

    df = pd.read_csv(trades_path)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['is_win'] = df['pnl_pct'] > 0

    logger.info(f"Loaded {len(df)} trades")
    logger.info(f"Period: {df['entry_time'].min()} to {df['entry_time'].max()}")

    # Extract time features
    df['hour'] = df['entry_time'].dt.hour
    df['day_of_week'] = df['entry_time'].dt.dayofweek
    df['week'] = df['entry_time'].dt.isocalendar().week
    df['date'] = df['entry_time'].dt.date

    # ========================================================================
    # 1. TIME OF DAY ANALYSIS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("1. TIME OF DAY ANALYSIS")
    logger.info("=" * 80)

    hourly_stats = df.groupby('hour').agg({
        'pnl_pct': ['sum', 'mean', 'count'],
        'is_win': ['sum', 'count'],
    }).reset_index()
    hourly_stats.columns = ['hour', 'total_pnl', 'avg_pnl', 'n_trades', 'n_wins', 'total']
    hourly_stats['win_rate'] = hourly_stats['n_wins'] / hourly_stats['total']
    hourly_stats['pnl_per_trade'] = hourly_stats['total_pnl'] / hourly_stats['n_trades']

    logger.info(f"\n{'Hour':<6} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<12} {'Avg P&L':<10} {'P&L/Trade':<12}")
    logger.info("-" * 70)

    best_hour = hourly_stats.loc[hourly_stats['total_pnl'].idxmax()]
    worst_hour = hourly_stats.loc[hourly_stats['total_pnl'].idxmin()]

    for _, row in hourly_stats.iterrows():
        marker = "🥇" if row['hour'] == best_hour['hour'] else "🥉" if row['hour'] == worst_hour['hour'] else ""
        logger.info(
            f"{int(row['hour']):<6} {int(row['n_trades']):<8} "
            f"{row['win_rate']:<10.2%} {row['total_pnl']:<12.2f} "
            f"{row['avg_pnl']:<10.4f} {row['pnl_per_trade']:<12.4f} {marker}"
        )

    logger.info(f"\n🥇 Best hour: {int(best_hour['hour'])}:00 (Total P&L: {best_hour['total_pnl']:.2f}%)")
    logger.info(f"🥉 Worst hour: {int(worst_hour['hour'])}:00 (Total P&L: {worst_hour['total_pnl']:.2f}%)")

    # ========================================================================
    # 2. DAY OF WEEK ANALYSIS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("2. DAY OF WEEK ANALYSIS")
    logger.info("=" * 80)

    dow_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    dow_stats = df.groupby('day_of_week').agg({
        'pnl_pct': ['sum', 'mean', 'count'],
        'is_win': 'sum',
    }).reset_index()
    dow_stats.columns = ['dow', 'total_pnl', 'avg_pnl', 'n_trades', 'n_wins']
    dow_stats['day_name'] = dow_stats['dow'].map(dow_names)
    dow_stats['win_rate'] = dow_stats['n_wins'] / dow_stats['n_trades']

    logger.info(f"\n{'Day':<10} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<12} {'Avg P&L':<10}")
    logger.info("-" * 60)

    best_dow = dow_stats.loc[dow_stats['total_pnl'].idxmax()]

    for _, row in dow_stats.iterrows():
        marker = "🥇" if row['dow'] == best_dow['dow'] else ""
        logger.info(
            f"{row['day_name']:<10} {int(row['n_trades']):<8} "
            f"{row['win_rate']:<10.2%} {row['total_pnl']:<12.2f} "
            f"{row['avg_pnl']:<10.4f} {marker}"
        )

    logger.info(f"\n🥇 Best day: {best_dow['day_name']} (P&L: {best_dow['total_pnl']:.2f}%)")

    # ========================================================================
    # 3. WEEKLY TRENDS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("3. WEEKLY PERFORMANCE TRENDS")
    logger.info("=" * 80)

    weekly_stats = df.groupby('week').agg({
        'pnl_pct': ['sum', 'count'],
        'is_win': 'sum',
    }).reset_index()
    weekly_stats.columns = ['week', 'total_pnl', 'n_trades', 'n_wins']
    weekly_stats['win_rate'] = weekly_stats['n_wins'] / weekly_stats['n_trades']
    weekly_stats = weekly_stats.sort_values('week')

    logger.info(f"\n{'Week':<6} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<12}")
    logger.info("-" * 50)

    for _, row in weekly_stats.iterrows():
        logger.info(
            f"{int(row['week']):<6} {int(row['n_trades']):<8} "
            f"{row['win_rate']:<10.2%} {row['total_pnl']:<12.2f}"
        )

    best_week = weekly_stats.loc[weekly_stats['total_pnl'].idxmax()]
    worst_week = weekly_stats.loc[weekly_stats['total_pnl'].idxmin()]

    logger.info(f"\n🥇 Best week: Week {int(best_week['week'])} (+{best_week['total_pnl']:.2f}%)")
    logger.info(f"🥉 Worst week: Week {int(worst_week['week'])} ({worst_week['total_pnl']:.2f}%)")

    # ========================================================================
    # 4. WIN VS LOSS TRADE CHARACTERISTICS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("4. WIN VS LOSS TRADE CHARACTERISTICS")
    logger.info("=" * 80)

    wins = df[df['is_win']]
    losses = df[df['is_win'] == False]

    logger.info("\nWinning Trades:")
    logger.info(f"  Count: {len(wins)} ({len(wins)/len(df)*100:.1f}%)")
    logger.info(f"  Avg P&L: {wins['pnl_pct'].mean():.4f}%")
    logger.info(f"  Std P&L: {wins['pnl_pct'].std():.4f}%")
    logger.info(f"  Median P&L: {wins['pnl_pct'].median():.4f}%")
    logger.info(f"  Avg Hold: {wins['bars_held'].mean():.1f} bars")

    logger.info("\nLosing Trades:")
    logger.info(f"  Count: {len(losses)} ({len(losses)/len(df)*100:.1f}%)")
    logger.info(f"  Avg P&L: {losses['pnl_pct'].mean():.4f}%")
    logger.info(f"  Std P&L: {losses['pnl_pct'].std():.4f}%")
    logger.info(f"  Median P&L: {losses['pnl_pct'].median():.4f}%")
    logger.info(f"  Avg Hold: {losses['bars_held'].mean():.1f} bars")

    # Exit reason breakdown
    logger.info("\nExit Reason Breakdown:")
    exit_by_win = df.groupby('exit_reason').agg({
        'pnl_pct': 'sum',
        'is_win': ['sum', 'count'],
    })
    exit_by_win.columns = ['total_pnl', 'n_wins', 'total']

    for reason, data in exit_by_win.iterrows():
        wr = data['n_wins'] / data['total']
        logger.info(f"  {reason}: {int(data['total'])} trades, {wr:.1%} win rate, {data['total_pnl']:.2f}% P&L")

    # ========================================================================
    # 5. TRADE DURATION ANALYSIS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("5. TRADE DURATION ANALYSIS")
    logger.info("=" * 80)

    duration_bins = [0, 5, 10, 15, 20, 30, 50, 100]
    df['duration_bucket'] = pd.cut(df['bars_held'], bins=duration_bins)

    duration_stats = df.groupby('duration_bucket').agg({
        'pnl_pct': ['mean', 'count'],
        'is_win': 'sum',
    })
    duration_stats.columns = ['avg_pnl', 'n_trades', 'n_wins']
    duration_stats['win_rate'] = duration_stats['n_wins'] / duration_stats['n_trades']

    logger.info(f"\n{'Duration (bars)':<18} {'Trades':<10} {'Win Rate':<12} {'Avg P&L':<10}")
    logger.info("-" * 60)

    for bucket, row in duration_stats.iterrows():
        logger.info(
            f"{str(bucket):<18} {int(row['n_trades']):<10} "
            f"{row['win_rate']:<12.2%} {row['avg_pnl']:<10.4f}"
        )

    # ========================================================================
    # 6. STREAKS AND RUNS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("6. STREAKS AND RUNS")
    logger.info("=" * 80)

    # Initialize defaults
    longest_win_streak = {'n_trades': 0}
    longest_loss_streak = {'n_trades': 0}

    df_sorted = df.sort_values('entry_time').reset_index(drop=True)
    df_sorted['streak_group'] = (df_sorted['is_win'] != df_sorted['is_win'].shift()).cumsum()

    streaks = df_sorted.groupby('streak_group').agg({
        'pnl_pct': 'sum',
        'is_win': 'first',
        'entry_time': ['min', 'max'],
        'bars_held': 'count',
    })
    streaks.columns = ['is_win', 'total_pnl', 'start_time', 'end_time', 'n_trades']

    if len(streaks[streaks['is_win'] == 1]) > 0:
        winning_streaks = streaks[streaks['is_win'] == 1].sort_values('total_pnl', ascending=False)

        logger.info(f"\nTop 5 Winning Streaks:")
        for i, (_, row) in enumerate(winning_streaks.head(5).iterrows(), 1):
            logger.info(f"  {i}. {row['n_trades']} trades, +{row['total_pnl']:.2f}%")

        longest_win_streak = winning_streaks.loc[winning_streaks['n_trades'].idxmax()]
        logger.info(f"\nLongest winning streak: {int(longest_win_streak['n_trades'])} trades (+{longest_win_streak['total_pnl']:.2f}%)")
    else:
        logger.info("\nNo winning streaks found")

    if len(streaks[streaks['is_win'] == 0]) > 0:
        losing_streaks = streaks[streaks['is_win'] == 0].sort_values('total_pnl')

        logger.info(f"\nTop 5 Losing Streaks:")
        for i, (_, row) in enumerate(losing_streaks.head(5).iterrows(), 1):
            logger.info(f"  {i}. {row['n_trades']} trades, {row['total_pnl']:.2f}%")

        longest_loss_streak = losing_streaks.loc[losing_streaks['n_trades'].idxmax()]
        logger.info(f"\nLongest losing streak: {int(longest_loss_streak['n_trades'])} trades ({longest_loss_streak['total_pnl']:.2f}%)")
    else:
        logger.info("\nNo losing streaks found")

    # ========================================================================
    # 7. DRAWDOWN ANALYSIS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("7. DRAWDOWN ANALYSIS")
    logger.info("=" * 80)

    df_sorted = df.sort_values('entry_time').reset_index(drop=True)
    df_sorted['cum_pnl'] = df_sorted['pnl_pct'].cumsum()

    # Find all drawdown periods
    running_max = df_sorted['cum_pnl'].expanding().max()
    df_sorted['drawdown'] = df_sorted['cum_pnl'] - running_max

    # Find drawdown periods (when in drawdown)
    in_drawdown = df_sorted['drawdown'] < 0
    df_sorted['drawdown_group'] = (in_drawdown != in_drawdown.shift()).cumsum()

    # First count duration before aggregation
    dd_duration = df_sorted[df_sorted['drawdown'] < 0].groupby('drawdown_group').size()

    drawdown_periods = df_sorted[df_sorted['drawdown'] < 0].groupby('drawdown_group').agg({
        'drawdown': 'min',
        'cum_pnl': ['min', 'max'],
        'entry_time': ['min', 'max'],
    })
    drawdown_periods.columns = ['max_dd', 'pnl_at_dd_start', 'pnl_at_dd_end', 'start', 'end']
    drawdown_periods['duration_bars'] = dd_duration.values

    worst_5_dd = drawdown_periods.nsmallest(5, 'max_dd')

    logger.info(f"\nTop 5 Worst Drawdowns:")
    for i, (_, row) in enumerate(worst_5_dd.iterrows(), 1):
        logger.info(f"  {i}. {row['max_dd']:.2f}% (Duration: {int(row['duration_bars'])} bars)")

    # Recovery time analysis
    logger.info(f"\nMax Drawdown: {df_sorted['drawdown'].min():.2f}%")
    max_dd_idx = df_sorted['drawdown'].idxmin()
    max_dd_time = df_sorted.loc[max_dd_idx, 'entry_time']

    # Find recovery (new high after max DD)
    after_dd = df_sorted.loc[max_dd_idx:, 'cum_pnl']
    pre_dd_peak = df_sorted.loc[max_dd_idx, 'cum_pnl']

    recovered = False
    recovery_time = None
    for idx, val in after_dd.items():
        if val >= pre_dd_peak:
            recovered = True
            recovery_time = df_sorted.loc[idx, 'entry_time']
            break

    if recovered:
        recovery_bars = len(after_dd.loc[:after_dd.index.get_loc(idx)])
        logger.info(f"Recovered in {recovery_bars} bars (on {recovery_time})")
    else:
        logger.info("Not yet recovered by end of test period")

    # ========================================================================
    # SAVE SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)

    summary = {
        'test_period': {
            'start': str(df['entry_time'].min()),
            'end': str(df['entry_time'].max()),
            'total_trades': len(df),
        },
        'performance': {
            'total_return_pct': df['pnl_pct'].sum(),
            'win_rate': df['is_win'].mean(),
            'max_drawdown_pct': df_sorted['drawdown'].min(),
        },
        'best_hour': int(best_hour['hour']),
        'worst_hour': int(worst_hour['hour']),
        'best_day': best_dow['day_name'],
        'best_week': int(best_week['week']),
        'longest_win_streak': int(longest_win_streak['n_trades']),
        'longest_loss_streak': int(longest_loss_streak['n_trades']),
    }

    summary_path = Path('_bmad-output/reports/performance_analysis_summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n✅ Summary saved to: {summary_path}")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
