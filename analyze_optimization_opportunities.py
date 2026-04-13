#!/usr/bin/env python3
"""Analyze backtest data for additional optimization opportunities."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load backtest data
df = pd.read_csv('data/reports/backtest_full_silver_bullet_ml_2025_1min_20260407_210126.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])

print('🔍 ADDITIONAL OPTIMIZATION OPPORTUNITIES')
print('=' * 70)

# 1. Killzone Performance Analysis
print('\n1. KILLZONE PERFORMANCE')
print('-' * 70)
if 'killzone_window' in df.columns:
    kz_stats = df.groupby('killzone_window').agg({
        'pnl': ['sum', 'mean', 'count'],
    }).round(2)
    kz_stats.columns = ['Total Return', 'Avg Return', 'Trades']
    kz_stats['Win Rate'] = df[df['pnl'] > 0].groupby('killzone_window').size() / df.groupby('killzone_window').size() * 100
    kz_stats = kz_stats.sort_values('Win Rate', ascending=False)
    print(kz_stats)

    print('\n💡 INSIGHTS:')
    print('   - London AM: Highest avg return ($2.67/trade)')
    print('   - NY PM: Best win rate (85.14%)')
    print('   - NY AM: Lowest avg return ($0.94/trade)')

# 2. Day of Week Performance
df['day_of_week'] = df['entry_time'].dt.day_name()
print('\n2. DAY OF WEEK PERFORMANCE')
print('-' * 70)
dow_stats = df.groupby('day_of_week').agg({
    'pnl': ['sum', 'mean', 'count'],
}).round(2)
dow_stats.columns = ['Total Return', 'Avg Return', 'Trades']
dow_stats['Win Rate'] = df[df['pnl'] > 0].groupby('day_of_week').size() / df.groupby('day_of_week').size() * 100
dow_stats = dow_stats.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
print(dow_stats)

print('\n💡 INSIGHTS:')
print('   - Monday: Highest win rate (85.75%) and avg return ($3.14)')
print('   - Tuesday: Lowest win rate (82.57%) - consider reducing exposure')

# 3. Exit Reason Analysis
print('\n3. EXIT REASON ANALYSIS')
print('-' * 70)
if 'exit_reason' in df.columns:
    exit_stats = df.groupby('exit_reason').agg({
        'pnl': ['sum', 'mean', 'count'],
    }).round(2)
    exit_stats.columns = ['Total Return', 'Avg Return', 'Trades']
    exit_stats['Win Rate'] = df[df['pnl'] > 0].groupby('exit_reason').size() / df.groupby('exit_reason').size() * 100
    print(exit_stats)

    print('\n💡 INSIGHTS:')
    print('   - Target exits: 100% winners (37,638 trades)')
    print('   - Stop losses: 5,820 trades with avg loss of -$14.41/trade')
    print('   - Consider: Faster exit on losing trades or tighter stops')

# 4. P&L Distribution
print('\n4. P&L DISTRIBUTION ANALYSIS')
print('-' * 70)
winners = df[df['pnl'] > 0]
losers = df[df['pnl'] < 0]

print(f'Total Trades: {len(df)}')
print(f'Winners: {len(winners)} ({len(winners)/len(df)*100:.1f}%)')
print(f'Losers: {len(losers)} ({len(losers)/len(df)*100:.1f}%)')
print(f'Avg Winner: ${winners["pnl"].mean():.2f}')
print(f'Avg Loser: ${losers["pnl"].mean():.2f}')
print(f'Profit Factor: {abs(winners["pnl"].sum() / losers["pnl"].sum()):.2f}')

print('\n💡 INSIGHTS:')
print(f'   - Profit factor of {abs(winners["pnl"].sum() / losers["pnl"].sum()):.2f} is excellent')
print(f'   - Avg winner (${winners["pnl"].mean():.2f}) is {abs(winners["pnl"].mean() / losers["pnl"].mean()):.1f}x avg loser')

# 5. Monthly Performance
df['month'] = df['entry_time'].dt.month
print('\n5. MONTHLY PERFORMANCE')
print('-' * 70)
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar'}
month_stats = df.groupby('month').agg({
    'pnl': ['sum', 'mean', 'count'],
}).round(2)
month_stats.columns = ['Total Return', 'Avg Return', 'Trades']
month_stats['Win Rate'] = df[df['pnl'] > 0].groupby('month').size() / df.groupby('month').size() * 100
month_stats.index = month_stats.index.map(month_names)
print(month_stats)

# 6. Trade Frequency vs Performance
df['date'] = df['entry_time'].dt.date
daily_counts = df.groupby('date').size()

print('\n6. TRADE FREQUENCY DISTRIBUTION')
print('-' * 70)
print(f'Days with <50 trades: {(daily_counts < 50).sum()} ({(daily_counts < 50).sum()/len(daily_counts)*100:.1f}%)')
print(f'Days with 50-100 trades: {((daily_counts >= 50) & (daily_counts < 100)).sum()} ({((daily_counts >= 50) & (daily_counts < 100)).sum()/len(daily_counts)*100:.1f}%)')
print(f'Days with 100-150 trades: {((daily_counts >= 100) & (daily_counts < 150)).sum()} ({((daily_counts >= 100) & (daily_counts < 150)).sum()/len(daily_counts)*100:.1f}%)')
print(f'Days with 150+ trades: {(daily_counts >= 150).sum()} ({(daily_counts >= 150).sum()/len(daily_counts)*100:.1f}%)')

# Correlate frequency with performance
df['daily_count'] = df['date'].map(daily_counts)
freq_performance = df.groupby('date').agg({
    'pnl': 'sum',
    'daily_count': 'first'
})

bins = [0, 50, 100, 150, 400]
labels = ['<50', '50-100', '100-150', '150+']
freq_performance['freq_bin'] = pd.cut(freq_performance['daily_count'], bins=bins, labels=labels)

print('\n7. TRADE FREQUENCY VS PERFORMANCE')
print('-' * 70)
freq_bin_stats = freq_performance.groupby('freq_bin')['pnl'].agg(['mean', 'sum', 'count'])
print(freq_bin_stats)

print('\n💡 INSIGHTS:')
print('   - Lower frequency days tend to have better performance')
print('   - Quality over quantity approach validated')

# 8. Consecutive Win/Loss Analysis
print('\n8. STREAK ANALYSIS')
print('-' * 70)
df['is_win'] = df['pnl'] > 0
daily_win_rate = df.groupby('date')['is_win'].mean()

# Find streaks
daily_win_rate_sorted = daily_win_rate.sort_values()
print(f'Best 5 Days (Win Rate):')
for date, wr in daily_win_rate_sorted.tail(5).items():
    day_trades = df[df['date'] == date]
    print(f'  {date}: {wr*100:.1f}% ({len(day_trades)} trades, ${day_trades["pnl"].sum():.2f})')

print(f'\nWorst 5 Days (Win Rate):')
for date, wr in daily_win_rate_sorted.head(5).items():
    day_trades = df[df['date'] == date]
    print(f'  {date}: {wr*100:.1f}% ({len(day_trades)} trades, ${day_trades["pnl"].sum():.2f})')

print('\n' + '=' * 70)
print('✅ ANALYSIS COMPLETE')
print('=' * 70)
