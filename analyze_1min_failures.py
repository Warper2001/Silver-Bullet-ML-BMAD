#!/usr/bin/env python3
"""
Diagnostic Analysis - Why Premium Strategy Failed on 1-Minute Data

Analyzes the catastrophic -$83K loss to find root causes.
"""

import pandas as pd
import numpy as np

print("="*70)
print("🔍 DIAGNOSTIC ANALYSIS - 1-MINUTE FAILURE")
print("="*70)

# Load results
df = pd.read_csv('data/reports/premium_backtest_1min_2025.csv')

print(f"\n📊 LOADING FAILURE DATA...")
print(f"Total Trades: {len(df)}")
print(f"Total Loss: ${df['pnl'].sum():,.0f}")
print(f"Win Rate: {(df['pnl'] > 0).sum() / len(df) * 100:.2f}%")

# Analysis 1: Stop loss distance
print(f"\n{'='*70}")
print("🎯 ANALYSIS 1: STOP LOSS DISTANCE")
print("="*70)

df['stop_distance'] = abs(df['entry'] - df['stop'])
df['risk_reward'] = abs(df['target'] - df['entry']) / df['stop_distance']

print(f"Avg Stop Distance: ${df['stop_distance'].mean():.0f}")
print(f"Avg Risk/Reward Ratio: {df['risk_reward'].mean():.2f}:1")

# Analyze by stop distance quartiles
df['stop_quartile'] = pd.qcut(df['stop_distance'], 4, labels=['Q1 (tightest)', 'Q2', 'Q3', 'Q4 (widest)'])
print(f"\nPerformance by Stop Distance:")
for quartile in df['stop_quartile'].unique():
    subset = df[df['stop_quartile'] == quartile]
    win_rate = (subset['pnl'] > 0).sum() / len(subset) * 100
    total_pnl = subset['pnl'].sum()
    stop_hit_rate = (subset['exit_reason'] == 'stop').sum() / len(subset) * 100
    print(f"  {quartile}: {win_rate:.1f}% win, ${total_pnl:,.0f}, {stop_hit_rate:.1f}% stop hits")

# Analysis 2: FVG size impact
print(f"\n{'='*70}")
print("📊 ANALYSIS 2: FVG SIZE IMPACT")
print("="*70)

df['fvg_quartile'] = pd.qcut(df['fvg_size'], 4, labels=['Q1 (smallest)', 'Q2', 'Q3', 'Q4 (largest)'])
print(f"Performance by FVG Size:")
for quartile in df['fvg_quartile'].unique():
    subset = df[df['fvg_quartile'] == quartile]
    win_rate = (subset['pnl'] > 0).sum() / len(subset) * 100
    total_pnl = subset['pnl'].sum()
    avg_fvg = subset['fvg_size'].mean()
    print(f"  {quartile} (avg ${avg_fvg:.0f}): {win_rate:.1f}% win, ${total_pnl:,.0f}")

# Analysis 3: Quality score analysis
print(f"\n{'='*70}")
print("🎯 ANALYSIS 3: QUALITY SCORE TRUTH")
print("="*70)

# Create quality ranges
df['quality_range'] = pd.cut(df['quality_score'], bins=[0, 60, 70, 80, 100], labels=['0-60', '60-70', '70-80', '80-100'])
print(f"Performance by Quality Score:")
for q_range in df['quality_range'].cat.categories:
    subset = df[df['quality_range'] == q_range]
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).sum() / len(subset) * 100
        total_pnl = subset['pnl'].sum()
        print(f"  {q_range}: {win_rate:.1f}% win ({len(subset)} trades), ${total_pnl:,.0f}")

# Analysis 4: Time of day
print(f"\n{'='*70}")
print("⏰ ANALYSIS 4: TIME OF DAY")
print("="*70)

df['hour'] = pd.to_datetime(df['entry_time']).dt.hour
print(f"Performance by Hour:")
for hour in sorted(df['hour'].unique()):
    subset = df[df['hour'] == hour]
    win_rate = (subset['pnl'] > 0).sum() / len(subset) * 100
    total_pnl = subset['pnl'].sum()
    print(f"  {hour:02d}:00: {win_rate:.1f}% win ({len(subset):2d} trades), ${total_pnl:7,.0f}")

# Analysis 5: Day of week
print(f"\n{'='*70}")
print("📅 ANALYSIS 5: DAY OF WEEK")
print("="*70)

df['day_of_week'] = pd.to_datetime(df['entry_time']).dt.day_name()
print(f"Performance by Day:")
for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
    subset = df[df['day_of_week'] == day]
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).sum() / len(subset) * 100
        total_pnl = subset['pnl'].sum()
        print(f"  {day}: {win_rate:.1f}% win ({len(subset):2d} trades), ${total_pnl:7,.0f}")

# Analysis 6: Volume ratio impact
print(f"\n{'='*70}")
print("📊 ANALYSIS 6: VOLUME RATIO IMPACT")
print("="*70)

df['vr_quartile'] = pd.qcut(df['volume_ratio'], 4, labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'])
print(f"Performance by Volume Ratio:")
for quartile in df['vr_quartile'].unique():
    subset = df[df['vr_quartile'] == quartile]
    win_rate = (subset['pnl'] > 0).sum() / len(subset) * 100
    total_pnl = subset['pnl'].sum()
    avg_vr = subset['volume_ratio'].mean()
    print(f"  {quartile} (avg {avg_vr:.1f}x): {win_rate:.1f}% win, ${total_pnl:,.0f}")

# Analysis 7: Bar distance impact
print(f"\n{'='*70}")
print("📏 ANALYSIS 7: BAR DISTANCE (MSS-FVG GAP)")
print("="*70)

df['bd_quartile'] = pd.qcut(df['bar_diff'], 4, labels=['Q1 (closest)', 'Q2', 'Q3', 'Q4 (farthest)'])
print(f"Performance by Bar Distance:")
for quartile in df['bd_quartile'].unique():
    subset = df[df['bd_quartile'] == quartile]
    win_rate = (subset['pnl'] > 0).sum() / len(subset) * 100
    total_pnl = subset['pnl'].sum()
    avg_bd = subset['bar_diff'].mean()
    print(f"  {quartile} (avg {avg_bd:.1f} bars): {win_rate:.1f}% win, ${total_pnl:,.0f}")

# Summary: What works?
print(f"\n{'='*70}")
print("✅ WHAT ACTUALLY WORKS (IF ANYTHING)")
print("="*70)

# Find any profitable segments
best_quality = None
best_win_rate = 0
best_pnl = -float('inf')

for q_range in df['quality_range'].cat.categories:
    subset = df[df['quality_range'] == q_range]
    if len(subset) >= 10:  # Minimum sample size
        win_rate = (subset['pnl'] > 0).sum() / len(subset) * 100
        total_pnl = subset['pnl'].sum()
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_quality = q_range
            best_win_rate = win_rate

if best_pnl > 0:
    print(f"Best Quality Range: {best_quality}")
    print(f"  Win Rate: {best_win_rate:.1f}%")
    print(f"  Total P&L: ${best_pnl:,.0f}")
else:
    print("❌ NO PROFITABLE SEGMENTS FOUND")
    print("   All quality ranges lose money")

# Check bearish vs bullish
print(f"\nDirection Analysis:")
for direction in ['bullish', 'bearish']:
    subset = df[df['direction'] == direction]
    win_rate = (subset['pnl'] > 0).sum() / len(subset) * 100
    total_pnl = subset['pnl'].sum()
    status = "✅ PROFITABLE" if total_pnl > 0 else "❌ LOSING"
    print(f"  {direction.capitalize()}: {status} - {win_rate:.1f}% win, ${total_pnl:,.0f}")

print(f"\n{'='*70}")
print("🔧 RECOMMENDATIONS")
print("="*70)

print("""
1. STOP TRADING THIS STRATEGY IMMEDIATELY
   - 20% win rate is catastrophic
   - -$83K loss is unacceptable

2. ROOT CAUSE: 1-MINUTE NOISE SWAMPS THE SIGNAL
   - 5-minute data smoothed out noise
   - 1-minute has 5x more false breakouts
   - MSS-FVG confluence fails on intraday

3. POSSIBLE FIXES (NOT GUARANTEED):
   a) Wider stops (3x instead of 1.5x)
   b) Higher FVG threshold ($200+ instead of $75)
   c) Trade ONLY bearish signals (41% vs 16% win rate)
   d) Filter out low-volume periods
   e) Switch back to 5-minute timeframe

4. ML TRAINING NOT VIABLE:
   - Need 500+ winning trades to train
   - Only 30 winners in entire dataset
   - Garbage in, garbage out

5. RECOMMENDATION:
   - ABANDON PREMIUM STRATEGY ON 1-MINUTE
   - EITHER: Use 5-minute timeframe
   - OR: Develop completely different strategy
""")

print("="*70)
