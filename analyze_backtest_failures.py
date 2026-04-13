#!/usr/bin/env python3
"""
Diagnostic Analysis of Premium Strategy Backtest Failures

This script analyzes why the premium strategy is failing (19.89% win rate)
by comparing winning vs losing trades to identify patterns and insights.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

print("🔍 PREMIUM STRATEGY - DIAGNOSTIC FAILURE ANALYSIS")
print("=" * 70)

# Load backtest data
df = pd.read_csv('data/reports/premium_backtest_2025_proper.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['date'] = df['entry_time'].dt.date
df['hour'] = df['entry_time'].dt.hour
df['day_of_week'] = df['entry_time'].dt.day_name()

# Separate winners and losers
winners = df[df['pnl'] > 0].copy()
losers = df[df['pnl'] < 0].copy()

print(f"\n📊 OVERVIEW:")
print(f"   Total Trades: {len(df)}")
print(f"   Winners: {len(winners)} ({len(winners)/len(df)*100:.1f}%)")
print(f"   Losers: {len(losers)} ({len(losers)/len(df)*100:.1f}%)")

# ANALYSIS 1: Trade Structure Differences
print(f"\n🎯 ANALYSIS 1: TRADE STRUCTURE")
print("-" * 70)

for direction in ['bullish', 'bearish']:
    dir_winners = winners[winners['direction'] == direction]
    dir_losers = losers[losers['direction'] == direction]

    if len(dir_winners) > 0 and len(dir_losers) > 0:
        print(f"\n{direction.capitalize()} Trades:")

        # Risk analysis
        winner_risk = (dir_winners['entry'] - dir_winners['stop']).abs()
        loser_risk = (dir_losers['entry'] - dir_losers['stop']).abs()

        print(f"  Avg Risk - Winners: {winner_risk.mean():.2f} pts")
        print(f"  Avg Risk - Losers:  {loser_risk.mean():.2f} pts")
        print(f"  Risk Difference: {winner_risk.mean() - loser_risk.mean():+.2f} pts")

        # Reward analysis
        winner_reward = (dir_winners['target'] - dir_winners['entry']).abs() if direction == 'bullish' else (dir_winners['entry'] - dir_winners['target']).abs()
        loser_reward = (dir_losers['target'] - dir_losers['entry']).abs() if direction == 'bullish' else (dir_losers['entry'] - dir_losers['target']).abs()

        print(f"  Avg Reward - Winners: {winner_reward.mean():.2f} pts")
        print(f"  Avg Reward - Losers:  {loser_reward.mean():.2f} pts")

# ANALYSIS 2: Exit Patterns
print(f"\n🎯 ANALYSIS 2: EXIT PATTERNS")
print("-" * 70)

for reason in ['stop', 'target', 'timeout']:
    reason_trades = df[df['exit_reason'] == reason]
    if len(reason_trades) > 0:
        win_rate = (reason_trades['pnl'] > 0).sum() / len(reason_trades) * 100
        avg_pnl = reason_trades['pnl'].mean()

        print(f"\n{reason.capitalize()} Exit:")
        print(f"  Trades: {len(reason_trades)} ({len(reason_trades)/len(df)*100:.1f}%)")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Avg P&L: ${avg_pnl:.2f}")

# ANALYSIS 3: Quality Score Analysis
print(f"\n🎯 ANALYSIS 3: QUALITY SCORE ANALYSIS")
print("-" * 70)

# Create quality bins
df['quality_bin'] = pd.cut(df['quality_score'], bins=[0, 65, 70, 75, 80, 85, 90, 100],
                             labels=['<65', '65-70', '70-75', '75-80', '80-85', '85-90', '90+'])

print("\nQuality Score vs Win Rate:")
for q_bin in ['<65', '65-70', '70-75', '75-80', '80-85', '85-90', '90+']:
    q_trades = df[df['quality_bin'] == q_bin]
    if len(q_trades) > 0:
        win_rate = (q_trades['pnl'] > 0).sum() / len(q_trades) * 100
        avg_pnl = q_trades['pnl'].mean()
        avg_quality = q_trades['quality_score'].mean()

        print(f"  Quality {q_bin:4s}: {len(q_trades):4d} trades, {win_rate:5.1f}% win, ${avg_pnl:7.2f} avg (avg score: {avg_quality:.1f})")

# ANALYSIS 4: Temporal Analysis
print(f"\n🎯 ANALYSIS 4: TEMPORAL PATTERNS")
print("-" * 70)

print("\nHour of Day vs Win Rate:")
for hour in sorted(df['hour'].unique()):
    hour_trades = df[df['hour'] == hour]
    if len(hour_trades) > 10:  # Only show hours with sufficient data
        win_rate = (hour_trades['pnl'] > 0).sum() / len(hour_trades) * 100
        avg_pnl = hour_trades['pnl'].mean()
        print(f"  Hour {hour:02d}: {len(hour_trades):4d} trades, {win_rate:5.1f}% win, ${avg_pnl:7.2f} avg")

print("\nDay of Week vs Win Rate:")
for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
    day_trades = df[df['day_of_week'] == day]
    if len(day_trades) > 0:
        win_rate = (day_trades['pnl'] > 0).sum() / len(day_trades) * 100
        avg_pnl = day_trades['pnl'].mean()
        print(f"  {day:9s}: {len(day_trades):4d} trades, {win_rate:5.1f}% win, ${avg_pnl:7.2f} avg")

# ANALYSIS 5: Killzone Analysis
print(f"\n🎯 ANALYSIS 5: KILLZONE PERFORMANCE")
print("-" * 70)

for kz in df['killzone_window'].dropna().unique():
    kz_trades = df[df['killzone_window'] == kz]
    if len(kz_trades) > 0:
        win_rate = (kz_trades['pnl'] > 0).sum() / len(kz_trades) * 100
        avg_pnl = kz_trades['pnl'].mean()
        print(f"  {str(kz):12s}: {len(kz_trades):4d} trades, {win_rate:5.1f}% win, ${avg_pnl:7.2f} avg")

# ANALYSIS 6: FVG Size Analysis
print(f"\n🎯 ANALYSIS 6: FVG SIZE DISTRIBUTION")
print("-" * 70)

df['fvg_bin'] = pd.cut(df['fvg_size'], bins=[0, 100, 200, 300, 500, 1000, 10000],
                       labels=['<100', '100-200', '200-300', '300-500', '500-1000', '1000+'])

print("\nFVG Size vs Win Rate:")
for fvg_bin in ['<100', '100-200', '200-300', '300-500', '500-1000', '1000+']:
    fvg_trades = df[df['fvg_bin'] == fvg_bin]
    if len(fvg_trades) > 0:
        win_rate = (fvg_trades['pnl'] > 0).sum() / len(fvg_trades) * 100
        avg_pnl = fvg_trades['pnl'].mean()
        avg_fvg = fvg_trades['fvg_size'].mean()

        print(f"  FVG ${fvg_bin:7s}: {len(fvg_trades):4d} trades, {win_rate:5.1f}% win, ${avg_pnl:7.2f} avg (avg: ${avg_fvg:.0f})")

# ANALYSIS 7: Volume Ratio Impact (SKIPPED - not in backtest data)
print(f"\n🎯 ANALYSIS 7: VOLUME RATIO IMPACT")
print("-" * 70)
print("  Skipped: volume_ratio not available in backtest CSV")

# ANALYSIS 8: Bar Distance Impact (SKIPPED - not in backtest data)
print(f"\n🎯 ANALYSIS 8: MSS-FVG BAR DISTANCE")
print("-" * 70)
print("  Skipped: bar_diff not available in backtest CSV")

# ANALYSIS 9: Sample Winners vs Losers
print(f"\n🎯 ANALYSIS 9: WINNER vs LOSER EXAMPLES")
print("-" * 70)

print(f"\n🏆 TOP 5 WINNERS:")
top_winners = winners.nlargest(5, 'pnl')
for i, row in top_winners.iterrows():
    print(f"  {row['direction']:7s} Entry: {row['entry']:8.2f} → Target: {row['target']:8.2f} P&L: ${row['pnl']:7.2f}")
    print(f"    Quality: {row['quality_score']:.1f}, FVG: ${row['fvg_size']:.0f}")

print(f"\n❌ TOP 5 LOSERS:")
worst_losers = losers.nsmallest(5, 'pnl')
for i, row in worst_losers.iterrows():
    print(f"  {row['direction']:7s} Entry: {row['entry']:8.2f} → Stop: {row['stop']:8.2f} P&L: ${row['pnl']:7.2f}")
    print(f"    Quality: {row['quality_score']:.1f}, FVG: ${row['fvg_size']:.0f}")

# ANALYSIS 10: Key Insights Summary
print(f"\n🎯 ANALYSIS 10: KEY INSIGHTS")
print("=" * 70)

insights = []

# Check quality score inversion
low_quality_win_rate = (df[df['quality_score'] < 70]['pnl'] > 0).sum() / len(df[df['quality_score'] < 70]) * 100
high_quality_win_rate = (df[df['quality_score'] > 85]['pnl'] > 0).sum() / len(df[df['quality_score'] > 85]) * 100

if low_quality_win_rate > high_quality_win_rate:
    insights.append("⚠️  QUALITY SCORE INVERSION: Low quality (<70) has HIGHER win rate than high quality (>85)")

# Check stop placement
stop_loss_rate = (df['exit_reason'] == 'stop').sum() / len(df) * 100
if stop_loss_rate > 70:
    insights.append(f"⚠️  STOP PLACEMENT ISSUE: {stop_loss_rate:.1f}% of trades hit stops - stops too tight!")

# Check direction bias
bullish_win_rate = (df[df['direction'] == 'bullish']['pnl'] > 0).sum() / len(df[df['direction'] == 'bullish']) * 100
bearish_win_rate = (df[df['direction'] == 'bearish']['pnl'] > 0).sum() / len(df[df['direction'] == 'bearish']) * 100

if abs(bullish_win_rate - bearish_win_rate) > 5:
    insights.append(f"⚠️  DIRECTION BIAS: Bullish {bullish_win_rate:.1f}% vs Bearish {bearish_win_rate:.1f}% win rate")

# Check temporal patterns
hourly_win_rates = {}
for hour in df['hour'].unique():
    hour_trades = df[df['hour'] == hour]
    if len(hour_trades) > 20:
        hourly_win_rates[hour] = (hour_trades['pnl'] > 0).sum() / len(hour_trades) * 100

if len(hourly_win_rates) > 0:
    best_hour = max(hourly_win_rates, key=hourly_win_rates.get)
    worst_hour = min(hourly_win_rates, key=hourly_win_rates.get)

    if hourly_win_rates[best_hour] - hourly_win_rates[worst_hour] > 15:
        insights.append(f"⚠️  TEMPORAL BIAS: Hour {best_hour}: {hourly_win_rates[best_hour]:.1f}% vs Hour {worst_hour}: {hourly_win_rates[worst_hour]:.1f}% win rate")

print("\nKEY FINDINGS:")
for insight in insights:
    print(f"  {insight}")

# Generate optimization recommendations
print(f"\n🚀 OPTIMIZATION RECOMMENDATIONS:")
print("=" * 70)

recommendations = []

# Quality score recommendation
if low_quality_win_rate > high_quality_win_rate:
    recommendations.append("1. INVERT quality score filtering: Focus on quality 60-75 range instead of 90-100")

# Stop placement recommendation
if stop_loss_rate > 70:
    recommendations.append("2. WIDEN stop losses: Use 2-3x current distance (100-130 points instead of 50)")

# Temporal recommendation
if len(hourly_win_rates) > 0 and hourly_win_rates[best_hour] - hourly_win_rates[worst_hour] > 15:
    recommendations.append(f"3. TIME filters: Focus trading on hours {best_hour}:00-{(best_hour+1)%24}:00 with highest win rates")

for rec in recommendations:
    print(f"  {rec}")

# Save analysis results
output = {
    'analysis_date': datetime.now().isoformat(),
    'total_trades': int(len(df)),
    'winners': int(len(winners)),
    'losers': int(len(losers)),
    'win_rate': float(len(winners) / len(df) * 100),
    'key_insights': insights,
    'recommendations': recommendations,
    'quality_inversion': bool(low_quality_win_rate > high_quality_win_rate),
    'stop_loss_rate': float(stop_loss_rate),
    'temporal_bias_exists': bool(len(hourly_win_rates) > 0 and hourly_win_rates[best_hour] - hourly_win_rates[worst_hour] > 15)
}

output_path = Path("data/reports/diagnostic_analysis.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n💾 Analysis saved: {output_path}")
print("\n" + "=" * 70)
print("✅ DIAGNOSTIC ANALYSIS COMPLETE")
print("=" * 70)
print("Use these insights to guide the optimization phase.")
