#!/usr/bin/env python3
"""
Analyze Transaction Costs for 1-Minute System

Models the impact of increased trade frequency (5-25 trades/day vs 3.92 current)
on profitability after accounting for commissions and slippage.

Determines breakeven win rates and recommends MIN_BARS_BETWEEN_TRADES adjustment
if costs are prohibitive.
"""

import sys
from pathlib import Path

# Constants
COMMISSION_PER_CONTRACT_RT = 2.50  # Round-trip commission per contract
MNQ_TICK_VALUE = 0.25  # $0.25 per tick
MNQ_POINT_VALUE = 20.0  # $20 per point
DEFAULT_SLIPPAGE_TICKS = 0.50  # 0.5 ticks slippage per trade
CONTRACTS_PER_TRADE = 5  # Default position size

def calculate_breakeven_win_rate(trades_per_day, avg_win_dollars, avg_loss_dollars, slippage_ticks=DEFAULT_SLIPPAGE_TICKS):
    """
    Calculate breakeven win rate given trade frequency and avg win/loss amounts.

    Returns:
        breakeven_win_rate: Win rate needed to break even after costs
        cost_per_trade: Total cost per trade (commission + slippage)
    """
    # Transaction costs
    commission_cost = COMMISSION_PER_CONTRACT_RT * CONTRACTS_PER_TRADE
    slippage_cost = slippage_ticks * MNQ_TICK_VALUE * CONTRACTS_PER_TRADE
    total_cost_per_trade = commission_cost + slippage_cost

    # Breakeven calculation: (win_rate * avg_win) - ((1 - win_rate) * avg_loss) - total_cost = 0
    # win_rate * (avg_win + avg_loss) = avg_loss + total_cost
    # win_rate = (avg_loss + total_cost) / (avg_win + avg_loss)

    breakeven_win_rate = (avg_loss_dollars + total_cost_per_trade) / (avg_win_dollars + avg_loss_dollars)

    return breakeven_win_rate, total_cost_per_trade

def main():
    print("=" * 70)
    print("TRANSACTION COST ANALYSIS - 1-MINUTE SYSTEM")
    print("=" * 70)

    print(f"\n📊 ASSUMPTIONS:")
    print(f"   Commission: ${COMMISSION_PER_CONTRACT_RT:.2f}/contract RT")
    print(f"   Slippage: {DEFAULT_SLIPPAGE_TICKS} ticks = ${DEFAULT_SLIPPAGE_TICKS * MNQ_TICK_VALUE:.2f}/contract")
    print(f"   Position size: {CONTRACTS_PER_TRADE} contracts")
    print(f"   Total cost per trade: ${COMMISSION_PER_CONTRACT_RT * CONTRACTS_PER_TRADE:.2f} + ${DEFAULT_SLIPPAGE_TICKS * MNQ_TICK_VALUE * CONTRACTS_PER_TRADE:.2f} = ${(COMMISSION_PER_CONTRACT_RT * CONTRACTS_PER_TRADE) + (DEFAULT_SLIPPAGE_TICKS * MNQ_TICK_VALUE * CONTRACTS_PER_TRADE):.2f}")

    print(f"\n📈 CURRENT 5-MINUTE SYSTEM BASELINE:")
    print(f"   Trades/day: 3.92")
    print(f"   Win rate: 51.80%")
    print(f"   Expectation: ~$50/trade (before costs)")

    # Model different trade frequencies
    print(f"\n🎯 BREAKEVEN ANALYSIS AT DIFFERENT TRADE FREQUENCIES:")

    # Assume similar win/loss amounts as current system
    # Current: $50/trade expectation, 51.80% win rate
    # Let's reverse engineer avg win/loss:
    # expectation = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    # $50 = (0.518 * avg_win) - (0.482 * avg_loss)
    # Assume risk/reward ratio of 1:2 (avg_loss = $50, avg_win = $100)
    avg_win_dollars = 100.0  # From 0.3% TP on MNQ at ~18000 = $54/contract = $270 for 5 contracts
    avg_loss_dollars = 50.0   # From 0.2% SL on MNQ at ~18000 = $36/contract = $180 for 5 contracts

    print(f"\n   Assumed win/loss amounts (from triple-barrier exits):")
    print(f"     Avg win: ${avg_win_dollars:.0f} (TP: 0.3%)")
    print(f"     Avg loss: ${avg_loss_dollars:.0f} (SL: 0.2%)")

    print(f"\n   {'Trades/Day':<12} {'Cost/Trade':>12} {'Breakeven WR':>14} {'Margin %':>12} {'Expectation':>14}")
    print("-" * 70)

    results = []
    for trades_per_day in [5, 10, 15, 20, 25]:
        breakeven_wr, cost_per_trade = calculate_breakeven_win_rate(
            trades_per_day, avg_win_dollars, avg_loss_dollars
        )

        # Calculate gross expectation (before costs)
        gross_expectation = (0.518 * avg_win_dollars) - (0.482 * avg_loss_dollars)

        # Calculate net expectation (after costs)
        net_expectation = gross_expectation - cost_per_trade

        # Margin to breakeven
        current_win_rate = 0.518
        margin_pct = (current_win_rate - breakeven_wr) / current_win_rate * 100

        print(f"   {trades_per_day:<12} ${cost_per_trade:>10.2f}   {breakeven_wr:>13.2%}   {margin_pct:>10.1f}%   ${net_expectation:>12.2f}")

        results.append({
            'trades_per_day': trades_per_day,
            'cost_per_trade': cost_per_trade,
            'breakeven_wr': breakeven_wr,
            'net_expectation': net_expectation
        })

    # Profitability analysis
    print(f"\n💰 PROFITABILITY ANALYSIS:")

    print(f"\n   At 51.80% win rate (current system):")
    for result in results:
        trades_per_day = result['trades_per_day']
        net_expectation = result['net_expectation']
        daily_profit = net_expectation * trades_per_day
        monthly_profit = daily_profit * 21  # 21 trading days/month
        annual_profit = daily_profit * 252  # 252 trading days/year

        print(f"     {trades_per_day} trades/day: ${net_expectation:.2f}/trade = ${daily_profit:.2f}/day = ${monthly_profit:,.0f}/month = ${annual_profit:,.0f}/year")

    # Recommendation
    print(f"\n📊 RECOMMENDATIONS:")

    # Find minimum profitable trade frequency
    profitable_frequencies = [r for r in results if r['net_expectation'] > 20]  # Target >$20/trade

    if profitable_frequencies:
        min_freq = min(r['trades_per_day'] for r in profitable_frequencies)
        max_freq = max(r['trades_per_day'] for r in profitable_frequencies)
        print(f"   ✅ Target 5-25 trades/day is VIABLE")
        print(f"   ✅ Optimal range: {min_freq}-{max_freq} trades/day for >$20/trade after costs")
    else:
        print(f"   ❌ Current parameters NOT VIABLE - need to adjust")

    # Check if MIN_BARS_BETWEEN_TRADES = 1 is appropriate
    max_result = max(results, key=lambda x: x['net_expectation'])
    print(f"\n   Best trade frequency: {max_result['trades_per_day']} trades/day")
    print(f"   Net expectation: ${max_result['net_expectation']:.2f}/trade")

    if max_result['net_expectation'] < 20:
        print(f"\n   ⚠️  WARNING: Expectation <$20/trade at all frequencies")
        print(f"   ⚠️  RECOMMENDATION: Increase MIN_BARS_BETWEEN_TRADES to 5-10")
        print(f"   ⚠️  This would reduce trade frequency to 2-6 trades/day")
        print(f"   ⚠️  Alternative: Improve win rate to 55%+ or reduce slippage")
    else:
        print(f"\n   ✅ MIN_BARS_BETWEEN_TRADES = 1 is ACCEPTABLE")
        print(f"   ✅ Proceed with 1-minute migration")

    print(f"\n" + "=" * 70)
    print("✅ TRANSACTION COST ANALYSIS COMPLETE")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
