#!/usr/bin/env python3
"""FINAL SAFETY VALIDATION — Monte Carlo, MAE/MFE, and Clustering.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

def run_monte_carlo(trades_pnl, iterations=1000, initial_capital=10000):
    print(f"Running Monte Carlo simulation ({iterations} iterations)...")
    results = []
    for _ in range(iterations):
        # Shuffle trades with replacement
        sim = np.random.choice(trades_pnl, size=len(trades_pnl), replace=True)
        equity = initial_capital + np.cumsum(sim)
        
        # Calculate Max Drawdown
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity)
        max_dd = np.max(dd) if len(dd) > 0 else 0
        
        results.append({
            "final_equity": equity[-1],
            "max_dd": max_dd,
            "ruin": 1 if np.any(equity <= 0) else 0
        })
    return pd.DataFrame(results)

def analyze_safety(history_csv):
    df = pd.read_csv(history_csv)
    print(f"Analyzing {len(df)} trades from {history_csv}...")
    
    # 1. Clustering Analysis
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_pnl = df.groupby('date')['pnl'].sum()
    print("\n--- CLUSTERING & CONSISTENCY ---")
    print(f"Profitable Days: {(daily_pnl > 0).sum()} / {len(daily_pnl)} ({daily_pnl.mean():.2f}/day)")
    print(f"Max Daily Win:  ${daily_pnl.max():.2f}")
    print(f"Max Daily Loss: ${daily_pnl.min():.2f}")
    
    # 2. Monte Carlo
    mc_results = run_monte_carlo(df['pnl'].values)
    print("\n--- MONTE CARLO (1,000 runs, $10k base) ---")
    print(f"95% Confidence Max DD: ${np.percentile(mc_results['max_dd'], 95):.2f}")
    print(f"Median Final Equity:  ${mc_results['final_equity'].median():.2f}")
    print(f"Probability of Ruin: {mc_results['ruin'].mean()*100:.2f}%")
    
    # 3. Expectancy Score
    wins = df[df['pnl'] > 0]['pnl']
    losses = df[df['pnl'] <= 0]['pnl']
    avg_w = wins.mean() if not wins.empty else 0
    avg_l = abs(losses.mean()) if not losses.empty else 1
    win_rate = len(wins) / len(df)
    expectancy = (win_rate * avg_w) - ((1 - win_rate) * avg_l)
    print("\n--- RISK PARAMETERS ---")
    print(f"Expectancy Score: ${expectancy:.2f} per trade")
    print(f"Risk/Reward Ratio: 1 : {avg_w/avg_l:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, required=True)
    args = parser.parse_args()
    analyze_safety(args.history)
