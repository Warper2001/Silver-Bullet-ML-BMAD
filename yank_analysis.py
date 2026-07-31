
import pandas as pd
from src.monitoring.trade_db import TradeDatabase
import numpy as np

db = TradeDatabase()
df = db.get_trades_df('trader-yank')

if df.empty:
    print("No trades found for trader-yank")
    exit()

# Ensure numeric types
df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# 1. PnL Distribution
total_trades = len(df)
wins = df[df['pnl'] > 0]
losses = df[df['pnl'] < 0]
avg_win = wins['pnl'].mean()
avg_loss = losses['pnl'].mean()
win_rate = len(wins) / total_trades

# 2. Outlier Analysis (The "Black Swan" check)
largest_loss = df['pnl'].min()
largest_win = df['pnl'].max()
std_dev = df['pnl'].std()
outliers = df[df['pnl'] < (avg_loss - 2 * abs(std_dev))]

# 3. Exit Reason Breakdown
reason_counts = df['exit_reason'].value_counts(normalize=True) * 100

# 4. Equity Curve & Drawdown Details
equity = df['pnl'].cumsum()
running_max = equity.cummax()
drawdown = equity - running_max
max_dd = drawdown.min()

print(f"--- YANK Performance Deep Dive ---")
print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
print(f"Profit Factor: {abs(wins['pnl'].sum() / losses['pnl'].sum()) if not losses.empty else 'Inf':.2f}")
print(f"Max Drawdown: ${max_dd:.2f}")
print(f"Largest Single Loss: ${largest_loss:.2f}")
print(f"Largest Single Win: ${largest_win:.2f}")
print(f"\n--- Exit Reason Distribution ---")
print(reason_counts)
print(f"\n--- Outlier Analysis ---")
print(f"Trades > 2 std dev from avg loss: {len(outliers)}")
if not outliers.empty:
    print("Largest outliers:")
    print(outliers[['timestamp', 'pnl', 'exit_reason']].sort_values('pnl').head())
