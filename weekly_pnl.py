
import pandas as pd
from src.monitoring.trade_db import TradeDatabase

db = TradeDatabase()
df = db.get_trades_df()

if df.empty:
    print("No trades found.")
    exit()

df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)

# Filter for current week (>= Monday)
weekly_trades = df[df['timestamp'] >= '2026-06-15']

# Aggregate PnL per trader
weekly_summary = weekly_trades.groupby('trader_id')['pnl'].sum()

print("Weekly PnL (from 2026-06-15):")
print(weekly_summary.to_string())
