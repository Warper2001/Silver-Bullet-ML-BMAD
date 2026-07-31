
import pandas as pd
from datetime import datetime, timedelta
from src.monitoring.trade_db import TradeDatabase

db = TradeDatabase()
# Calculate start of the current trading week (Monday)
today = datetime.utcnow()
start_of_week = today - timedelta(days=today.weekday())
start_of_week_str = start_of_week.strftime('%Y-%m-%d')

df = db.get_trades_df()
if df.empty:
    print("No trades found in database.")
    exit()

df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)

# Filter for current week
weekly_trades = df[df['timestamp'] >= start_of_week_str]

if weekly_trades.empty:
    print(f"No trades logged since {start_of_week_str}.")
else:
    weekly_summary = weekly_trades.groupby('trader_id')['pnl'].sum()
    print(f"Weekly PnL (from {start_of_week_str}):")
    print(weekly_summary.to_string())
