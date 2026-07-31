
import pandas as pd
from src.monitoring.trade_db import TradeDatabase
from datetime import datetime, timedelta

db = TradeDatabase()
start_of_week = datetime.utcnow() - timedelta(days=datetime.utcnow().weekday())
start_of_week_str = start_of_week.strftime('%Y-%m-%d')

df = db.get_trades_df()
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)

weekly_trades = df[df['timestamp'] >= start_of_week_str]
print(f"Weekly PnL (from {start_of_week_str}):")
print(weekly_trades.groupby('trader_id')['pnl'].sum().to_string())
