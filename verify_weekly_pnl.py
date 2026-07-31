
import pandas as pd
from src.monitoring.trade_db import TradeDatabase

db = TradeDatabase()
df = db.get_trades_df()
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)

weekly_trades = df[df['timestamp'] >= '2026-06-15']

if weekly_trades.empty:
    print("Weekly PnL (from 2026-06-15): $0.00")
else:
    print(weekly_trades.groupby('trader_id')['pnl'].sum().to_string())
