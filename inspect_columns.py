
import pandas as pd
from src.monitoring.trade_db import TradeDatabase

db = TradeDatabase()
df = db.get_trades_df('trader-yank')
print(f"Columns: {df.columns.tolist()}")
print(df.head())
