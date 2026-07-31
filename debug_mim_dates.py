
import pandas as pd
from src.monitoring.trade_db import TradeDatabase

db = TradeDatabase()
df = db.get_trades_df('trader-mim-nb')
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)

# Show the most recent trades for mim-nb
print(df.sort_values('timestamp', ascending=False).head(5)[['timestamp', 'pnl']])
