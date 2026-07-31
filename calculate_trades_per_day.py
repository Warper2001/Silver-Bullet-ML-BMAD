
import pandas as pd
from src.monitoring.trade_db import TradeDatabase
from datetime import datetime

db = TradeDatabase()
df = db.get_trades_df()

if df.empty:
    print("No trades found in database.")
    exit()

df['timestamp'] = pd.to_datetime(df['timestamp'])
results = []

for trader_id in df['trader_id'].unique():
    t_df = df[df['trader_id'] == trader_id].sort_values('timestamp')
    
    total_trades = len(t_df)
    start_date = t_df['timestamp'].min()
    end_date = t_df['timestamp'].max()
    
    # Calculate days span (min 1 day to avoid division by zero)
    days_span = (end_date - start_date).days + 1
    trades_per_day = total_trades / days_span
    
    results.append({
        'Trader ID': trader_id,
        'Total Trades': total_trades,
        'Days Active': days_span,
        'Trades/Day': round(trades_per_day, 2)
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
