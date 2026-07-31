
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/trades.db")
df = pd.read_sql_query("SELECT * FROM trades WHERE trader_id = 'trader-s26' ORDER BY created_at DESC LIMIT 10", conn)
print("Last 10 records for trader-s26 in DB:")
print(df[['timestamp', 'pnl', 'created_at']].to_string())
conn.close()
