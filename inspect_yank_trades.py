
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/trades.db")
df = pd.read_sql_query("SELECT * FROM trades WHERE trader_id = 'trader-yank' LIMIT 20", conn)
print(df.to_string())
conn.close()
