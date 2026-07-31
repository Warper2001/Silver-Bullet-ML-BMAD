
import sqlite3
import pandas as pd
from pathlib import Path

csv_path = Path("logs/s26_soft_fvg_trade_log.csv")
db_path = Path("data/trades.db")

df = pd.read_csv(csv_path)
df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
new_trades = df[df['entry_time'].dt.date == pd.to_datetime('2026-06-15').date()]

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

count = 0
for _, trade in new_trades.iterrows():
    # Force insert without the timestamp check to ensure all 4 are in
    cursor.execute("""
        INSERT OR IGNORE INTO trades (trader_id, timestamp, pnl, direction, entry_price, exit_price, exit_reason, ml_proba)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, ('trader-s26', trade['entry_time'].isoformat(), trade['pnl'], trade['direction'], trade['entry_price'], trade['exit_price'], trade['reason'], trade['ml_proba']))
    count += 1

conn.commit()
print(f"Sync complete. {count} total trades inserted.")
conn.close()
