
import sqlite3
import pandas as pd
from pathlib import Path

csv_path = Path("logs/s26_soft_fvg_trade_log.csv")
db_path = Path("data/trades.db")

# Read CSV
df = pd.read_csv(csv_path)
df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)

# Filter for June 15th
new_trades = df[df['entry_time'].dt.date == pd.to_datetime('2026-06-15').date()]

# Sync to DB
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

for _, trade in new_trades.iterrows():
    # Only insert if not already exists (naive check by timestamp)
    cursor.execute("""
        INSERT OR IGNORE INTO trades (trader_id, timestamp, pnl, direction, entry_price, exit_price, exit_reason, ml_proba)
        SELECT ?, ?, ?, ?, ?, ?, ?, ?
        WHERE NOT EXISTS (SELECT 1 FROM trades WHERE trader_id = ? AND timestamp = ?)
    """, ('trader-s26', trade['entry_time'].isoformat(), trade['pnl'], trade['direction'], trade['entry_price'], trade['exit_price'], trade['reason'], trade['ml_proba'], 'trader-s26', trade['entry_time'].isoformat()))

conn.commit()
print(f"Sync complete. {cursor.rowcount} trades added to DB.")
conn.close()
