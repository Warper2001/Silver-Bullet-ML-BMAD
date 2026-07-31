
import pandas as pd
from pathlib import Path
import os

log_dir = Path("logs")
csv_logs = list(log_dir.glob("*trade_log.csv"))

print(f"Checking {len(csv_logs)} log files for today's trades...")

for log in csv_logs:
    if not log.exists():
        continue
    try:
        # Read the file to see the last entries
        df = pd.read_csv(log)
        # Find time column
        date_col = [c for c in df.columns if 'time' in c][0]
        df[date_col] = pd.to_datetime(df[date_col], format='mixed', utc=True)
        today_trades = df[df[date_col].dt.date >= pd.to_datetime('2026-06-15').date()]
        if not today_trades.empty:
            print(f"Found {len(today_trades)} trades in {log.name} today:")
            print(today_trades.tail())
        else:
            print(f"No trades in {log.name} for today.")
    except Exception as e:
        print(f"Error checking {log.name}: {e}")
