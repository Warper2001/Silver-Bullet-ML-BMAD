
import pandas as pd
from src.monitoring.trade_db import TradeDatabase
from src.research.performance_metrics_calculator import PerformanceMetricsCalculator
import sqlite3

db = TradeDatabase()
calc = PerformanceMetricsCalculator()

# 1. Identify all services that have trade logs (legacy CSVs)
import glob
csv_logs = glob.glob("logs/*trade_log.csv") + glob.glob("logs/*combine*.log")

# 2. Get traders already in DB
db_traders = db.get_all_trader_ids()

print(f"Traders currently in DB: {db_traders}")

results = []
# Loop through all found logs and build metrics
# We will consolidate everything into the report table
for trader_id in db_traders:
    df = db.get_trades_df(trader_id=trader_id)
    if df.empty:
        results.append(f"| {trader_id:<20} | No data |")
        continue
    
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['exit_time'] = df['timestamp']
    df['entry_time'] = df['timestamp']
    
    try:
        metrics = calc.calculate_all_metrics(df)
        fmt = calc.format_metrics_dict(metrics)
        metrics_str = f"PNL: {fmt['Total Return']} | WR: {fmt['Win Rate']} | PF: {fmt['Profit Factor']}"
        results.append(f"| {trader_id:<20} | {metrics_str} |")
    except Exception as e:
        results.append(f"| {trader_id:<20} | Error: {e} |")

header = f"| {'Trader ID':<20} | {'Performance Summary':<100} |"
sep = "|" + "-"*22 + "|" + "-"*102 + "|"
print("\n".join([sep, header, sep] + results))
