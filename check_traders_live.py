
import pandas as pd
from src.monitoring.trade_db import TradeDatabase
from src.research.performance_metrics_calculator import PerformanceMetricsCalculator
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
db = TradeDatabase()
calc = PerformanceMetricsCalculator()

# Define the date when "Live" trading actually started for the project.
# Based on the logs, we see trades from 2025, which the user identifies as backtests.
# I will use 2026-01-01 as a conservative cutoff for "Live" trades.
LIVE_START_DATE = "2026-01-01"

def get_report():
    traders = db.get_all_trader_ids()
    if not traders:
        return "No trades found in database."
    
    results = []
    for trader_id in traders:
        df = db.get_trades_df(trader_id)
        if df.empty or len(df) == 0:
            metrics_metrics_str = "No trades"
        else:
            try:
                # FILTER: Only keep trades from 2026 onwards (Live)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                live_df = df[df['timestamp'] >= LIVE_START_DATE].copy()
                
                if live_df.empty:
                    metrics_str = "No live trades (only backtest data)"
                else:
                    # Ensure numeric types
                    numeric_cols = ['pnl', 'entry_price', 'exit_price']
                    for col in numeric_cols:
                        if col in live_df.columns:
                            live_df[col] = pd.to_numeric(live_df[col], errors='coerce').fillna(0.0)
                    
                    metrics = calc.calculate_all_metrics(live_df)
                    formatted = calc.format_metrics_dict(metrics)
                    metrics_str = f"PNL: {formatted['Total Return']} | WR: {formatted['Win Rate']} | PF: {formatted['Profit Factor']} | DD: {formatted['Max Drawdown']}"
            except Exception as e:
                metrics_str = f"Error: {e}"
        
        results.append(f"| {trader_id:<20} | {metrics_str} |")

    header = f"| {'Trader ID':<20} | {'Live Performance Metrics (Since 2026)':<100} |"
    sep = "|" + "-"*22 + "|" + "-"*102 + "|"
    return "\n".join([sep, header, sep] + results)

if __name__ == "__main__":
    print(get_report())
