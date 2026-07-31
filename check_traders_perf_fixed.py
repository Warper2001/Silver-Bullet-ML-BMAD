
import pandas as pd
from src.research.performance_metrics_calculator import PerformanceMetricsCalculator
import os

# Updated Trader to Log Mapping & Handling logic
LOG_MAP = {
    'trader-yank.service': 'logs/tier2_trade_log.csv',
    'trader-s26.service': 'logs/s26_soft_fvg_trade_log.csv',
    'trader-s27.service': 'logs/s27_squeeze_trade_log.csv',
    'trader-mim-nb.service': 'data/mim_nb/trades.csv',
    'trader-btc-carry.service': 'logs/carry_positions.csv',
    'trader-btc-combine.service': 'logs/s26_combine_bot.log',
    'trader-s26-combine.service': 'logs/s26_combine_bot.log',
    'trader-stat-arb.service': 'logs/stat_arb_bot.log',
}

calc = PerformanceMetricsCalculator()
results = []

for service, log_path in LOG_MAP.items():
    status = "Active"
    metrics_str = "N/A"
    
    if os.path.exists(log_path):
        try:
            if log_path.endswith('.csv'):
                # FIX: Use on_bad_lines='skip' to handle the tokenization errors 
                # observed in S27 and BTC-Carry (extra columns/commas)
                df = pd.read_csv(log_path, on_bad_lines='skip')
                
                if not df.empty:
                    # Map specific columns to 'pnl' if they differ (like pnl_usd for carry)
                    if 'pnl_usd' in df.columns and 'pnl' not in df.columns:
                        df = df.rename(columns={'pnl_usd': 'pnl'})
                    
                    if 'pnl' in df.columns:
                        metrics = calc.calculate_all_metrics(df)
                        formatted = calc.format_metrics_dict(metrics)
                        metrics_str = f"PNL: {formatted['Total Return']} | WR: {formatted['Win Rate']} | PF: {formatted['Profit Factor']} | DD: {formatted['Max Drawdown']}"
                    else:
                        metrics_str = "CSV missing 'pnl' column"
                else:
                    metrics_str = "No trades found in CSV"
            else:
                metrics_str = "Log is not CSV (Text Log)"
        except Exception as e:
            metrics_str = f"Error processing log: {str(e)}"
    else:
        metrics_str = "Log file not found"
    
    results.append(f"| {service:<30} | {status:<10} | {metrics_str} |")

print("|" + "-"*32 + "|" + "-"*12 + "|" + "-"*100 + "|")
print(f"| {'Service Name':<30} | {'Status':<10} | {'Performance Metrics':<100} |")
print("|" + "-"*32 + "|" + "-"*12 + "|" + "-"*100 + "|")
for r in results:
    print(r)
