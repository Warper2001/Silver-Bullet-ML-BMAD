
import pandas as pd
from src.monitoring.trade_db import TradeDatabase
from src.research.performance_metrics_calculator import PerformanceMetricsCalculator

db = TradeDatabase()
calc = PerformanceMetricsCalculator()

def get_report():
    traders = db.get_all_trader_ids()
    results = []
    
    for trader_id in traders:
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
    return "\n".join([sep, header, sep] + results)

if __name__ == "__main__":
    print(get_report())
