
import pandas as pd
from src.monitoring.trade_db import TradeDatabase
from src.research.performance_metrics_calculator import PerformanceMetricsCalculator
from datetime import datetime

db = TradeDatabase()
calc = PerformanceMetricsCalculator()

def get_today_report():
    today_str = datetime.utcnow().strftime('%Y-%m-%d')
    df = db.get_trades_df()
    
    if df.empty:
        return "No trades found in database."
    
    # Filter for trades created today
    today_trades = df[df['created_at'].str.startswith(today_str)].copy()
    
    if today_trades.empty:
        return f"No trades were logged today ({today_str})."
    
    traders = today_trades['trader_id'].unique()
    results = []
    
    for tid in traders:
        t_df = today_trades[today_trades['trader_id'] == tid].copy()
        
        numeric_cols = ['pnl', 'entry_price', 'exit_price']
        for col in numeric_cols:
            if col in t_df.columns:
                t_df[col] = pd.to_numeric(t_df[col], errors='coerce').fillna(0.0)
        
        if 'timestamp' in t_df.columns:
            t_df['timestamp'] = pd.to_datetime(t_df['timestamp'])
        
        try:
            metrics = calc.calculate_all_metrics(t_df)
            formatted = calc.format_metrics_dict(metrics)
            metrics_str = f"PNL: {formatted['Total Return']} | WR: {formatted['Win Rate']} | PF: {formatted['Profit Factor']}"
        except Exception as e:
            metrics_str = f"Error calculating: {e}"
            
        results.append(f"| {tid:<20} | {metrics_str} |")

    header = f"| {'Trader ID':<20} | {'Today\'s Performance':<100} |"
    sep = "|" + "-"*22 + "|" + "-"*102 + "|"
    return "\n".join([sep, header, sep] + results)

if __name__ == "__main__":
    print(get_today_report())
