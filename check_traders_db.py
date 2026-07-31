
import pandas as pd
from src.monitoring.trade_db import TradeDatabase
from src.research.performance_metrics_calculator import PerformanceMetricsCalculator
import logging

logging.basicConfig(level=logging.INFO)
db = TradeDatabase()
calc = PerformanceMetricsCalculator()

def get_report():
    traders = db.get_all_trader_ids()
    if not traders:
        return "No trades found in database."
    
    results = []
    for trader_id in traders:
        df = db.get_trades_df(trader_id)
        if df.empty:
            metrics_str = "No trades"
        else:
            try:
                # Force ALL numeric columns to be float
                numeric_cols = ['pnl', 'entry_price', 'exit_price']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                
                # The error "unsupported operand type(s) for -: 'str' and 'str'" 
                # likely comes from timestamp handling in the calculator's _build_equity_curve.
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                metrics = calc.calculate_all_metrics(df)
                formatted = calc.format_metrics_dict(metrics)
                metrics_str = f"PNL: {formatted['Total Return']} | WR: {formatted['Win Rate']} | PF: {formatted['Profit Factor']} | DD: {formatted['Max Drawdown']}"
            except Exception as e:
                import traceback
                logger.error(traceback.format_exc())
                metrics_str = f"Error: {e}"
        
        results.append(f"| {trader_id:<20} | {metrics_str} |")

    header = f"| {'Trader ID':<20} | {'Performance Metrics':<100} |"
    sep = "|" + "-"*22 + "|" + "-"*102 + "|"
    return "\n".join([sep, header, sep] + results)

if __name__ == "__main__":
    print(get_report())
