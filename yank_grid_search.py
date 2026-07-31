
import pandas as pd
import numpy as np
from src.monitoring.trade_db import TradeDatabase
from src.research.performance_metrics_calculator import PerformanceMetricsCalculator
import itertools

db = TradeDatabase()
calc = PerformanceMetrics = PerformanceMetricsCalculator()

def run_simulation(df, ml_thresh, tp_mult):
    # In a real backtest, we'd re-run the strategy logic.
    # Since we are optimizing based on existing trade logs, we simulate the effect:
    # 1. Tighter Entry: We assume trades with lower ml_proba are filtered out.
    # 2. Wider Exit: We simulate increasing the TP. 
    # Note: Since the DB doesn't store the raw 'proba' for all legacy YANK trades, 
    # we will simulate the 'Tighter Entry' by filtering the most recent trades 
    # where proba is available, or by simulating a win-rate decay/increase.
    
    # However, a better way is to use the actual trade records and simulate 
    # the PnL shift. If we increase TP mult, we assume a portion of TP trades 
    # gain more, while some TP trades become SL/TimeStops.
    
    # For a precise grid search, we should use the backtest engine.
    # But for a quick "direction" check, we can use the current trade distribution.
    
    # Let's use the actual backtest_tier2_1year_validation.py logic if possible,
    # or a simplified version here.
    return 0 # Placeholder for the logic below

def main():
    df = db.get_trades_df('trader-yank')
    if df.empty:
        print("No data")
        return

    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
    
    # Grid parameters
    ml_thresholds = [0.60, 0.65, 0.70, 0.75]
    tp_multipliers = [5.0, 6.0, 7.0, 8.0]
    
    results = []
    
    # For this a-priori analysis, we'll simulate:
    # - Increasing ML Threshold reduces trade count but increases Win Rate (estimated +2% per 0.05 bump)
    # - Increasing TP multiplier increases Avg Win but decreases Win Rate (estimated -3% per 1.0 bump)
    
    base_wr = 0.523
    base_avg_win = 480.24
    base_avg_loss = 458.49
    base_count = len(df)
    
    for ml, tp in itertools.product(ml_thresholds, tp_multipliers):
        # Simulation Model:
        # ML Threshold impact: Trade count drops, WR increases
        count_factor = 1.0 - ((ml - 0.60) * 2.0) # 0.75 threshold -> 70% of trades
        wr = base_wr + ((ml - 0.60) * 0.1) # WR increases slightly
        
        # TP Multiplier impact: Avg Win increases, WR drops
        win_boost = (tp - 6.0) * 100 # +$100 per multiplier point
        wr -= (tp - 6.0) * 0.03 # WR drops 3% per point
        
        sim_count = int(base_count * count_factor)
        sim_wins = int(sim_count * wr)
        sim_losses = sim_count - sim_wins
        
        total_pnl = (sim_wins * (base_avg_win + win_boost)) - (sim_losses * base_avg_loss)
        pf = (sim_wins * (base_avg_win + win_boost)) / (sim_losses * base_avg_loss) if sim_losses > 0 else float('inf')
        
        results.append({
            'ml_thresh': ml,
            'tp_mult': tp,
            'total_pnl': total_pnl,
            'pf': pf,
            'wr': wr,
            'count': sim_count
        })

    res_df = pd.DataFrame(results)
    print(res_df.sort_values('pf', ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
