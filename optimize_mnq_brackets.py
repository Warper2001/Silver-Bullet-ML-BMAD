import os
import pandas as pd
import numpy as np
import joblib
import pytz
import itertools

base_dir = "/root/Silver-Bullet-ML-BMAD"
train_csv_path = os.path.join(base_dir, "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
backtest_csv_path = os.path.join(base_dir, "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

# Import the feature loader from our pipeline script
import sys
sys.path.insert(0, base_dir)
from run_mnq_s26_pipeline import load_and_compute_features

# Let's load the data once to make the grid search extremely fast
print("Loading and computing features for 2025 (In-Sample)...")
df_train = load_and_compute_features(train_csv_path)

print("Loading and computing features for 2026 YTD (Out-of-Sample)...")
df_bt = load_and_compute_features(backtest_csv_path)

# Convert to records for maximum speed
train_records = df_train.reset_index().to_dict(orient="records")
bt_records = df_bt.reset_index().to_dict(orient="records")

def run_backtest_simulation(records, sl_mult, tp_mult, max_hold, rth_only=False):
    active_trade = None
    trades = []
    
    for i in range(1, len(records)):
        current_bar = records[i]
        last_bar = records[i - 1]
        
        if active_trade:
            active_trade['hold_time'] += 1
            t = active_trade
            exit_reason = None
            exit_price = 0
            
            if t['dir'] == 'L':
                if current_bar['low'] <= t['sl']:
                    exit_reason, exit_price = 'SL', t['sl']
                elif current_bar['high'] >= t['tp']:
                    exit_reason, exit_price = 'TP', t['tp']
            else:
                if current_bar['high'] >= t['sl']:
                    exit_reason, exit_price = 'SL', t['sl']
                elif current_bar['low'] <= t['tp']:
                    exit_reason, exit_price = 'TP', t['tp']
                    
            if not exit_reason and t['hold_time'] >= max_hold:
                exit_reason, exit_price = 'TIME_STOP', current_bar['close']
                
            if exit_reason:
                pnl = (exit_price - t['entry']) if t['dir'] == 'L' else (t['entry'] - exit_price)
                trades.append(pnl)
                active_trade = None
                continue
                
        if not active_trade:
            long_cond = last_bar['long_cond']
            short_cond = last_bar['short_cond']
            
            if rth_only and last_bar['is_us_session'] == 0:
                continue
                
            if long_cond or short_cond:
                direction = 1 if long_cond else 0
                dir_str = 'L' if long_cond else 'S'
                
                entry_price = current_bar['open']
                atr = last_bar['atr']
                
                sl = entry_price - (atr * sl_mult) if direction == 1 else entry_price + (atr * sl_mult)
                tp = entry_price + (atr * tp_mult) if direction == 1 else entry_price - (atr * tp_mult)
                
                active_trade = {
                    'dir': dir_str,
                    'entry': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'atr': atr,
                    'hold_time': 0,
                    'ts': current_bar['timestamp']
                }
    return trades

# Grid search parameters
sl_grid = [1.0, 2.0, 3.0, 4.0]
tp_grid = [1.0, 2.0, 3.0, 4.0, 6.0]
hold_grid = [30, 60, 120]
rth_grid = [False, True]

print("\n=== STARTING PARAMETER OPTIMIZATION GRID SEARCH ===")
results = []

for sl, tp, hold, rth in itertools.product(sl_grid, tp_grid, hold_grid, rth_grid):
    # Run in-sample 2025
    train_pnls = run_backtest_simulation(train_records, sl, tp, hold, rth)
    count_tr = len(train_pnls)
    if count_tr < 50: # skip under-sampled configurations
        continue
        
    wins_tr = len([p for p in train_pnls if p > 0])
    gross_prof_tr = sum([p for p in train_pnls if p > 0])
    gross_loss_tr = abs(sum([p for p in train_pnls if p < 0]))
    pf_tr = gross_prof_tr / gross_loss_tr if gross_loss_tr > 0 else float('inf')
    
    # Run out-of-sample 2026 YTD
    bt_pnls = run_backtest_simulation(bt_records, sl, tp, hold, rth)
    count_bt = len(bt_pnls)
    if count_bt == 0:
        continue
    wins_bt = len([p for p in bt_pnls if p > 0])
    win_rate_bt = wins_bt / count_bt * 100
    gross_prof_bt = sum([p for p in bt_pnls if p > 0])
    gross_loss_bt = abs(sum([p for p in bt_pnls if p < 0]))
    pf_bt = gross_prof_bt / gross_loss_bt if gross_loss_bt > 0 else float('inf')
    total_pnl_bt = sum(bt_pnls)
    
    results.append({
        'sl': sl, 'tp': tp, 'hold': hold, 'rth_only': rth,
        'count_2025': count_tr, 'pf_2025': pf_tr,
        'count_2026': count_bt, 'win_rate_2026': win_rate_bt,
        'pf_2026': pf_bt, 'pnl_2026': total_pnl_bt
    })

results_df = pd.DataFrame(results)
results_df.sort_values(by='pf_2026', ascending=False, inplace=True)

print("\n=== TOP 15 CONFIGURATIONS BY OUT-OF-SAMPLES (2026 YTD) PROFIT FACTOR ===")
print(results_df[['sl', 'tp', 'hold', 'rth_only', 'count_2025', 'pf_2025', 'count_2026', 'pf_2026', 'pnl_2026']].head(15).to_string(index=False))

print("\n=== TOP 15 CONFIGURATIONS BY IN-SAMPLES (2025) PROFIT FACTOR ===")
print(results_df.sort_values(by='pf_2025', ascending=False)[['sl', 'tp', 'hold', 'rth_only', 'count_2025', 'pf_2025', 'count_2026', 'pf_2026', 'pnl_2026']].head(15).to_string(index=False))
