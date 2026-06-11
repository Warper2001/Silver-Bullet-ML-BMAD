import os
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta

base_dir = "/root/Silver-Bullet-ML-BMAD"
train_csv_path = os.path.join(base_dir, "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
bt_csv_path = os.path.join(base_dir, "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

et_tz = pytz.timezone("America/New_York")

def run_vwap_reversion(csv_path, year_label):
    print(f"\nLoading data for {year_label}...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert(et_tz)
    
    # Calculate VWAP manually
    # VWAP = cumulative(Volume * Typical Price) / cumulative(Volume)
    df_et['typical_price'] = (df_et['high'] + df_et['low'] + df_et['close']) / 3
    df_et['vol_x_tp'] = df_et['volume'] * df_et['typical_price']
    
    # Group by date to calculate daily VWAP
    df_et['date'] = df_et.index.date
    df_et['cum_vol'] = df_et.groupby('date')['volume'].cumsum()
    df_et['cum_vol_x_tp'] = df_et.groupby('date')['vol_x_tp'].cumsum()
    df_et['vwap'] = df_et['cum_vol_x_tp'] / df_et['cum_vol']
    
    # ATR for stop loss
    df_et['prev_close'] = df_et['close'].shift(1)
    df_et['tr0'] = abs(df_et['high'] - df_et['low'])
    df_et['tr1'] = abs(df_et['high'] - df_et['prev_close'])
    df_et['tr2'] = abs(df_et['low'] - df_et['prev_close'])
    df_et['tr'] = df_et[['tr0', 'tr1', 'tr2']].max(axis=1)
    df_et['atr'] = df_et['tr'].rolling(20).mean()
    
    # RTH Filter
    df_et['hour'] = df_et.index.hour
    df_et['minute'] = df_et.index.minute
    df_et['is_rth'] = (((df_et['hour'] == 9) & (df_et['minute'] >= 30)) | 
                       ((df_et['hour'] >= 10) & (df_et['hour'] < 16))).astype(int)
    
    df_et.dropna(inplace=True)
    
    completed_trades = []
    
    DEV_THRESHOLD = 3.0 # Divergence of 3x ATR from VWAP
    
    days = df_et.groupby('date')
    
    for day, day_df in days:
        active_trade = None
        records = day_df.reset_index().to_dict('records')
        
        for i, row in enumerate(records):
            current_time = row['timestamp']
            
            if active_trade:
                t = active_trade
                exit_reason = None
                exit_price = 0
                
                # Dynamic TP is the VWAP!
                current_vwap = row['vwap']
                
                if t['dir'] == 'SHORT':
                    if row['high'] >= t['sl']:
                        exit_reason, exit_price = 'SL', t['sl']
                    elif row['low'] <= current_vwap:
                        exit_reason, exit_price = 'TP', current_vwap
                else:
                    if row['low'] <= t['sl']:
                        exit_reason, exit_price = 'SL', t['sl']
                    elif row['high'] >= current_vwap:
                        exit_reason, exit_price = 'TP', current_vwap
                        
                if not exit_reason and i == len(records) - 1:
                    exit_reason, exit_price = 'TIME_STOP', row['close']
                    
                if exit_reason:
                    pnl = (t['entry'] - exit_price) if t['dir'] == 'SHORT' else (exit_price - t['entry'])
                    completed_trades.append({
                        'day': day,
                        'entry_time': t['entry_time'],
                        'exit_time': current_time,
                        'dir': t['dir'],
                        'entry': t['entry'],
                        'exit': exit_price,
                        'pnl': pnl,
                        'reason': exit_reason
                    })
                    active_trade = None
                    break # One reversion trade per day
                    
            elif row['is_rth'] == 1 and current_time.time() > pd.to_datetime("10:00:00").time():
                # Wait for market to settle
                dist_from_vwap = row['close'] - row['vwap']
                atr = row['atr']
                
                if atr > 0:
                    z_score = dist_from_vwap / atr
                    
                    if z_score > DEV_THRESHOLD:
                        # Price is too high, short back to VWAP
                        active_trade = {
                            'dir': 'SHORT',
                            'entry': row['close'],
                            'sl': row['close'] + (atr * 2.0),
                            'entry_time': current_time
                        }
                    elif z_score < -DEV_THRESHOLD:
                        # Price is too low, long back to VWAP
                        active_trade = {
                            'dir': 'LONG',
                            'entry': row['close'],
                            'sl': row['close'] - (atr * 2.0),
                            'entry_time': current_time
                        }

    print(f"\n--- {year_label} VWAP REVERSION BACKTEST RESULTS ---")
    if not completed_trades:
        print("No trades generated.")
        return
        
    df_trades = pd.DataFrame(completed_trades)
    
    total_trades = len(df_trades)
    wins = len(df_trades[df_trades['pnl'] > 0])
    losses = len(df_trades[df_trades['pnl'] < 0])
    win_rate = wins / total_trades * 100
    
    gross_prof = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
    pf = gross_prof / gross_loss if gross_loss > 0 else float('inf')
    
    net_pnl_pts = df_trades['pnl'].sum()
    net_pnl_usd = net_pnl_pts * 2.0
    
    print(f"Total Trades:  {total_trades}")
    print(f"Win Rate:      {win_rate:.1f}% ({wins} W / {losses} L)")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Net P&L:       {net_pnl_pts:+.2f} points (${net_pnl_usd:+.2f})")
    
    reasons = df_trades['reason'].value_counts().to_dict()
    print(f"Exit Reasons:  {reasons}")

run_vwap_reversion(train_csv_path, "2025")
run_vwap_reversion(bt_csv_path, "2026 YTD")
