import os
import csv
import sys
import pandas as pd
import numpy as np
import joblib
import pytz
from datetime import datetime

base_dir = "/root/Silver-Bullet-ML-BMAD"
csv_path = os.path.join(base_dir, "data/kraken/PF_XBTUSD_1min.csv")
model_path = os.path.join(base_dir, "models/s26_soft_fvg_ml_model.pkl")

# S26 Parameters
length = 20
sl_mult = 2.0
tp_mult = 4.0
max_hold = 60
et_tz = pytz.timezone("America/New_York")

def main():
    # Allow passing custom threshold as command line arg
    ml_thresh = 0.62
    if len(sys.argv) > 1:
        ml_thresh = float(sys.argv[1])
        
    print(f"Loading Kraken 1m data for 2026...")
    df = pd.read_csv(csv_path)
    df = df[df["timestamp"].str.startswith("2026")].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Loaded {len(df)} bars. Computing vectorized indicators...")
    
    # 1. Indicators
    df['prev_close'] = df['close'].shift(1)
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['prev_close'])
    df['tr2'] = abs(df['low'] - df['prev_close'])
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['atr'] = df['tr'].rolling(length).mean()
    
    df['h1_high'] = df['high'].rolling(360).max()
    df['h1_low'] = df['low'].rolling(360).min()
    df['sweep_bear'] = (df['high'] >= df['h1_high'].shift(1))
    df['sweep_bull'] = (df['low'] <= df['h1_low'].shift(1))
    df['recent_sweep_bear'] = df['sweep_bear'].astype(int).rolling(60).max() > 0
    df['recent_sweep_bull'] = df['sweep_bull'].astype(int).rolling(60).max() > 0
    
    df['soft_fvg_bear'] = (df['low'].shift(2) - df['high']) > (0.2 * df['atr'])
    df['soft_fvg_bull'] = (df['low'] - df['high'].shift(2)) > (0.2 * df['atr'])
    
    df['long_cond'] = df['recent_sweep_bull'] & df['soft_fvg_bull']
    df['short_cond'] = df['recent_sweep_bear'] & df['soft_fvg_bear']
    
    # Fill fillna with standard parameters to replicate pandas shift fillna behavior
    df['long_cond'] = df['long_cond'] & (~df['long_cond'].shift(1).fillna(False))
    df['short_cond'] = df['short_cond'] & (~df['short_cond'].shift(1).fillna(False))
    
    df['vol_sma'] = df['volume'].rolling(50).mean()
    df['rvol'] = df['volume'] / df['vol_sma'].replace(0, np.nan)
    df['rvol'] = df['rvol'].fillna(1.0)
    
    df['macro_ema'] = df['close'].ewm(span=200).mean()
    df['dist_macro_ema'] = (df['close'] - df['macro_ema']) / df['atr']
    
    df['ema'] = df['close'].ewm(span=length, adjust=False).mean()
    df['dist_ema'] = (df['close'] - df['ema']) / df['atr']
    
    df_et = df.index.tz_convert(et_tz)
    df['hour_et'] = df_et.hour
    df['minute_et'] = df_et.minute
    df['dow'] = df_et.dayofweek
    df['is_us_session'] = (((df['hour_et'] == 9) & (df['minute_et'] >= 30)) | 
                           ((df['hour_et'] >= 10) & (df['hour_et'] < 16)) | 
                           ((df['hour_et'] == 16) & (df['minute_et'] == 0))).astype(int)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in ['atr', 'rvol', 'dist_ema', 'dist_macro_ema']:
        df[col] = df[col].fillna(0)
        
    print("Loading ML model...")
    model = joblib.load(model_path)
    
    # 2. Replay loop (extremely fast since indicators are pre-computed)
    print(f"Replaying {len(df)} bars with threshold={ml_thresh:.2f}...")
    
    active_trade = None
    completed_trades = []
    
    # Convert dataframe to a fast array/list of dicts or records for maximum speed
    records = df.reset_index().to_dict(orient="records")
    
    for i in range(1, len(records)):
        current_bar = records[i]
        last_bar = records[i - 1]
        
        # Check active trade exit
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
                completed_trades.append({
                    'entry_time': t['ts'],
                    'exit_time': current_bar['timestamp'],
                    'dir': t['dir'],
                    'entry': t['entry'],
                    'exit': exit_price,
                    'reason': exit_reason,
                    'pnl': pnl,
                    'proba': t['proba']
                })
                active_trade = None
                continue  # Exit processed, no entry on the same bar
                
        # Check entry (only if flat)
        if not active_trade:
            long_cond = last_bar['long_cond']
            short_cond = last_bar['short_cond']
            
            if long_cond or short_cond:
                direction = 1 if long_cond else 0
                dir_str = 'L' if long_cond else 'S'
                
                # Extract features for ML
                features = pd.DataFrame([{
                    'dir': direction,
                    'atr': last_bar['atr'],
                    'rvol': last_bar['rvol'],
                    'dist_ema': last_bar['dist_ema'],
                    'dist_macro_ema': last_bar['dist_macro_ema'],
                    'hour_et': last_bar['hour_et'],
                    'dow': last_bar['dow'],
                    'is_us_session': last_bar['is_us_session']
                }])
                
                proba = model.predict_proba(features)[0, 1]
                
                if proba >= ml_thresh:
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
                        'proba': proba,
                        'hold_time': 0,
                        'ts': current_bar['timestamp']
                    }
                    
    print()
    print("=" * 70)
    print(f"S26 CRYPTO BACKTEST TRADES (2026 YTD | THRESHOLD={ml_thresh:.2f})")
    print("=" * 70)
    
    total_pnl = 0.0
    wins = 0
    losses = 0
    
    for t in completed_trades:
        total_pnl += t['pnl']
        if t['pnl'] > 0:
            wins += 1
        elif t['pnl'] < 0:
            losses += 1
            
        print(f"  {t['reason']:<9}  {t['dir']}  entry={t['entry']:.2f}  exit={t['exit']:.2f}  P&L=${t['pnl']:+.2f}  P(Success)={t['proba']:.3f}  {t['entry_time'].strftime('%Y-%m-%d %H:%M')} UTC")
        
    count = len(completed_trades)
    win_rate = wins / count * 100 if count > 0 else 0.0
    gross_profit = sum([t['pnl'] for t in completed_trades if t['pnl'] > 0])
    gross_loss = abs(sum([t['pnl'] for t in completed_trades if t['pnl'] < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    print("-" * 70)
    print(f"  Trades Filled: {count}")
    print(f"  Win Rate:      {win_rate:.1f}% ({wins} Wins, {losses} Losses)")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Net P&L:       ${total_pnl:+.2f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
