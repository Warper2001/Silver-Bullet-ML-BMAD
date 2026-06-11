import os
import sys
import pandas as pd
import numpy as np
import pytz
import joblib

base_dir = "/root/Silver-Bullet-ML-BMAD"
csv_path = os.path.join(base_dir, "data/kraken/PF_XBTUSD_1min.csv")
print("Loading Kraken BTC data...")
df = pd.read_csv(csv_path)
df = df[df["timestamp"].str.startswith("2026")].copy()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df.sort_index(inplace=True)

# Recompute indicators for BTC
length = 20
sl_mult = 2.0
tp_mult = 4.0
max_hold = 60
et_tz = pytz.timezone("America/New_York")

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

# Simulate 0.62 threshold
model_path = os.path.join(base_dir, "models/s26_soft_fvg_ml_model.pkl")
model = joblib.load(model_path)
records = df.reset_index().to_dict(orient="records")

active_trade = None
completed_trades = []

print("Running trade simulation...")
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
            continue
            
    if not active_trade:
        long_cond = last_bar['long_cond']
        short_cond = last_bar['short_cond']
        
        if long_cond or short_cond:
            direction = 1 if long_cond else 0
            dir_str = 'L' if long_cond else 'S'
            
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
            
            if proba >= 0.62:
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

print(f"Generated {len(completed_trades)} trades. Simulating 50k Combine...")

# Prop Firm Rules:
# - Account Size: $50,000
# - Target: +$3,000
# - Daily Loss Limit: -$1,000
# - Max Trailing Drawdown: -$2,000

for size in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]:
    equity = 50000.0
    peak_equity = 50000.0
    max_dd = 0.0
    passed = False
    fail_dd = False
    fail_daily = False
    
    trades_till_pass = 0
    days_till_pass = 0
    
    daily_pnls = {}
    
    for i, t in enumerate(completed_trades):
        pnl_usd = t['pnl'] * size
        equity += pnl_usd
        peak_equity = max(peak_equity, equity)
        dd = peak_equity - equity
        max_dd = max(max_dd, dd)
        
        # Track daily P&L
        day_key = t['exit_time'].date()
        daily_pnls[day_key] = daily_pnls.get(day_key, 0.0) + pnl_usd
        
        if dd >= 2000.0:
            fail_dd = True
            
        if daily_pnls[day_key] <= -1000.0:
            fail_daily = True
            
        if equity >= 53000.0 and not fail_dd and not fail_daily:
            passed = True
            trades_till_pass = i + 1
            days_till_pass = len(set([tr['exit_time'].date() for tr in completed_trades[:i+1]]))
            break
            
    print(f"Size: {size:.2f} BTC | Passed: {passed:<5} | Max DD: ${max_dd:,.2f} | Daily Fail: {fail_daily:<5} | Drawdown Fail: {fail_dd:<5} | Trades: {trades_till_pass:<3} | Days: {days_till_pass}")
