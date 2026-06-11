import os
import csv
import sys
import pandas as pd
import numpy as np
import joblib
import pytz
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

base_dir = "/root/Silver-Bullet-ML-BMAD"
train_csv_path = os.path.join(base_dir, "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
backtest_csv_path = os.path.join(base_dir, "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
model_path = os.path.join(base_dir, "models/mnq_s26_xgboost_model.pkl")

# MNQ Parameters (tuned for stock index volatility)
length = 20
sl_mult = 2.0
tp_mult = 4.0
max_hold = 60
et_tz = pytz.timezone("America/New_York")

def load_and_compute_features(csv_path):
    print(f"Reading {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    # 1. Indicators
    df['prev_close'] = df['close'].shift(1)
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['prev_close'])
    df['tr2'] = abs(df['low'] - df['prev_close'])
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['atr'] = df['tr'].rolling(length).mean()
    
    # Stock indices require different sweep lookbacks (360m ≈ 6 hours)
    df['h1_high'] = df['high'].rolling(360).max()
    df['h1_low'] = df['low'].rolling(360).min()
    df['sweep_bear'] = (df['high'] >= df['h1_high'].shift(1))
    df['sweep_bull'] = (df['low'] <= df['h1_low'].shift(1))
    df['recent_sweep_bear'] = df['sweep_bear'].astype(int).rolling(60).max() > 0
    df['recent_sweep_bull'] = df['sweep_bull'].astype(int).rolling(60).max() > 0
    
    # FVG size threshold (0.2x ATR is soft FVG)
    df['soft_fvg_bear'] = (df['low'].shift(2) - df['high']) > (0.2 * df['atr'])
    df['soft_fvg_bull'] = (df['low'] - df['high'].shift(2)) > (0.2 * df['atr'])
    
    df['long_cond'] = df['recent_sweep_bull'] & df['soft_fvg_bull']
    df['short_cond'] = df['recent_sweep_bear'] & df['soft_fvg_bear']
    
    df['long_cond'] = df['long_cond'] & (~df['long_cond'].shift(1).fillna(False))
    df['short_cond'] = df['short_cond'] & (~df['short_cond'].shift(1).fillna(False))
    
    # Volatility and EMAs
    df['vol_sma'] = df['volume'].rolling(50).mean()
    df['rvol'] = df['volume'] / df['vol_sma'].replace(0, np.nan)
    df['rvol'] = df['rvol'].fillna(1.0)
    
    df['macro_ema'] = df['close'].ewm(span=200).mean()
    df['dist_macro_ema'] = (df['close'] - df['macro_ema']) / df['atr']
    
    df['ema'] = df['close'].ewm(span=length, adjust=False).mean()
    df['dist_ema'] = (df['close'] - df['ema']) / df['atr']
    
    # US Session Flags (crucial for MNQ stock index trading!)
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
        
    return df

def simulate_trades(df):
    records = df.reset_index().to_dict(orient="records")
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
                trades.append({
                    'entry_time': t['ts'],
                    'dir': t['dir'],
                    'entry': t['entry'],
                    'exit': exit_price,
                    'reason': exit_reason,
                    'pnl': pnl,
                    'features': t['features'],
                    'outcome': 1 if pnl > 0 else 0
                })
                active_trade = None
                continue
                
        if not active_trade:
            long_cond = last_bar['long_cond']
            short_cond = last_bar['short_cond']
            
            if long_cond or short_cond:
                direction = 1 if long_cond else 0
                dir_str = 'L' if long_cond else 'S'
                
                # Features at entry time
                feat_dict = {
                    'dir': direction,
                    'atr': last_bar['atr'],
                    'rvol': last_bar['rvol'],
                    'dist_ema': last_bar['dist_ema'],
                    'dist_macro_ema': last_bar['dist_macro_ema'],
                    'hour_et': last_bar['hour_et'],
                    'dow': last_bar['dow'],
                    'is_us_session': last_bar['is_us_session']
                }
                
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
                    'proba': 1.0,
                    'hold_time': 0,
                    'ts': current_bar['timestamp'],
                    'features': feat_dict
                }
    return trades

def main():
    print("=" * 70)
    print("MNQ S26 SWEEP + SOFT FVG STRATEGY PIPELINE")
    print("=" * 70)
    
    # Step 1: Feature Extraction on 2025 Training Data
    print("\n--- STEP 1: Feature Extraction (2025 Dataset) ---")
    df_train = load_and_compute_features(train_csv_path)
    trades_train = simulate_trades(df_train)
    print(f"Generated {len(trades_train)} trade setups from 2025.")
    
    # Prepare training sets
    X_train = pd.DataFrame([t['features'] for t in trades_train])
    y_train = np.array([t['outcome'] for t in trades_train])
    
    # Step 2: ML Model Training (calibrated HistGradientBoosting)
    print("\n--- STEP 2: ML Model Training (HistGradientBoosting) ---")
    clf = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    
    # Use Sigmoid Calibration for robust probabilities
    calibrated_clf = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv=5)
    calibrated_clf.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(calibrated_clf, model_path)
    print(f"Trained model saved successfully to {model_path}!")
    
    # Step 3: Run Backtest on 2026 YTD Data
    print("\n--- STEP 3: Out-Of-Sample Backtest (2026 YTD Dataset) ---")
    df_bt = load_and_compute_features(backtest_csv_path)
    trades_bt = simulate_trades(df_bt)
    print(f"Generated {len(trades_bt)} raw trade setups from 2026.")
    
    # Replay with thresholds
    for threshold in [0.0, 0.55, 0.60, 0.65]:
        print(f"\nEvaluating performance at threshold >= {threshold:.2f}...")
        bt_trades_executed = []
        
        for t in trades_bt:
            feat_df = pd.DataFrame([t['features']])
            proba = calibrated_clf.predict_proba(feat_df)[0, 1]
            if proba >= threshold:
                bt_trades_executed.append({
                    'dir': t['dir'],
                    'pnl': t['pnl'],
                    'proba': proba
                })
                
        count = len(bt_trades_executed)
        if count > 0:
            wins = len([t for t in bt_trades_executed if t['pnl'] > 0])
            losses = len([t for t in bt_trades_executed if t['pnl'] < 0])
            win_rate = wins / count * 100
            total_pnl = sum([t['pnl'] for t in bt_trades_executed])
            gross_profit = sum([t['pnl'] for t in bt_trades_executed if t['pnl'] > 0])
            gross_loss = abs(sum([t['pnl'] for t in bt_trades_executed if t['pnl'] < 0]))
            pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            print(f"  Trades Filled: {count}")
            print(f"  Win Rate:      {win_rate:.1f}% ({wins} W / {losses} L)")
            print(f"  Profit Factor: {pf:.2f}")
            print(f"  Net P&L:       {total_pnl:+.2f} index points")
        else:
            print("  No trades met the ML threshold.")
            
    print("\n" + "=" * 70)
    print("MNQ Pipeline Run Completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
