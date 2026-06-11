import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
import pytz

csv_path = Path("/root/Silver-Bullet-ML-BMAD/data/kraken/PF_XBTUSD_1min.csv")
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()
df = df[df.index.year >= 2026].copy()

# Base Params
length = 20
sl_mult = 2.0
tp_mult = 4.0
max_hold = 60
et_tz = pytz.timezone("America/New_York")
df_et = df.index.tz_convert(et_tz)

# S27 Indicators
df['prev_close'] = df['close'].shift(1)
df['tr0'] = abs(df['high'] - df['low'])
df['tr1'] = abs(df['high'] - df['prev_close'])
df['tr2'] = abs(df['low'] - df['prev_close'])
df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
df['atr'] = df['tr'].rolling(length).mean()
df['sma'] = df['close'].rolling(length).mean()
df['std'] = df['close'].rolling(length).std()
df['bb_upper'] = df['sma'] + (2.0 * df['std'])
df['bb_lower'] = df['sma'] - (2.0 * df['std'])
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
df['ema'] = df['close'].ewm(span=length, adjust=False).mean()
df['kc_upper'] = df['ema'] + (1.5 * df['atr'])
df['kc_lower'] = df['ema'] - (1.5 * df['atr'])
df['squeeze_on'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
df['recent_squeeze'] = df['squeeze_on'].astype(int).rolling(window=5).max() > 0
df['upper_band'] = df['high'].rolling(length).max()
df['lower_band'] = df['low'].rolling(length).min()
df['vol_sma'] = df['volume'].rolling(50).mean()
df['rvol'] = df['volume'] / df['vol_sma'].replace(0, np.nan)
df['rvol'] = df['rvol'].fillna(1.0)
df['macro_ema'] = df['close'].ewm(span=200).mean()
df['dist_macro_ema'] = (df['close'] - df['macro_ema']) / df['atr']
df['dist_ema'] = (df['close'] - df['ema']) / df['atr']
df['hour_et'] = df_et.hour
df['dow'] = df_et.dayofweek
df['is_us_session'] = (((df['hour_et'] == 9) & (df_et.minute >= 30)) | ((df['hour_et'] >= 10) & (df['hour_et'] < 16)) | ((df['hour_et'] == 16) & (df_et.minute == 0))).astype(int)

df['s27_long'] = (df['close'] > df['upper_band'].shift(1)) & df['recent_squeeze']
df['s27_short'] = (df['close'] < df['lower_band'].shift(1)) & df['recent_squeeze']
df['s27_long'] = df['s27_long'] & (~df['s27_long'].shift(1).fillna(value=False).astype(bool))
df['s27_short'] = df['s27_short'] & (~df['s27_short'].shift(1).fillna(value=False).astype(bool))

# S26 Indicators
df['h1_high'] = df['high'].rolling(360).max()
df['h1_low'] = df['low'].rolling(360).min()
df['sweep_bear'] = (df['high'] >= df['h1_high'].shift(1))
df['sweep_bull'] = (df['low'] <= df['h1_low'].shift(1))
df['recent_sweep_bear'] = df['sweep_bear'].astype(int).rolling(60).max() > 0
df['recent_sweep_bull'] = df['sweep_bull'].astype(int).rolling(60).max() > 0
df['soft_fvg_bear'] = (df['low'].shift(2) - df['high']) > (0.2 * df['atr'])
df['soft_fvg_bull'] = (df['low'] - df['high'].shift(2)) > (0.2 * df['atr'])

df['s26_short'] = df['recent_sweep_bear'] & df['soft_fvg_bear']
df['s26_short'] = df['s26_short'] & (~df['s26_short'].shift(1).fillna(value=False).astype(bool))
df['s26_long'] = df['recent_sweep_bull'] & df['soft_fvg_bull']
df['s26_long'] = df['s26_long'] & (~df['s26_long'].shift(1).fillna(value=False).astype(bool))

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

closes = df['close'].values
highs = df['high'].values
lows = df['low'].values
atrs = df['atr'].values

def simulate_strategy(long_col, short_col, model_path, features, threshold=0.56):
    model = joblib.load(model_path)
    
    long_sigs = np.where(df[long_col])[0]
    short_sigs = np.where(df[short_col])[0]
    
    f_dict = {
        'atr': atrs, 'rvol': df['rvol'].values, 'dist_ema': df['dist_ema'].values,
        'dist_macro_ema': df['dist_macro_ema'].values, 'hour_et': df['hour_et'].values,
        'dow': df['dow'].values, 'bb_width': df['bb_width'].values, 'is_us_session': df['is_us_session'].values
    }
    
    trades = []
    def sim(sigs, dir_val):
        for i in sigs:
            if i+1 >= len(closes): continue
            entry = closes[i]
            atr = atrs[i]
            if pd.isna(atr) or atr == 0: continue
            
            feat_row = {'dir': dir_val}
            for f in features:
                if f != 'dir': feat_row[f] = f_dict[f][i]
                
            proba = model.predict_proba(pd.DataFrame([feat_row])[features])[0, 1]
            if proba < threshold: continue
            
            is_long = dir_val == 1
            sl = entry - (atr * sl_mult) if is_long else entry + (atr * sl_mult)
            tp = entry + (atr * tp_mult) if is_long else entry - (atr * tp_mult)
            
            pnl = 0
            for j in range(i+1, min(i+max_hold+1, len(closes))):
                if is_long:
                    if lows[j] <= sl: pnl = -sl_mult; break
                    elif highs[j] >= tp: pnl = tp_mult; break
                else:
                    if highs[j] >= sl: pnl = -sl_mult; break
                    elif lows[j] <= tp: pnl = tp_mult; break
            else:
                exit_price = closes[min(i+max_hold, len(closes)-1)]
                pnl = (exit_price - entry) / atr if is_long else (entry - exit_price) / atr
                
            trades.append({'idx': i, 'pnl_r': pnl})
            
    sim(long_sigs, 1)
    sim(short_sigs, 0)
    
    if not trades: return None
    t_df = pd.DataFrame(trades).sort_values('idx')
    
    # Calculate Max Drawdown from Cumulative PnL
    t_df['cum_pnl'] = t_df['pnl_r'].cumsum()
    t_df['peak'] = t_df['cum_pnl'].cummax()
    t_df['drawdown'] = t_df['cum_pnl'] - t_df['peak']
    max_dd = t_df['drawdown'].min()
    
    # Isolate Out of Sample (last 30%)
    split_idx = int(len(t_df) * 0.7)
    oos_df = t_df.iloc[split_idx:].copy()
    oos_df['cum_pnl'] = oos_df['pnl_r'].cumsum()
    oos_df['peak'] = oos_df['cum_pnl'].cummax()
    oos_df['drawdown'] = oos_df['cum_pnl'] - oos_df['peak']
    oos_max_dd = oos_df['drawdown'].min()
    
    gp = oos_df[oos_df['pnl_r'] > 0]['pnl_r'].sum()
    gl = abs(oos_df[oos_df['pnl_r'] < 0]['pnl_r'].sum())
    pf = gp / gl if gl > 0 else float('inf')
    
    # Return OOS Metrics
    return {
        'trades': len(oos_df),
        'pf': pf,
        'net_r': oos_df['pnl_r'].sum(),
        'max_dd_r': oos_max_dd
    }

s27_feats = ['dir', 'atr', 'vol_15m', 'dist_ema', 'hour_et', 'dow', 'bb_width'] 
s26_feats = ['dir', 'atr', 'rvol', 'dist_ema', 'dist_macro_ema', 'hour_et', 'dow', 'is_us_session']

# Quick patch for S27 features that differ slightly
df['volume_15m'] = df['volume'].rolling(15).sum()
f_dict_s27 = {
    'atr': atrs, 'vol_15m': df['volume_15m'].values, 'dist_ema': df['dist_ema'].values,
    'hour': df['hour_et'].values, 'dow': df['dow'].values, 'bb_width': df['bb_width'].values
}

def sim_s27(threshold=0.56):
    model = joblib.load("/root/Silver-Bullet-ML-BMAD/models/vol_squeeze_ml_model.pkl")
    # S27 was trained with feature name 'hour' not 'hour_et'
    s27_feats_model = ['dir', 'atr', 'vol_15m', 'dist_ema', 'hour', 'dow', 'bb_width']
    long_sigs = np.where(df['s27_long'])[0]
    short_sigs = np.where(df['s27_short'])[0]
    trades = []
    def s(sigs, dir_val):
        for i in sigs:
            if i+1 >= len(closes): continue
            entry = closes[i]
            atr = atrs[i]
            if pd.isna(atr) or atr == 0: continue
            
            feat_row = {'dir': dir_val}
            for f in s27_feats_model:
                if f != 'dir': feat_row[f] = f_dict_s27[f][i]
                
            proba = model.predict_proba(pd.DataFrame([feat_row])[s27_feats_model])[0, 1]
            if proba < threshold: continue
            
            is_long = dir_val == 1
            sl = entry - (atr * sl_mult) if is_long else entry + (atr * sl_mult)
            tp = entry + (atr * tp_mult) if is_long else entry - (atr * tp_mult)
            
            pnl = 0
            for j in range(i+1, min(i+max_hold+1, len(closes))):
                if is_long:
                    if lows[j] <= sl: pnl = -sl_mult; break
                    elif highs[j] >= tp: pnl = tp_mult; break
                else:
                    if highs[j] >= sl: pnl = -sl_mult; break
                    elif lows[j] <= tp: pnl = tp_mult; break
            else:
                exit_price = closes[min(i+max_hold, len(closes)-1)]
                pnl = (exit_price - entry) / atr if is_long else (entry - exit_price) / atr
                
            trades.append({'idx': i, 'pnl_r': pnl})
    s(long_sigs, 1)
    s(short_sigs, 0)
    t_df = pd.DataFrame(trades).sort_values('idx')
    split_idx = int(len(t_df) * 0.7)
    oos_df = t_df.iloc[split_idx:].copy()
    oos_df['cum_pnl'] = oos_df['pnl_r'].cumsum()
    oos_df['peak'] = oos_df['cum_pnl'].cummax()
    oos_max_dd = oos_df['cum_pnl'] - oos_df['peak']
    gp = oos_df[oos_df['pnl_r'] > 0]['pnl_r'].sum()
    gl = abs(oos_df[oos_df['pnl_r'] < 0]['pnl_r'].sum())
    return len(oos_df), gp/gl if gl>0 else 0, oos_df['pnl_r'].sum(), oos_max_dd.min()

print("Comparing Out-of-Sample Performance (ML > 0.56)")
s27_res = sim_s27(0.56)
s26_res = simulate_strategy('s26_long', 's26_short', '/root/Silver-Bullet-ML-BMAD/models/s26_soft_fvg_ml_model.pkl', s26_feats, 0.56)

print(f"\nS27 (Volatility Squeeze):")
print(f"  PF:     {s27_res[1]:.2f}")
print(f"  Max DD: {s27_res[3]:.2f} R")

print(f"\nS26 (Soft FVG + Sweep):")
print(f"  PF:     {s26_res['pf']:.2f}")
print(f"  Max DD: {s26_res['max_dd_r']:.2f} R")
