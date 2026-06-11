import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("/root/Silver-Bullet-ML-BMAD/data/kraken/PF_XBTUSD_1min.csv")
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()
df = df[df.index.year >= 2026].copy()

# Base S27 Indicators
length = 20
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
df['ema'] = df['close'].ewm(span=length, adjust=False).mean()
df['kc_upper'] = df['ema'] + (1.5 * df['atr'])
df['kc_lower'] = df['ema'] - (1.5 * df['atr'])
df['squeeze_on'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
df['recent_squeeze'] = df['squeeze_on'].astype(int).rolling(window=5).max() > 0

df['upper_band'] = df['high'].rolling(length).max()
df['lower_band'] = df['low'].rolling(length).min()

# S27 Signals
df['s27_long'] = (df['close'] > df['upper_band'].shift(1)) & df['recent_squeeze']
df['s27_short'] = (df['close'] < df['lower_band'].shift(1)) & df['recent_squeeze']
df['s27_long'] = df['s27_long'] & (~df['s27_long'].shift(1).fillna(value=False).astype(bool))
df['s27_short'] = df['s27_short'] & (~df['s27_short'].shift(1).fillna(value=False).astype(bool))

# Path C: Soft FVG S26
# Approximate H1 sweep using rolling 360-min extreme
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

def run_sim(long_col, short_col, sl_type='fixed', exit_type='fixed'):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['atr'].values
    bb_ups = df['bb_upper'].values
    bb_dns = df['bb_lower'].values
    emas = df['ema'].values
    
    long_sigs = np.where(df[long_col])[0]
    short_sigs = np.where(df[short_col])[0]
    
    results = []
    max_hold = 60
    
    def process(sigs, dir_val):
        for i in sigs:
            if i+1 >= len(closes): continue
            entry = closes[i]
            atr = atrs[i]
            if pd.isna(atr) or atr == 0: continue
            
            is_long = dir_val == 1
            
            # SL Logic
            if sl_type == 'structural':
                sl = bb_dns[i] if is_long else bb_ups[i]
                max_risk = 3 * atr
                if is_long and (entry - sl) > max_risk: sl = entry - max_risk
                if not is_long and (sl - entry) > max_risk: sl = entry + max_risk
            else:
                sl = entry - (atr * 2.0) if is_long else entry + (atr * 2.0)
                
            risk = entry - sl if is_long else sl - entry
            if risk <= 0: continue
            
            # TP Logic
            tp = entry + (risk * 2.0) if is_long else entry - (risk * 2.0) # 2:1 RR
            
            pnl = 0
            win = 0
            mfe = 0
            
            for j in range(i+1, min(i+max_hold+1, len(closes))):
                curr_mfe = highs[j] - entry if is_long else entry - lows[j]
                mfe = max(mfe, curr_mfe)
                
                if exit_type == 'trailing':
                    # Trail stop once in profit
                    if is_long and closes[j] < emas[j] and curr_mfe > risk:
                        pnl = closes[j] - entry
                        win = 1 if pnl > 0 else 0
                        break
                    elif not is_long and closes[j] > emas[j] and curr_mfe > risk:
                        pnl = entry - closes[j]
                        win = 1 if pnl > 0 else 0
                        break
                        
                # Standard stops
                if is_long:
                    if lows[j] <= sl: pnl = -risk; break
                    elif exit_type == 'fixed' and highs[j] >= tp: pnl = tp - entry; win = 1; break
                else:
                    if highs[j] >= sl: pnl = -risk; break
                    elif exit_type == 'fixed' and lows[j] <= tp: pnl = entry - tp; win = 1; break
            else:
                exit_price = closes[min(i+max_hold, len(closes)-1)]
                pnl = (exit_price - entry) if is_long else (entry - exit_price)
                win = 1 if pnl > 0 else 0
                
            results.append({'pnl_r': pnl / risk, 'win': win, 'mfe_r': mfe / risk})
            
    process(long_sigs, 1)
    process(short_sigs, 0)
    
    df_res = pd.DataFrame(results)
    if len(df_res) == 0: return "0 Trades"
    
    wr = df_res['win'].mean() * 100
    gp = df_res[df_res['pnl_r'] > 0]['pnl_r'].sum()
    gl = abs(df_res[df_res['pnl_r'] < 0]['pnl_r'].sum())
    pf = gp / gl if gl > 0 else float('inf')
    net_r = df_res['pnl_r'].sum()
    avg_mfe = df_res['mfe_r'].mean()
    
    return f"Trades: {len(df_res):>4} | WR: {wr:>5.2f}% | PF: {pf:>4.2f} | Net R: {net_r:>7.2f}R | Avg Max Excursion: {avg_mfe:.2f}R"

print("--- S27 Base (Fixed 2.0 ATR SL, 4.0 ATR TP) ---")
print(run_sim('s27_long', 's27_short', 'fixed', 'fixed'))

print("\n--- S27 + Path B (Structural BB Stop Loss, 2:1 RR) ---")
print(run_sim('s27_long', 's27_short', 'structural', 'fixed'))

print("\n--- S27 + Path A+B (Structural Stop + Dynamic EMA Trailing Exit) ---")
print(run_sim('s27_long', 's27_short', 'structural', 'trailing'))

print("\n--- S26 + Path C (Soft FVG + H1 Sweep, Fixed Stops) ---")
print(run_sim('s26_long', 's26_short', 'fixed', 'fixed'))

print("\n--- S26 + Path C+B (Soft FVG + Structural Stops) ---")
print(run_sim('s26_long', 's26_short', 'structural', 'fixed'))
