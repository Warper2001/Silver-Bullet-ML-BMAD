import pandas as pd
import numpy as np
from pathlib import Path

# Load data
csv_path = Path("/root/Silver-Bullet-ML-BMAD/data/kraken/PF_XBTUSD_1min.csv")
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()

# Filter for 2026 data
df = df[df.index.year >= 2026].copy()

# Strategy Parameters
atr_period = 14
multiplier = 1.5
sl_mult = 2.0
tp_mult = 4.0
max_hold = 60 # 1 hour max hold

# Indicators
df['atr'] = df['high'].rolling(atr_period).max() - df['low'].rolling(atr_period).min()
df['upper_band'] = df['high'].rolling(atr_period).max()
df['lower_band'] = df['low'].rolling(atr_period).min()

# Signals: Breakout of the recent channel
df['long_entry'] = (df['close'] > df['upper_band'].shift(1))
df['short_entry'] = (df['close'] < df['lower_band'].shift(1))

# Prevent consecutive signals in the same direction (cooldown)
df['long_entry'] = df['long_entry'] & (~df['long_entry'].shift(1).fillna(False))
df['short_entry'] = df['short_entry'] & (~df['short_entry'].shift(1).fillna(False))

# Fast Vectorized-ish Backtest over signal indices
print("Running fast forward-scan backtest...")

closes = df['close'].values
highs = df['high'].values
lows = df['low'].values
atrs = df['atr'].values
long_sigs = np.where(df['long_entry'])[0]
short_sigs = np.where(df['short_entry'])[0]

results = []

for i in long_sigs:
    if i + 1 >= len(closes): continue
    entry = closes[i]
    atr = atrs[i]
    if pd.isna(atr) or atr == 0: continue
    
    sl = entry - (atr * sl_mult)
    tp = entry + (atr * tp_mult)
    
    # Scan forward
    pnl = 0
    win = 0
    for j in range(i + 1, min(i + max_hold + 1, len(closes))):
        if lows[j] <= sl:
            pnl = -sl_mult
            win = 0
            break
        elif highs[j] >= tp:
            pnl = tp_mult
            win = 1
            break
    else:
        # Time stop
        exit_price = closes[min(i + max_hold, len(closes)-1)]
        pnl = (exit_price - entry) / atr
        win = 1 if pnl > 0 else 0
        
    results.append({'dir': 'L', 'pnl_r': pnl, 'win': win})

for i in short_sigs:
    if i + 1 >= len(closes): continue
    entry = closes[i]
    atr = atrs[i]
    if pd.isna(atr) or atr == 0: continue
    
    sl = entry + (atr * sl_mult)
    tp = entry - (atr * tp_mult)
    
    # Scan forward
    pnl = 0
    win = 0
    for j in range(i + 1, min(i + max_hold + 1, len(closes))):
        if highs[j] >= sl:
            pnl = -sl_mult
            win = 0
            break
        elif lows[j] <= tp:
            pnl = tp_mult
            win = 1
            break
    else:
        # Time stop
        exit_price = closes[min(i + max_hold, len(closes)-1)]
        pnl = (entry - exit_price) / atr
        win = 1 if pnl > 0 else 0
        
    results.append({'dir': 'S', 'pnl_r': pnl, 'win': win})

res_df = pd.DataFrame(results)
if len(res_df) > 0:
    total_trades = len(res_df)
    win_rate = res_df['win'].mean() * 100
    gross_profit = res_df[res_df['pnl_r'] > 0]['pnl_r'].sum()
    gross_loss = abs(res_df[res_df['pnl_r'] < 0]['pnl_r'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    net_r = res_df['pnl_r'].sum()
    
    print(f"--- S27 Raw Strategy Performance (2026 YTD) ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Profit Fact:  {pf:.2f}")
    print(f"Net R:        {net_r:.2f}R")
else:
    print("No valid trades executed.")
