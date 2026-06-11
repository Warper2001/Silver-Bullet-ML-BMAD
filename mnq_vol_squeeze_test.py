import pandas as pd
import numpy as np
from pathlib import Path

# Load MNQ data
csv_path = Path("/root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()

# Squeeze Parameters
length = 20
bb_mult = 2.0
kc_mult = 1.5

# Trade Management Parameters
sl_mult = 2.0
tp_mult = 4.0
max_hold = 60

# 1. True Range & ATR
df['prev_close'] = df['close'].shift(1)
df['tr0'] = abs(df['high'] - df['low'])
df['tr1'] = abs(df['high'] - df['prev_close'])
df['tr2'] = abs(df['low'] - df['prev_close'])
df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
df['atr'] = df['tr'].rolling(length).mean()

# 2. Bollinger Bands
df['sma'] = df['close'].rolling(length).mean()
df['std'] = df['close'].rolling(length).std()
df['bb_upper'] = df['sma'] + (bb_mult * df['std'])
df['bb_lower'] = df['sma'] - (bb_mult * df['std'])

# 3. Keltner Channels
df['ema'] = df['close'].ewm(span=length, adjust=False).mean()
df['kc_upper'] = df['ema'] + (kc_mult * df['atr'])
df['kc_lower'] = df['ema'] - (kc_mult * df['atr'])

# 4. Squeeze Condition (BB is entirely inside KC)
df['squeeze_on'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
df['recent_squeeze'] = df['squeeze_on'].astype(int).rolling(window=5).max() > 0

# 5. Momentum Breakout Channels
df['upper_band'] = df['high'].rolling(length).max()
df['lower_band'] = df['low'].rolling(length).min()

# Signals: Price breaks 20-period high/low AND we are in or just exiting a Volatility Squeeze
df['long_entry'] = (df['close'] > df['upper_band'].shift(1)) & df['recent_squeeze']
df['short_entry'] = (df['close'] < df['lower_band'].shift(1)) & df['recent_squeeze']

# Prevent consecutive signals in the same direction (cooldown)
df['long_entry'] = df['long_entry'] & (~df['long_entry'].shift(1).fillna(value=False).astype(bool))
df['short_entry'] = df['short_entry'] & (~df['short_entry'].shift(1).fillna(value=False).astype(bool))

# Fast Vectorized-ish Backtest over signal indices
print("Running MNQ Volatility Squeeze Breakout backtest...")

closes = df['close'].values
highs = df['high'].values
lows = df['low'].values
atrs = df['atr'].values
long_sigs = np.where(df['long_entry'])[0]
short_sigs = np.where(df['short_entry'])[0]

results = []

def simulate(sigs, direction):
    for i in sigs:
        if i + 1 >= len(closes): continue
        entry = closes[i]
        atr = atrs[i]
        if pd.isna(atr) or atr == 0: continue
        
        is_long = direction == 1
        sl = entry - (atr * sl_mult) if is_long else entry + (atr * sl_mult)
        tp = entry + (atr * tp_mult) if is_long else entry - (atr * tp_mult)
        
        pnl = 0
        win = 0
        for j in range(i + 1, min(i + max_hold + 1, len(closes))):
            if is_long:
                if lows[j] <= sl:
                    pnl = -sl_mult
                    break
                elif highs[j] >= tp:
                    pnl = tp_mult; win = 1
                    break
            else:
                if highs[j] >= sl:
                    pnl = -sl_mult
                    break
                elif lows[j] <= tp:
                    pnl = tp_mult; win = 1
                    break
        else:
            # Time stop
            exit_price = closes[min(i + max_hold, len(closes)-1)]
            pnl = (exit_price - entry) / atr if is_long else (entry - exit_price) / atr
            win = 1 if pnl > 0 else 0
            
        results.append({'dir': 'L' if is_long else 'S', 'pnl_r': pnl, 'win': win})

simulate(long_sigs, 1)
simulate(short_sigs, 0)

res_df = pd.DataFrame(results)
if len(res_df) > 0:
    total_trades = len(res_df)
    win_rate = res_df['win'].mean() * 100
    gross_profit = res_df[res_df['pnl_r'] > 0]['pnl_r'].sum()
    gross_loss = abs(res_df[res_df['pnl_r'] < 0]['pnl_r'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    net_r = res_df['pnl_r'].sum()
    
    print(f"--- MNQ Squeeze Breakout Raw Strategy Performance (2026 YTD) ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Profit Fact:  {pf:.2f}")
    print(f"Net R:        {net_r:.2f}R")
else:
    print("No valid trades executed.")