import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

# Load MNQ data
csv_path = Path("/root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()

# Filter for 2026 data
df = df[df.index.year >= 2026].copy()

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
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']

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

# ML Features
df['volume_15m'] = df['volume'].rolling(15).sum()
df['dist_ema'] = (df['close'] - df['ema']) / df['atr']
df['hour'] = df.index.hour
df['dow'] = df.index.dayofweek

# Signals: Price breaks 20-period high/low AND we are in or just exiting a Volatility Squeeze
df['long_entry'] = (df['close'] > df['upper_band'].shift(1)) & df['recent_squeeze']
df['short_entry'] = (df['close'] < df['lower_band'].shift(1)) & df['recent_squeeze']

# Prevent consecutive signals in the same direction (cooldown)
df['long_entry'] = df['long_entry'] & (~df['long_entry'].shift(1).fillna(value=False).astype(bool))
df['short_entry'] = df['short_entry'] & (~df['short_entry'].shift(1).fillna(value=False).astype(bool))

# Fast Vectorized-ish Backtest over signal indices
closes = df['close'].values
highs = df['high'].values
lows = df['low'].values
atrs = df['atr'].values

# Feature arrays
vols = df['volume_15m'].values
dist_emas = df['dist_ema'].values
hours = df['hour'].values
dows = df['dow'].values
bb_widths = df['bb_width'].values

long_sigs = np.where(df['long_entry'])[0]
short_sigs = np.where(df['short_entry'])[0]

trades = []

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
            
        trades.append({
            'idx': i,
            'dir': direction,
            'atr': atr,
            'vol_15m': vols[i],
            'dist_ema': dist_emas[i],
            'hour': hours[i],
            'dow': dows[i],
            'bb_width': bb_widths[i],
            'pnl_r': pnl,
            'target': win
        })

simulate(long_sigs, 1)
simulate(short_sigs, 0)

trades_df = pd.DataFrame(trades).sort_values('idx').dropna()

# Train/Test Split (Chronological: first 70% train, last 30% test)
split_idx = int(len(trades_df) * 0.7)
train_df = trades_df.iloc[:split_idx]
test_df = trades_df.iloc[split_idx:]

features = ['dir', 'atr', 'vol_15m', 'dist_ema', 'hour', 'dow', 'bb_width']

X_train, y_train = train_df[features], train_df['target']
X_test, y_test = test_df[features], test_df['target']

print(f"Training on {len(X_train)} MNQ trades...")
model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
model.fit(X_train, y_train)

test_df = test_df.copy()
test_df['proba'] = model.predict_proba(X_test)[:, 1]

def calc_metrics(df_sub, name):
    if len(df_sub) == 0:
        print(f"[{name}] No trades.")
        return
    wins = df_sub['target'].sum()
    total = len(df_sub)
    wr = wins / total * 100
    gross_profit = df_sub[df_sub['pnl_r'] > 0]['pnl_r'].sum()
    gross_loss = abs(df_sub[df_sub['pnl_r'] < 0]['pnl_r'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    net_r = df_sub['pnl_r'].sum()
    print(f"[{name:<20}] Trades: {total:>5} | WR: {wr:>5.2f}% | PF: {pf:>4.2f} | Net R: {net_r:>7.2f}R")

print("\n--- Out-of-Sample (Test Set) Results ---")
calc_metrics(test_df, "Raw Strategy")

for thresh in [0.5, 0.52, 0.54, 0.56, 0.58, 0.6]:
    filtered = test_df[test_df['proba'] > thresh]
    calc_metrics(filtered, f"ML Filtered > {thresh}")

model_path = Path("/root/Silver-Bullet-ML-BMAD/models/mnq_vol_squeeze_ml_model.pkl")
model_path.parent.mkdir(exist_ok=True, parents=True)
joblib.dump(model, model_path)
print(f"\nMNQ Model successfully trained and saved to {model_path}")
