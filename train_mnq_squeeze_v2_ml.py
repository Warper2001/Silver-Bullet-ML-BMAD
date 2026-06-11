import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
import pytz

# Load MNQ data
csv_path = Path("/root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()

# Filter for 2026 data
df = df[df.index.year >= 2026].copy()

# =====================================================================
# STRATEGY & INDICATOR KNOBS (Hyperparameters)
# =====================================================================
# Squeeze Parameters
length = 20
bb_mult = 2.0     # Bollinger Band standard deviation multiplier
kc_mult = 1.5     # Keltner Channel ATR multiplier

# Trade Management
sl_mult = 2.0     # Stop Loss ATR multiplier
tp_mult = 4.0     # Take Profit ATR multiplier
max_hold = 60     # Time stop (minutes)

# Feature Engineering Parameters
rvol_period = 50  # Period for Relative Volume baseline
macro_ema = 200   # Macro trend filter period
# =====================================================================

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
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close'] # Width as % of price

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

# =====================================================================
# NEW: NASDAQ-SPECIFIC FEATURE ENGINEERING
# =====================================================================

# Timezone Conversion (Critical for MNQ)
et_tz = pytz.timezone("America/New_York")
df_et = df.index.tz_convert(et_tz)
df['hour_et'] = df_et.hour
df['minute_et'] = df_et.minute
df['dow'] = df_et.dayofweek

# RTH Flag (Regular Trading Hours: 9:30 AM - 4:00 PM ET)
df['is_rth'] = ((df['hour_et'] == 9) & (df['minute_et'] >= 30)) | \
               ((df['hour_et'] >= 10) & (df['hour_et'] < 16)) | \
               ((df['hour_et'] == 16) & (df['minute_et'] == 0))
df['is_rth'] = df['is_rth'].astype(int)

# Relative Volume (RVOL)
df['vol_sma'] = df['volume'].rolling(rvol_period).mean()
df['rvol'] = df['volume'] / df['vol_sma'].replace(0, np.nan) # Prevent div by zero
df['rvol'] = df['rvol'].fillna(1.0) # Default to normal volume if baseline is 0

# Macro Trend Alignment
df['macro_ema'] = df['close'].ewm(span=macro_ema).mean()
df['dist_macro_ema'] = (df['close'] - df['macro_ema']) / df['atr'] # Normalized distance

# Normalize standard EMA distance
df['dist_ema'] = (df['close'] - df['ema']) / df['atr']

# Clean up infinite values from division
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True) # Safe default for features

# =====================================================================

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

# Feature arrays (Extracted at moment of breakout)
rvols = df['rvol'].values
dist_emas = df['dist_ema'].values
dist_macro_emas = df['dist_macro_ema'].values
hours_et = df['hour_et'].values
dows = df['dow'].values
bb_widths = df['bb_width'].values
is_rths = df['is_rth'].values

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
            'rvol': rvols[i],
            'dist_ema': dist_emas[i],
            'dist_macro_ema': dist_macro_emas[i],
            'hour_et': hours_et[i],
            'dow': dows[i],
            'bb_width': bb_widths[i],
            'is_rth': is_rths[i],
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

features = ['dir', 'atr', 'rvol', 'dist_ema', 'dist_macro_ema', 'hour_et', 'dow', 'bb_width', 'is_rth']

X_train, y_train = train_df[features], train_df['target']
X_test, y_test = test_df[features], test_df['target']

print(f"Training on {len(X_train)} MNQ trades with Nasdaq-specific features...")
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

model_path = Path("/root/Silver-Bullet-ML-BMAD/models/mnq_vol_squeeze_v2_ml_model.pkl")
model_path.parent.mkdir(exist_ok=True, parents=True)
joblib.dump(model, model_path)
print(f"\nMNQ Model (v2) successfully trained and saved to {model_path}")

# Feature Importance Check
# HistGradientBoosting doesn't expose feature_importances_ natively, so we use permutation importance
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
print("\n--- Feature Importance (Out-of-Sample Permutation) ---")
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{features[i]:<15}: {r.importances_mean[i]:.3f} +/- {r.importances_std[i]:.3f}")
    else:
        print(f"{features[i]:<15}: {r.importances_mean[i]:.3f} (noise)")