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
# S27 SQUEEZE BASELINE
# =====================================================================
length = 20
bb_mult = 2.0     
kc_mult = 1.5     
sl_mult = 2.0     
tp_mult = 4.0     
max_hold = 60     

df['prev_close'] = df['close'].shift(1)
df['tr0'] = abs(df['high'] - df['low'])
df['tr1'] = abs(df['high'] - df['prev_close'])
df['tr2'] = abs(df['low'] - df['prev_close'])
df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
df['atr'] = df['tr'].rolling(length).mean()

df['sma'] = df['close'].rolling(length).mean()
df['std'] = df['close'].rolling(length).std()
df['bb_upper'] = df['sma'] + (bb_mult * df['std'])
df['bb_lower'] = df['sma'] - (bb_mult * df['std'])

df['ema'] = df['close'].ewm(span=length, adjust=False).mean()
df['kc_upper'] = df['ema'] + (kc_mult * df['atr'])
df['kc_lower'] = df['ema'] - (kc_mult * df['atr'])

df['squeeze_on'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
df['recent_squeeze'] = df['squeeze_on'].astype(int).rolling(window=5).max() > 0

df['upper_band'] = df['high'].rolling(length).max()
df['lower_band'] = df['low'].rolling(length).min()

df['long_entry'] = (df['close'] > df['upper_band'].shift(1)) & df['recent_squeeze']
df['short_entry'] = (df['close'] < df['lower_band'].shift(1)) & df['recent_squeeze']
df['long_entry'] = df['long_entry'] & (~df['long_entry'].shift(1).fillna(value=False).astype(bool))
df['short_entry'] = df['short_entry'] & (~df['short_entry'].shift(1).fillna(value=False).astype(bool))

# =====================================================================
# BMAD RESEARCH FEATURES (Validation & Enhancement Report)
# =====================================================================

# 1. Daily Bias (SMA 50 on Daily timeframe equivalent)
# We approximate daily bias by taking the SMA over 50 days (approx 50 * 1440 mins)
# But since this is a 1-minute chart, we'll use a 24-hour EMA as a proxy for Daily Bias slope
df['daily_ema'] = df['close'].ewm(span=1440, adjust=False).mean()
df['daily_bias_slope'] = (df['daily_ema'] - df['daily_ema'].shift(1440)) / df['atr']

# 2. Volatility Filter (ATR% requirement)
# Ratio of current 20-min ATR to 24-hour ATR
df['daily_atr'] = df['tr'].rolling(1440).mean()
df['atr_ratio'] = df['atr'] / df['daily_atr'].replace(0, np.nan)

# 3. Soft-FVG (Imbalance Proxy)
# Max gap between high of i-2 and low of i within the last 15 minutes
df['fvg_bull'] = (df['low'] - df['high'].shift(2)) / df['atr']
df['fvg_bear'] = (df['low'].shift(2) - df['high']) / df['atr']
# We take the maximum imbalance in the rolling 15m window leading up to the breakout
df['max_imbalance_bull'] = df['fvg_bull'].rolling(15).max()
df['max_imbalance_bear'] = df['fvg_bear'].rolling(15).max()

# 4. Nasdaq Specific Base Features
et_tz = pytz.timezone("America/New_York")
df_et = df.index.tz_convert(et_tz)
df['hour_et'] = df_et.hour
df['minute_et'] = df_et.minute
df['is_rth'] = (((df['hour_et'] == 9) & (df['minute_et'] >= 30)) | 
               ((df['hour_et'] >= 10) & (df['hour_et'] < 16)) | 
               ((df['hour_et'] == 16) & (df['minute_et'] == 0))).astype(int)

df['vol_sma'] = df['volume'].rolling(50).mean()
df['rvol'] = df['volume'] / df['vol_sma'].replace(0, np.nan)

# Clean up
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# =====================================================================
# FAST BACKTEST & ML EXTRACTION
# =====================================================================
closes = df['close'].values
highs = df['high'].values
lows = df['low'].values
atrs = df['atr'].values

# Feature Arrays
bias_slopes = df['daily_bias_slope'].values
atr_ratios = df['atr_ratio'].values
imb_bulls = df['max_imbalance_bull'].values
imb_bears = df['max_imbalance_bear'].values
rvols = df['rvol'].values
is_rths = df['is_rth'].values
hours_et = df['hour_et'].values

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
            exit_price = closes[min(i + max_hold, len(closes)-1)]
            pnl = (exit_price - entry) / atr if is_long else (entry - exit_price) / atr
            win = 1 if pnl > 0 else 0
            
        trades.append({
            'idx': i,
            'dir': direction,
            'atr': atr,
            'daily_bias': bias_slopes[i],
            'atr_ratio': atr_ratios[i],
            'imbalance': imb_bulls[i] if is_long else imb_bears[i],
            'rvol': rvols[i],
            'is_rth': is_rths[i],
            'hour_et': hours_et[i],
            'pnl_r': pnl,
            'target': win
        })

simulate(long_sigs, 1)
simulate(short_sigs, 0)

trades_df = pd.DataFrame(trades).sort_values('idx').dropna()

# Train/Test Split (70/30)
split_idx = int(len(trades_df) * 0.7)
train_df = trades_df.iloc[:split_idx]
test_df = trades_df.iloc[split_idx:]

features = ['dir', 'daily_bias', 'atr_ratio', 'imbalance', 'rvol', 'is_rth', 'hour_et']

X_train, y_train = train_df[features], train_df['target']
X_test, y_test = test_df[features], test_df['target']

print(f"Training on {len(X_train)} MNQ trades using BMAD Research Features...")
model = HistGradientBoostingClassifier(max_iter=150, random_state=42)
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

model_path = Path("/root/Silver-Bullet-ML-BMAD/models/mnq_bmad_v3_ml_model.pkl")
model_path.parent.mkdir(exist_ok=True, parents=True)
joblib.dump(model, model_path)
print(f"\nBMAD V3 Model successfully trained and saved to {model_path}")

# Feature Importance
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
print("\n--- Feature Importance (Out-of-Sample) ---")
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] > 0.001:
        print(f"{features[i]:<15}: {r.importances_mean[i]:.4f} +/- {r.importances_std[i]:.4f}")
    else:
        print(f"{features[i]:<15}: {r.importances_mean[i]:.4f} (noise)")