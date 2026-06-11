import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
import pytz

# Load Crypto data
csv_path = Path("/root/Silver-Bullet-ML-BMAD/data/kraken/PF_XBTUSD_1min.csv")
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()

# Filter for 2026 data
df = df[df.index.year >= 2026].copy()

# =====================================================================
# PATH C: SOFT FVG + SWEEP (S26 RESURRECTED)
# =====================================================================
length = 20
sl_mult = 2.0
tp_mult = 4.0
max_hold = 60

# ATR Calculation
df['prev_close'] = df['close'].shift(1)
df['tr0'] = abs(df['high'] - df['low'])
df['tr1'] = abs(df['high'] - df['prev_close'])
df['tr2'] = abs(df['low'] - df['prev_close'])
df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
df['atr'] = df['tr'].rolling(length).mean()

# Approximate H1 sweep using rolling 360-min extreme
df['h1_high'] = df['high'].rolling(360).max()
df['h1_low'] = df['low'].rolling(360).min()
df['sweep_bear'] = (df['high'] >= df['h1_high'].shift(1))
df['sweep_bull'] = (df['low'] <= df['h1_low'].shift(1))
df['recent_sweep_bear'] = df['sweep_bear'].astype(int).rolling(60).max() > 0
df['recent_sweep_bull'] = df['sweep_bull'].astype(int).rolling(60).max() > 0

# Soft FVG logic
df['soft_fvg_bear'] = (df['low'].shift(2) - df['high']) > (0.2 * df['atr'])
df['soft_fvg_bull'] = (df['low'] - df['high'].shift(2)) > (0.2 * df['atr'])

df['s26_short'] = df['recent_sweep_bear'] & df['soft_fvg_bear']
df['s26_short'] = df['s26_short'] & (~df['s26_short'].shift(1).fillna(value=False).infer_objects(copy=False).astype(bool))

df['s26_long'] = df['recent_sweep_bull'] & df['soft_fvg_bull']
df['s26_long'] = df['s26_long'] & (~df['s26_long'].shift(1).fillna(value=False).infer_objects(copy=False).astype(bool))

# =====================================================================
# ML FEATURE EXTRACTION
# =====================================================================
# RVOL
rvol_period = 50
df['vol_sma'] = df['volume'].rolling(rvol_period).mean()
df['rvol'] = df['volume'] / df['vol_sma'].replace(0, np.nan)
df['rvol'] = df['rvol'].fillna(1.0)

# Macro Trend
macro_ema = 200
df['macro_ema'] = df['close'].ewm(span=macro_ema).mean()
df['dist_macro_ema'] = (df['close'] - df['macro_ema']) / df['atr']

# EMA
df['ema'] = df['close'].ewm(span=length, adjust=False).mean()
df['dist_ema'] = (df['close'] - df['ema']) / df['atr']

# Time Context
et_tz = pytz.timezone("America/New_York")
df_et = df.index.tz_convert(et_tz)
df['hour_et'] = df_et.hour
df['dow'] = df_et.dayofweek
df['is_us_session'] = (((df['hour_et'] == 9) & (df_et.minute >= 30)) | 
                       ((df['hour_et'] >= 10) & (df['hour_et'] < 16)) | 
                       ((df['hour_et'] == 16) & (df_et.minute == 0))).astype(int)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# =====================================================================
# SIMULATE & TRAIN
# =====================================================================
closes = df['close'].values
highs = df['high'].values
lows = df['low'].values
atrs = df['atr'].values

rvols = df['rvol'].values
dist_emas = df['dist_ema'].values
dist_macro_emas = df['dist_macro_ema'].values
hours_et = df['hour_et'].values
dows = df['dow'].values
is_us_sessions = df['is_us_session'].values

long_sigs = np.where(df['s26_long'])[0]
short_sigs = np.where(df['s26_short'])[0]

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
                if lows[j] <= sl: pnl = -sl_mult; break
                elif highs[j] >= tp: pnl = tp_mult; win = 1; break
            else:
                if highs[j] >= sl: pnl = -sl_mult; break
                elif lows[j] <= tp: pnl = tp_mult; win = 1; break
        else:
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
            'is_us_session': is_us_sessions[i],
            'pnl_r': pnl,
            'target': win
        })

simulate(long_sigs, 1)
simulate(short_sigs, 0)

trades_df = pd.DataFrame(trades).sort_values('idx').dropna()
split_idx = int(len(trades_df) * 0.7)
train_df = trades_df.iloc[:split_idx]
test_df = trades_df.iloc[split_idx:]

features = ['dir', 'atr', 'rvol', 'dist_ema', 'dist_macro_ema', 'hour_et', 'dow', 'is_us_session']

X_train, y_train = train_df[features], train_df['target']
X_test, y_test = test_df[features], test_df['target']

print(f"Training on {len(X_train)} Soft-FVG S26 Crypto trades...")
model = HistGradientBoostingClassifier(max_iter=150, random_state=42)
model.fit(X_train, y_train)

test_df = test_df.copy()
test_df['proba'] = model.predict_proba(X_test)[:, 1]

def calc_metrics(df_sub, name):
    if len(df_sub) == 0: return
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

model_path = Path("/root/Silver-Bullet-ML-BMAD/models/s26_soft_fvg_ml_model.pkl")
model_path.parent.mkdir(exist_ok=True, parents=True)
joblib.dump(model, model_path)
print(f"\nModel successfully trained and saved to {model_path}")
