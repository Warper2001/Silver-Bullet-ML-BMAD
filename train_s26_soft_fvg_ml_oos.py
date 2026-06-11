"""
train_s26_soft_fvg_ml_oos.py
----------------------------
OOS diagnostic version of train_s26_soft_fvg_ml.py.

CHANGES vs. original:
  - Train window: pre-2026 only (Nov 2024 – Dec 2025) — 2026 is the holdout.
  - Saves to models/s26_soft_fvg_ml_model_oos.pkl (live model is untouched).
  - No internal 70/30 split needed — 2026 YTD is the real test set.

Everything else (indicators, features, model class/params) is identical to
the live training script so the feature contract matches backtest_s26_oos_ytd.py.
"""
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

# -----------------------------------------------------------------------
# KEY DIFFERENCE: train on pre-2026 data only (2026 is the OOS holdout)
# -----------------------------------------------------------------------
df = df[df.index.year < 2026].copy()

print(f"Training data: {df.index.min()} → {df.index.max()}  ({len(df):,} bars)")

# =====================================================================
# PATH C: SOFT FVG + SWEEP (S26)  — identical to live training script
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
rvol_period = 50
df['vol_sma'] = df['volume'].rolling(rvol_period).mean()
df['rvol'] = df['volume'] / df['vol_sma'].replace(0, np.nan)
df['rvol'] = df['rvol'].fillna(1.0)

macro_ema = 200
df['macro_ema'] = df['close'].ewm(span=macro_ema).mean()
df['dist_macro_ema'] = (df['close'] - df['macro_ema']) / df['atr']

df['ema'] = df['close'].ewm(span=length, adjust=False).mean()
df['dist_ema'] = (df['close'] - df['ema']) / df['atr']

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
# SIMULATE TRADES ON PRE-2026 DATA
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
print(f"Pre-2026 simulated trades for training: {len(trades_df)}")
print(f"  Win rate (raw, no ML): {trades_df['target'].mean()*100:.1f}%")

features = ['dir', 'atr', 'rvol', 'dist_ema', 'dist_macro_ema', 'hour_et', 'dow', 'is_us_session']

X_train = trades_df[features]
y_train = trades_df['target']

print(f"\nFitting HistGradientBoostingClassifier on {len(X_train)} trades...")
model = HistGradientBoostingClassifier(max_iter=150, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------------------------------------------
# Save to OOS-specific path — DOES NOT overwrite the live model
# -----------------------------------------------------------------------
model_path = Path("/root/Silver-Bullet-ML-BMAD/models/s26_soft_fvg_ml_model_oos.pkl")
model_path.parent.mkdir(exist_ok=True, parents=True)
joblib.dump(model, model_path)
print(f"\nOOS model saved → {model_path}")
print("Live model (s26_soft_fvg_ml_model.pkl) is UNCHANGED.")
