#!/usr/bin/env python3
"""TIER 2 STRATEGY REFINEMENT — H1 Liquidity Sweeps + ML Meta-Labeling.

Implementing:
  1. H1 Liquidity Sweep (match Deep Sweep logic).
  2. 1m FVG Entry (Midpoint).
  3. Symmetric 5.0x SL and 5.0x TP.
  4. ML Meta-Labeling Filter integration.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Config ──────────────────────────────────────────────────────────────────── #
DATA_PATH              = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
START_DATE             = "2025-08-01"
END_DATE               = "2025-12-31"

ATR_THRESHOLD          = 0.5 
MAX_GAP_DOLLARS        = 60.0
MAX_HOLD_BARS          = 120 # 2 Hours
LIMIT_CANCEL_BARS      = 15
MNQ_CONTRACT_VALUE     = 2.0 
TRANSACTION_COST       = 1.80

MODEL_PATH = Path("models/xgboost/tier2_meta_labeling_model.pkl")

# ── Data Preparation ────────────────────────────────────────────────────────── #

def detect_swings(df, window=2):
    highs, lows = df['high'].values, df['low'].values
    n = len(df)
    sh, sl = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(window, n - window):
        if all(highs[i] > highs[i-window:i]) and all(highs[i] > highs[i+1:i+window+1]):
            sh[i] = highs[i]
        if all(lows[i] < lows[i-window:i]) and all(lows[i] < lows[i+1:i+window+1]):
            sl[i] = lows[i]
    return sh, sl

def get_mitigation_map(df, swing_prices, is_high=True):
    n = len(df)
    mit_map = np.full(n, -1, dtype=int)
    sw_idx = np.where(~np.isnan(swing_prices))[0]
    for idx in sw_idx:
        p = swing_prices[idx]
        later = df['high' if is_high else 'low'].values[idx+1:]
        hits = np.where(later > p if is_high else later < p)[0]
        mit_map[idx] = idx + 1 + hits[0] if len(hits) > 0 else n
    return mit_map

def get_et_hour(timestamps):
    ts = pd.to_datetime(timestamps)
    if ts.tz is None: ts = ts.tz_localize('UTC')
    else: ts = ts.tz_convert('UTC') 
    return ts.tz_convert('US/Eastern').hour.values

def get_day_of_week(timestamps):
    ts = pd.to_datetime(timestamps)
    if ts.tz is None: ts = ts.tz_localize('UTC')
    return ts.dayofweek.values

def prepare_data(df_1m: pd.DataFrame) -> pd.DataFrame:
    df_1h = df_1m.set_index('timestamp').resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()
    
    sh, sl = detect_swings(df_1h, window=2)
    mit_sh = get_mitigation_map(df_1h, sh, True)
    mit_sl = get_mitigation_map(df_1h, sl, False)
    
    br_sw, bl_sw = np.zeros(len(df_1h), dtype=bool), np.zeros(len(df_1h), dtype=bool)
    for i in range(len(df_1h)):
        m_sh = np.where(mit_sh[:i-2] == i)[0]
        if any(df_1h.loc[i, 'close'] < sh[idx] for idx in m_sh): br_sw[i] = True
        m_sl = np.where(mit_sl[:i-2] == i)[0]
        if any(df_1h.loc[i, 'close'] > sl[idx] for idx in m_sl): bl_sw[i] = True
            
    df_1h['br_sw_act'] = pd.Series(br_sw).rolling(6, min_periods=1).max().astype(bool)
    df_1h['bl_sw_act'] = pd.Series(bl_sw).rolling(6, min_periods=1).max().astype(bool)
    
    df_1h_s = df_1h[['timestamp', 'br_sw_act', 'bl_sw_act']].copy()
    df_1h_s['timestamp'] = df_1h_s['timestamp'] + pd.Timedelta(hours=1)

    df_1m = pd.merge_asof(df_1m.sort_values('timestamp'), df_1h_s, on='timestamp', direction='backward')
    df_1m[['br_sw_act', 'bl_sw_act']] = df_1m[['br_sw_act', 'bl_sw_act']].fillna(False)
    return df_1m

# ── ML Filter ───────────────────────────────────────────────────────────────── #

class MetaLabelingFilter:
    def __init__(self, model_path, threshold=0.55):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.model = None
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                print(f"Loaded ML model from {model_path}")
            except Exception as e:
                print(f"Error loading ML model: {e}")
        else:
            print(f"ML model not found at {model_path}")

    def predict_proba(self, features):
        if self.model is None:
            return 1.0 # Pass all if no model
        
        # Ensure features are in correct order for model
        feature_cols = ['atr', 'gap_size', 'volume_ratio', 'et_hour', 'day_of_week', 'signal_direction']
        df_feat = pd.DataFrame([features])[feature_cols]
        
        # XGBoost expect signal_direction to be encoded if it was categorical
        # Let's assume 1 for bullish, 0 for bearish
        df_feat['signal_direction'] = 1 if df_feat['signal_direction'].iloc[0] == "bullish" else 0
        
        proba = self.model.predict_proba(df_feat)[0, 1]
        return proba

# ── Simulation ─────────────────────────────────────────────────────────────── #

def simulate_trade(direction, entry, tp, sl, highs, lows, closes, start_idx, n):
    for j in range(1, MAX_HOLD_BARS + 1):
        idx = start_idx + j
        if idx >= n: break
        h, l = highs[idx], lows[idx]
        if direction == "bullish":
            if l <= sl: return {"pnl": (sl - entry) * MNQ_CONTRACT_VALUE - TRANSACTION_COST, "win": 0}
            if h >= tp: return {"pnl": (tp - entry) * MNQ_CONTRACT_VALUE - TRANSACTION_COST, "win": 1}
        else:
            if h >= sl: return {"pnl": (entry - sl) * MNQ_CONTRACT_VALUE - TRANSACTION_COST, "win": 0}
            if l <= tp: return {"pnl": (entry - tp) * MNQ_CONTRACT_VALUE - TRANSACTION_COST, "win": 1}
    ep = closes[min(start_idx + MAX_HOLD_BARS, n - 1)]
    pnl = ((ep - entry) if direction == "bullish" else (entry - ep)) * MNQ_CONTRACT_VALUE - TRANSACTION_COST
    return {"pnl": pnl, "win": 1 if pnl > 0 else 0}

def run_backtest(df, blocked_hours, sl_mult, tp_mult, entry_pct, ml_filter=None, export_path=None):
    n = len(df)
    highs, lows, opens, closes, volumes = df["high"].values, df["low"].values, df["open"].values, df["close"].values, df["volume"].values
    timestamps = df["timestamp"].values
    et_hours = get_et_hour(timestamps)
    days_of_week = get_day_of_week(timestamps)
    bl_sw, br_sw = df["bl_sw_act"].values, df["br_sw_act"].values
    
    prev_close = pd.Series(closes).shift(1).values
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = pd.Series(tr).rolling(20, min_periods=5).mean().values
    
    is_bull_bar = (closes > opens).astype(float)
    up_vol = pd.Series(volumes * is_bull_bar).rolling(20, min_periods=1).sum().values
    dn_vol = pd.Series(volumes * (1-is_bull_bar)).rolling(20, min_periods=1).sum().values

    trades = []
    ml_data = []
    next_bar = 0
    
    for i in range(2, n):
        if i < next_bar: continue
        if et_hours[i] in blocked_hours: continue
        
        c1_h, c1_l, c3_o, c3_l, c3_h = highs[i-2], lows[i-2], opens[i], lows[i], highs[i]
        
        for d in ("bullish", "bearish"):
            if d == "bullish":
                # FVG Confluence
                if not (bl_sw[i] and c1_h < c3_l and closes[i-1] > opens[i-1]): continue
                gap_top, gap_bot = c3_l, c1_h
                gap_size = gap_top - gap_bot
                ent = gap_top - gap_size * entry_pct
                tp, sl = ent + gap_size * tp_mult, ent - gap_size * sl_mult
            else:
                if not (br_sw[i] and c1_l > c3_h and closes[i-1] < opens[i-1]): continue
                gap_top, gap_bot = c1_l, c3_h
                gap_size = gap_top - gap_bot
                ent = gap_bot + gap_size * entry_pct
                tp, sl = ent - gap_size * tp_mult, ent + gap_size * sl_mult
            
            if gap_size <= 0 or gap_size < atr[i] * ATR_THRESHOLD or gap_size * MNQ_CONTRACT_VALUE > MAX_GAP_DOLLARS:
                continue
            
            uv, dv = up_vol[i], dn_vol[i]
            vol_ratio = (uv/dv if dv>0 else 99) if d == "bullish" else (dv/uv if uv>0 else 99)

            # Features for ML
            features = {
                "atr": atr[i],
                "gap_size": gap_size,
                "volume_ratio": vol_ratio,
                "et_hour": et_hours[i],
                "day_of_week": days_of_week[i],
                "signal_direction": d
            }
            
            # ML Filter
            if ml_filter:
                proba = ml_filter.predict_proba(features)
                if proba < ml_filter.threshold:
                    continue

            fill_idx = -1
            for k in range(1, LIMIT_CANCEL_BARS + 1):
                idx = i + k
                if idx >= n: break
                if (d == "bullish" and lows[idx] <= ent) or (d == "bearish" and highs[idx] >= ent):
                    fill_idx = idx; break
            
            if fill_idx != -1:
                res = simulate_trade(d, ent, tp, sl, highs, lows, closes, fill_idx, n)
                trades.append(res)
                
                # Collect ML label
                features["label"] = 1 if res["pnl"] > 0 else 0
                ml_data.append(features)
                
                next_bar = fill_idx + 5
                break
    
    if export_path and ml_data:
        export_df = pd.DataFrame(ml_data)
        # Encode signal_direction for CSV
        export_df['signal_direction'] = export_df['signal_direction'].map({"bullish": 1, "bearish": 0})
        export_df.to_csv(export_path, index=False)
        print(f"Exported {len(ml_data)} trade setups to {export_path}")

    if not trades: return {"wr": 0, "pf": 0, "pnl": 0, "total": 0}
    pnl = [t["pnl"] for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [abs(p) for p in pnl if p <= 0]
    return {
        "wr": len(wins)/len(trades)*100,
        "pf": sum(wins)/sum(losses) if losses else 99,
        "pnl": sum(pnl),
        "total": len(trades)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="Export trade metadata for ML training")
    parser.add_argument("--meta-labeling", action="store_true", help="Apply ML meta-labeling filter")
    parser.add_argument("--threshold", type=float, default=0.55, help="ML probability threshold")
    args = parser.parse_args()

    raw_df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df_p = prepare_data(raw_df)
    df = df_p[(df_p['timestamp'] >= START_DATE) & (df_p['timestamp'] <= END_DATE + " 23:59")].reset_index(drop=True)
    
    blocked = {0, 1, 6, 8, 16, 17, 22, 23}
    
    ml_filter = None
    if args.meta_labeling:
        ml_filter = MetaLabelingFilter(MODEL_PATH, threshold=args.threshold)
    
    export_path = "data/ml_training/tier2_meta_labeling.csv" if args.export else None
    
    # We use Symmetric 5.0x SL and 5.0x TP as instructed
    res = run_backtest(df, blocked, 5.0, 5.0, 0.5, ml_filter=ml_filter, export_path=export_path)
    
    print("\n" + "="*60 + "\n TIER 2 META-LABELING BACKTEST\n" + "="*60)
    print(f"ML Filter: {'ENABLED' if args.meta_labeling else 'DISABLED'}")
    if args.meta_labeling:
        print(f"Threshold: {args.threshold}")
    print(f"Total Trades: {res['total']}")
    print(f"Win Rate:     {res['wr']:.2f}%")
    print(f"Profit Factor: {res['pf']:.2f}")
    print(f"Total P&L:    ${res['pnl']:.2f}")
    print("="*60)
