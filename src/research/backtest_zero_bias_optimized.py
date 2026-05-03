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

def find_dol_target(highs, lows, closes, i, direction, lookback=200):
    """Return the nearest unswept swing in the trade direction, or None."""
    if direction == "bullish":
        # nearest unswept swing HIGH above current close
        candidates = []
        for j in range(i - 1, max(0, i - lookback) - 1, -1):
            # local high: higher than neighbours
            if j > 0 and j < len(highs) - 1 and highs[j] > highs[j-1] and highs[j] > highs[j+1]:
                if highs[j] > closes[i]:  # above current price
                    # check it hasn't been swept (no close above it between j and i)
                    if not any(closes[j+1:i+1] > highs[j]):
                        candidates.append(highs[j])
        return min(candidates) if candidates else None
    else:
        candidates = []
        for j in range(i - 1, max(0, i - lookback) - 1, -1):
            if j > 0 and j < len(lows) - 1 and lows[j] < lows[j-1] and lows[j] < lows[j+1]:
                if lows[j] < closes[i]:
                    if not any(closes[j+1:i+1] < lows[j]):
                        candidates.append(lows[j])
        return max(candidates) if candidates else None

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
    ts = pd.to_datetime(timestamps, utc=True)
    return ts.tz_convert('US/Eastern').hour.values

def get_day_of_week(timestamps):
    ts = pd.to_datetime(timestamps, utc=True)
    return ts.tz_convert('US/Eastern').dayofweek.values

def prepare_data(df_1m: pd.DataFrame) -> pd.DataFrame:
    # Daily for ADR + prior-day midpoint (HTF bias gate)
    df_daily = df_1m.set_index('timestamp').resample('1D').agg({
        'high': 'max', 'low': 'min'
    }).dropna()
    df_daily['range'] = df_daily['high'] - df_daily['low']
    df_daily['adr_20'] = df_daily['range'].rolling(20, min_periods=5).mean()
    df_daily['prev_day_high'] = df_daily['high'].shift(1)
    df_daily['prev_day_low']  = df_daily['low'].shift(1)
    df_daily['prev_day_mid']  = (df_daily['prev_day_high'] + df_daily['prev_day_low']) / 2
    df_daily = df_daily.reset_index()[['timestamp', 'adr_20', 'prev_day_mid', 'prev_day_high', 'prev_day_low']]
    df_daily['timestamp'] = df_daily['timestamp'] + pd.Timedelta(days=1)

    df_1m = pd.merge_asof(df_1m.sort_values('timestamp'), df_daily, on='timestamp', direction='backward')

    # H4 bias: bullish if last close > prior H4 high (break of structure), bearish if < prior H4 low
    df_4h = df_1m.set_index('timestamp').resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()
    df_4h['h4_bias'] = 0
    for k in range(2, len(df_4h)):
        if df_4h['close'].iloc[k] > df_4h['high'].iloc[k-2]:
            df_4h.loc[k, 'h4_bias'] = 1   # bullish structure
        elif df_4h['close'].iloc[k] < df_4h['low'].iloc[k-2]:
            df_4h.loc[k, 'h4_bias'] = -1  # bearish structure
        else:
            df_4h.loc[k, 'h4_bias'] = df_4h['h4_bias'].iloc[k-1]  # carry forward
    df_4h_s = df_4h[['timestamp', 'h4_bias']].copy()
    df_4h_s['timestamp'] = df_4h_s['timestamp'] + pd.Timedelta(hours=4)
    df_1m = pd.merge_asof(df_1m.sort_values('timestamp'), df_4h_s, on='timestamp', direction='backward')
    df_1m['h4_bias'] = df_1m['h4_bias'].fillna(0)

    df_1h = df_1m.set_index('timestamp').resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()
    
    sh, sl = detect_swings(df_1h, window=2)
    mit_sh = get_mitigation_map(df_1h, sh, True)
    mit_sl = get_mitigation_map(df_1h, sl, False)
    
    br_sw, bl_sw = np.zeros(len(df_1h), dtype=bool), np.zeros(len(df_1h), dtype=bool)
    # Track the 1m bar index where the 1h bar ended
    # We can't easily do it here, but we can track the 1h bar index.
    last_br_idx, last_bl_idx = np.full(len(df_1h), -1), np.full(len(df_1h), -1)
    
    cur_br, cur_bl = -1, -1
    for i in range(3, len(df_1h)):  # start at 3 so mit_sh[:i-2] is always a positive slice
        m_sh = np.where(mit_sh[:i-2] == i)[0]
        if any(df_1h.loc[i, 'close'] < sh[idx] for idx in m_sh): 
            br_sw[i] = True
            cur_br = i
        m_sl = np.where(mit_sl[:i-2] == i)[0]
        if any(df_1h.loc[i, 'close'] > sl[idx] for idx in m_sl): 
            bl_sw[i] = True
            cur_bl = i
        last_br_idx[i] = cur_br
        last_bl_idx[i] = cur_bl
            
    df_1h['br_sw_act'] = pd.Series(br_sw).rolling(6, min_periods=1).max().astype(bool)
    df_1h['bl_sw_act'] = pd.Series(bl_sw).rolling(6, min_periods=1).max().astype(bool)
    df_1h['last_br_idx'] = last_br_idx
    df_1h['last_bl_idx'] = last_bl_idx
    
    # H1 ATR
    df_1h['tr'] = np.maximum(df_1h['high'] - df_1h['low'], 
                            np.maximum(np.abs(df_1h['high'] - df_1h['close'].shift(1)), 
                                      np.abs(df_1h['low'] - df_1h['close'].shift(1))))
    df_1h['h1_atr'] = df_1h['tr'].rolling(20, min_periods=5).mean()
    
    # Shift H1 data forward so it's only available AFTER the bar completes
    df_1h_s = df_1h[['timestamp', 'br_sw_act', 'bl_sw_act', 'h1_atr', 'last_br_idx', 'last_bl_idx']].copy()
    
    # We also need the last 6 H1 closes for slope. This is tricky to do in merge_asof nicely.
    # We'll attach the h1_idx and then fetch them in run_backtest or pre-calculate slope.
    df_1h['h1_slope'] = np.nan
    closes = df_1h['close'].values
    atrs = df_1h['h1_atr'].values
    for i in range(20, len(df_1h)):
        if atrs[i] > 0:
            slope = np.polyfit(range(6), closes[i-5:i+1], 1)[0]
            df_1h.loc[i, 'h1_slope'] = slope / atrs[i]
            
    df_1h_s = df_1h[['timestamp', 'br_sw_act', 'bl_sw_act', 'h1_atr', 'last_br_idx', 'last_bl_idx', 'h1_slope']].copy()
    df_1h_s['timestamp'] = df_1h_s['timestamp'] + pd.Timedelta(hours=1)

    df_1m = pd.merge_asof(df_1m.sort_values('timestamp'), df_1h_s, on='timestamp', direction='backward')
    df_1m[['br_sw_act', 'bl_sw_act']] = df_1m[['br_sw_act', 'bl_sw_act']].fillna(False)
    
    # Map H1 bar index to 1m bar index for fvg_to_sweep_bars
    # We'll do this in run_backtest by tracking when the H1 bar actually ends in 1m terms.
    
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

        feature_cols = [
            'fvg_fill_pct', 'sweep_window_vol', 'volume_ratio', 'signal_direction',
            'h1_trend_slope', 'atr', 'session_displacement', 'session_volume_ratio',
        ]

        df_feat = pd.DataFrame([features])[feature_cols]
        if isinstance(df_feat['signal_direction'].iloc[0], str):
            df_feat['signal_direction'] = 1 if df_feat['signal_direction'].iloc[0] == "bullish" else 0

        # Model is a dict {"base_model": XGBClassifier, "platt": LogisticRegression}
        raw = self.model["base_model"].predict_proba(df_feat)[0, 1]
        return float(self.model["platt"].predict_proba(np.array([[raw]]))[0, 1])

# ── Simulation ─────────────────────────────────────────────────────────────── #

def simulate_trade(direction, entry, tp, sl, highs, lows, closes, start_idx, n, cost):
    for j in range(1, MAX_HOLD_BARS + 1):
        idx = start_idx + j
        if idx >= n: break
        h, l = highs[idx], lows[idx]
        if direction == "bullish":
            if l <= sl: return {"pnl": (sl - entry) * MNQ_CONTRACT_VALUE - cost, "win": 0, "bars_held": j, "exit_type": "sl"}
            if h >= tp: return {"pnl": (tp - entry) * MNQ_CONTRACT_VALUE - cost, "win": 1, "bars_held": j, "exit_type": "tp"}
        else:
            if h >= sl: return {"pnl": (entry - sl) * MNQ_CONTRACT_VALUE - cost, "win": 0, "bars_held": j, "exit_type": "sl"}
            if l <= tp: return {"pnl": (entry - tp) * MNQ_CONTRACT_VALUE - cost, "win": 1, "bars_held": j, "exit_type": "tp"}
    ep = closes[min(start_idx + MAX_HOLD_BARS, n - 1)]
    pnl = ((ep - entry) if direction == "bullish" else (entry - ep)) * MNQ_CONTRACT_VALUE - cost
    return {"pnl": pnl, "win": 1 if pnl > 0 else 0, "bars_held": MAX_HOLD_BARS, "exit_type": "time"}

def _compute_context_features(i, df, atr_val, session_high, session_low, adr_20, last_sweep_bar, last_entry_bar, session_open_price, et_hour, session_volume_ratio, fvg_fill_pct, bar_body_ratio, silver_bullet_window, direction_sign):
    # session_displacement
    session_displacement = (df['close'].iloc[i] - session_open_price) / atr_val if not np.isnan(session_open_price) and atr_val > 0 else 0.0

    # adr_pct_used
    adr_pct_used = np.clip((session_high - session_low) / adr_20, 0, 2) if adr_20 > 0 else 0.5

    # fvg_to_sweep_bars
    fvg_to_sweep_bars = min(i - last_sweep_bar, 20) if last_sweep_bar >= 0 else 20

    # prior_setup_proximity
    prior_setup_proximity = min(i - last_entry_bar, 120) if last_entry_bar >= 0 else 120

    # h1_trend_slope is already in df as 'h1_slope'
    h1_trend_slope = df['h1_slope'].iloc[i] if 'h1_slope' in df.columns else 0.0
    if np.isnan(h1_trend_slope): h1_trend_slope = 0.0

    # sin_hour, cos_hour
    sin_hour = np.sin(2 * np.pi * et_hour / 24)
    cos_hour = np.cos(2 * np.pi * et_hour / 24)

    # Interaction Features
    sweep_window_vol = silver_bullet_window * session_volume_ratio
    slope_direction_match = 1 if np.sign(h1_trend_slope) == direction_sign else 0

    return {
        "session_displacement": session_displacement,
        "adr_pct_used": adr_pct_used,
        "fvg_to_sweep_bars": fvg_to_sweep_bars,
        "prior_setup_proximity": prior_setup_proximity,
        "h1_trend_slope": h1_trend_slope,
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "session_volume_ratio": session_volume_ratio,
        "fvg_fill_pct": fvg_fill_pct,
        "bar_body_ratio": bar_body_ratio,
        "sweep_window_vol": sweep_window_vol,
        "slope_direction_match": slope_direction_match
    }
def run_backtest(df, blocked_hours, sl_mult, tp_mult, entry_pct, ml_filter=None,
                 htf_bias_filter=False, premium_discount_filter=False, dol_gate_filter=False,
                 export_path=None, cost_override=None, return_trades=False):
    cost = cost_override if cost_override is not None else TRANSACTION_COST
    n = len(df)
    highs, lows, opens, closes, volumes = df["high"].values, df["low"].values, df["open"].values, df["close"].values, df["volume"].values
    timestamps = df["timestamp"].values
    et_hours = get_et_hour(timestamps)
    days_of_week = get_day_of_week(timestamps)
    bl_sw, br_sw = df["bl_sw_act"].values, df["br_sw_act"].values
    adr_20_vals = df["adr_20"].values if "adr_20" in df.columns else np.zeros(n)
    h4_bias_vals = df["h4_bias"].values if "h4_bias" in df.columns else np.zeros(n)
    prev_day_mid_vals = df["prev_day_mid"].values if "prev_day_mid" in df.columns else np.full(n, np.nan)

    last_br_h1_idx = df["last_br_idx"].values if "last_br_idx" in df.columns else np.full(n, -1)
    last_bl_h1_idx = df["last_bl_idx"].values if "last_bl_idx" in df.columns else np.full(n, -1)
    
    prev_close = pd.Series(closes).shift(1).values
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = pd.Series(tr).rolling(20, min_periods=5).mean().values
    
    is_bull_bar = (closes > opens).astype(float)
    up_vol = pd.Series(volumes * is_bull_bar).rolling(20, min_periods=1).sum().values
    dn_vol = pd.Series(volumes * (1-is_bull_bar)).rolling(20, min_periods=1).sum().values

    trades = []
    ml_data = []
    next_bar = 0
    
    # Context tracking
    session_high, session_low = -1.0, -1.0
    session_open_price = np.nan
    last_entry_bar = -1
    last_sweep_bar = -1
    # P4: separate per-direction sweep idx to avoid cross-direction contamination
    last_h1_bullish_sweep_idx = -1
    last_h1_bearish_sweep_idx = -1
    current_date = None

    for i in tqdm(range(2, n), desc="Simulating Trades"):
        if i < next_bar: continue

        ts_i = pd.to_datetime(timestamps[i], utc=True).tz_convert('US/Eastern')
        if current_date != ts_i.date():
            current_date = ts_i.date()
            session_high, session_low = highs[i], lows[i]
            session_open_price = np.nan  # P3: only set when hour >= 6 (match live trader)
        else:
            session_high = max(session_high, highs[i])
            session_low = min(session_low, lows[i])

        # P3: set session open from first bar at or after 6 AM ET (matches live trader logic)
        if np.isnan(session_open_price) and et_hours[i] >= 6:
            session_open_price = opens[i]

        if et_hours[i] in blocked_hours: continue
        
        c1_h, c1_l, c3_o, c3_l, c3_h = highs[i-2], lows[i-2], opens[i], lows[i], highs[i]
        
        for d in ("bullish", "bearish"):
            if d == "bullish":
                if not (bl_sw[i] and c1_h < c3_l and closes[i-1] > opens[i-1]): continue
                gap_top, gap_bot = c3_l, c1_h
                gap_size = gap_top - gap_bot
                ent = gap_top - gap_size * entry_pct
                tp, sl = ent + gap_size * tp_mult, ent - gap_size * sl_mult
                
                # Identify last sweep bar in 1m terms
                h1_sw_idx = last_bl_h1_idx[i]
            else:
                if not (br_sw[i] and c1_l > c3_h and closes[i-1] < opens[i-1]): continue
                gap_top, gap_bot = c1_l, c3_h
                gap_size = gap_top - gap_bot
                ent = gap_bot + gap_size * entry_pct
                tp, sl = ent - gap_size * tp_mult, ent + gap_size * sl_mult
                
                h1_sw_idx = last_br_h1_idx[i]
            
            if gap_size <= 0 or gap_size < atr[i] * ATR_THRESHOLD or gap_size * MNQ_CONTRACT_VALUE > MAX_GAP_DOLLARS:
                continue

            # HTF bias gate: H4 structure must align; price must be in correct half of prior day range
            if htf_bias_filter:
                h4 = h4_bias_vals[i]
                pdm = prev_day_mid_vals[i]
                if not np.isnan(pdm):
                    if d == "bullish" and not (h4 >= 0 and closes[i] < pdm):
                        continue
                    if d == "bearish" and not (h4 <= 0 and closes[i] > pdm):
                        continue

            # Premium/discount gate: FVG must form in discount (bullish) or premium (bearish) vs session range
            if premium_discount_filter:
                s_range = session_high - session_low
                if s_range > 0:
                    session_mid = session_low + s_range / 2
                    if d == "bullish" and gap_bot >= session_mid:
                        continue
                    if d == "bearish" and gap_top <= session_mid:
                        continue

            # DOL gate: nearest unswept opposing swing must be >= 2x SL distance away
            if dol_gate_filter:
                dol = find_dol_target(highs, lows, closes, i, d)
                if dol is None:
                    continue
                sl_dist = abs(ent - sl)
                dol_dist = abs(dol - ent)
                if dol_dist < 2 * sl_dist:
                    continue

            # P4: use per-direction sweep idx to avoid cross-direction contamination
            last_h1_sweep_idx_for_d = last_h1_bullish_sweep_idx if d == "bullish" else last_h1_bearish_sweep_idx
            if h1_sw_idx != last_h1_sweep_idx_for_d:
                if d == "bullish":
                    last_h1_bullish_sweep_idx = h1_sw_idx
                else:
                    last_h1_bearish_sweep_idx = h1_sw_idx
                for j in range(i, -1, -1):
                    if (d == "bullish" and last_bl_h1_idx[j] != h1_sw_idx) or \
                       (d == "bearish" and last_br_h1_idx[j] != h1_sw_idx):
                        last_sweep_bar = j + 1
                        break
            
            uv, dv = up_vol[i], dn_vol[i]
            vol_ratio = (uv/dv if dv>0 else 99) if d == "bullish" else (dv/uv if uv>0 else 99)

            # New features computation for ML
            et_h = et_hours[i]
            et_m = pd.to_datetime(timestamps[i], utc=True).tz_convert('US/Eastern').minute
            silver_bullet_window = 1 if (et_h == 3) or (et_h == 4 and et_m == 0) or (et_h == 9 and et_m >= 30) or (et_h == 10) else 0

            recent_vol_mean = np.mean(volumes[max(0, i-20):i]) if i > 0 else 0
            session_volume_ratio = volumes[i] / recent_vol_mean if recent_vol_mean > 0 else 1.0

            # P10/D3: use bar close relative to FVG (varies meaningfully; entry_est is constant 0.5)
            fvg_fill_pct = (closes[i] - gap_bot) / gap_size if gap_size > 0 else 0.5

            bar_range = highs[i] - lows[i]
            bar_body_ratio = abs(closes[i] - opens[i]) / bar_range if bar_range > 0 else 0.5

            direction_sign = 1 if d == "bullish" else -1

            # Context features
            ctx = _compute_context_features(
                i, df, atr[i], session_high, session_low, adr_20_vals[i], 
                last_sweep_bar, last_entry_bar, session_open_price, et_h, 
                session_volume_ratio, fvg_fill_pct, bar_body_ratio, silver_bullet_window, direction_sign
            )

            # Features for ML
            features = {
                "atr": atr[i],
                "gap_size": gap_size,
                "volume_ratio": vol_ratio,
                "et_hour": et_h,
                "day_of_week": days_of_week[i],
                "signal_direction": d,
                **ctx
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
                res = simulate_trade(d, ent, tp, sl, highs, lows, closes, fill_idx, n, cost)
                res['timestamp'] = timestamps[fill_idx]
                res['direction'] = d
                trades.append(res)
                
                # Collect ML label
                features["label"] = 1 if res["pnl"] > 0 else 0
                ml_data.append(features)
                
                last_entry_bar = fill_idx
                next_bar = fill_idx + res["bars_held"] + 1
                break

    
    if export_path and ml_data:
        export_df = pd.DataFrame(ml_data)
        # Encode signal_direction for CSV
        export_df['signal_direction'] = export_df['signal_direction'].map({"bullish": 1, "bearish": 0})
        export_df.to_csv(export_path, index=False)
        print(f"Exported {len(ml_data)} trade setups to {export_path}")

    stats = {"wr": 0, "pf": 0, "pnl": 0, "total": 0}
    if trades:
        pnl = [t["pnl"] for t in trades]
        wins = [p for p in pnl if p > 0]
        losses = [abs(p) for p in pnl if p <= 0]
        stats = {
            "wr": len(wins)/len(trades)*100,
            "pf": sum(wins)/sum(losses) if losses else 99,
            "pnl": sum(pnl),
            "total": len(trades)
        }
        
    if return_trades:
        return stats, trades
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="Export trade metadata for ML training")
    parser.add_argument("--meta-labeling", action="store_true", help="Apply ML meta-labeling filter")
    parser.add_argument("--threshold", type=float, default=0.55, help="ML probability threshold")
    parser.add_argument("--start", type=str, default=START_DATE, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=END_DATE, help="End date (YYYY-MM-DD)")
    parser.add_argument("--cost", type=float, default=TRANSACTION_COST, help="Round-trip transaction cost")
    parser.add_argument("--history", type=str, default=None, help="Export trade history to CSV")
    parser.add_argument("--htf-bias", action="store_true", help="Require H4 structural alignment + price in correct prior-day zone")
    parser.add_argument("--premium-discount", action="store_true", help="Require FVG in discount (long) or premium (short) vs session range")
    parser.add_argument("--dol-gate", action="store_true", help="Require unswept opposing swing >= 2x SL distance (draw on liquidity)")
    parser.add_argument("--no-tuesday", action="store_true", help="Skip all Tuesday bars (consistently PF<1.0 across backtest period)")
    args = parser.parse_args()

    raw_df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df_p = prepare_data(raw_df)
    df = df_p[(df_p['timestamp'] >= args.start) & (df_p['timestamp'] <= args.end + " 23:59")].reset_index(drop=True)

    if args.no_tuesday:
        ts_et = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('US/Eastern')
        df = df[ts_et.dt.dayofweek != 1].reset_index(drop=True)
    
    blocked = {0, 1, 6, 8, 16, 17, 22, 23}
    
    if args.export and args.meta_labeling:
        print("ERROR: --export and --meta-labeling cannot be used together. "
              "Export generates unfiltered training labels; applying the ML filter "
              "creates survivorship bias in the CSV. Run --export alone first.")
        sys.exit(1)

    ml_filter = None
    if args.meta_labeling:
        ml_filter = MetaLabelingFilter(MODEL_PATH, threshold=args.threshold)

    export_path = "data/ml_training/tier2_meta_labeling.csv" if args.export else None
    current_cost = args.cost
    
    res_dict, trades_list = run_backtest(
        df, blocked, 5.0, 5.0, 0.5,
        ml_filter=ml_filter,
        htf_bias_filter=args.htf_bias,
        premium_discount_filter=args.premium_discount,
        dol_gate_filter=args.dol_gate,
        export_path=export_path,
        cost_override=current_cost,
        return_trades=True,
    )

    if args.history:
        history_df = pd.DataFrame(trades_list)
        history_df.to_csv(args.history, index=False)
        print(f"Exported {len(trades_list)} trade details to {args.history}")

    print("\n" + "="*60 + "\n TIER 2 META-LABELING BACKTEST\n" + "="*60)
    print(f"ML Filter:          {'ENABLED' if args.meta_labeling else 'DISABLED'}")
    if args.meta_labeling:
        print(f"Threshold:          {args.threshold}")
    print(f"HTF Bias Gate:      {'ON' if args.htf_bias else 'off'}")
    print(f"Premium/Discount:   {'ON' if args.premium_discount else 'off'}")
    print(f"DOL Gate:           {'ON' if args.dol_gate else 'off'}")
    print(f"No-Tuesday:         {'ON' if args.no_tuesday else 'off'}")
    print(f"Total Trades:       {res_dict['total']}")
    print(f"Win Rate:           {res_dict['wr']:.2f}%")
    print(f"Profit Factor:      {res_dict['pf']:.2f}")
    print(f"Total P&L:          ${res_dict['pnl']:.2f}")
    print("="*60)
