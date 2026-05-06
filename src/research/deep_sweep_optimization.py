#!/usr/bin/env python3
"""DEEP SWEEP OPTIMIZER (FINAL) — H1 Liquidity Sweep + 1m FVG Confluence.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools

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
    if ts.dt.tz is None: ts = ts.dt.tz_localize('UTC')
    else: ts = ts.dt.tz_convert('UTC') 
    return ts.dt.tz_convert('US/Eastern').dt.hour.values

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

# ── Simulation ─────────────────────────────────────────────────────────────── #

def simulate_trade(direction, entry, tp, sl, highs, lows, closes, start_idx, n):
    for j in range(1, MAX_HOLD_BARS + 1):
        idx = start_idx + j
        if idx >= n: break
        h, l = highs[idx], lows[idx]
        if direction == "bullish":
            if l <= sl: return {"pnl": (sl - entry) * 2.0 - 1.80}
            if h >= tp: return {"pnl": (tp - entry) * 2.0 - 1.80}
        else:
            if h >= sl: return {"pnl": (entry - sl) * 2.0 - 1.80}
            if l <= tp: return {"pnl": (entry - tp) * 2.0 - 1.80}
    ep = closes[min(start_idx + MAX_HOLD_BARS, n - 1)]
    pnl = ((ep - entry) if direction == "bullish" else (entry - ep)) * 2.0 - 1.80
    return {"pnl": pnl}

def run_backtest(df, blocked_hours, sl_mult, tp_mult, entry_pct):
    n = len(df)
    highs, lows, opens, closes = df["high"].values, df["low"].values, df["open"].values, df["close"].values
    et_hours = get_et_hour(df["timestamp"])
    bl_sw, br_sw = df["bl_sw_act"].values, df["br_sw_act"].values
    
    prev_close = pd.Series(closes).shift(1).values
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = pd.Series(tr).rolling(20, min_periods=5).mean().values
    
    trades = []
    next_bar = 0
    for i in range(2, n):
        if i < next_bar: continue
        if et_hours[i] in blocked_hours: continue
        
        c1_h, c1_l, c3_o, c3_l, c3_h = highs[i-2], lows[i-2], opens[i], lows[i], highs[i]
        
        for d in ("bullish", "bearish"):
            if d == "bullish":
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
            
            if gap_size <= 0 or gap_size < atr[i] * 0.5 or gap_size * 2.0 > MAX_GAP_DOLLARS:
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
                next_bar = fill_idx + 5
                break
    
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
    raw_df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df_p = prepare_data(raw_df)
    df = df_p[(df_p['timestamp'] >= START_DATE) & (df_p['timestamp'] <= END_DATE + " 23:59")].reset_index(drop=True)
    
    blocked = {0, 1, 6, 8, 16, 17, 22, 23}
    sl_m, tp_m, ent_p = [2.0, 3.5, 5.0], [5.0, 8.0, 12.0], [0.5] # Focused on Midpoint
    
    results = []
    print(f"Running final grid search on {len(sl_m)*len(tp_m)*len(ent_p)} combinations...")
    for sl, tp, ep in itertools.product(sl_m, tp_m, ent_p):
        res = run_backtest(df, blocked, sl, tp, ep)
        res.update({"sl": sl, "tp": tp, "ep": ep})
        results.append(res)
        
    print("\n" + "="*60 + "\n FINAL STRATEGY RANKING (Aug-Dec 2025)\n" + "="*60)
    res_df = pd.DataFrame(results).sort_values("pnl", ascending=False)
    print(res_df.head(10).to_string(index=False))
