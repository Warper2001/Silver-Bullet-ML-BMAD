#!/usr/bin/env python3
"""TIER 2 STRATEGY REFINEMENT — H4 Trend Alignment & Liquidity Sweeps.

Implementing stateful multi-timeframe confluence:
  1. H4 EMA Alignment (9 > 21 > 50).
  2. H1 Liquidity Sweep (pierce-and-reject).
  3. 1m FVG Entry with Zero-Bias realistic constraints.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Config ──────────────────────────────────────────────────────────────────── #
DATA_PATH              = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
START_DATE             = "2025-08-01"
END_DATE               = "2025-12-31"

ATR_THRESHOLD          = 0.7
VOLUME_RATIO_THRESHOLD = 2.25
MAX_GAP_DOLLARS        = 50.0
MAX_HOLD_BARS          = 20
LIMIT_CANCEL_BARS      = 10
MNQ_CONTRACT_VALUE     = 2.0  # MNQ is $2 per point
TRANSACTION_COST       = 1.80  # $0.80 commission + 2 ticks slippage ($1.00)

# ── Data Preparation ────────────────────────────────────────────────────────── #

def detect_swings(df, window=2):
    """Detect fractal swing highs and lows."""
    if len(df) < window * 2 + 1:
        return np.full(len(df), np.nan), np.full(len(df), np.nan)
        
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)
    
    # Vectorized check for rolling max/min
    for i in range(window, n - window):
        is_high = True
        for j in range(1, window + 1):
            if highs[i] <= highs[i-j] or highs[i] <= highs[i+j]:
                is_high = False; break
        if is_high: swing_highs[i] = highs[i]
        
        is_low = True
        for j in range(1, window + 1):
            if lows[i] >= lows[i-j] or lows[i] >= lows[i+j]:
                is_low = False; break
        if is_low: swing_lows[i] = lows[i]
        
    return swing_highs, swing_lows

def get_mitigation_map(df, swing_prices, is_high=True):
    """Pre-compute the index where each swing point is first crossed by price."""
    n = len(df)
    if n == 0: return np.array([], dtype=int)
    
    prices = df['high'].values if is_high else df['low'].values
    mitigation_indices = np.full(n, -1, dtype=int)
    swing_indices = np.where(~np.isnan(swing_prices))[0]
    
    for idx in swing_indices:
        price = swing_prices[idx]
        # Start looking for mitigation AFTER the swing is confirmed (idx + window)
        # However, for the 'touch' we start from idx+1 to find the death point.
        # We handle confirmation logic in the sweep detector.
        if is_high:
            later_prices = df['high'].values[idx+1:]
            mit_rel = np.where(later_prices > price)[0]
        else:
            later_prices = df['low'].values[idx+1:]
            mit_rel = np.where(later_prices < price)[0]
            
        if len(mit_rel) > 0:
            mitigation_indices[idx] = idx + 1 + mit_rel[0]
        else:
            mitigation_indices[idx] = n # Never mitigated
            
    return mitigation_indices

def get_et_hour(timestamps):
    """Convert UTC timestamps to ET hours correctly handling DST."""
    ts = pd.to_datetime(timestamps)
    if ts.tz is None:
        ts = ts.tz_localize('UTC')
    et_times = ts.tz_convert('US/Eastern')
    return et_times.hour.values

def prepare_tier2_data(df_1m: pd.DataFrame) -> pd.DataFrame:
    if df_1m.empty:
        return df_1m
        
    # H4 EMA Alignment
    df_4h = df_1m.set_index('timestamp').resample('4h').agg({
        'close': 'last'
    }).dropna()
    
    if len(df_4h) >= 50:
        df_4h['ema9'] = df_4h['close'].ewm(span=9, adjust=False).mean()
        df_4h['ema21'] = df_4h['close'].ewm(span=21, adjust=False).mean()
        df_4h['ema50'] = df_4h['close'].ewm(span=50, adjust=False).mean()
        df_4h['h4_bullish_align'] = (df_4h['ema9'] > df_4h['ema21']) & (df_4h['ema21'] > df_4h['ema50'])
        df_4h['h4_bearish_align'] = (df_4h['ema9'] < df_4h['ema21']) & (df_4h['ema21'] < df_4h['ema50'])
    else:
        df_4h['h4_bullish_align'] = False
        df_4h['h4_bearish_align'] = False
        
    df_4h_shifted = df_4h[['h4_bullish_align', 'h4_bearish_align']].shift(1)

    # H1 Liquidity Sweeps
    df_1h = df_1m.set_index('timestamp').resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()
    
    h1_window = 2
    sh, sl = detect_swings(df_1h, window=h1_window)
    mit_sh = get_mitigation_map(df_1h, sh, is_high=True)
    mit_sl = get_mitigation_map(df_1h, sl, is_high=False)
    
    bearish_sweeps = np.zeros(len(df_1h), dtype=bool)
    bullish_sweeps = np.zeros(len(df_1h), dtype=bool)
    
    for i in range(len(df_1h)):
        # A sweep at bar i is valid if it mitigates an ALREADY CONFIRMED swing point
        # A swing at sh_idx is confirmed at sh_idx + h1_window
        lookback = 24 # Look back 24 hours for unmitigated levels
        start_search = max(0, i - lookback)
        
        # Bearish Sweep: price mitigates a SH and closes below it
        # The SH must have been confirmed before this bar started
        mitigated_sh_here = np.where(mit_sh[start_search:i-h1_window] == i)[0] + start_search
        for sh_idx in mitigated_sh_here:
            if df_1h.loc[i, 'close'] < sh[sh_idx]:
                bearish_sweeps[i] = True
        
        # Bullish Sweep: price mitigates a SL and closes above it
        mitigated_sl_here = np.where(mit_sl[start_search:i-h1_window] == i)[0] + start_search
        for sl_idx in mitigated_sl_here:
            if df_1h.loc[i, 'close'] > sl[sl_idx]:
                bullish_sweeps[i] = True
            
    df_1h['h1_bearish_sweep'] = bearish_sweeps
    df_1h['h1_bullish_sweep'] = bullish_sweeps
    
    # A sweep signal stays active for a few hours
    df_1h['h1_bearish_sweep_active'] = df_1h['h1_bearish_sweep'].rolling(3, min_periods=1).max().astype(bool)
    df_1h['h1_bullish_sweep_active'] = df_1h['h1_bullish_sweep'].rolling(3, min_periods=1).max().astype(bool)
    
    df_1h_shifted = df_1h[['timestamp', 'h1_bearish_sweep_active', 'h1_bullish_sweep_active']].copy()
    df_1h_shifted['timestamp'] = df_1h_shifted['timestamp'] + pd.Timedelta(hours=1)

    # Merge
    df_1m = df_1m.sort_values('timestamp')
    df_1m = pd.merge_asof(df_1m, df_4h_shifted, left_on='timestamp', right_index=True, direction='backward')
    df_1m = pd.merge_asof(df_1m, df_1h_shifted, on='timestamp', direction='backward')
    
    # Fill NaNs
    cols = ['h4_bullish_align', 'h4_bearish_align', 'h1_bullish_sweep_active', 'h1_bearish_sweep_active']
    df_1m[cols] = df_1m[cols].fillna(False)
        
    return df_1m

# ── Exit simulation ───────────────────────────────────────────────────────────── #

def simulate_trade(direction, entry, tp, sl, highs, lows, closes, start_idx, n, max_hold):
    for j in range(1, max_hold + 1):
        idx = start_idx + j
        if idx >= n: break
        h, l = highs[idx], lows[idx]
        if direction == "bullish":
            if l <= sl:
                # Use SL price, or low of bar if it gaped past
                return {"exit": "sl", "bars": j, "pnl": (min(sl, l) - entry) * MNQ_CONTRACT_VALUE - TRANSACTION_COST}
            if h >= tp:
                return {"exit": "tp", "bars": j, "pnl": (tp - entry) * MNQ_CONTRACT_VALUE - TRANSACTION_COST}
        else:
            if h >= sl:
                return {"exit": "sl", "bars": j, "pnl": (entry - max(sl, h)) * MNQ_CONTRACT_VALUE - TRANSACTION_COST}
            if l <= tp:
                return {"exit": "tp", "bars": j, "pnl": (entry - tp) * MNQ_CONTRACT_VALUE - TRANSACTION_COST}
    ep = closes[min(start_idx + max_hold, n - 1)]
    pnl = ((ep - entry) if direction == "bullish" else (entry - ep)) * MNQ_CONTRACT_VALUE - TRANSACTION_COST
    return {"exit": "time", "bars": max_hold, "pnl": pnl}

# ── Backtest Engine ───────────────────────────────────────────────────────────── #

def run_backtest(df, blocked_hours, sl_mult, tp_mult, entry_pct, use_tier2=True):
    n = len(df)
    if n < 3: return {"total": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0}
    
    highs, lows, opens, closes = df["high"].values, df["low"].values, df["open"].values, df["close"].values
    timestamps = df["timestamp"].values
    et_hours = get_et_hour(timestamps)
    
    # Tier 2 Flags
    h4_bull = df["h4_bullish_align"].values
    h4_bear = df["h4_bearish_align"].values
    h1_bull_sweep = df["h1_bullish_sweep_active"].values
    h1_bear_sweep = df["h1_bearish_sweep_active"].values
    
    # Precompute filters
    prev_close = pd.Series(closes).shift(1).values
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = pd.Series(tr).rolling(20, min_periods=5).mean().values
    is_bull = (closes > opens).astype(float)
    up_vol = pd.Series(df["volume"].values * is_bull).rolling(20, min_periods=1).sum().values
    dn_vol = pd.Series(df["volume"].values * (1-is_bull)).rolling(20, min_periods=1).sum().values
    
    trades = []
    next_entry_bar = 0
    
    for i in range(2, n):
        if i < next_entry_bar: continue
        if et_hours[i] in blocked_hours: continue
        
        c1_close, c1_high, c1_low = closes[i-2], highs[i-2], lows[i-2]
        c3_open, c3_low, c3_high = opens[i], lows[i], highs[i]
        
        for direction in ("bullish", "bearish"):
            # Tier 2 Confluence (Required by Spec)
            if use_tier2:
                if direction == "bullish":
                    if not (h4_bull[i] and h1_bull_sweep[i]): continue
                else:
                    if not (h4_bear[i] and h1_bear_sweep[i]): continue

            if direction == "bullish":
                if not (c1_close > c3_open): continue
                gap_top, gap_bottom = c1_high, c3_low
                entry_level = gap_bottom + (gap_top - gap_bottom) * entry_pct
                tp_level = entry_level + (gap_top - gap_bottom) * tp_mult
                sl_level = entry_level - (gap_top - gap_bottom) * sl_mult
            else:
                if not (c1_close < c3_open): continue
                gap_top, gap_bottom = c3_high, c1_low
                entry_level = gap_top - (gap_top - gap_bottom) * entry_pct
                tp_level = entry_level - (gap_top - gap_bottom) * tp_mult
                sl_level = entry_level + (gap_top - gap_bottom) * sl_mult
            
            if gap_top <= gap_bottom: continue
            if (gap_top - gap_bottom) < atr[i] * ATR_THRESHOLD: continue
            if (gap_top - gap_bottom) * MNQ_CONTRACT_VALUE > MAX_GAP_DOLLARS: continue
            
            # Volume Filter
            uv, dv = up_vol[i], dn_vol[i]
            ratio = (uv/dv if dv>0 else 99) if direction == "bullish" else (dv/uv if uv>0 else 99)
            if ratio < VOLUME_RATIO_THRESHOLD: continue
            
            # Limit Entry: Search from i+1 onwards (No same-bar fill)
            fill_idx = -1
            for k in range(1, LIMIT_CANCEL_BARS + 1):
                idx = i + k
                if idx >= n: break
                if direction == "bullish" and lows[idx] <= entry_level:
                    fill_idx = idx; break
                if direction == "bearish" and highs[idx] >= entry_level:
                    fill_idx = idx; break
            
            if fill_idx != -1:
                res = simulate_trade(direction, entry_level, tp_level, sl_level, highs, lows, closes, fill_idx, n, MAX_HOLD_BARS)
                trades.append(res)
                next_entry_bar = fill_idx + res["bars"] + 1
                break
    
    if not trades: return {"total": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0}
    
    wins = [t for t in trades if t["pnl"] > 0]
    total_pnl = sum(t["pnl"] for t in trades)
    gross_p = sum(t["pnl"] for t in wins)
    gross_l = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    
    return {
        "total": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "pf": gross_p / gross_l if gross_l > 0 else 99,
        "pnl": total_pnl
    }

# ── Main ─────────────────────────────────────────────────────────────────────── #

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print(f"Error: Data path {DATA_PATH} not found.")
        return pd.DataFrame()
        
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df = df[
        (df["timestamp"] >= pd.Timestamp(START_DATE, tz="UTC"))
        & (df["timestamp"] <= pd.Timestamp(END_DATE + " 23:59", tz="UTC"))
    ]
    return df.sort_values("timestamp").reset_index(drop=True)

if __name__ == "__main__":
    print(f"Loading data from {DATA_PATH}...")
    df_raw = load_data()
    if df_raw.empty:
        print("No data loaded. Check date range and file path.")
        sys.exit(1)
        
    print(f"Loaded {len(df_raw)} bars. Preparing Tier 2 data...")
    df = prepare_tier2_data(df_raw)
    
    blocked_hours = {0, 1, 6, 8, 16, 17, 22, 23}
    
    # Tier 2 Parameter Sweep
    sl_mults = [2.0, 3.5]
    tp_mults = [2.0, 4.0, 8.0]
    entry_pcts = [0.0, 0.25, 0.5] 
    
    results = []
    print(f"Running Tier 2 grid search on {len(sl_mults)*len(tp_mults)*len(entry_pcts)} combinations...")
    
    for sl, tp, entry in itertools.product(sl_mults, tp_mults, entry_pcts):
        res = run_backtest(df, blocked_hours, sl, tp, entry, use_tier2=True)
        res.update({"sl": sl, "tp": tp, "entry": entry})
        results.append(res)
        
    res_df = pd.DataFrame(results).sort_values("pnl", ascending=False)
    
    print("\n" + "="*80)
    print(" TIER 2 STRATEGY RESULTS (H4 Align + H1 Sweep + 1m FVG)")
    print("="*80)
    print(res_df.head(10).to_string(index=False))
    print("="*80)
    
    if not res_df.empty and res_df.iloc[0]['total'] > 0:
        best = res_df.iloc[0]
        print(f"\nBEST TIER 2 CONFIG: SL={best['sl']}, TP={best['tp']}, EntryPct={best['entry']}")
        print(f"WR: {best['wr']:.2f}%, PF: {best['pf']:.2f}, P&L: ${best['pnl']:.2f}, Trades: {best['total']}")
    else:
        print("\nNo trades found with Tier 2 confluence. Consider relaxing filters or extending date range.")
