import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
sys.path.insert(0, "/root/Silver-Bullet-ML-BMAD")

from src.research.strategy_core import StrategyConfig, detect_liquidity_sweep
from src.data.models import DollarBar

# Load Data
csv_path = Path("/root/Silver-Bullet-ML-BMAD/data/kraken/PF_XBTUSD_1min.csv")
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[df['timestamp'].dt.year == 2026]

# Build H1
h1_df = df.set_index('timestamp').resample('h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

cfg = StrategyConfig(h1_sweep_lookback=6)

print(f"--- Frequency Scan Diagnostic ---")
print(f"Total H1 intervals: {len(h1_df)}")

# Log every H1 bar that registers a liquidity sweep
sweeps = 0
for i in range(cfg.h1_sweep_lookback + 10, len(h1_df)):
    chunk = h1_df.iloc[i - (cfg.h1_sweep_lookback + 10) : i + 1]
    if detect_liquidity_sweep(chunk, cfg):
        sweeps += 1
        if sweeps <= 5: # Just log the first few
            print(f"Sweep found at {chunk.index[-1]}")

print(f"Total Liquidity Sweeps detected: {sweeps}")

# Log M15 CHoCH (logic manual mimic)
# CHoCH = Close below swing low by 0.3*ATR
m15_df = df.set_index('timestamp').resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
chochs = 0
# Simplified CHoCH logic
for i in range(20, len(m15_df)):
    swing_low = m15_df['low'].iloc[i-5:i].min()
    if m15_df['close'].iloc[i] < swing_low:
        chochs += 1

print(f"Total M15 CHoCH-like events: {chochs}")
