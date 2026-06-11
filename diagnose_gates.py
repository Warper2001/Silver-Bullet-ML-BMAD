import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
sys.path.insert(0, "/root/Silver-Bullet-ML-BMAD")

from src.research.strategy_core import StrategyConfig, detect_liquidity_sweep, detect_fvg, Direction
from src.data.models import DollarBar

csv_path = Path("/root/Silver-Bullet-ML-BMAD/data/kraken/PF_XBTUSD_1min.csv")
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[df['timestamp'].dt.year == 2026]

cfg = StrategyConfig(h1_sweep_lookback=6)
h1_df = df.set_index('timestamp').resample('h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

print(f"Total H1 bars: {len(h1_df)}")
sweep = detect_liquidity_sweep(h1_df, cfg)
print(f"Sweep result: {sweep}")

m1_df = df.set_index('timestamp').tail(20)
print(f"Tail bars (last 5):\n{m1_df.tail()}")
fvg = detect_fvg(m1_df, cfg, 500.0)
print(f"FVG result: {fvg}")
