import pandas as pd
import os
base_dir = "/root/Silver-Bullet-ML-BMAD"
files = [
    "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv",
    "data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv",
    "data/kraken/PF_XBTUSD_1min.csv"
]
for f in files:
    path = os.path.join(base_dir, f)
    if os.path.exists(path):
        df = pd.read_csv(path, nrows=1)
        print(f"{f}: {df.columns.tolist()}")
    else:
        print(f"{f}: NOT FOUND")
