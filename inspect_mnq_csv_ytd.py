import pandas as pd
import os

base_dir = "/root/Silver-Bullet-ML-BMAD"
bt_csv_path = os.path.join(base_dir, "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

if os.path.exists(bt_csv_path):
    print("=== mnq_1min_2026_ytd.csv tail ===")
    df = pd.read_csv(bt_csv_path)
    print(df.tail(10))
    print("\nSummary Statistics of 'close':")
    print(df['close'].describe())
else:
    print("Backtest CSV not found")
