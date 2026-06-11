import pandas as pd
import os

base_dir = "/root/Silver-Bullet-ML-BMAD"
train_csv_path = os.path.join(base_dir, "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")

if os.path.exists(train_csv_path):
    print("=== mnq_1min_2025.csv tail ===")
    df = pd.read_csv(train_csv_path)
    print(df.tail(10))
    print("\nSummary Statistics of 'close':")
    print(df['close'].describe())
else:
    print("Train CSV not found")
