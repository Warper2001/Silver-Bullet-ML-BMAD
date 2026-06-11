import joblib
import pandas as pd
import numpy as np
import os

base_dir = "/root/Silver-Bullet-ML-BMAD"
model_path = os.path.join(base_dir, "models/mnq_s26_xgboost_model.pkl")

import sys
sys.path.insert(0, base_dir)
from run_mnq_s26_pipeline import load_and_compute_features, simulate_trades

model = joblib.load(model_path)
df_bt = load_and_compute_features(os.path.join(base_dir, "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"))
trades_bt = simulate_trades(df_bt)

probs = []
for t in trades_bt:
    feat_df = pd.DataFrame([t['features']])
    proba = model.predict_proba(feat_df)[0, 1]
    probs.append(proba)

probs = np.array(probs)
print("=== MNQ 2026 Trade Probability Distribution ===")
print(f"Total Trades: {len(probs)}")
print(f"Min Prob:     {probs.min():.4f}")
print(f"Max Prob:     {probs.max():.4f}")
print(f"Mean Prob:    {probs.mean():.4f}")
print(f"Median Prob:  {np.median(probs):.4f}")
print(f"Count >= 0.35: {sum(probs >= 0.35)}")
print(f"Count >= 0.40: {sum(probs >= 0.40)}")
print(f"Count >= 0.42: {sum(probs >= 0.42)}")
print(f"Count >= 0.45: {sum(probs >= 0.45)}")
print(f"Count >= 0.50: {sum(probs >= 0.50)}")

# Let's see the performance for these lower thresholds!
for th in [0.0, 0.35, 0.38, 0.40, 0.42, 0.45]:
    executed = [t for i, t in enumerate(trades_bt) if probs[i] >= th]
    count = len(executed)
    if count > 0:
        wins = len([t for t in executed if t['pnl'] > 0])
        win_rate = wins / count * 100
        total_pnl = sum([t['pnl'] for t in executed])
        gross_profit = sum([t['pnl'] for t in executed if t['pnl'] > 0])
        gross_loss = abs(sum([t['pnl'] for t in executed if t['pnl'] < 0]))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        print(f"Threshold >= {th:.2f}: Count={count}, WinRate={win_rate:.1f}%, PF={pf:.2f}, PnL={total_pnl:+.1f}")
