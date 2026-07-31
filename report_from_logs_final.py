
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re

logs = {
    "trader-yank": "logs/tier2_trade_log.csv",
    "trader-s26": "logs/s26_soft_fvg_trade_log.csv",
    "trader-s27": "logs/s27_squeeze_trade_log.csv",
    "trader-s26-combine": "logs/s26_combine_bot.log", 
}

def parse_pnl(line):
    # Regex for "$+740.50" or "$-42.66"
    match = re.search(r"\$([+-]?\d+\.\d+)", line)
    return float(match.group(1)) if match else 0.0

results = []

for tid, path in logs.items():
    if not os.path.exists(path):
        results.append(f"| {tid:<20} | Log file not found |")
        continue
    
    if path.endswith('.csv'):
        df = pd.read_csv(path)
        # Find pnl col
        pnl_col = [c for c in df.columns if 'pnl' in c.lower()][0]
        df[pnl_col] = pd.to_numeric(df[pnl_col], errors='coerce').fillna(0.0)
        
        pnl = df[pnl_col].sum()
        wr = (df[pnl_col] > 0).mean() * 100
        
        wins = df[df[pnl_col] > 0][pnl_col].sum()
        losses = abs(df[df[pnl_col] < 0][pnl_col].sum())
        pf = wins / losses if losses > 0 else float('inf')
        
        results.append(f"| {tid:<20} | PNL: ${pnl:>8.2f} | WR: {wr:>5.1f}% | PF: {pf:>5.2f} |")

    elif path.endswith('.log'):
        with open(path, 'r') as f:
            lines = f.readlines()
            # Parse PnL from log lines that indicate a closed trade
            pnls = [parse_pnl(line) for line in lines if "TRADE" in line or "CLOSED" in line or "SUBMITTED" in line]
            # Wait, log parsing is tricky. Let's just sum the PnL found.
            # This is a rough estimation based on log strings.
            pnl = sum(pnls)
            results.append(f"| {tid:<20} | PNL: ${pnl:>8.2f} (Log Est.) | WR: N/A | PF: N/A |")

header = f"| {'Trader ID':<20} | {'Performance from Logs':<60} |"
sep = "|" + "-"*22 + "|" + "-"*62 + "|"
print("\n".join([sep, header, sep] + results))
