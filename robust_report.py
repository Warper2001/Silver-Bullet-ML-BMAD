
import csv
import os

def get_pnl_from_csv(file_path):
    pnl_total = 0.0
    wins = 0
    losses = 0
    gross_wins = 0.0
    gross_losses = 0.0
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            pnl_idx = [i for i, h in enumerate(header) if 'pnl' in h.lower()][0]
        except:
            return 0, 0, 0
            
        for row in reader:
            try:
                if len(row) > pnl_idx:
                    val = float(row[pnl_idx])
                    pnl_total += val
                    if val > 0:
                        wins += 1
                        gross_wins += val
                    elif val < 0:
                        losses += 1
                        gross_losses += abs(val)
            except:
                continue
    
    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    return pnl_total, wr, pf

results = []
logs = {
    "trader-yank": "logs/tier2_trade_log.csv",
    "trader-s26": "logs/s26_soft_fvg_trade_log.csv",
    "trader-s27": "logs/s27_squeeze_trade_log.csv",
}

for tid, path in logs.items():
    if os.path.exists(path):
        pnl, wr, pf = get_pnl_from_csv(path)
        results.append(f"| {tid:<20} | PNL: ${pnl:>10.2f} | WR: {wr:>5.1f}% | PF: {pf:>5.2f} |")
    else:
        results.append(f"| {tid:<20} | File missing |")

header = f"| {'Trader ID':<20} | {'Performance from Logs':<60} |"
sep = "|" + "-"*22 + "|" + "-"*62 + "|"
print("\n".join([sep, header, sep] + results))
