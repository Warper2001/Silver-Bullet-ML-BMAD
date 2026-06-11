import os
import pandas as pd
import numpy as np
import pytz

base_dir = "/root/Silver-Bullet-ML-BMAD"
mnq_csv_path = os.path.join(base_dir, "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
es_csv_path = os.path.join(base_dir, "data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv")

et_tz = pytz.timezone("America/New_York")

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df.index = df.index.tz_convert(et_tz)
    return df

print("Loading MNQ and ES data for 2026 YTD...")
mnq = load_data(mnq_csv_path)
es = load_data(es_csv_path)

es = es[es.index.year == 2026]

# Resample to 5-minute bars to reduce noise and align timestamps perfectly
mnq_5m = mnq['close'].resample('5min').last().dropna()
es_5m = es['close'].resample('5min').last().dropna()

df = pd.concat([mnq_5m, es_5m], axis=1, join='inner')
df.columns = ['MNQ', 'ES']

# Calculate rolling beta (hedge ratio)
window = 60 # 5 hours of 5-minute bars
df['MNQ_ret'] = df['MNQ'].pct_change()
df['ES_ret'] = df['ES'].pct_change()

# Calculate rolling covariance and variance
cov = df['MNQ_ret'].rolling(window).cov(df['ES_ret'])
var = df['ES_ret'].rolling(window).var()
df['beta'] = cov / var

# Calculate the spread: MNQ - (beta * ES)
df['spread'] = df['MNQ_ret'] - (df['beta'] * df['ES_ret'])

# Calculate Z-Score of the spread
df['spread_mean'] = df['spread'].rolling(window).mean()
df['spread_std'] = df['spread'].rolling(window).std()
df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']

df.dropna(inplace=True)

# Backtest logic
Z_ENTRY = 2.5
Z_EXIT = 0.0

active_trade = None
completed_trades = []

for i in range(1, len(df)):
    row = df.iloc[i]
    prev_row = df.iloc[i-1]
    
    if active_trade:
        # Check exit
        if active_trade['dir'] == 'SHORT_MNQ': # We shorted the spread
            if row['z_score'] <= Z_EXIT:
                # Exit
                mnq_ret = (row['MNQ'] - active_trade['mnq_entry']) / active_trade['mnq_entry']
                es_ret = (row['ES'] - active_trade['es_entry']) / active_trade['es_entry']
                
                # PnL logic: Short MNQ, Long ES (weighted by beta)
                pnl_pct = -mnq_ret + (active_trade['beta'] * es_ret)
                
                completed_trades.append({
                    'entry_time': active_trade['time'],
                    'exit_time': df.index[i],
                    'pnl_pct': pnl_pct,
                    'type': 'SHORT_MNQ_LONG_ES'
                })
                active_trade = None
                
        elif active_trade['dir'] == 'LONG_MNQ': # We longed the spread
            if row['z_score'] >= Z_EXIT:
                mnq_ret = (row['MNQ'] - active_trade['mnq_entry']) / active_trade['mnq_entry']
                es_ret = (row['ES'] - active_trade['es_entry']) / active_trade['es_entry']
                
                # PnL logic: Long MNQ, Short ES
                pnl_pct = mnq_ret - (active_trade['beta'] * es_ret)
                
                completed_trades.append({
                    'entry_time': active_trade['time'],
                    'exit_time': df.index[i],
                    'pnl_pct': pnl_pct,
                    'type': 'LONG_MNQ_SHORT_ES'
                })
                active_trade = None
                
    else:
        # Check entry
        if prev_row['z_score'] < Z_ENTRY and row['z_score'] >= Z_ENTRY:
            # Spread is too high, short the spread (Short MNQ, Long ES)
            active_trade = {
                'dir': 'SHORT_MNQ',
                'mnq_entry': row['MNQ'],
                'es_entry': row['ES'],
                'beta': row['beta'],
                'time': df.index[i]
            }
        elif prev_row['z_score'] > -Z_ENTRY and row['z_score'] <= -Z_ENTRY:
            # Spread is too low, long the spread (Long MNQ, Short ES)
            active_trade = {
                'dir': 'LONG_MNQ',
                'mnq_entry': row['MNQ'],
                'es_entry': row['ES'],
                'beta': row['beta'],
                'time': df.index[i]
            }

print("\n--- STATISTICAL ARBITRAGE (MNQ vs ES) BACKTEST RESULTS (2026 YTD) ---")
if not completed_trades:
    print("No trades generated.")
else:
    df_trades = pd.DataFrame(completed_trades)
    
    total_trades = len(df_trades)
    wins = len(df_trades[df_trades['pnl_pct'] > 0])
    losses = len(df_trades[df_trades['pnl_pct'] < 0])
    win_rate = wins / total_trades * 100
    
    gross_prof = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(df_trades[df_trades['pnl_pct'] < 0]['pnl_pct'].sum())
    pf = gross_prof / gross_loss if gross_loss > 0 else float('inf')
    
    print(f"Total Trades:  {total_trades}")
    print(f"Win Rate:      {win_rate:.1f}% ({wins} W / {losses} L)")
    print(f"Profit Factor: {pf:.2f}")
    
    # Assume $100k account allocating $50k per leg
    total_return_usd = df_trades['pnl_pct'].sum() * 50000.0
    print(f"Estimated Net P&L: ${total_return_usd:,.2f}")
