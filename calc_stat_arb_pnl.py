import pandas as pd
import numpy as np
import pytz
import os

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

mnq = load_data(mnq_csv_path)
es = load_data(es_csv_path)

es = es[es.index.year == 2026]

mnq_5m = mnq['close'].resample('5min').last().dropna()
es_5m = es['close'].resample('5min').last().dropna()

df = pd.concat([mnq_5m, es_5m], axis=1, join='inner')
df.columns = ['MNQ', 'ES']

window = 60
df['MNQ_ret'] = df['MNQ'].pct_change()
df['ES_ret'] = df['ES'].pct_change()
cov = df['MNQ_ret'].rolling(window).cov(df['ES_ret'])
var = df['ES_ret'].rolling(window).var()
df['beta'] = cov / var
df['spread'] = df['MNQ_ret'] - (df['beta'] * df['ES_ret'])
df['spread_mean'] = df['spread'].rolling(window).mean()
df['spread_std'] = df['spread'].rolling(window).std()
df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']

df.dropna(inplace=True)

Z_ENTRY = 2.5
Z_EXIT = 0.0

# Using exactly what we deployed in stat_arb_bot.py
QTY_MNQ = 2
QTY_MES = 3

# Point values for the micro contracts
MNQ_PT_VAL = 2.0
MES_PT_VAL = 5.0

active_trade = None
completed_trades = []

for i in range(1, len(df)):
    row = df.iloc[i]
    prev_row = df.iloc[i-1]
    
    if active_trade:
        if active_trade['dir'] == 'SHORT_MNQ': 
            if row['z_score'] <= Z_EXIT:
                mnq_diff = active_trade['mnq_entry'] - row['MNQ'] # Short
                es_diff = row['ES'] - active_trade['es_entry'] # Long
                
                pnl_mnq = mnq_diff * QTY_MNQ * MNQ_PT_VAL
                pnl_es = es_diff * QTY_MES * MES_PT_VAL
                total_pnl = pnl_mnq + pnl_es
                
                completed_trades.append({'time': df.index[i], 'pnl': total_pnl})
                active_trade = None
                
        elif active_trade['dir'] == 'LONG_MNQ':
            if row['z_score'] >= Z_EXIT:
                mnq_diff = row['MNQ'] - active_trade['mnq_entry'] # Long
                es_diff = active_trade['es_entry'] - row['ES'] # Short
                
                pnl_mnq = mnq_diff * QTY_MNQ * MNQ_PT_VAL
                pnl_es = es_diff * QTY_MES * MES_PT_VAL
                total_pnl = pnl_mnq + pnl_es
                
                completed_trades.append({'time': df.index[i], 'pnl': total_pnl})
                active_trade = None
                
    else:
        if prev_row['z_score'] < Z_ENTRY and row['z_score'] >= Z_ENTRY:
            active_trade = {'dir': 'SHORT_MNQ', 'mnq_entry': row['MNQ'], 'es_entry': row['ES']}
        elif prev_row['z_score'] > -Z_ENTRY and row['z_score'] <= -Z_ENTRY:
            active_trade = {'dir': 'LONG_MNQ', 'mnq_entry': row['MNQ'], 'es_entry': row['ES']}

df_trades = pd.DataFrame(completed_trades)
df_trades['date'] = df_trades['time'].dt.date

total_trades = len(df_trades)
net_pnl = df_trades['pnl'].sum()
trading_days = df_trades['date'].nunique()

daily_pnl = df_trades.groupby('date')['pnl'].sum()
avg_daily_pnl = daily_pnl.mean()
median_daily_pnl = daily_pnl.median()
avg_trades_per_day = total_trades / trading_days

print(f"Total Trades: {total_trades}")
print(f"Total Net P&L: ${net_pnl:,.2f}")
print(f"Trading Days: {trading_days}")
print(f"Projected Trades / Day: {avg_trades_per_day:.2f}")
print(f"Projected Avg PnL / Day: ${avg_daily_pnl:.2f}")
print(f"Projected Median PnL / Day: ${median_daily_pnl:.2f}")
print(f"Average PnL / Trade: ${net_pnl/total_trades:.2f}")
