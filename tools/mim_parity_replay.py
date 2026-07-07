"""MIM-NB live-vs-sealed-engine parity replay (halt-and-review, 2026-07-07).

Replays the SEALED engine (study_mim_nb_catstop.py, seal 6957daa) over
warmup (mnq_1min_2026_ytd.csv, Jan..Jun-10) + live archived bars
(data/mim_nb/bars_raw.csv, Jun-11..Jul-07) and diffs:
  A) trade-level vs data/mim_nb/trades.csv  (S=500 era <=06-24, S=250 era >=06-25)
  B) mark-level sigma/UB/LB/action vs data/mim_nb/decisions.csv
Engine code is exec'd from the sealed file verbatim (no reimplementation).
"""
import pandas as pd, numpy as np
from collections import defaultdict, deque

BASE = "/root/Silver-Bullet-ML-BMAD"

# --- exec sealed engine defs verbatim (everything above the run section) ---
src = open(f"{BASE}/study_mim_nb_catstop.py").read()
src_defs = src.split('print("Loading data...')[0]
ns = {}
exec(src_defs, ns)
load, run_catstop, LOOKBACK = ns['load'], ns['run_catstop'], ns['LOOKBACK']
ET = ns['ET']

# --- build spliced bar set in engine schema ---
ytd = pd.read_csv(f"{BASE}/data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv",
                  usecols=['timestamp','open','high','low','close','volume'])
ytd['timestamp'] = pd.to_datetime(ytd['timestamp'], utc=True, format='ISO8601')
ytd = ytd[ytd['timestamp'] < '2026-06-11']

raw = pd.read_csv(f"{BASE}/data/mim_nb/bars_raw.csv")
raw = raw.rename(columns={'ts_utc':'timestamp'})[['timestamp','open','high','low','close','volume']]
raw['timestamp'] = pd.to_datetime(raw['timestamp'], utc=True, format='ISO8601')
raw = raw.drop_duplicates(subset='timestamp', keep='last')

allb = pd.concat([ytd, raw], ignore_index=True).sort_values('timestamp')
allb['et'] = allb['timestamp'].dt.tz_convert(ET)
allb['day'] = allb['et'].dt.date
allb['hm'] = allb['et'].dt.strftime('%H:%M')
df = allb[(allb['hm'] >= '09:31') & (allb['hm'] <= '16:00')].copy()

# report skipped/incomplete live days (engine requires 09:31 first bar + 16:00 present)
live_days = sorted(d for d in df['day'].unique() if str(d) >= '2026-06-11')
print("== live-era day completeness (engine criteria) ==")
for d in live_days:
    g = df[df['day'] == d]
    ok = (g['hm'].iloc[0] == '09:31') and ('16:00' in set(g['hm']))
    n = len(g)
    if not ok or n < 300:
        print(f"  {d}: bars={n} first={g['hm'].iloc[0]} has16:00={'16:00' in set(g['hm'])}  -> ENGINE SKIPS" if not ok else f"  {d}: bars={n} (partial but engine-accepted)")

# --- A) trade-level replay, both arms ---
for S in (250, 500):
    t = run_catstop(df, S)
    t = t[t['day'].astype(str) >= '2026-06-11']
    t.to_csv(f"/root/.claude/jobs/aad73931/tmp/replay_trades_s{S}.csv", index=False)
    print(f"\n== engine trades S={S}, live era ==")
    print(t.to_string(index=False) if len(t) else "  (none)")

# --- B) mark-level instrumented replay (same formulas, for sigma/UB/LB diff) ---
rows = []
hist = defaultdict(lambda: deque(maxlen=LOOKBACK))
day_count, prev_close = 0, np.nan
check_marks = {f"{h:02d}:{m}" for h in range(10, 16) for m in ('00','30')}
for day, g in df.groupby('day', sort=True):
    g = g.sort_values('et')
    hms, closes = g['hm'].values, g['close'].values
    if hms[0] != '09:31' or '16:00' not in set(hms):
        prev_close = closes[-1] if len(closes) else prev_close
        continue
    O = g['open'].values[0]
    day_count += 1
    tradeable = day_count > LOOKBACK and not np.isnan(prev_close)
    gu = max(O - prev_close, 0) if not np.isnan(prev_close) else 0.0
    gd = max(prev_close - O, 0) if not np.isnan(prev_close) else 0.0
    for i, hm in enumerate(hms):
        if hm in check_marks and tradeable and len(hist[hm]) == LOOKBACK:
            sig = float(np.mean(hist[hm]))
            rows.append({'day': str(day), 'mark': hm, 'e_open': O, 'e_prev': prev_close,
                         'e_sigma': sig, 'e_ub': O*(1+sig)+gd, 'e_lb': O*(1-sig)-gu,
                         'e_close': closes[i]})
    for i, hm in enumerate(hms):
        hist[hm].append(abs(closes[i]/O - 1.0))
    prev_close = closes[-1]
eng = pd.DataFrame(rows)
eng = eng[eng['day'] >= '2026-06-11']

dec = pd.read_csv(f"{BASE}/data/mim_nb/decisions.csv")
dec['day'] = dec['ts_et'].str[:10]
m = dec.merge(eng, left_on=['day','mark'], right_on=['day','mark'], how='left')
m['d_sigma'] = (m['sigma'] - m['e_sigma']).abs()
m['d_ub'] = (m['ub'] - m['e_ub']).abs()
m['d_lb'] = (m['lb'] - m['e_lb']).abs()
m['d_close'] = (m['close'] - m['e_close']).abs()
m['d_open'] = (m['open_d'] - m['e_open']).abs()
print(f"\n== mark-level diff, {len(m)} live decision rows ({m['e_sigma'].notna().sum()} matched) ==")
print(m[['d_open','d_close','d_sigma','d_ub','d_lb']].describe().loc[['mean','max']].to_string())
bad = m[(m['d_ub'] > 2.0) | (m['d_lb'] > 2.0) | (m['d_close'] > 0.25) | m['e_sigma'].isna()]
print(f"\nrows with UB/LB diff >2pt, close mismatch, or no engine match: {len(bad)}")
if len(bad):
    print(bad[['day','mark','close','e_close','sigma','e_sigma','ub','e_ub','lb','e_lb','action']].to_string(index=False))
m.to_csv("/root/.claude/jobs/aad73931/tmp/mark_parity.csv", index=False)
