"""
study_mim_nb_catstop.py
-----------------------
Gates A+B for prereg mim-nb-v2-catstop (sealed commit 6957daa).

V2 spec (noise bands, wide band-stop, reversals, EOD exit) + intrabar catastrophe
stop at entry -/+ S, S in {250 (S-A), 500 (S-B)} pts, monitored on every 1-min bar
(low/high touch, fill at stop level), live from the entry bar.

Gate A: pooled net PF >= 1.10, pooled exp > 0, each window net PF >= 1.00.
Gate B: combine MC (corrected rules) pass% >= 50% at some size 1-10 and pass > blow.
"""
import pandas as pd
import numpy as np
from collections import defaultdict, deque

BASE = "/root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute"
COST_PTS = 1.12
PT_VAL = 2.0
ET = "America/New_York"
LOOKBACK = 14

def load(path):
    df = pd.read_csv(path, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, format='ISO8601')
    df['et'] = df['timestamp'].dt.tz_convert(ET)
    df['day'] = df['et'].dt.date
    df['hm'] = df['et'].dt.strftime('%H:%M')
    return df[(df['hm'] >= '09:31') & (df['hm'] <= '16:00')].copy()

def run_catstop(df, S):
    hist = defaultdict(lambda: deque(maxlen=LOOKBACK))
    trades = []
    day_count = 0
    prev_close = np.nan
    check_marks = {f"{h:02d}:{m}" for h in range(10, 16) for m in ('00', '30')} | {'16:00'}
    entry_marks = {f"{h:02d}:{m}" for h in range(10, 16) for m in ('00', '30')} - {'16:00'}

    for day, g in df.groupby('day', sort=True):
        g = g.sort_values('et')
        hms = g['hm'].values
        opens, highs, lows = g['open'].values, g['high'].values, g['low'].values
        closes, vols = g['close'].values, g['volume'].values
        if hms[0] != '09:31' or '16:00' not in set(hms):
            continue
        O = opens[0]
        day_count += 1
        tradeable = day_count > LOOKBACK and not np.isnan(prev_close)
        gap_up_adj = max(O - prev_close, 0) if not np.isnan(prev_close) else 0.0
        gap_dn_adj = max(prev_close - O, 0) if not np.isnan(prev_close) else 0.0
        cum_pv = np.cumsum(closes * vols)
        cum_v = np.cumsum(vols)
        vwap = cum_pv / np.where(cum_v == 0, 1, cum_v)

        pos, entry_px, entry_t, cat = 0, 0.0, None, np.nan
        pending = None

        for i, hm in enumerate(hms):
            # 1) fill pending at open
            if tradeable and pending is not None:
                action, why = pending
                px = opens[i]
                if action == 'exit' and pos != 0:
                    trades.append({'day': day, 'dir': pos, 'reason': why,
                                   'pnl_pts': pos * (px - entry_px),
                                   'entry_t': entry_t, 'exit_t': hms[i]})
                    pos = 0
                elif action in ('long', 'short'):
                    new = 1 if action == 'long' else -1
                    if pos != 0 and pos != new:
                        trades.append({'day': day, 'dir': pos, 'reason': 'REVERSAL',
                                       'pnl_pts': pos * (px - entry_px),
                                       'entry_t': entry_t, 'exit_t': hms[i]})
                        pos = 0
                    if pos == 0:
                        pos, entry_px, entry_t = new, px, hms[i]
                        cat = entry_px - S if pos == 1 else entry_px + S
                pending = None

            # 2) intrabar catastrophe stop (live from entry bar)
            if pos == 1 and lows[i] <= cat:
                trades.append({'day': day, 'dir': 1, 'reason': 'CAT_STOP',
                               'pnl_pts': cat - entry_px, 'entry_t': entry_t, 'exit_t': hms[i]})
                pos, pending = 0, None
            elif pos == -1 and highs[i] >= cat:
                trades.append({'day': day, 'dir': -1, 'reason': 'CAT_STOP',
                               'pnl_pts': entry_px - cat, 'entry_t': entry_t, 'exit_t': hms[i]})
                pos, pending = 0, None

            # 3) half-hour checks
            if hm in check_marks:
                sig = hist[hm]
                if tradeable and len(sig) == LOOKBACK:
                    sigma = float(np.mean(sig))
                    ub = O * (1 + sigma) + gap_dn_adj
                    lb = O * (1 - sigma) - gap_up_adj
                    c = closes[i]
                    if pos == 1 and c < lb:
                        pending = ('exit', 'STOP')
                    elif pos == -1 and c > ub:
                        pending = ('exit', 'STOP')
                    if hm in entry_marks:
                        if c > ub and pos != 1:
                            pending = ('long', 'BREAK_UP')
                        elif c < lb and pos != -1:
                            pending = ('short', 'BREAK_DN')

            if hm == '16:00':
                if pos != 0:
                    trades.append({'day': day, 'dir': pos, 'reason': 'EOD',
                                   'pnl_pts': pos * (closes[i] - entry_px),
                                   'entry_t': entry_t, 'exit_t': '16:00'})
                    pos = 0
                pending = None

        for i, hm in enumerate(hms):
            hist[hm].append(abs(closes[i] / O - 1.0))
        prev_close = closes[-1]
    return pd.DataFrame(trades)

def stats(tdf):
    gross = tdf['pnl_pts'].values
    net = gross - COST_PTS
    npf = net[net > 0].sum()/abs(net[net < 0].sum()) if (net < 0).any() else float('inf')
    return len(tdf), npf, net.mean()*PT_VAL, gross.min()

def run_mc(pooled, contracts, n_sim=5000, max_days=90, seed=42):
    rng = np.random.default_rng(seed)
    t = pooled.copy()
    t['net1'] = (t['pnl_pts'] - COST_PTS) * PT_VAL
    by_day = [g['net1'].values for _, g in t.groupby('day', sort=True)]
    nd = len(by_day)
    pn = bn = 0
    dtp = []
    for _ in range(n_sim):
        bal, floor, best_day, outcome = 50_000.0, 48_000.0, 0.0, None
        for dn, di in enumerate(rng.integers(0, nd, size=max_days)):
            day_pnl = 0.0
            for p1 in by_day[di]:
                p = p1 * contracts
                bal += p
                day_pnl += p
                if bal <= floor:
                    outcome = 'blow'
                    break
                if day_pnl <= -1000.0:
                    break
            if outcome:
                break
            best_day = max(best_day, day_pnl)
            profit = bal - 50_000.0
            if profit >= 3000.0 and best_day < 0.5 * profit:
                outcome = 'pass'
                dtp.append(dn+1)
                break
            floor = min(50_000.0, max(floor, bal - 2000.0))
        if outcome == 'pass': pn += 1
        elif outcome == 'blow': bn += 1
    med = int(np.median(dtp)) if dtp else -1
    return pn/n_sim, bn/n_sim, med

print("Loading data...", flush=True)
dev = load(f"{BASE}/mnq_1min_2025.csv")
oos = load(f"{BASE}/mnq_1min_2026_ytd.csv")

for S, label in [(250, 'S-A 250pts (1/2 DLL)'), (500, 'S-B 500pts (DLL)')]:
    print(f"\n{'='*70}\nVARIANT {label}\n{'='*70}", flush=True)
    td = run_catstop(dev, S)
    to = run_catstop(oos, S)
    pooled = pd.concat([td, to], ignore_index=True)
    pooled.to_csv(f"data/reports/mim_nb_catstop_s{S}_pooled.csv", index=False)

    nd_, pfd, expd, wd = stats(td)
    no_, pfo, expo, wo = stats(to)
    np_, pfp, expp, wp = stats(pooled)
    print(f"  dev 2025:  N={nd_:>3}  net PF={pfd:.3f}  exp=${expd:+.2f}/ct  worst={wd:+.0f} pts")
    print(f"  OOS 2026:  N={no_:>3}  net PF={pfo:.3f}  exp=${expo:+.2f}/ct  worst={wo:+.0f} pts")
    print(f"  pooled:    N={np_:>3}  net PF={pfp:.3f}  exp=${expp:+.2f}/ct  worst={wp:+.0f} pts")
    for r, g in pooled.groupby('reason'):
        print(f"    {r:<9} N={len(g):>4}  avg={g['pnl_pts'].mean():>+8.2f} pts")
    ga = {"pooled net PF >= 1.10": pfp >= 1.10,
          "pooled exp > 0": expp > 0,
          "dev net PF >= 1.00": pfd >= 1.00,
          "OOS net PF >= 1.00": pfo >= 1.00}
    for k, v in ga.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    gateA = all(ga.values())
    print(f"  ==> GATE A {'PASS' if gateA else 'FAIL'}")

    print(f"\n  Gate B — combine MC (pooled):")
    print(f"  {'Size':>5} | {'Pass%':>6} | {'Blow%':>6} | {'Stall%':>6} | {'Resolved':>8} | MedDays")
    bestk = (0, 0.0, 1.0, -1)
    for k in range(1, 11):
        p, b, med = run_mc(pooled, k)
        res = p/(p+b)*100 if (p+b) > 0 else 0
        flag = "  <== gate met" if (p >= 0.50 and p > b) else ""
        print(f"  {k:>4}ct | {p*100:>5.1f}% | {b*100:>5.1f}% | {(1-p-b)*100:>5.1f}% | {res:>7.1f}% | {med:>5}{flag}")
        if p > bestk[1]:
            bestk = (k, p, b, med)
    k, p, b, med = bestk
    gateB = p >= 0.50 and p > b
    print(f"  Best: {k}ct pass={p*100:.1f}% blow={b*100:.1f}% median {med} days")
    print(f"  ==> GATE B {'PASS' if gateB else 'FAIL'}")

print("\nDone.", flush=True)
