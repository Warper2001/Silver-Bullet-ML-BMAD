"""
study_mim_noise_bands_gate0.py
------------------------------
Gate 0 (dev 2025) for pre-registration mim-noise-bands-mnq
(_bmad-output/preregistration_mim_noise_bands.md, sealed commit 50f111b).

Noise bounds: UB(t)=O*(1+sigma(t))+max(Cprev-O,0); LB(t)=O*(1-sigma(t))-max(O-Cprev,0)
sigma(t) = 14-day mean of |close(d,t)/O_d - 1| per minute label.
Checks at HH:00/HH:30 closes (entries 10:00-15:30, stops 10:00-16:00), fills next bar open.
V1 stop: long max(UB,VWAP) / short min(LB,VWAP).  V2 stop: long LB / short UB.
Reversals allowed. EOD exit at 16:00 close. Cost 1.12 pts per completed trade.
Dev 2025 only; OOS 2026 NOT touched. Diagnostics: 2023/24 Sep-Nov (non-gating).
"""
import pandas as pd
import numpy as np
from collections import defaultdict, deque

BASE = "/root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute"
COST_PTS = 1.12
ET = "America/New_York"
LOOKBACK = 14

def load(path):
    df = pd.read_csv(path, usecols=['timestamp', 'open', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, format='ISO8601')
    df['et'] = df['timestamp'].dt.tz_convert(ET)
    df['day'] = df['et'].dt.date
    df['hm'] = df['et'].dt.strftime('%H:%M')
    rth = df[(df['hm'] >= '09:31') & (df['hm'] <= '16:00')].copy()
    return rth

def run(df, variant):
    """variant: 'V1' (tight stop) or 'V2' (wide stop). Returns trades DataFrame."""
    hist = defaultdict(lambda: deque(maxlen=LOOKBACK))  # minute label -> last 14 |move|
    trades = []
    day_count = 0
    prev_close = np.nan

    check_marks = {f"{h:02d}:{m}" for h in range(10, 16) for m in ('00', '30')} | {'16:00'}
    entry_marks = {f"{h:02d}:{m}" for h in range(10, 16) for m in ('00', '30')} - {'16:00'}

    for day, g in df.groupby('day', sort=True):
        g = g.sort_values('et')
        hms = g['hm'].values
        opens = g['open'].values
        closes = g['close'].values
        vols = g['volume'].values
        if hms[0] != '09:31' or '16:00' not in set(hms):
            # incomplete session: still update history if full open exists? skip entirely
            continue
        O = opens[0]
        day_count += 1
        tradeable = day_count > LOOKBACK and not np.isnan(prev_close)

        gap_up_adj = max(O - prev_close, 0) if not np.isnan(prev_close) else 0.0
        gap_dn_adj = max(prev_close - O, 0) if not np.isnan(prev_close) else 0.0

        cum_pv = np.cumsum(closes * vols)
        cum_v = np.cumsum(vols)
        vwap = cum_pv / np.where(cum_v == 0, 1, cum_v)

        pos = 0          # +1/-1/0
        entry_px = 0.0
        entry_t = None
        pending = None   # ('long'|'short'|'exit', reason) to fill at next bar open

        for i, hm in enumerate(hms):
            # fill pending order at this bar's open
            if tradeable and pending is not None:
                action, why = pending
                px = opens[i]
                if action == 'exit' and pos != 0:
                    trades.append({'day': day, 'dir': pos, 'entry': entry_px, 'exit': px,
                                   'reason': why, 'pnl_pts': pos * (px - entry_px),
                                   'entry_t': entry_t, 'exit_t': hms[i]})
                    pos = 0
                elif action in ('long', 'short'):
                    new = 1 if action == 'long' else -1
                    if pos != 0 and pos != new:
                        trades.append({'day': day, 'dir': pos, 'entry': entry_px, 'exit': px,
                                       'reason': 'REVERSAL', 'pnl_pts': pos * (px - entry_px),
                                       'entry_t': entry_t, 'exit_t': hms[i]})
                        pos = 0
                    if pos == 0:
                        pos, entry_px, entry_t = new, px, hms[i]
                pending = None

            # evaluate at check marks using this bar's close
            if hm in check_marks:
                sig = hist[hm]
                if tradeable and len(sig) == LOOKBACK:
                    sigma = float(np.mean(sig))
                    ub = O * (1 + sigma) + gap_dn_adj
                    lb = O * (1 - sigma) - gap_up_adj
                    c = closes[i]
                    v = vwap[i]
                    # stop first
                    if pos == 1:
                        lvl = max(ub, v) if variant == 'V1' else lb
                        if c < lvl:
                            pending = ('exit', 'STOP')
                    elif pos == -1:
                        lvl = min(lb, v) if variant == 'V1' else ub
                        if c > lvl:
                            pending = ('exit', 'STOP')
                    # entry/reversal (entry marks only); breakout overrides stop intent
                    if hm in entry_marks:
                        if c > ub and pos != 1:
                            pending = ('long', 'BREAK_UP')
                        elif c < lb and pos != -1:
                            pending = ('short', 'BREAK_DN')

            # EOD exit at 16:00 close
            if hm == '16:00':
                if pos != 0:
                    trades.append({'day': day, 'dir': pos, 'entry': entry_px, 'exit': closes[i],
                                   'reason': 'EOD', 'pnl_pts': pos * (closes[i] - entry_px),
                                   'entry_t': entry_t, 'exit_t': '16:00'})
                    pos = 0
                pending = None

        # update sigma history with today's moves (after trading the day)
        for i, hm in enumerate(hms):
            hist[hm].append(abs(closes[i] / O - 1.0))
        prev_close = closes[-1]

    return pd.DataFrame(trades)

def gate0(name, tdf, min_n, gating=True):
    print(f"\n{'='*68}\n{name}\n{'='*68}")
    n = len(tdf)
    if n == 0:
        print("  ZERO trades — INCONCLUSIVE")
        return
    gross = tdf['pnl_pts'].values
    net = gross - COST_PTS
    gpf = gross[gross > 0].sum() / abs(gross[gross < 0].sum()) if (gross < 0).any() else float('inf')
    npf = net[net > 0].sum() / abs(net[net < 0].sum()) if (net < 0).any() else float('inf')
    exp_usd = net.mean() * 2.0
    avg_gw = gross[gross > 0].mean() if (gross > 0).any() else 0.0
    cf = 1.12 / avg_gw * 100 if avg_gw > 0 else float('inf')
    days = tdf['day'].nunique()
    print(f"  N={n} trades over {days} traded days ({n/days:.2f}/day)")
    print(f"  WR gross={(gross>0).mean()*100:.1f}%  gross PF={gpf:.3f}  NET PF={npf:.3f}")
    print(f"  NET expectancy = ${exp_usd:+.2f}/contract/trade")
    print(f"  avg gross win={avg_gw:.2f} pts  cost fraction={cf:.1f}% (gate <=25%)")
    for r, gr in tdf.groupby('reason'):
        print(f"    {r:<9} N={len(gr):>4}  avg={gr['pnl_pts'].mean():>+8.2f} pts  WR={(gr['pnl_pts']>0).mean()*100:.0f}%")
    if not gating:
        print("  (secondary diagnostic — non-gating)")
        return
    checks = {f"N >= {min_n}": n >= min_n,
              "net PF >= 1.10": npf >= 1.10,
              "net expectancy > 0": exp_usd > 0,
              "cost <= 25% avg gross win": cf <= 25.0}
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if n < min_n:
        print("  ==> INCONCLUSIVE (N below minimum)")
    else:
        print(f"  ==> {'GATE 0 PASS' if all(checks.values()) else 'GATE 0 FAIL'}")

print("Loading dev 2025...", flush=True)
dev = load(f"{BASE}/mnq_1min_2025.csv")
for variant in ('V1', 'V2'):
    t = run(dev, variant)
    t.to_csv(f"data/reports/mim_nb_gate0_{variant.lower()}_2025.csv", index=False)
    gate0(f"{variant} — {'tight stop max(UB,VWAP)' if variant=='V1' else 'wide stop (opposite band)'}, dev 2025", t, 100)

print("\nSecondary diagnostics (non-gating):", flush=True)
for f, lbl in [("mnq_1min_2023_sepnov.csv", "2023 Sep–Nov"),
               ("mnq_1min_2024_sepnov.csv", "2024 Sep–Nov")]:
    try:
        d = load(f"{BASE}/{f}")
        for variant in ('V1', 'V2'):
            gate0(f"{variant} — {lbl}", run(d, variant), 0, gating=False)
    except Exception as e:
        print(f"  {lbl}: unavailable ({e})")

print("\nDone. OOS 2026 NOT touched.", flush=True)
