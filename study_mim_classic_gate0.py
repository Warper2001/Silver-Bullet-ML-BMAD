"""
study_mim_classic_gate0.py
--------------------------
Gate 0 (dev 2025) for pre-registration mim-classic-mnq
(_bmad-output/preregistration_mim_classic.md, sealed commit 56a1759).

V1: direction = sign(r1), r1 = close(10:00 ET) - prior session 16:00 ET close
V2: trade only if sign(r1) == sign(r12), r12 = close(15:30) - close(15:00)
Entry: open of bar labeled 15:31 ET. Exit: close of bar labeled 16:00 ET. No stop.
Costs: $2.24/contract RT = 1.12 MNQ points per trade.

Dev = 2025 file only. 2026 OOS is NOT touched.
Secondary diagnostic (non-gating): 2023/2024 Sep-Nov files.
"""
import pandas as pd
import numpy as np

BASE = "/root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute"
COST_PTS = 1.12
ET = "America/New_York"

def load(path):
    df = pd.read_csv(path, usecols=['timestamp', 'open', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, format='ISO8601')
    df['et'] = df['timestamp'].dt.tz_convert(ET)
    df['day'] = df['et'].dt.date
    df['hm'] = df['et'].dt.strftime('%H:%M')
    return df

def build_trades(df):
    """Per-day session table -> V1/V2 trade lists (pnl in gross points)."""
    need = {'10:00', '15:00', '15:30', '15:31', '16:00'}
    px = {}
    for day, g in df.groupby('day'):
        bars = g.set_index('hm')
        rec = {}
        for hm in need:
            if hm in bars.index:
                row = bars.loc[hm]
                rec[hm + '_close'] = row['close'] if np.ndim(row['close']) == 0 else row['close'].iloc[0]
                rec[hm + '_open'] = row['open'] if np.ndim(row['open']) == 0 else row['open'].iloc[0]
        # session close for next day's prev_close: bar labeled 16:00, fallback last bar <= 16:00
        rth = g[g['hm'] <= '16:00']
        rec['sess_close'] = rth['close'].iloc[-1] if len(rth) else np.nan
        px[day] = rec
    days = sorted(px)
    v1, v2 = [], []
    prev_close = np.nan
    for d in days:
        r = px[d]
        ok = all(k in r for k in
                 ['10:00_close', '15:00_close', '15:30_close', '15:31_open', '16:00_close'])
        if ok and not np.isnan(prev_close):
            r1 = r['10:00_close'] - prev_close
            r12 = r['15:30_close'] - r['15:00_close']
            entry, exitp = r['15:31_open'], r['16:00_close']
            if r1 != 0:
                d1 = 1 if r1 > 0 else -1
                pnl = d1 * (exitp - entry)
                v1.append({'day': d, 'dir': d1, 'pnl_pts': pnl})
                if r12 != 0 and (r12 > 0) == (r1 > 0):
                    v2.append({'day': d, 'dir': d1, 'pnl_pts': pnl})
        if not np.isnan(r.get('sess_close', np.nan)):
            prev_close = r['sess_close']
    return pd.DataFrame(v1), pd.DataFrame(v2)

def gate0(name, tdf, min_n, gating=True):
    print(f"\n{'='*66}\n{name}\n{'='*66}")
    n = len(tdf)
    if n == 0:
        print("  ZERO trades — INCONCLUSIVE")
        return
    gross = tdf['pnl_pts'].values
    net = gross - COST_PTS
    wr_g = (gross > 0).mean() * 100
    wr_n = (net > 0).mean() * 100
    gpf = gross[gross > 0].sum() / abs(gross[gross < 0].sum())
    npf = net[net > 0].sum() / abs(net[net < 0].sum())
    exp_usd = net.mean() * 2.0
    avg_gw = gross[gross > 0].mean()
    cost_frac = 1.12 / avg_gw * 100
    print(f"  N={n}  WR gross={wr_g:.1f}% / net={wr_n:.1f}%")
    print(f"  gross PF={gpf:.3f}   NET PF={npf:.3f}")
    print(f"  NET expectancy = ${exp_usd:+.2f}/contract/trade")
    print(f"  avg gross win={avg_gw:.2f} pts  cost fraction={cost_frac:.1f}% (gate <=25%)")
    print(f"  long/short split: {(tdf['dir']==1).sum()} L / {(tdf['dir']==-1).sum()} S")
    if not gating:
        print("  (secondary diagnostic — non-gating)")
        return
    checks = {f"N >= {min_n}": n >= min_n,
              "net PF >= 1.10": npf >= 1.10,
              "net expectancy > 0": exp_usd > 0,
              "cost <= 25% avg gross win": cost_frac <= 25.0}
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if n < min_n:
        print("  ==> INCONCLUSIVE (N below minimum)")
    else:
        print(f"  ==> {'GATE 0 PASS' if all(checks.values()) else 'GATE 0 FAIL'}")

print("Loading dev 2025...", flush=True)
dev = load(f"{BASE}/mnq_1min_2025.csv")
v1, v2 = build_trades(dev)
v1.to_csv("data/reports/mim_classic_gate0_v1_2025.csv", index=False)
v2.to_csv("data/reports/mim_classic_gate0_v2_2025.csv", index=False)
gate0("V1 — sign(r1), dev 2025", v1, 200)
gate0("V2 — agreement filter sign(r1)==sign(r12), dev 2025", v2, 100)

print("\nSecondary diagnostic (non-gating): 2023/2024 Sep–Nov", flush=True)
for f, lbl in [("mnq_1min_2023_sepnov.csv", "2023 Sep–Nov"),
               ("mnq_1min_2024_sepnov.csv", "2024 Sep–Nov")]:
    try:
        d = load(f"{BASE}/{f}")
        a, b = build_trades(d)
        gate0(f"V1 — {lbl}", a, 0, gating=False)
        gate0(f"V2 — {lbl}", b, 0, gating=False)
    except Exception as e:
        print(f"  {lbl}: unavailable ({e})")

print("\nDone. OOS 2026 NOT touched.", flush=True)
