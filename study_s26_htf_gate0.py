"""
study_s26_htf_gate0.py
----------------------
Gate 0 (dev window 2025) for pre-registration s26-htf-cost-viability
(_bmad-output/preregistration_s26_htf.md, sealed commit 4d43200).

Variants:
  V1: 15m raw signal (no ML)
  V2: 15m + walk-forward ML (180-day rolling train, monthly retrain, thresh 0.62)
  V3: 1h raw signal (no ML)

Dev window: trades ENTERED 2025-01-01 -> 2025-12-31 UTC. 2026 OOS is NOT computed.
Cost model: $6.00 RT/contract = 60 BTC points deduction per trade (MBT $0.10/pt).

Gate 0 criteria (net of costs):
  15m: N>=100 ; 1h: N>=30
  Net PF >= 1.10 ; Net expectancy > $0/contract
  Cost fraction: $6 <= 25% of avg gross |win| per contract (avg gross win >= 240 pts)

Touches nothing live.
"""
import pandas as pd
import numpy as np
import pytz
from sklearn.ensemble import HistGradientBoostingClassifier

CSV = "/root/Silver-Bullet-ML-BMAD/data/kraken/PF_XBTUSD_1min.csv"
ET = pytz.timezone("America/New_York")

LENGTH, SL_MULT, TP_MULT, MAX_HOLD = 20, 2.0, 4.0, 60
COST_PTS = 60.0          # $6.00 RT / $0.10 per pt
THRESH = 0.62
DEV_START = pd.Timestamp("2025-01-01", tz="UTC")
DEV_END = pd.Timestamp("2026-01-01", tz="UTC")   # exclusive; entries only

print("Loading 1-min data...", flush=True)
m1 = pd.read_csv(CSV)
m1['timestamp'] = pd.to_datetime(m1['timestamp'])
m1 = m1.set_index('timestamp').sort_index()

def resample(df, rule):
    out = df.resample(rule, label='left', closed='left').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    return out.dropna(subset=['open'])

def build(df, sweep_win, recent_win):
    """Indicators + signals per sealed spec. Full-history warm-up."""
    df = df.copy()
    pc = df['close'].shift(1)
    tr = pd.concat([(df['high']-df['low']).abs(),
                    (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(LENGTH).mean()
    hh = df['high'].rolling(sweep_win).max()
    ll = df['low'].rolling(sweep_win).min()
    sweep_bear = df['high'] >= hh.shift(1)
    sweep_bull = df['low'] <= ll.shift(1)
    df['recent_sweep_bear'] = sweep_bear.astype(int).rolling(recent_win).max() > 0
    df['recent_sweep_bull'] = sweep_bull.astype(int).rolling(recent_win).max() > 0
    df['soft_fvg_bear'] = (df['low'].shift(2) - df['high']) > (0.2*df['atr'])
    df['soft_fvg_bull'] = (df['low'] - df['high'].shift(2)) > (0.2*df['atr'])
    df['sig_long'] = df['recent_sweep_bull'] & df['soft_fvg_bull']
    df['sig_long'] = df['sig_long'] & (~df['sig_long'].shift(1).fillna(False).astype(bool))
    df['sig_short'] = df['recent_sweep_bear'] & df['soft_fvg_bear']
    df['sig_short'] = df['sig_short'] & (~df['sig_short'].shift(1).fillna(False).astype(bool))
    # ML features (used by V2 only; harmless elsewhere)
    df['rvol'] = (df['volume']/df['volume'].rolling(50).mean().replace(0, np.nan)).fillna(1.0)
    df['dist_macro_ema'] = (df['close'] - df['close'].ewm(span=200).mean())/df['atr']
    df['dist_ema'] = (df['close'] - df['close'].ewm(span=LENGTH, adjust=False).mean())/df['atr']
    idx_et = df.index.tz_convert(ET)
    df['hour_et'] = idx_et.hour
    df['dow'] = idx_et.dayofweek
    df['is_us_session'] = (((df['hour_et'] == 9) & (idx_et.minute >= 30)) |
                           ((df['hour_et'] >= 10) & (df['hour_et'] < 16)) |
                           ((df['hour_et'] == 16) & (idx_et.minute == 0))).astype(int)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in ['atr', 'rvol', 'dist_ema', 'dist_macro_ema']:
        df[c] = df[c].fillna(0)
    return df

def replay(df, entry_lo, entry_hi, ml_model_for_month=None):
    """Sequential one-at-a-time replay. Entries only in [entry_lo, entry_hi).
    ml_model_for_month: dict {Period: model} or None for raw variant."""
    ts = df.index
    o, h, l, c, atr = (df[x].values for x in ['open', 'high', 'low', 'close', 'atr'])
    sigL, sigS = df['sig_long'].values, df['sig_short'].values
    feats = df[['atr', 'rvol', 'dist_ema', 'dist_macro_ema',
                'hour_et', 'dow', 'is_us_session']].values
    lo = ts.searchsorted(entry_lo)
    hi = ts.searchsorted(entry_hi)
    trades, active = [], None
    for i in range(lo, min(hi + MAX_HOLD + 1, len(ts))):
        if active:
            t = active
            t['hold'] += 1
            er, ep = None, 0.0
            if t['dir'] == 1:
                if l[i] <= t['sl']: er, ep = 'SL', t['sl']
                elif h[i] >= t['tp']: er, ep = 'TP', t['tp']
            else:
                if h[i] >= t['sl']: er, ep = 'SL', t['sl']
                elif l[i] <= t['tp']: er, ep = 'TP', t['tp']
            if not er and t['hold'] >= MAX_HOLD:
                er, ep = 'TIME_STOP', c[i]
            if er:
                pnl = (ep - t['entry']) if t['dir'] == 1 else (t['entry'] - ep)
                trades.append({'entry_time': t['ts'], 'exit_time': ts[i], 'dir': t['dir'],
                               'reason': er, 'pnl_pts': pnl, 'proba': t['proba']})
                active = None
                continue
        if not active and lo < i < hi:
            s = i - 1  # signal bar; enter this bar's open
            d = 1 if sigL[s] else (0 if sigS[s] else None)
            if d is None or np.isnan(atr[s]) or atr[s] == 0:
                continue
            p = np.nan
            if ml_model_for_month is not None:
                mdl = ml_model_for_month.get(ts[i].to_period('M'))
                if mdl is None:
                    continue
                p = mdl.predict_proba(np.array([[d] + list(feats[s])]))[0, 1]
                if p < THRESH:
                    continue
            e = o[i]
            active = {'dir': d, 'entry': e,
                      'sl': e - atr[s]*SL_MULT if d == 1 else e + atr[s]*SL_MULT,
                      'tp': e + atr[s]*TP_MULT if d == 1 else e - atr[s]*TP_MULT,
                      'hold': 0, 'proba': p, 'ts': ts[i]}
    return pd.DataFrame(trades)

def label_for_training(df, t_lo, t_hi):
    """Per-signal independent simulation for ML labels (entry = signal bar close),
    identical scheme to the live 1-min training script."""
    ts = df.index
    c, h, l, atr = (df[x].values for x in ['close', 'high', 'low', 'atr'])
    feats = df[['atr', 'rvol', 'dist_ema', 'dist_macro_ema',
                'hour_et', 'dow', 'is_us_session']].values
    rows = []
    for sig_col, d in (('sig_long', 1), ('sig_short', 0)):
        idxs = np.where(df[sig_col].values)[0]
        idxs = idxs[(ts[idxs] >= t_lo) & (ts[idxs] < t_hi)]
        for i in idxs:
            if i + 1 >= len(c) or np.isnan(atr[i]) or atr[i] == 0:
                continue
            e = c[i]
            sl = e - atr[i]*SL_MULT if d == 1 else e + atr[i]*SL_MULT
            tp = e + atr[i]*TP_MULT if d == 1 else e - atr[i]*TP_MULT
            win = 0
            for j in range(i+1, min(i+MAX_HOLD+1, len(c))):
                if d == 1:
                    if l[j] <= sl: break
                    if h[j] >= tp: win = 1; break
                else:
                    if h[j] >= sl: break
                    if l[j] <= tp: win = 1; break
            else:
                ep = c[min(i+MAX_HOLD, len(c)-1)]
                win = 1 if ((ep-e) if d == 1 else (e-ep)) > 0 else 0
            rows.append(([d] + list(feats[i]), win))
    return rows

def gate0(name, tdf, min_n):
    print(f"\n{'='*68}\n{name}\n{'='*68}")
    n = len(tdf)
    if n == 0:
        print("  ZERO trades — INCONCLUSIVE")
        return
    gross = tdf['pnl_pts'].values
    net = gross - COST_PTS
    wr = (gross > 0).mean()*100
    gpf = gross[gross > 0].sum()/abs(gross[gross < 0].sum()) if (gross < 0).any() else float('inf')
    npf = net[net > 0].sum()/abs(net[net < 0].sum()) if (net < 0).any() else float('inf')
    exp_usd = net.mean()*0.10
    avg_gross_win = gross[gross > 0].mean() if (gross > 0).any() else 0.0
    cost_frac = 60.0/avg_gross_win*100 if avg_gross_win > 0 else float('inf')
    print(f"  N={n}  WR(gross)={wr:.1f}%  gross PF={gpf:.3f}")
    print(f"  NET PF={npf:.3f}   NET expectancy=${exp_usd:+.2f}/ct/trade")
    print(f"  avg gross win={avg_gross_win:.0f} pts  cost fraction={cost_frac:.1f}% (gate: <=25%)")
    for r, g in tdf.groupby('reason'):
        print(f"    {r:<10} N={len(g):>4}  avg={g['pnl_pts'].mean():>+8.1f} pts")
    checks = {
        f"N >= {min_n}": n >= min_n,
        "net PF >= 1.10": npf >= 1.10,
        "net expectancy > 0": exp_usd > 0,
        "cost <= 25% of avg gross win": cost_frac <= 25.0,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if n < min_n:
        verdict = "INCONCLUSIVE (N below minimum)"
    else:
        verdict = "GATE 0 PASS" if all(checks.values()) else "GATE 0 FAIL"
    print(f"  ==> {verdict}")

# ---------------- 15m ----------------
print("Resampling to 15m / 1h...", flush=True)
b15 = build(resample(m1, '15min'), sweep_win=24, recent_win=4)
b60 = build(resample(m1, '60min'), sweep_win=6, recent_win=1)

print("Running V1 (15m raw)...", flush=True)
v1 = replay(b15, DEV_START, DEV_END)
v1.to_csv("data/reports/s26_htf_gate0_v1_15m_raw.csv", index=False)
gate0("V1 — 15m RAW (dev 2025)", v1, 100)

print("\nRunning V2 (15m + WF-ML, 180d rolling, monthly retrain)...", flush=True)
models = {}
for m in pd.period_range('2025-01', '2025-12', freq='M'):
    t_start = m.to_timestamp().tz_localize('UTC')
    rows = label_for_training(b15, t_start - pd.Timedelta(days=180), t_start)
    if len(rows) < 50:
        print(f"  {m}: only {len(rows)} training trades — month skipped (no model)", flush=True)
        continue
    X = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    mdl = HistGradientBoostingClassifier(max_iter=150, random_state=42)
    mdl.fit(X, y)
    models[m] = mdl
    print(f"  {m}: trained on {len(rows)} trades (base WR {y.mean()*100:.1f}%)", flush=True)
v2 = replay(b15, DEV_START, DEV_END, ml_model_for_month=models)
v2.to_csv("data/reports/s26_htf_gate0_v2_15m_ml.csv", index=False)
gate0("V2 — 15m + WALK-FORWARD ML (dev 2025)", v2, 100)

print("\nRunning V3 (1h raw)...", flush=True)
v3 = replay(b60, DEV_START, DEV_END)
v3.to_csv("data/reports/s26_htf_gate0_v3_1h_raw.csv", index=False)
gate0("V3 — 1h RAW (dev 2025)", v3, 30)

print("\nDone. OOS (2026) was NOT computed.", flush=True)
