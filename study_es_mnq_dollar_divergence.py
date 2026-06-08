"""
ES/MNQ — Dollar-Meaningful Divergence Diagnostic
Computes spread in ABSOLUTE MNQ INDEX POINTS (not return z-score)
so we can see how often the divergence is large enough to trade.

Spread construction:
  - Use a rolling N-bar (5-min) cumulative MNQ move vs beta-predicted ES move
  - beta_price = rolling_cov(mnq_pts, es_pts) / rolling_var(es_pts)  over BETA_WIN bars
  - spread_pts = cumulative_mnq_pts - beta_price * cumulative_es_pts
  - Threshold scan: $10, $20, $30, $50/contract (= 5, 10, 15, 25 MNQ pts)

RTH: 09:30–16:00 ET | In-sample: 2025-05-01 → 2026-02-28
"""
import pandas as pd
import numpy as np
from pathlib import Path

MNQ_PATH   = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
MNQ_2026   = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
ES_PATH    = Path("data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv")

BETA_WIN   = 60    # bars for rolling beta (1-hour rolling)
SPREAD_WIN = 5     # bars for cumulative divergence measurement (5-min spread)
THRESHOLDS = [5, 10, 15, 25]   # MNQ pts = $10, $20, $30, $50/contract
STOP_MULT  = 2.0   # stop at 2× the entry divergence in the wrong direction
HOLD_MAX   = 30    # max bars to hold before forcing exit
MNQ_PV     = 2.0   # $/point
COMMISSION = 4.80  # round-trip $/contract (typical retail)
RTH_START  = "09:30"
RTH_END    = "15:55"   # leave 5-min buffer before close

def load_et(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()

print("Loading bars…")
mnq = pd.concat([load_et(MNQ_PATH), load_et(MNQ_2026)])
mnq = mnq[~mnq.index.duplicated(keep="first")]
es  = load_et(ES_PATH)

both = mnq[["close"]].rename(columns={"close":"mnq"}).join(
       es[["close"]].rename(columns={"close":"es"}), how="inner")
both = both["2025-05-01":"2026-02-28"]
rth  = both.between_time(RTH_START, RTH_END).copy()
print(f"  RTH bars: {len(rth):,}  across {rth.index.normalize().nunique()} days")

# ── rolling beta (price-space, MNQ pts vs ES pts) ─────────────────────────
rth["mnq_chg"] = rth["mnq"].diff()
rth["es_chg"]  = rth["es"].diff()
roll_cov = rth["mnq_chg"].rolling(BETA_WIN).cov(rth["es_chg"])
roll_var = rth["es_chg"].rolling(BETA_WIN).var()
rth["beta"] = (roll_cov / roll_var.replace(0, np.nan)).fillna(method="ffill").clip(0, 10)

# ── N-bar cumulative spread (divergence over SPREAD_WIN bars) ───────────────
rth["mnq_cum"] = rth["mnq_chg"].rolling(SPREAD_WIN).sum()
rth["es_cum"]  = rth["es_chg"].rolling(SPREAD_WIN).sum()
rth["spread_pts"] = rth["mnq_cum"] - rth["beta"] * rth["es_cum"]
rth = rth.dropna(subset=["spread_pts", "beta"])
rth["date"]    = rth.index.date

print(f"\n{'='*70}")
print(f"THRESHOLD SCAN — divergence in {SPREAD_WIN}-bar cumulative MNQ points")
print(f"Beta window: {BETA_WIN} bars | Hold max: {HOLD_MAX} bars | Stop: {STOP_MULT}× entry divergence")
print(f"Commission: ${COMMISSION}/contract round-trip | MNQ pt value: ${MNQ_PV}")
print(f"{'='*70}")
print(f"\n  {'Thresh':>8}  {'$/contract':>12}  {'Freq/day':>9}  {'N':>6}  {'WR':>7}  {'Net $/trade':>12}  {'Worst-mo WR':>12}")
print(f"  {'-------':>8}  {'----------':>12}  {'--------':>9}  {'----':>6}  {'---':>7}  {'-----------':>12}  {'-----------':>12}")

z_arr  = rth["spread_pts"].values
idx    = rth.index
pos_map = {ts: i for i, ts in enumerate(idx)}
dates  = rth["date"].values
n_days = rth["date"].nunique()

for thresh_pts in THRESHOLDS:
    thresh_usd = thresh_pts * MNQ_PV
    triggers = rth[rth["spread_pts"].abs() > thresh_pts].index
    n_trig = len(triggers)
    freq = n_trig / n_days

    wins = 0
    total_net_pnl = 0.0
    by_month = {}

    for ts in triggers:
        i = pos_map[ts]
        entry_spread = z_arr[i]
        fade_dir = -np.sign(entry_spread)  # fade: bet spread returns to 0
        stop_level = entry_spread - fade_dir * abs(entry_spread) * STOP_MULT

        win = False
        exit_spread = entry_spread
        for j in range(i+1, min(i+HOLD_MAX+1, len(z_arr))):
            s_j = z_arr[j]
            # win: spread crosses zero toward fade_dir
            if fade_dir > 0 and s_j >= 0:
                win = True; exit_spread = s_j; break
            if fade_dir < 0 and s_j <= 0:
                win = True; exit_spread = s_j; break
            # stop: spread went to 2× in wrong direction
            if fade_dir > 0 and s_j < stop_level:
                exit_spread = s_j; break
            if fade_dir < 0 and s_j > stop_level:
                exit_spread = s_j; break

        # p&l: MNQ pts moved in our direction × $2/pt - commission
        pnl_pts = (entry_spread - exit_spread) * fade_dir   # positive = profit
        net_pnl = pnl_pts * MNQ_PV - COMMISSION

        wins += int(win)
        total_net_pnl += net_pnl

        m = idx[i].to_period("M")
        if m not in by_month:
            by_month[m] = [0, 0]
        by_month[m][0 if win else 1] += 1

    n_trades = len(triggers)
    wr = wins / n_trades if n_trades else 0
    avg_net = total_net_pnl / n_trades if n_trades else 0
    worst_wr = min((w/(w+l) if (w+l) else 0) for w, l in by_month.values()) if by_month else 0

    print(f"  {thresh_pts:>5} pts  ${thresh_usd:>9.0f}  {freq:>9.1f}/d  {n_trades:>6}  {wr:>7.1%}  ${avg_net:>10.2f}  {worst_wr:>11.1%}")

# ── deep dive at most promising threshold ────────────────────────────────────
FOCUS_THRESH = 15  # $30/contract — large enough to survive commission
print(f"\n{'='*70}")
print(f"DEEP DIVE — {FOCUS_THRESH} pts (${FOCUS_THRESH*MNQ_PV:.0f}/contract) threshold")
print(f"{'='*70}")

triggers = rth[rth["spread_pts"].abs() > FOCUS_THRESH].index
by_month_dd = {}
pnl_list = []

for ts in triggers:
    i = pos_map[ts]
    entry_spread = z_arr[i]
    fade_dir = -np.sign(entry_spread)
    stop_level = entry_spread - fade_dir * abs(entry_spread) * STOP_MULT

    win = False
    exit_spread = entry_spread
    bars_held = HOLD_MAX
    for j in range(i+1, min(i+HOLD_MAX+1, len(z_arr))):
        s_j = z_arr[j]
        if fade_dir > 0 and s_j >= 0:
            win = True; exit_spread = s_j; bars_held = j - i; break
        if fade_dir < 0 and s_j <= 0:
            win = True; exit_spread = s_j; bars_held = j - i; break
        if fade_dir > 0 and s_j < stop_level:
            exit_spread = s_j; bars_held = j - i; break
        if fade_dir < 0 and s_j > stop_level:
            exit_spread = s_j; bars_held = j - i; break

    pnl_pts = (entry_spread - exit_spread) * fade_dir
    net_pnl = pnl_pts * MNQ_PV - COMMISSION
    pnl_list.append(net_pnl)

    m = idx[i].to_period("M")
    if m not in by_month_dd:
        by_month_dd[m] = {"wins": 0, "losses": 0, "pnl": 0.0, "n": 0}
    by_month_dd[m]["wins" if win else "losses"] += 1
    by_month_dd[m]["pnl"] += net_pnl
    by_month_dd[m]["n"] += 1

pnl_arr = np.array(pnl_list)
total = len(pnl_arr)
wr = np.sum(pnl_arr > 0) / total if total else 0
avg_pnl = pnl_arr.mean() if total else 0
median_pnl = np.median(pnl_arr) if total else 0

print(f"\n  Trades:        {total}  ({total/n_days:.1f}/day)")
print(f"  Win rate:      {wr:.1%}")
print(f"  Avg net P&L:   ${avg_pnl:.2f}/contract")
print(f"  Median net P&L: ${median_pnl:.2f}/contract")
print(f"  Total P&L est: ${pnl_arr.sum():.0f}  (1 contract, {n_days} days)")

print(f"\n  By month:")
print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}  {'Avg P&L':>10}  {'freq/d':>8}")
for m in sorted(by_month_dd):
    d = by_month_dd[m]
    n = d["n"]
    mwr = d["wins"]/n if n else 0
    avg = d["pnl"]/n if n else 0
    est_days = n / (n_days / len(by_month_dd))
    freq_est = n / max(1, round(est_days))
    flag = "❌" if mwr < 0.35 else "✅"
    print(f"  {str(m):<10}  {n:>5}  {mwr:>7.1%}  ${avg:>8.2f}  {freq_est:>7.1f}/d  {flag}")

print(f"\n{'='*70}")
print(f"SUMMARY — is ES/MNQ divergence structurally viable?")
print(f"{'='*70}")
viable = (total/n_days >= 1.0) and (wr >= 0.50) and (avg_pnl > 0)
print(f"  Freq ≥ 1.0/day:    {'✅' if total/n_days >= 1.0 else '❌'}  [{total/n_days:.1f}/day]")
print(f"  WR ≥ 50%:          {'✅' if wr >= 0.50 else '❌'}  [{wr:.1%}]")
print(f"  Avg net P&L > 0:   {'✅' if avg_pnl > 0 else '❌'}  [${avg_pnl:.2f}]")
print(f"\n  {'✅ Structurally viable — proceed to Gate 0 pre-registration study.' if viable else '❌ Not yet viable at this threshold — check grid above for better level.'}")
