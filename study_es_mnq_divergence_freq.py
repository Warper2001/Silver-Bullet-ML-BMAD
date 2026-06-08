"""
ES/MNQ Spread Divergence — Gate 0 Frequency Check
Mary's 3-step diagnostic: frequency → restoration speed → crude WR sim

Spread = MNQ_ret - beta * ES_ret  (rolling 20-bar OLS beta)
Signal: |z_score| > 1.5σ from rolling 20-bar std of spread residuals
RTH only: 09:30–16:00 ET (same window as combine sessions)
Date range: 2025-05-01 → 2026-02-28 (ES data availability)
"""
import pandas as pd
import numpy as np
from pathlib import Path

MNQ_PATH = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
MNQ_2026  = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
ES_PATH   = Path("data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv")

ROLLWIN      = 20       # bars for rolling beta / z-score
ZSCORE_THRESH = 1.5     # divergence trigger
REST_THRESH   = 0.5     # reversion target (z returns within ±0.5σ)
REST_MAX_BARS = 30      # restoration timeout
STOP_Z        = 2.5     # crude stop: spread widens to 2.5σ in wrong direction
RTH_START     = "09:30"
RTH_END       = "16:00"
INSAMPLE_END  = pd.Timestamp("2026-02-28", tz="America/New_York")
ES_START      = pd.Timestamp("2025-05-01", tz="America/New_York")

# ── load ──────────────────────────────────────────────────────────────────────
def load_et(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    df = df.set_index("timestamp").sort_index()
    return df

print("Loading bars…")
mnq = pd.concat([load_et(MNQ_PATH), load_et(MNQ_2026)])
mnq = mnq[~mnq.index.duplicated(keep="first")]
es  = load_et(ES_PATH)

# ── align: restrict to overlap window ────────────────────────────────────────
start = max(ES_START, mnq.index[0].normalize())
end   = INSAMPLE_END
mnq = mnq[(mnq.index >= start) & (mnq.index <= end)]
es  = es [(es.index  >= start) & (es.index  <= end)]

# inner join on timestamp
both = mnq[["close"]].rename(columns={"close":"mnq"}).join(
        es[["close"]].rename(columns={"close":"es"}), how="inner")
print(f"  Aligned bars: {len(both):,}  ({both.index[0].date()} → {both.index[-1].date()})")

# ── RTH filter ────────────────────────────────────────────────────────────────
rth = both.between_time(RTH_START, RTH_END)
print(f"  RTH bars:     {len(rth):,}")

# ── returns + rolling beta + z-score ─────────────────────────────────────────
rth = rth.copy()
rth["mnq_ret"] = rth["mnq"].pct_change()
rth["es_ret"]  = rth["es"].pct_change()
rth = rth.dropna()

# rolling beta (cov/var) using ROLLWIN bars
roll_cov = rth["mnq_ret"].rolling(ROLLWIN).cov(rth["es_ret"])
roll_var = rth["es_ret"].rolling(ROLLWIN).var()
rth["beta"] = roll_cov / roll_var.replace(0, np.nan)

# spread residual
rth["spread"] = rth["mnq_ret"] - rth["beta"] * rth["es_ret"]

# rolling z-score of the spread
roll_mean = rth["spread"].rolling(ROLLWIN).mean()
roll_std  = rth["spread"].rolling(ROLLWIN).std()
rth["z"] = (rth["spread"] - roll_mean) / roll_std.replace(0, np.nan)
rth = rth.dropna(subset=["z", "beta"])

# ── STEP 1: frequency check ──────────────────────────────────────────────────
rth["trigger"] = rth["z"].abs() > ZSCORE_THRESH
rth["date"] = rth.index.date

daily_trig = rth.groupby("date")["trigger"].sum()
trading_days = len(daily_trig)
total_triggers = rth["trigger"].sum()
median_daily = daily_trig.median()
mean_daily   = daily_trig.mean()
pct_days_zero = (daily_trig == 0).mean() * 100

print(f"\n{'='*68}")
print(f"STEP 1 — FREQUENCY CHECK  (z > ±{ZSCORE_THRESH})")
print(f"{'='*68}")
print(f"  Trading days in sample:   {trading_days}")
print(f"  Total divergence triggers:{total_triggers}")
print(f"  Mean triggers/day:        {mean_daily:.1f}")
print(f"  Median triggers/day:      {median_daily:.1f}")
print(f"  Days with 0 triggers:     {(daily_trig == 0).sum()} ({pct_days_zero:.0f}%)")

freq_kill = median_daily < 1.5
print(f"\n  Kill condition (median < 1.5/day): {'❌ KILL' if freq_kill else '✅ PASS'}  [{median_daily:.1f}]")

# by-month
rth["month"] = rth.index.to_period("M")
monthly = rth.groupby("month").agg(
    trig_total=("trigger","sum"),
    trading_bars=("trigger","count"),
).reset_index()
monthly["trading_days_est"] = (monthly["trading_bars"] / (390)).round(0).astype(int)
monthly["trig_per_day"] = monthly["trig_total"] / monthly["trading_days_est"].replace(0, np.nan)

print(f"\n  By month:")
print(f"  {'Month':<10}  {'Triggers':>8}  {'Days':>5}  {'Trig/day':>9}")
for _, row in monthly.iterrows():
    print(f"  {str(row['month']):<10}  {row['trig_total']:>8.0f}  {row['trading_days_est']:>5.0f}  {row['trig_per_day']:>9.1f}")

# ── STEP 2: restoration speed ────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"STEP 2 — RESTORATION SPEED  (revert to |z| < {REST_THRESH} within {REST_MAX_BARS} bars)")
print(f"{'='*68}")

trigger_idx = rth.index[rth["trigger"]]
z_arr = rth["z"].values
idx_arr = rth.index

# build fast lookup: timestamp → positional index
pos_map = {ts: i for i, ts in enumerate(idx_arr)}

restoration_bars = []
timed_out = 0
for ts in trigger_idx:
    i = pos_map[ts]
    entry_dir = np.sign(z_arr[i])
    reverted = False
    for j in range(i+1, min(i+REST_MAX_BARS+1, len(z_arr))):
        if abs(z_arr[j]) < REST_THRESH:
            restoration_bars.append(j - i)
            reverted = True
            break
    if not reverted:
        timed_out += 1

if restoration_bars:
    median_rest = np.median(restoration_bars)
    p75_rest    = np.percentile(restoration_bars, 75)
    print(f"  Triggers evaluated:       {len(trigger_idx)}")
    print(f"  Timed out (>{REST_MAX_BARS} bars): {timed_out} ({timed_out/len(trigger_idx)*100:.0f}%)")
    print(f"  Median restoration:       {median_rest:.1f} bars")
    print(f"  75th-pct restoration:     {p75_rest:.1f} bars")
    rest_kill = median_rest > 15
    print(f"\n  Kill condition (median > 15 bars): {'❌ KILL' if rest_kill else '✅ PASS'}  [{median_rest:.1f}]")
else:
    print("  No restoration data — check data.")
    rest_kill = True

# ── STEP 3: crude WR simulation ───────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"STEP 3 — CRUDE WR SIMULATION  (enter at z>{ZSCORE_THRESH}, exit at |z|<{REST_THRESH} or stop at z>{STOP_Z})")
print(f"{'='*68}")

# For each trigger: direction = fade (if z>1.5 → short spread; if z<-1.5 → long spread)
# Win = z returns to <0.5 before z reaches 2.5 in same direction as entry
# We track what z does after entry
wins = 0
losses = 0
by_month_res = {}

for ts in trigger_idx:
    i = pos_map[ts]
    entry_z   = z_arr[i]
    fade_dir  = -np.sign(entry_z)   # fading: if z>0 we expect z to drop
    win = False
    m = rth.index[i].to_period("M")

    for j in range(i+1, min(i+REST_MAX_BARS+1, len(z_arr))):
        z_j = z_arr[j]
        # win: z crossed to opposite side of threshold
        if abs(z_j) < REST_THRESH:
            win = True
            break
        # stop: z widened past STOP_Z in same direction
        if fade_dir > 0 and z_j < -STOP_Z:
            break
        if fade_dir < 0 and z_j > STOP_Z:
            break
    # if timed out → loss

    if win:
        wins += 1
    else:
        losses += 1

    if m not in by_month_res:
        by_month_res[m] = [0, 0]
    by_month_res[m][0 if win else 1] += 1

total_trades = wins + losses
wr = wins / total_trades if total_trades else 0

print(f"  Total simulated trades:   {total_trades}")
print(f"  Wins / Losses:            {wins} / {losses}")
print(f"  Aggregate WR:             {wr:.1%}")

print(f"\n  By month:")
worst_wr = 1.0
print(f"  {'Month':<10}  {'N':>5}  {'WR':>7}")
for m in sorted(by_month_res):
    w, l = by_month_res[m]
    n = w + l
    mwr = w / n if n else 0
    worst_wr = min(worst_wr, mwr)
    flag = "❌" if mwr < 0.38 else "✅"
    print(f"  {str(m):<10}  {n:>5}  {mwr:>7.1%}  {flag}")

sim_kill_wr       = wr < 0.52
sim_kill_wm       = worst_wr < 0.38
print(f"\n  Kill condition WR < 52%:          {'❌ KILL' if sim_kill_wr else '✅ PASS'}  [{wr:.1%}]")
print(f"  Kill condition worst-month < 38%: {'❌ KILL' if sim_kill_wm else '✅ PASS'}  [{worst_wr:.1%}]")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"GATE 0 FREQUENCY DIAGNOSTIC — SUMMARY")
print(f"{'='*68}")
any_kill = freq_kill or rest_kill or sim_kill_wr or sim_kill_wm
print(f"  Step 1 — Frequency (median ≥ 1.5/day):   {'❌ KILL' if freq_kill  else '✅ PASS'}  [{median_daily:.1f}/day]")
print(f"  Step 2 — Restoration (median ≤ 15 bars):  {'❌ KILL' if rest_kill  else '✅ PASS'}  [{median_rest:.1f} bars]")
print(f"  Step 3 — WR ≥ 52%:                        {'❌ KILL' if sim_kill_wr else '✅ PASS'}  [{wr:.1%}]")
print(f"  Step 3 — Worst-month WR ≥ 38%:            {'❌ KILL' if sim_kill_wm else '✅ PASS'}  [{worst_wr:.1%}]")
print()
if any_kill:
    print("  ❌ CANDIDATE KILLED — one or more kill conditions triggered.")
else:
    print("  ✅ ALL CHECKS PASS — proceed to full strategy development.")
print(f"{'='*68}")
