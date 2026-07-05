"""PL combine-fit gate — simulate the Topstep 50K trailing-MLL math on the FROZEN
PL structural trade path (1 full 50oz contract), net of the MEASURED $34/RT slippage.

Combine rules (from repo MC scripts study_mim_noise_bands_gate2_mc.py / rerun_sigc_combined_mc.py):
  start balance $50,000; floor starts $48,000
  EOD ratchet: floor = min(50000, max(floor, EOD_balance - 2000))
  BUST if balance <= floor at any point (trailing MLL touched)
  PASS target: profit >= $3,000 AND biggest_day < 0.5*profit (consistency)
This is NOT a holdout test — it reuses the frozen in-sample trade list only to ask a
sizing/risk question: can 1 full PL contract survive the $2,000 trailing DD at all?"""
import pandas as pd

CSV = "data/reports/backtest_1year_20260626_025416.csv"
SLIP_RT = 34.0   # measured PLV26 all-in $/RT (Amendment 1 PASS, 2026-07-05)
START, FLOOR0, MLL, TARGET = 50000.0, 48000.0, 2000.0, 3000.0

df = pd.read_csv(CSV)
df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
df["day"] = df["exit_time"].dt.tz_convert("America/New_York").dt.date
df = df.sort_values("exit_time").reset_index(drop=True)

for label, cost in [("GROSS", 0.0), ("NET @ $34/RT slippage", SLIP_RT)]:
    df["p"] = df["pnl"] - cost
    # per-trade risk
    worst = df["p"].min()
    sl = df[df["exit_type"] == "sl"]["p"]
    print(f"\n===== {label} =====")
    print(f"  N={len(df)}  net P&L=${df['p'].sum():,.0f}  worst single trade=${worst:,.0f}  "
          f"median SL=${sl.median():,.0f}  worst SL=${sl.min():,.0f}")
    # single-trade vs limits
    print(f"  worst single-trade loss ${-worst:,.0f} vs $1,000 daily-halt: "
          f"{'EXCEEDS in one trade' if -worst > 1000 else 'within'};  vs $2,000 MLL: "
          f"{'EXCEEDS' if -worst > 2000 else 'within'}")

    # daily P&L
    daily = df.groupby("day")["p"].sum()
    print(f"  worst day=${daily.min():,.0f}   best day=${daily.max():,.0f}   "
          f"days with P&L<=-$1,000: {(daily<=-1000).sum()}")

    # trailing-MLL simulation (intraday touch on closed-trade balance; EOD ratchet)
    bal, floor = START, FLOOR0
    min_gap = 1e18   # smallest (balance - floor) reached
    busted, bust_day = False, None
    peak = START
    max_dd = 0.0
    for day, g in df.groupby("day"):
        for p in g["p"]:
            bal += p
            peak = max(peak, bal)
            max_dd = max(max_dd, peak - bal)
            min_gap = min(min_gap, bal - floor)
            if bal <= floor and not busted:
                busted, bust_day = True, day
        floor = min(START, max(floor, bal - MLL))   # EOD ratchet
    profit = bal - START
    biggest_day = daily.max()
    consistency_ok = biggest_day < 0.5 * profit if profit > 0 else False
    print(f"  trailing-MLL sim: final balance=${bal:,.0f}  max DD from HWM=${max_dd:,.0f}  "
          f"closest approach to floor=${min_gap:,.0f}")
    print(f"  BUST (touched trailing floor)?  {'YES on '+str(bust_day) if busted else 'no'}")
    print(f"  target: profit ${profit:,.0f} >= $3,000? {profit>=TARGET}   "
          f"consistency biggest-day ${biggest_day:,.0f} < 50% of profit (${0.5*profit:,.0f})? {consistency_ok}")
