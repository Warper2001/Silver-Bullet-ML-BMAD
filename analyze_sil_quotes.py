"""
SIL slippage measurement — frozen analysis (prereg addendum 3, 2026-06-12).

Evaluates data/quotes/sil_quote_capture.csv against the PRE-COMMITTED rule in
_bmad-output/precommit_sil_slippage_measurement_2026-06.md:

  Valid sample:        SILN26, bid & ask present, ask > bid, 09:30-15:55 ET
  Qualifying session:  >= 3,000 valid samples
  Minimum evidence:    >= 5 qualifying sessions
  PASS iff:            pooled median spread <= $8.74/RT
                       AND every qualifying session's median <= $10

PASS authorizes only the writing of a Gate 1 prereg for SI-GC 5m LONG.

Usage:
  .venv/bin/python analyze_sil_quotes.py
"""
from pathlib import Path

import pandas as pd

CSV = Path("data/quotes/sil_quote_capture.csv")
PV = 1000.0                 # $ per 1.00 SIL move
TICK = 0.005                # = $5/contract
PASS_MEDIAN_USD = 8.74      # frozen: slip_RT bound for WR 57.8% to clear
SESSION_GUARD_USD = 10.0    # frozen: no qualifying session median above this
MIN_SAMPLES = 3000
MIN_SESSIONS = 5
RTH = ("09:30", "15:55")


def main():
    df = pd.read_csv(CSV)
    if df.empty:
        print(f"INSUFFICIENT DATA — {CSV} has no samples yet. Keep capturing.")
        return
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["et"] = df["ts_utc"].dt.tz_convert("America/New_York")

    sil = df[(df["symbol"] == "SILN26")
             & df["bid"].notna() & df["ask"].notna()].copy()
    sil["spread_usd"] = (sil["ask"] - sil["bid"]) * PV
    sil = sil[sil["spread_usd"] > 0]
    sil = sil.set_index("et").between_time(*RTH)
    sil["session"] = sil.index.date

    per = sil.groupby("session")["spread_usd"].agg(["count", "median",
                                                    lambda s: s.quantile(0.75)])
    per.columns = ["n", "median_usd", "p75_usd"]
    qual = per[per["n"] >= MIN_SAMPLES]

    print("Per-session (SILN26, RTH 09:30-15:55 ET):")
    for s, r in per.iterrows():
        q = "qualifying" if r["n"] >= MIN_SAMPLES else f"NOT qualifying (<{MIN_SAMPLES})"
        print(f"  {s}  n={int(r['n']):>5}  median=${r['median_usd']:.2f}  "
              f"p75=${r['p75_usd']:.2f}  [{q}]")

    if len(qual) < MIN_SESSIONS:
        print(f"\nINSUFFICIENT DATA — {len(qual)} qualifying session(s), "
              f"need {MIN_SESSIONS}. Keep capturing.")
        return

    pooled = sil[sil["session"].isin(qual.index)]["spread_usd"]
    med = pooled.median()
    in_ticks = med / (TICK * PV)
    pct_1tick = (pooled <= TICK * PV + 1e-9).mean()
    sizes = sil[sil["session"].isin(qual.index)][["bid_size", "ask_size"]].median()

    print(f"\nPooled over {len(qual)} qualifying sessions "
          f"(n={len(pooled):,} valid samples):")
    print(f"  median spread: ${med:.2f}  ({in_ticks:.2f} ticks)")
    print(f"  p75: ${pooled.quantile(0.75):.2f}   p90: ${pooled.quantile(0.90):.2f}")
    print(f"  % samples at <=1 tick: {pct_1tick:.1%}")
    print(f"  median sizes: bid {sizes['bid_size']:.0f} × ask {sizes['ask_size']:.0f}")

    g_med = med <= PASS_MEDIAN_USD
    g_sess = (qual["median_usd"] <= SESSION_GUARD_USD).all()
    print(f"\n  {'✅' if g_med else '❌'} pooled median ${med:.2f} <= ${PASS_MEDIAN_USD}")
    print(f"  {'✅' if g_sess else '❌'} all session medians <= ${SESSION_GUARD_USD} "
          f"(worst: ${qual['median_usd'].max():.2f})")

    if g_med and g_sess:
        print("\n🟢 PASS — measured cost clears the frozen bar. Authorized next "
              "step: WRITE Gate 1 prereg for SI-GC 5m LONG using the measured "
              "spread as cost basis (deployment remains a separate decision).")
    else:
        print("\n🔴 FAIL — family closure (98cd4ad) is confirmed final.")

    # context only — not part of the rule
    si = df[(df["symbol"] == "SIN26") & df["bid"].notna() & df["ask"].notna()].copy()
    if len(si):
        si["spread_usd"] = (si["ask"] - si["bid"]) * 5000.0
        si = si[si["spread_usd"] > 0].set_index("et").between_time(*RTH)
        print(f"\n[context] SIN26 (full SI, $5,000/pt) median RTH spread: "
              f"${si['spread_usd'].median():.2f}")


if __name__ == "__main__":
    main()
