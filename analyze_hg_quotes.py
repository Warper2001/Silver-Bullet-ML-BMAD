"""
HG (copper) slippage measurement — frozen analysis (precommit 2026-06-26).

Evaluates data/quotes/hg_quote_capture.csv against the PRE-COMMITTED rule in
_bmad-output/precommit_hg_slippage_measurement_2026-06.md:

  Binding symbol:      front-month micro copper MHGN26 (roll-switch to MHGU26
                       iff MHGN26 valid-sample count < 50% of MHGU26's).
  Valid sample:        binding symbol, bid & ask present, ask > bid, 09:30-15:55 ET
  Qualifying session:  >= 3,000 valid samples
  Minimum evidence:    >= 5 qualifying sessions
  All-in cost:         c = pooled_median_spread + COMMISSION ($1.50/RT)
  PASS iff:            c <= $4.63/RT  AND  every qualifying session's all-in
                       median <= $5.63/RT (pure-breakeven guard)

PASS authorizes only WRITING a Gate-1 prereg for HG on its sealed holdout.

Usage:
  .venv/bin/python analyze_hg_quotes.py
"""
from pathlib import Path

import pandas as pd

CSV = Path("data/quotes/hg_quote_capture.csv")
PV = 2500.0                 # $ per 1.00 copper move, 1 micro contract
TICK = 0.0005               # = $1.25/contract
COMMISSION_USD = 1.50       # frozen: assumed micro-copper commission+fees /RT
PASS_ALLIN_USD = 4.63       # frozen: all-in RT cost ceiling (net$/trade>=$2 binds)
SESSION_GUARD_USD = 5.63    # frozen: no qualifying session all-in median above this
MIN_SAMPLES = 3000
MIN_SESSIONS = 5
RTH = ("09:30", "15:55")
PRIMARY = "MHGN26"
FALLBACK = "MHGU26"


def session_table(df, symbol):
    s = df[(df["symbol"] == symbol) & df["bid"].notna() & df["ask"].notna()].copy()
    s["spread_usd"] = (s["ask"] - s["bid"]) * PV
    s = s[s["spread_usd"] > 0].set_index("et").between_time(*RTH)
    s["session"] = s.index.date
    return s


def main():
    if not CSV.exists():
        print(f"INSUFFICIENT DATA — {CSV} not created yet. Keep capturing.")
        return
    df = pd.read_csv(CSV)
    if df.empty:
        print(f"INSUFFICIENT DATA — {CSV} has no samples yet. Keep capturing.")
        return
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["et"] = df["ts_utc"].dt.tz_convert("America/New_York")

    prim = session_table(df, PRIMARY)
    fall = session_table(df, FALLBACK)

    # Roll clause: switch binding symbol iff primary valid-sample count < 50% of fallback
    n_prim, n_fall = len(prim), len(fall)
    binding = PRIMARY
    if n_fall > 0 and n_prim < 0.5 * n_fall:
        binding = FALLBACK
        print(f"[roll clause TRIGGERED] {PRIMARY} samples ({n_prim}) < 50% of "
              f"{FALLBACK} ({n_fall}) → binding switches to {FALLBACK}")
    sil = prim if binding == PRIMARY else fall

    per = sil.groupby("session")["spread_usd"].agg(["count", "median",
                                                    lambda s: s.quantile(0.75)])
    per.columns = ["n", "median_usd", "p75_usd"]
    qual = per[per["n"] >= MIN_SAMPLES]

    print(f"Binding symbol: {binding}")
    print(f"Per-session ({binding}, RTH 09:30-15:55 ET):")
    for s, r in per.iterrows():
        q = "qualifying" if r["n"] >= MIN_SAMPLES else f"NOT qualifying (<{MIN_SAMPLES})"
        allin = r["median_usd"] + COMMISSION_USD
        print(f"  {s}  n={int(r['n']):>5}  spread_median=${r['median_usd']:.2f}  "
              f"all-in=${allin:.2f}  p75=${r['p75_usd']:.2f}  [{q}]")

    if len(qual) < MIN_SESSIONS:
        print(f"\nINSUFFICIENT DATA — {len(qual)} qualifying session(s), "
              f"need {MIN_SESSIONS}. Keep capturing.")
        return

    pooled = sil[sil["session"].isin(qual.index)]["spread_usd"]
    med = pooled.median()
    allin = med + COMMISSION_USD
    in_ticks = med / (TICK * PV)
    pct_1t = (pooled <= TICK * PV + 1e-9).mean()
    pct_2t = (pooled <= 2 * TICK * PV + 1e-9).mean()
    sizes = sil[sil["session"].isin(qual.index)][["bid_size", "ask_size"]].median()

    print(f"\nPooled over {len(qual)} qualifying sessions (n={len(pooled):,} samples):")
    print(f"  median spread: ${med:.2f}  ({in_ticks:.2f} ticks)")
    print(f"  all-in (spread + ${COMMISSION_USD:.2f} comm): ${allin:.2f}/RT")
    print(f"  p75 spread: ${pooled.quantile(0.75):.2f}   p90: ${pooled.quantile(0.90):.2f}")
    print(f"  % samples <=1 tick: {pct_1t:.1%}   <=2 ticks: {pct_2t:.1%}")
    print(f"  median sizes: bid {sizes['bid_size']:.0f} x ask {sizes['ask_size']:.0f}")

    sess_allin = qual["median_usd"] + COMMISSION_USD
    g_cost = allin <= PASS_ALLIN_USD
    g_sess = (sess_allin <= SESSION_GUARD_USD).all()
    print(f"\n  {'✅' if g_cost else '❌'} pooled all-in ${allin:.2f} <= ${PASS_ALLIN_USD}")
    print(f"  {'✅' if g_sess else '❌'} all session all-in medians <= ${SESSION_GUARD_USD} "
          f"(worst: ${sess_allin.max():.2f})")

    if g_cost and g_sess:
        print("\n🟢 PASS — measured copper cost clears the frozen bar. Authorized "
              "next step: WRITE a Gate-1 prereg for HG on its sealed holdout, using "
              "the MEASURED spread as cost basis (holdout run + deployment remain "
              "separate gates).")
    else:
        print("\n🔴 FAIL — HG reclassified gross-only; no holdout spent. Logged as "
              "'structural edge real but not cost-survivable at micro size.'")

    if binding != FALLBACK and len(fall):
        fq = fall.groupby("session")["spread_usd"].median()
        print(f"\n[context] {FALLBACK} median RTH spread: ${fall['spread_usd'].median():.2f} "
              f"({fall['spread_usd'].median()/(TICK*PV):.2f} ticks)")


if __name__ == "__main__":
    main()
