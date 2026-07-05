"""
PL (platinum) slippage measurement — Amendment 1 analysis (sealed 2026-07-05).

Identical to the frozen analyze_pl_quotes.py except the binding symbol is
PLV26 (Oct full-size platinum) per _bmad-output/precommit_pl_slippage_amendment1_plv26.md:
the parent's binding PLN26 (July) died into delivery mid-capture and returned
stale ballooning quotes ($45→$540 spread, 9→108 ticks, never mean-reverting), an
invalid measurement (same dead-front artifact as copper's MHGN26). All thresholds
inherited unchanged from the 2026-06-26 parent seal; the sample-count roll clause
is removed (it cannot detect a dead contract that keeps answering the quote
endpoint — PLN26 held sample parity while dead).

  Valid sample:        PLV26, bid & ask present, ask > bid, 09:30-15:55 ET
  Qualifying session:  >= 3,000 valid samples
  Minimum evidence:    >= 5 qualifying sessions
  All-in cost:         c = pooled_median_spread + $4.00 commission
  PASS iff:            c <= $41.71/RT (netPF>=1.10) AND every qualifying session's
                       all-in median <= $62.02/RT (pure-breakeven guard)

PASS authorizes only WRITING a Gate-1 prereg for PL on its sealed holdout.
Combine-fit (full 50oz notional ~$78K vs 50K account) is a separate downstream gate.

Usage:
  .venv/bin/python analyze_pl_quotes_amendment1.py
"""
from pathlib import Path

import pandas as pd

CSV = Path("data/quotes/pl_quote_capture.csv")
PV = 50.0                   # $ per 1.00 platinum move, 1 full contract (50 oz)
TICK = 0.10                 # = $5.00/contract
COMMISSION_USD = 4.00       # frozen: assumed full-contract commission+fees /RT
PASS_ALLIN_USD = 41.71      # frozen: all-in RT cost ceiling (netPF>=1.10)
SESSION_GUARD_USD = 62.02   # frozen: pure-breakeven guard (netPF=1.0)
MIN_SAMPLES = 3000
MIN_SESSIONS = 5
RTH = ("09:30", "15:55")
BINDING = "PLV26"           # Amendment 1: Oct front; July (PLN26) died mid-capture
OTHER = "PLN26"             # non-binding context (the dead front)


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

    sil = session_table(df, BINDING)
    print(f"Binding symbol (Amendment 1): {BINDING}")

    per = sil.groupby("session")["spread_usd"].agg(["count", "median",
                                                    lambda s: s.quantile(0.75)])
    per.columns = ["n", "median_usd", "p75_usd"]
    qual = per[per["n"] >= MIN_SAMPLES]

    print(f"Per-session ({BINDING}, RTH 09:30-15:55 ET):")
    for s, r in per.iterrows():
        q = "qualifying" if r["n"] >= MIN_SAMPLES else f"NOT qualifying (<{MIN_SAMPLES})"
        allin = r["median_usd"] + COMMISSION_USD
        print(f"  {s}  n={int(r['n']):>5}  spread_median=${r['median_usd']:.2f} "
              f"({r['median_usd']/(TICK*PV):.1f}t)  all-in=${allin:.2f}  p75=${r['p75_usd']:.2f}  [{q}]")

    if len(qual) < MIN_SESSIONS:
        print(f"\nINSUFFICIENT DATA — {len(qual)} qualifying session(s), "
              f"need {MIN_SESSIONS}. Keep capturing.")
        return

    pooled = sil[sil["session"].isin(qual.index)]["spread_usd"]
    med = pooled.median()
    allin = med + COMMISSION_USD
    in_ticks = med / (TICK * PV)
    pct_8t = (pooled <= 8 * TICK * PV + 1e-9).mean()
    sizes = sil[sil["session"].isin(qual.index)][["bid_size", "ask_size"]].median()

    print(f"\nPooled over {len(qual)} qualifying sessions (n={len(pooled):,} samples):")
    print(f"  median spread: ${med:.2f}  ({in_ticks:.2f} ticks)")
    print(f"  all-in (spread + ${COMMISSION_USD:.2f} comm): ${allin:.2f}/RT")
    print(f"  p75 spread: ${pooled.quantile(0.75):.2f}   p90: ${pooled.quantile(0.90):.2f}")
    print(f"  % samples <=8 ticks: {pct_8t:.1%}")
    print(f"  median sizes: bid {sizes['bid_size']:.0f} x ask {sizes['ask_size']:.0f}")

    sess_allin = qual["median_usd"] + COMMISSION_USD
    g_cost = allin <= PASS_ALLIN_USD
    g_sess = (sess_allin <= SESSION_GUARD_USD).all()
    print(f"\n  {'✅' if g_cost else '❌'} pooled all-in ${allin:.2f} <= ${PASS_ALLIN_USD}")
    print(f"  {'✅' if g_sess else '❌'} all session all-in medians <= ${SESSION_GUARD_USD} "
          f"(worst: ${sess_allin.max():.2f})")

    if g_cost and g_sess:
        print("\n🟢 PASS — measured platinum cost clears the frozen bar. Authorized "
              "next step: WRITE a Gate-1 prereg for PL on its sealed holdout using the "
              "MEASURED spread as cost basis (combine-fit + holdout remain separate gates).")
    else:
        print("\n🔴 FAIL — PL reclassified gross-only; no holdout spent. Logged as "
              "'orthogonal edge real but not cost-survivable at full-contract size.'")

    ot = session_table(df, OTHER)
    if len(ot):
        om = ot["spread_usd"].median()
        print(f"\n[context] {OTHER} (dead July front) median RTH spread: "
              f"${om:.2f} ({om/(TICK*PV):.1f} ticks) — non-binding")


if __name__ == "__main__":
    main()
