"""study_intraday_momentum_mnq.py — MIM-X cost screen.

Pre-registration: _bmad-output/preregistration_intraday_momentum_mnq.md (seal e865655)

Baltussen, Da, Lammers & Martens, JFE 142 (2021) 377-403:
  r_LH (last 30 min) is positively predicted by r_ROD (prior close -> last 30 min).
Rule: LONG if r_ROD > 0, SHORT if r_ROD < 0. Enter 15:30 ET close, exit 15:59 ET close.
Primary friction $3.00 RT. 1 MNQ, $2/point.
"""
import statistics, math
import pandas as pd, pytz

ET = pytz.timezone("US/Eastern")
PV, CONTRACTS = 2.0, 1
FRICTION_PRIMARY, FRICTION_SECONDARY = 3.00, 2.00
NOTIONAL_PER_CT = 59000.0
FEE_BPS_BENCH = 0.51


def load():
    frames = []
    for p in ("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv",
              "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"):
        df = pd.read_csv(p, parse_dates=["timestamp"])
        ts = df["timestamp"]
        ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
        df["timestamp"] = ts.dt.tz_convert(ET)
        frames.append(df.set_index("timestamp"))
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated()]


def build(df):
    d = df.copy()
    d["date"] = d.index.date
    d["hm"] = d.index.strftime("%H:%M")
    p1530 = d[d["hm"] == "15:30"].groupby("date")["close"].last()
    p1559 = d[d["hm"] == "15:59"].groupby("date")["close"].last()
    days = sorted(set(p1530.index) & set(p1559.index))
    rows = []
    for i, day in enumerate(days):
        if i == 0:
            continue
        prev = days[i - 1]
        if prev not in p1559.index:
            continue
        prior_close = p1559[prev]
        px1530, px1559 = p1530[day], p1559[day]
        r_rod = px1530 / prior_close - 1.0
        r_lh = px1559 / px1530 - 1.0
        if r_rod == 0:
            continue
        direction = 1 if r_rod > 0 else -1
        gross = direction * (px1559 - px1530) * PV * CONTRACTS
        rows.append({"date": day, "r_rod": r_rod, "r_lh": r_lh, "dir": direction,
                     "entry": px1530, "exit": px1559, "gross": gross,
                     "net3": gross - FRICTION_PRIMARY,
                     "net2": gross - FRICTION_SECONDARY})
    return pd.DataFrame(rows)


def tstat(v):
    n = len(v)
    sd = statistics.stdev(v)
    return statistics.mean(v) / (sd / math.sqrt(n)), sd


def main():
    t = build(load())
    n = len(t)
    print("=" * 84)
    print("MIM-X — Market Intraday Momentum, MNQ cost screen   |   seal e865655")
    print("=" * 84)
    print(f"  JFE 142 (2021) rule, unmodified. 1 MNQ @ ${PV}/pt. Enter 15:30 ET, exit 15:59 ET.")
    print(f"  Window: {t['date'].min()} -> {t['date'].max()}   N = {n} sessions")
    print(f"  (entirely outside the paper's 1974-2020 sample -- out-of-sample in time)")

    for label, col, fee in (("GROSS (no costs)", "gross", 0.0),
                            ("NET @ $3.00 RT  [PRIMARY]", "net3", FRICTION_PRIMARY),
                            ("NET @ $2.00 RT  [secondary]", "net2", FRICTION_SECONDARY)):
        v = t[col].tolist()
        ts_, sd = tstat(v)
        wr = 100 * sum(1 for x in v if x > 0) / n
        gp = sum(x for x in v if x > 0); gl = -sum(x for x in v if x < 0)
        print(f"\n  {label}")
        print(f"    total = ${sum(v):>10,.2f}   mean = ${statistics.mean(v):>7.2f}/trade"
              f"   sd = ${sd:>7.2f}")
        print(f"    t = {ts_:>6.3f}   win rate = {wr:5.1f}%   PF = {gp/gl if gl else float('inf'):.3f}")

    print("\n" + "=" * 84)
    print("VERDICT (seal §4) — primary is mean NET at $3.00 RT")
    print("=" * 84)
    v = t["net3"].tolist()
    m = statistics.mean(v); ts_, _ = tstat(v)
    print(f"  mean net @ $3.00 = ${m:+.2f}/trade   t = {ts_:.3f}   N = {n}")
    if m <= 0:
        print("\n  >>> FAILS COST SCREEN <<<")
        print("  The JFE effect does not survive MNQ friction at 1 contract.")
        print("  Pre-committed: record, close, no variant search (§5).")
    elif ts_ < 2.0:
        print("\n  >>> POSITIVE, UNPROVEN <<<  (mean > 0 but t < 2.0)")
        print("  Eligible for a prospective seal. No deployment (§4).")
    else:
        print("\n  >>> SURVIVES COST SCREEN <<<  Triggers §7. Still no deployment.")

    print("\n" + "-" * 84)
    print("SECONDARY (reported, never decision-bearing, §4)")
    # the paper's own predictive regression
    x = t["r_rod"].values; y = t["r_lh"].values
    xm, ym = x.mean(), y.mean()
    beta = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    alpha = ym - beta * xm
    resid = y - (alpha + beta * x)
    se = math.sqrt((resid ** 2).sum() / (n - 2) / ((x - xm) ** 2).sum())
    r2 = 1 - (resid ** 2).sum() / ((y - ym) ** 2).sum()
    print(f"  Predictive regression r_LH = a + b*r_ROD:")
    print(f"    beta = {beta:+.5f}   t(beta) = {beta/se:+.3f}   R^2 = {r2:.5f}")
    print(f"    (JFE predicts beta > 0)")
    gross_mean = statistics.mean(t["gross"])
    bps = 10000 * gross_mean / NOTIONAL_PER_CT
    print(f"\n  Friction screen (project standard):")
    print(f"    gross edge = ${gross_mean:.2f}/trade = {bps:.3f} bps of notional")
    print(f"    friction benchmark = {FEE_BPS_BENCH} bps   ratio = {bps/FEE_BPS_BENCH:.2f}x  (bar: 3x)")
    print(f"\n  Frequency: 1 trade/session, {n} sessions -> meets the >=1/day objective.")
    t.to_csv("data/reports/intraday_momentum_mnq_trades.csv", index=False)
    print("\n  per-trade ledger -> data/reports/intraday_momentum_mnq_trades.csv")


if __name__ == "__main__":
    main()
