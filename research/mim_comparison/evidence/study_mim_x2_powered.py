"""study_mim_x2_powered.py — MIM-X2, powered redo. Seal b890967.

Identical rule to MIM-X (JFE 142 (2021)); only N and data source differ.
Returns computed WITHIN a single contract's active window — no roll splice.
"""
import math, statistics
import pandas as pd

PV, FRIC_P, FRIC_S = 2.0, 3.00, 2.00
NOTIONAL, FEE_BPS = 59000.0, 0.51

s = pd.read_csv("data/mim_x/sessions.csv", parse_dates=["date"])
s["r_rod"] = s["p1530"] / s["p_prior"] - 1.0
s["r_lh"] = s["p1559"] / s["p1530"] - 1.0
s = s[s["r_rod"] != 0].copy()
s["dir"] = (s["r_rod"] > 0).map({True: 1, False: -1})
s["gross"] = s["dir"] * (s["p1559"] - s["p1530"]) * PV
s["net3"] = s["gross"] - FRIC_P
s["net2"] = s["gross"] - FRIC_S
n = len(s)

def stats(v):
    m, sd = statistics.mean(v), statistics.stdev(v)
    return m, sd, m / (sd / math.sqrt(len(v)))

print("=" * 86)
print("MIM-X2 — Market Intraday Momentum on MNQ, powered   |   seal b890967")
print("=" * 86)
print(f"  JFE 142 (2021) rule, unmodified. 1 MNQ @ ${PV}/pt, 15:30->15:59 ET.")
print(f"  N = {n} sessions   {s['date'].min().date()} -> {s['date'].max().date()}"
      f"   contracts = {s['contract'].nunique()}")
print(f"  SEALED POWER: min detectable @t=2 = $5.31/trade; 80% power = $7.44;"
      f" JFE range $5.43-$10.80")

for lbl, col in (("GROSS (no costs)", "gross"),
                 ("NET @ $3.00 RT  [PRIMARY]", "net3"),
                 ("NET @ $2.00 RT  [secondary]", "net2")):
    v = s[col].tolist(); m, sd, t = stats(v)
    gp = sum(x for x in v if x > 0); gl = -sum(x for x in v if x < 0)
    print(f"\n  {lbl}")
    print(f"    total = ${sum(v):>11,.2f}   mean = ${m:>7.2f}/trade   sd = ${sd:>6.2f}")
    print(f"    t = {t:>6.3f}   WR = {100*sum(1 for x in v if x>0)/n:5.1f}%"
          f"   PF = {gp/gl if gl else float('inf'):.4f}")

print("\n" + "=" * 86)
print("VERDICT (seal §5) — primary is mean NET at $3.00 RT")
print("=" * 86)
m, sd, t = stats(s["net3"].tolist())
print(f"  mean net = ${m:+.2f}/trade   t = {t:.3f}   N = {n}")
if m <= 0:
    print("\n  >>> FAILS <<<  Effect absent or negative on MNQ at deployable cost.")
    print("  Pre-committed: record, close, no variant search (§6).")
elif t < 2.0:
    print("\n  >>> POSITIVE, UNPROVEN <<<  (mean > 0, t < 2.0). No deployment (§5).")
else:
    print("\n  >>> SURVIVES <<<  Triggers §8. Still no deployment.")

print("\n" + "-" * 86)
print("SECONDARY (reported, never decision-bearing, §5)")
x, y = s["r_rod"].values, s["r_lh"].values
xm, ym = x.mean(), y.mean()
b = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
a = ym - b * xm
res = y - (a + b * x)
se = math.sqrt((res ** 2).sum() / (n - 2) / ((x - xm) ** 2).sum())
r2 = 1 - (res ** 2).sum() / ((y - ym) ** 2).sum()
print(f"  r_LH = a + b*r_ROD :  beta = {b:+.5f}   t(beta) = {b/se:+.3f}   R^2 = {r2:.6f}")
print(f"  (JFE predicts beta > 0)")
gm = statistics.mean(s["gross"])
print(f"  gross edge = ${gm:.2f}/trade = {10000*gm/NOTIONAL:+.3f} bps"
      f"   vs {FEE_BPS} bps friction -> ratio {10000*gm/NOTIONAL/FEE_BPS:+.2f}x (bar 3x)")

print("\n  Per-year (NOT decision-bearing, §5):")
s["yr"] = s["date"].dt.year
for yr, g in s.groupby("yr"):
    v = g["net3"].tolist()
    print(f"    {yr}: N={len(v):>4}  mean=${statistics.mean(v):>7.2f}  total=${sum(v):>9,.0f}")
s.to_csv("data/reports/mim_x2_trades.csv", index=False)
print("\n  ledger -> data/reports/mim_x2_trades.csv")
