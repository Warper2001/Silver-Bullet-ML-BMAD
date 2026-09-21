"""READ-ONLY comparison: MIM-NB live ledger vs the third-party 'edge compressed since 2025' report
(github.com/giovannibrusco/zarattini-2024-momentum-spy; single author, unreviewed, fetched 2026-09-21).
Reads data/mim_nb/trades.csv only (never writes to data/mim_nb). Descriptive; no hypothesis is tested against a threshold.

Report's own numbers (its README): full-period 2020-2026 Sharpe 1.11, +2.6 bp/trade, 41% win rate, payoff 1.69;
2020-2024 'Sharpe 1.4-2.0 every year'; 2025-2026 'below zero', 'recent Sharpe ~0' (no magnitudes, no counts).
Spec differences (so this is a like-for-like on TRADE STATISTICS only): report = SPY/ES, 2% vol targeting, VWAP/band trailing
stop; MIM-NB = MNQ x1, band stop + 250pt cat-stop, HH:00/HH:30 checks, reversals.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/root/Silver-Bullet-ML-BMAD")
OUT = Path(__file__).parent
t = pd.read_csv(REPO / "data/mim_nb/trades.csv")
for c in ("pnl_pts", "pnl_usd", "day_pnl_usd"):
    t[c] = t[c].astype(str).str.replace("+", "", regex=False).astype(float)
t["day"] = pd.to_datetime(t["day"])
t["bp"] = t["pnl_pts"] / t["entry_px"] * 1e4          # return per trade in bp of entry price (gross of fees: ledger has none)
rng = np.random.default_rng(20260921)

def stats(d: pd.DataFrame, label: str) -> dict:
    w, l = d[d.pnl_usd > 0], d[d.pnl_usd < 0]
    n = len(d)
    boot = rng.choice(d["bp"].to_numpy(), size=(20000, n)).mean(axis=1)
    return {"label": label, "N": n, "net_usd": d.pnl_usd.sum(), "PF": w.pnl_usd.sum() / -l.pnl_usd.sum(),
            "win_rate": len(w) / n, "payoff": w.pnl_usd.mean() / -l.pnl_usd.mean(),
            "mean_bp": d.bp.mean(), "sd_bp": d.bp.std(ddof=1), "median_bp": d.bp.median(),
            "mean_bp_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "t_mean_bp": d.bp.mean() / (d.bp.std(ddof=1) / np.sqrt(n)), "mean_usd": d.pnl_usd.mean()}

res = {"all": stats(t, "all ledger trades")}
res["ex_0915_roll_contaminated"] = stats(t[t.day != "2026-09-15"], "excluding 2026-09-15 (roll-contaminated spurious LONG)")
res["ex_top4_run_0729_0804"] = stats(t[~t.day.isin(pd.to_datetime(["2026-07-29","2026-07-30","2026-08-03","2026-08-04"]))],
                                     "excluding the 07-29..08-04 run (4 winning days)")
res["longs"] = stats(t[t.dir == 1], "long trades (descriptive only)")
res["shorts"] = stats(t[t.dir == -1], "short trades (descriptive only)")

# monthly
t["month"] = t.day.dt.strftime("%Y-%m")
res["monthly"] = t.groupby("month").agg(N=("pnl_usd", "size"), net_usd=("pnl_usd", "sum"),
                                       mean_bp=("bp", "mean")).round(2).reset_index().to_dict("records")

# daily-P&L Sharpe over ALL sessions in the window (zero days included; bot downtime days unknown -> counted as flat)
from pandas.tseries.holiday import USFederalHolidayCalendar
hol = USFederalHolidayCalendar().holidays("2026-06-01", "2026-09-30")
sessions = pd.bdate_range("2026-06-11", t.day.max()).difference(hol)
daily = t.groupby("day")["pnl_usd"].sum().reindex(sessions, fill_value=0.0)
notional = (t.entry_px.mean() * 2.0)                    # 1 MNQ notional, USD
r = daily / notional
sh = r.mean() / r.std(ddof=1) * np.sqrt(252)
bs = [ (lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252))(rng.choice(r.to_numpy(), size=len(r))) for _ in range(20000)]
res["daily_sharpe"] = {"sessions": int(len(sessions)), "trading_days": int((daily != 0).sum()), "sharpe_annualised": float(sh),
                       "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                       "note": "iid day bootstrap; zero days include any bot downtime; unlevered 1-contract notional"}
# what N could distinguish the report's +2.6 bp/trade from 0?
sd = res["all"]["sd_bp"]
res["resolution"] = {"report_mean_bp": 2.6, "ledger_sd_bp": sd, "d_report": 2.6 / sd,
                     "N_needed_80pct_one_sided": float(((1.645 + 0.842) / (2.6 / sd)) ** 2),
                     "MDE_bp_at_current_N": float((1.645 + 0.842) * sd / np.sqrt(res['all']['N']))}
# report's stats vs ledger
res["report_vs_ledger"] = {"report": {"mean_bp": 2.6, "win_rate": 0.41, "payoff": 1.69},
                           "ledger": {k: res["all"][k] for k in ("mean_bp", "win_rate", "payoff")}}
(OUT / "compare.json").write_text(json.dumps(res, indent=2, default=float))
print(json.dumps({k: v for k, v in res.items() if k not in ("monthly",)}, indent=1, default=lambda x: round(float(x), 4)))
print(pd.DataFrame(res["monthly"]).to_string(index=False))
