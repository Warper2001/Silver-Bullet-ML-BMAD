"""COT positioning Gate 0 — E-mini Nasdaq-100 net non-commercial vs forward MNQ returns.

Pre-registration: _bmad-output/preregistration_cot_positioning.md (sealed first).

Alignment (the anti-look-ahead rule): a COT report dated Tuesday T is published the
FOLLOWING Friday ~15:30 ET, so it is knowable only from that Friday's close. Forward
return runs from the Monday AFTER publication to the Monday after that.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytz
from scipy import stats

REPO = "/root/Silver-Bullet-ML-BMAD"
PRICE_FILES = [
    f"{REPO}/data/mim_x/mnq_1min_2021_2024_frontmonth.csv",
    f"{REPO}/data/processed/dollar_bars/1_minute/mnq_1min_2025.csv",
    f"{REPO}/data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv",
]
CONTRACT = "209742"        # NASDAQ MINI - CME (E-mini NQ)
ZWIN = 104                 # trailing z-score window, weeks -- fixed in the seal
MNQ_PV = 2.0
RT_COST = 4.00
ET = pytz.timezone("US/Eastern")
N_FLOOR = 200


def fetch_cot() -> pd.DataFrame:
    url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
    args = [
        "curl", "-s", "--max-time", "90", "--get", url,
        "--data-urlencode", "$limit=5000",
        "--data-urlencode", f"$where=cftc_contract_market_code='{CONTRACT}'",
        "--data-urlencode",
        "$select=report_date_as_yyyy_mm_dd,noncomm_positions_long_all,noncomm_positions_short_all",
        "--data-urlencode", "$order=report_date_as_yyyy_mm_dd",
    ]
    raw = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    df = pd.DataFrame(json.loads(raw))
    df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.date
    for c in ("noncomm_positions_long_all", "noncomm_positions_short_all"):
        df[c] = pd.to_numeric(df[c])
    df["net_spec"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
    return df[["report_date", "net_spec"]].sort_values("report_date").reset_index(drop=True)


def load_weekly_closes() -> pd.Series:
    frames = []
    for f in PRICE_FILES:
        d = pd.read_csv(f, usecols=["timestamp", "close"], parse_dates=["timestamp"])
        ts = d["timestamp"]
        ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
        d["timestamp"] = ts.dt.tz_convert(ET)
        frames.append(d)
    px = pd.concat(frames).sort_values("timestamp").reset_index(drop=True)
    px["date"] = px["timestamp"].dt.date
    daily = px.groupby("date")["close"].last()
    return daily


def main() -> None:
    cot = fetch_cot()
    daily = load_weekly_closes()
    daily_idx = pd.to_datetime(pd.Series(list(daily.index)))
    print(f"COT reports: {len(cot)}  ({cot.report_date.min()} .. {cot.report_date.max()})")
    print(f"Price days:  {len(daily)}  ({min(daily.index)} .. {max(daily.index)})")

    # z-score the signal over the trailing 104 weeks (sealed), on the FULL COT history
    cot["z"] = (cot["net_spec"] - cot["net_spec"].rolling(ZWIN).mean()) / cot["net_spec"].rolling(ZWIN).std()

    def first_close_on_or_after(d: dt.date):
        later = daily_idx[daily_idx >= pd.Timestamp(d)]
        if later.empty:
            return None, None
        day = later.iloc[0].date()
        return day, daily.loc[day]

    rows = []
    for r in cot.itertuples():
        if not np.isfinite(r.z):
            continue
        # report dated Tuesday r.report_date -> published the following Friday
        pub_friday = r.report_date + dt.timedelta(days=(4 - r.report_date.weekday()) % 7 or 7)
        entry_monday = pub_friday + dt.timedelta(days=3)     # Monday after publication
        exit_monday = entry_monday + dt.timedelta(days=7)
        d0, p0 = first_close_on_or_after(entry_monday)
        d1, p1 = first_close_on_or_after(exit_monday)
        if p0 is None or p1 is None or d1 <= d0:
            continue
        rows.append({"report_date": r.report_date, "z": r.z, "entry": d0, "exit": d1,
                     "ret_pts": p1 - p0, "pnl_long": (p1 - p0) * MNQ_PV - RT_COST})

    df = pd.DataFrame(rows)
    print(f"\nAligned weekly observations: {len(df)}  ({df.entry.min()} .. {df.exit.max()})")

    rho, p = stats.spearmanr(df["z"], df["ret_pts"])
    print(f"\nSpearman(z, forward weekly return): rho={rho:+.4f}  p={p:.4f}   (pre-declared sign: NEGATIVE)")

    # decile spread, economic check
    df["dec"] = pd.qcut(df["z"], 10, labels=False, duplicates="drop")
    top = df[df["dec"] == df["dec"].max()]      # most net-long specs
    bot = df[df["dec"] == df["dec"].min()]      # most net-short specs
    spread = bot["pnl_long"].mean() - top["pnl_long"].mean()
    print(f"Most net-LONG decile  (n={len(top)}): mean ${top['pnl_long'].mean():+.2f}/wk")
    print(f"Most net-SHORT decile (n={len(bot)}): mean ${bot['pnl_long'].mean():+.2f}/wk")
    print(f"Contrarian decile spread: ${spread:+.2f}/wk   (needs > ${RT_COST:.2f})")

    baseline = df["pnl_long"].mean()
    print(f"\nUnconditional always-long baseline: ${baseline:+.2f}/wk over {len(df)} weeks")

    checks = {
        f"N >= {N_FLOOR} (got {len(df)})": len(df) >= N_FLOOR,
        f"sign is NEGATIVE (rho={rho:+.4f})": rho < 0,
        f"p < 0.05 (p={p:.4f})": p < 0.05,
        f"decile spread ${spread:+.2f} > ${RT_COST:.2f}": spread > RT_COST,
        f"net-short-decile leg ${bot['pnl_long'].mean():+.2f} beats baseline ${baseline:+.2f}": (
            bot["pnl_long"].mean() > baseline
        ),
    }
    print(f"\n{'='*62}\nGATE 0 — COT positioning\n{'='*62}")
    for c, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {c}")
    verdict = "PASS" if all(checks.values()) else "FAIL"
    print(f"\n  VERDICT: {verdict}")

    if verdict == "PASS":
        wk = bot["pnl_long"]
        print(f"\n  Combine screen: best week = {wk.max()/wk[wk>0].sum()*100:.1f}% of gross profit")


if __name__ == "__main__":
    main()
