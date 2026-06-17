"""
build_yank_topstep_constrained.py — transform the YANK seal trade series into a
Topstep-Combine-compliant series: force-flatten any position open at 15:10 CT
(using the real 1-min price), and drop entries in the 15:08-17:00 CT blocked
window. Globex/evening trades (17:00 CT onward) are kept (allowed by Topstep).

Output: data/reports/yank_topstep_constrained.csv (same schema as the seal file).
Faithful because YANK setups are independent events: removing/truncating some
trades does not alter the others (the only second-order effect is on the daily
DLL, which the joint MC re-applies anyway).
"""
import numpy as np
import pandas as pd
import pytz

M = "/root/Silver-Bullet-ML-BMAD"
CT = pytz.timezone("America/Chicago")
PV, CONTR, COMM = 2.0, 5, 4.0
FLAT_MIN = 15 * 60 + 10      # 15:10 CT
BLOCK_LO, BLOCK_HI = 15 * 60 + 8, 17 * 60   # no new entries 15:08-17:00 CT


def main():
    px = pd.concat([
        pd.read_csv(f"{M}/data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"),
        pd.read_csv(f"{M}/data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"),
    ], ignore_index=True)
    px["ts"] = pd.to_datetime(px["timestamp"], utc=True, format="ISO8601").dt.floor("min")
    price = dict(zip(px["ts"], px["close"]))

    def price_at(ts):
        t = ts.floor("min")
        for _ in range(15):
            if t in price:
                return price[t]
            t -= pd.Timedelta(minutes=1)
        return None

    yk = pd.read_csv(f"{M}/data/reports/backtest_1year_20260615_181838.csv")
    en = pd.to_datetime(yk["entry_time"], utc=True, format="ISO8601")
    ex = pd.to_datetime(yk["exit_time"], utc=True, format="ISO8601")
    en_ct = en.dt.tz_convert(CT)
    dirn = np.where(yk["direction"] == "LONG", 1, -1)

    rows = []
    n_flat = n_block = 0
    for i in range(len(yk)):
        ent, ext, ect = en.iloc[i], ex.iloc[i], en_ct.iloc[i]
        tod = ect.hour * 60 + ect.minute
        if BLOCK_LO <= tod < BLOCK_HI:
            n_block += 1
            continue
        flat = CT.localize(pd.Timestamp(ect.year, ect.month, ect.day, 15, 10)).tz_convert("UTC")
        pnl = yk["pnl"].iloc[i]
        exit_t, exit_p, etype = yk["exit_time"].iloc[i], yk["exit_price"].iloc[i], yk["exit_type"].iloc[i]
        if ent < flat < ext:
            fp = price_at(flat)
            if fp is not None:
                pnl = float(dirn[i] * (fp - yk["entry_price"].iloc[i]) * PV * CONTR - COMM)
                exit_t, exit_p, etype = flat.isoformat(), fp, "topstep_flat"
                n_flat += 1
        rows.append({
            "entry_time": yk["entry_time"].iloc[i], "exit_time": exit_t,
            "direction": yk["direction"].iloc[i], "entry_price": yk["entry_price"].iloc[i],
            "exit_price": exit_p, "exit_type": etype, "bars_held": yk["bars_held"].iloc[i],
            "pnl": round(pnl, 2),
        })
    out = pd.DataFrame(rows)
    out.to_csv(f"{M}/data/reports/yank_topstep_constrained.csv", index=False)
    print(f"seal trades {len(yk)} -> constrained {len(out)} "
          f"(force-flattened {n_flat}, blocked {n_block})")
    print(f"P&L 5ct: ${yk['pnl'].sum():,.0f} -> ${out['pnl'].sum():,.0f} "
          f"({(out['pnl'].sum()-yk['pnl'].sum())/yk['pnl'].sum()*100:+.0f}%)")


if __name__ == "__main__":
    main()
