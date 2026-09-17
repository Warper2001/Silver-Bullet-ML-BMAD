"""Fresh MIM-NB live-vs-sealed-engine parity replay (2026-09-18).

Why: the last parity check was `halt_review_mim_nb_parity_20260707.md` (2026-07-07).
It found a real bug (DLL_GUARD_USD -500 instead of the MC-validated -1000, amputating
post-cat-stop re-entries) and it was fixed the same day
(`preregistration_mim_nb_dll_parity_reversion.md`; confirmed still live:
`src/research/mim_nb_live.py:53` reads -1000.0). Nobody has re-run the replay since —
not after that fix, not after the 2026-09-13 MNQ Z26 roll, not over the ~2.5 months of
live trading since. This re-runs it against the current live record.

Engine: `run_catstop()` copied verbatim from `study_mim_nb_catstop.py` (sealed commit
6957daa; current on-disk sha256 210518d6...) — the exact function the deployment MC
ran, at S=250 (the live cat-stop since 2026-06-25). No re-entry day-cut in the engine
itself (matches the sealed spec: "flat; re-entry permitted at any subsequent HH:00/HH:30
check" -- the DLL guard is a live-only risk control layered on top, now parity-reverted
to -1000 which lets one re-entry through before it can bind).

Live-recorded bars: `data/mim_nb/bars_raw.csv` (the bot's own captured 1-min bars,
2026-06-11 -> present) -- NOT the `mnq_1min_*.csv` files (known contaminated in roll
weeks, see AGENTS.md pitfall). One malformed row (ts_utc 2026-11-26, a single bad bar)
is dropped; 1,782 duplicate-timestamp rows (reconnect re-sends) are deduped keeping the
first-received value, i.e. what the live bot actually saw when it acted.

Window: 2026-06-25 (start of the 250pt cat-stop era) -> latest complete RTH session.

Run: .venv/bin/python mim_parity_replay_refresh.py
"""
import sqlite3
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
LOOKBACK = 14
COST_PTS = 1.12
PT_VAL = 2.0
S = 250  # live cat-stop since 2026-06-25 ("S-A" era)
ERA_START = "2026-06-25"
BARS_RAW = Path("/root/Silver-Bullet-ML-BMAD/data/mim_nb/bars_raw.csv")
TRADES_DB = Path("/root/Silver-Bullet-ML-BMAD/data/trades.db")
WARMUP_SRC = Path(
    "/root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"
)
WARMUP_CUTOFF = "2026-06-11"  # bars_raw.csv starts here; same split as the 2026-07-07 review


def _sessionize(df: pd.DataFrame) -> pd.DataFrame:
    df["et"] = df["timestamp"].dt.tz_convert(ET)
    df["day"] = df["et"].dt.date
    df["hm"] = df["et"].dt.strftime("%H:%M")
    return df[(df["hm"] >= "09:31") & (df["hm"] <= "16:00")].sort_values("timestamp").copy()


def load_warmup() -> pd.DataFrame:
    """Jan-Jun 10 2026 sigma-warmup, from the same source the sealed engine trains on.

    Caveat: this file's Jan-Feb rows are the known deferred-contract defect (see AGENTS.md
    pitfall). Used ONLY to seed the 14-day sigma lookback, not for any trade whose P&L is
    counted (everything before 2026-06-25 is dropped from the final comparison) -- same
    scope the 2026-07-07 review used it for.
    """
    df = pd.read_csv(WARMUP_SRC, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df = df[df["timestamp"] < pd.Timestamp(WARMUP_CUTOFF, tz="UTC")]
    return _sessionize(df)


def load_live_bars() -> pd.DataFrame:
    df = pd.read_csv(BARS_RAW, usecols=["ts_utc", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601")
    df = df[df["timestamp"] <= pd.Timestamp.now(tz="UTC")]  # drop the one malformed future row
    df = df.drop_duplicates("ts_utc", keep="first")
    return _sessionize(df)


def run_catstop(df: pd.DataFrame, S: int) -> pd.DataFrame:
    """Verbatim from study_mim_nb_catstop.py (sealed 6957daa)."""
    hist = defaultdict(lambda: deque(maxlen=LOOKBACK))
    trades = []
    day_count = 0
    prev_close = np.nan
    check_marks = {f"{h:02d}:{m}" for h in range(10, 16) for m in ("00", "30")} | {"16:00"}
    entry_marks = {f"{h:02d}:{m}" for h in range(10, 16) for m in ("00", "30")} - {"16:00"}

    for day, g in df.groupby("day", sort=True):
        g = g.sort_values("et")
        hms = g["hm"].values
        opens, highs, lows = g["open"].values, g["high"].values, g["low"].values
        closes, vols = g["close"].values, g["volume"].values
        if hms[0] != "09:31" or "16:00" not in set(hms):
            continue
        O = opens[0]
        day_count += 1
        tradeable = day_count > LOOKBACK and not np.isnan(prev_close)
        gap_up_adj = max(O - prev_close, 0) if not np.isnan(prev_close) else 0.0
        gap_dn_adj = max(prev_close - O, 0) if not np.isnan(prev_close) else 0.0
        cum_pv = np.cumsum(closes * vols)
        cum_v = np.cumsum(vols)
        vwap = cum_pv / np.where(cum_v == 0, 1, cum_v)  # noqa: F841 (parity with source)

        pos, entry_px, entry_t, cat = 0, 0.0, None, np.nan
        pending = None

        for i, hm in enumerate(hms):
            if tradeable and pending is not None:
                action, why = pending
                px = opens[i]
                if action == "exit" and pos != 0:
                    trades.append({"day": day, "dir": pos, "reason": why,
                                    "pnl_pts": pos * (px - entry_px),
                                    "entry_t": entry_t, "exit_t": hms[i]})
                    pos = 0
                elif action in ("long", "short"):
                    new = 1 if action == "long" else -1
                    if pos != 0 and pos != new:
                        trades.append({"day": day, "dir": pos, "reason": "REVERSAL",
                                        "pnl_pts": pos * (px - entry_px),
                                        "entry_t": entry_t, "exit_t": hms[i]})
                        pos = 0
                    if pos == 0:
                        pos, entry_px, entry_t = new, px, hms[i]
                        cat = entry_px - S if pos == 1 else entry_px + S
                pending = None

            if pos == 1 and lows[i] <= cat:
                trades.append({"day": day, "dir": 1, "reason": "CAT_STOP",
                                "pnl_pts": cat - entry_px, "entry_t": entry_t, "exit_t": hms[i]})
                pos, pending = 0, None
            elif pos == -1 and highs[i] >= cat:
                trades.append({"day": day, "dir": -1, "reason": "CAT_STOP",
                                "pnl_pts": entry_px - cat, "entry_t": entry_t, "exit_t": hms[i]})
                pos, pending = 0, None

            if hm in check_marks:
                sig = hist[hm]
                if tradeable and len(sig) == LOOKBACK:
                    sigma = float(np.mean(sig))
                    ub = O * (1 + sigma) + gap_dn_adj
                    lb = O * (1 - sigma) - gap_up_adj
                    c = closes[i]
                    if pos == 1 and c < lb:
                        pending = ("exit", "STOP")
                    elif pos == -1 and c > ub:
                        pending = ("exit", "STOP")
                    if hm in entry_marks:
                        if c > ub and pos != 1:
                            pending = ("long", "BREAK_UP")
                        elif c < lb and pos != -1:
                            pending = ("short", "BREAK_DN")

            if hm == "16:00":
                if pos != 0:
                    trades.append({"day": day, "dir": pos, "reason": "EOD",
                                    "pnl_pts": pos * (closes[i] - entry_px),
                                    "entry_t": entry_t, "exit_t": "16:00"})
                    pos = 0
                pending = None

        for i, hm in enumerate(hms):
            hist[hm].append(abs(closes[i] / O - 1.0))
        prev_close = closes[-1]
    return pd.DataFrame(trades)


def load_live_trades() -> pd.DataFrame:
    con = sqlite3.connect(TRADES_DB)
    df = pd.read_sql(
        "SELECT * FROM trades WHERE trader_id='trader-mim-nb' AND write_mode='realtime'", con
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df.sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    warmup = load_warmup()
    live_bars = load_live_bars()
    bars = pd.concat([warmup, live_bars], ignore_index=True).sort_values("timestamp")
    print(f"warmup bars: {len(warmup):,} RTH rows ({warmup['day'].min()} -> {warmup['day'].max()})")
    print(f"live bars: {len(live_bars):,} RTH rows ({live_bars['day'].min()} -> {live_bars['day'].max()})")

    engine = run_catstop(bars, S)  # LOOKBACK warmup drawn from before the era
    engine = engine[engine["day"] >= pd.Timestamp(ERA_START).date()].copy()
    engine["pnl_net_usd"] = (engine["pnl_pts"] - COST_PTS) * PT_VAL
    engine["pnl_gross_usd"] = engine["pnl_pts"] * PT_VAL

    live = load_live_trades()
    live = live[live["timestamp"] >= pd.Timestamp(ERA_START, tz="UTC")].copy()
    live["day"] = live["timestamp"].dt.tz_convert(ET).dt.date

    print(f"\n=== Engine (S={S}, sealed run_catstop, {ERA_START} -> present) ===")
    print(f"N={len(engine)}  net=${engine['pnl_net_usd'].sum():,.2f}  "
          f"gross=${engine['pnl_gross_usd'].sum():,.2f}")
    print(engine.groupby("reason").size())

    gw = live.loc[live.pnl > 0, "pnl"].sum()
    gl = -live.loc[live.pnl < 0, "pnl"].sum()
    print(f"\n=== Live ({ERA_START} -> present) ===")
    print(f"N={len(live)}  net=${live['pnl'].sum():,.2f}  PF={gw/gl if gl else float('inf'):.3f}")
    print(live.groupby("exit_reason").size())

    print(f"\n=== Day-by-day diff ===")
    days = sorted(set(engine["day"]) | set(live["day"]))
    rows = []
    for d in days:
        e = engine[engine["day"] == d]
        l = live[live["day"] == d]
        e_net = e["pnl_net_usd"].sum() if len(e) else 0.0
        l_net = l["pnl"].sum() if len(l) else 0.0
        e_reasons = ",".join(e["reason"]) if len(e) else "-"
        l_reasons = ",".join(l["exit_reason"]) if len(l) else "-"
        diff = l_net - e_net
        rows.append({"day": d, "engine_n": len(e), "engine_net": round(e_net, 2),
                      "engine_reasons": e_reasons, "live_n": len(l), "live_net": round(l_net, 2),
                      "live_reasons": l_reasons, "diff_usd": round(diff, 2)})
    diffdf = pd.DataFrame(rows)
    diffdf.to_csv("day_diff.csv", index=False)
    big = diffdf[diffdf["diff_usd"].abs() >= 100].sort_values("diff_usd")
    print(big.to_string(index=False))
    print(f"\nTotal divergence (live - engine net): ${diffdf['diff_usd'].sum():,.2f}")
    print(f"Sum |divergence| across days: ${diffdf['diff_usd'].abs().sum():,.2f}")

    engine.to_csv("engine_trades.csv", index=False)
    live.to_csv("live_trades.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
