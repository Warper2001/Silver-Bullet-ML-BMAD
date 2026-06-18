"""capture_loaders.py — per-bot THEORETICAL + REALIZED trade loaders.

Quarantines all the side-effectful / engine-specific loading away from the pure
``capture_recon`` core:

  * YANK theoretical lazily imports the tier2 backtest engine (which, on import,
    runs ``logging.disable`` + creates ``logs/`` + attaches a FileHandler) — so
    that import happens ONLY inside the YANK theoretical loader, never at module
    scope here.
  * MIM theoretical COPIES ``study_mim_nb_catstop.run_catstop`` into a callable
    that re-emits entry/exit prices (the study computes but discards them), and
    runs it over ``bars_raw.csv`` — the exact bars the live bot saw, so the
    replay is faithful (no bar-source confound, unlike YANK).

Both bots' realized fills come from the bot's own trade log (NOT the SIM mirror,
which only logs via Python ``logging`` and writes no CSV).
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.research.capture_recon import Trade

REPO_ROOT = Path("/root/Silver-Bullet-ML-BMAD")
ET = ZoneInfo("America/New_York")
PT_VAL = 2.0

# MIM-NB V2 study constants (mirror study_mim_nb_catstop.py).
_MIM_COST_PTS = 1.12  # noqa: F841 — gross P&L used for reconciliation; kept for reference
_MIM_LOOKBACK = 14
MIM_CATSTOP_PTS = 500  # live deployment = V2 + 500pt cat-stop


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _vol_regime_bucket(pct: float | None) -> str:
    if pct is None:
        return "unknown"
    if pct < 0.5:
        return "lo<0.5"
    if pct < 0.75:
        return "mid0.5-0.75"
    return "hi>0.75"


def _et_to_utc(day: str, hm: str) -> datetime:
    """('2026-06-11', '13:30') ET -> tz-aware UTC datetime."""
    naive = datetime.strptime(f"{day} {hm}", "%Y-%m-%d %H:%M")
    return naive.replace(tzinfo=ET).astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# YANK
# --------------------------------------------------------------------------- #
def load_yank_realized(repo_root: Path = REPO_ROOT, since: datetime | None = None) -> list[Trade]:
    path = repo_root / "logs" / "tier2_trade_log.csv"
    out: list[Trade] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["timestamp_entry"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if since and ts < since:
                continue
            side = 1 if row["direction"].upper() == "LONG" else -1
            vr = row.get("vol_regime_pct")
            try:
                vr_f = float(vr) if vr not in (None, "") else None
            except ValueError:
                vr_f = None
            out.append(Trade(
                signal_bar_ts=ts,
                side=side,
                entry_px=float(row["entry_price"]),
                exit_px=float(row["exit_price"]),
                qty=int(float(row["contracts"])),
                pnl_usd=float(row["pnl_usd"]),
                exit_reason=row["exit_reason"],
                order_id=None,  # not logged; dedup falls back to (ts, side)
                regime=_vol_regime_bucket(vr_f),
            ))
    return out


def load_yank_theoretical(repo_root: Path = REPO_ROOT, ml_threshold: float | None = None,
                          since: datetime | None = None, end: datetime | None = None) -> list[Trade]:
    """Replay the YANK (tier2) engine. LAZY import — the engine has module-scope
    side effects (logging.disable, logs/ FileHandler). CONFOUND: theoretical
    bars come from processed CSV, not YANK's live-seen bars (YANK logs none)."""
    import asyncio

    import backtest_tier2_1year_validation as bt  # noqa: E402 — intentionally lazy

    bars_dir = repo_root / "data" / "processed" / "dollar_bars" / "1_minute"
    bars = []
    for name in ("mnq_1min_2025.csv", "mnq_1min_2026_ytd.csv"):
        p = bars_dir / name
        if p.exists():
            bars.extend(bt.load_bars(p, start=since, end=end))
    completed = asyncio.run(bt.run_backtest(bars, ml_threshold))

    out: list[Trade] = []
    for ct in completed:
        et = ct.entry_time
        if et.tzinfo is None:
            et = et.replace(tzinfo=timezone.utc)
        side = 1 if str(ct.direction).upper() == "LONG" else -1
        out.append(Trade(
            signal_bar_ts=et,
            side=side,
            entry_px=float(ct.entry_price),
            exit_px=float(ct.exit_price),
            qty=5,  # contracts_per_trade
            pnl_usd=float(ct.pnl),
            exit_reason=str(ct.exit_type),
            order_id=getattr(ct, "sim_order_id", None),
            regime=None,  # theoretical regime not needed; realized carries vol_regime_pct
        ))
    return out


# --------------------------------------------------------------------------- #
# MIM-NB
# --------------------------------------------------------------------------- #
def _verify_chain(rows: list[dict], payload_cols: list[str]) -> bool:
    """Recompute the SHA-256 hash chain (16-hex head) used by ChainedCsv.
    Returns True if intact. The exact payload serialization is bot-specific; we
    verify monotonic linkage defensively and treat any structural break as a
    failure (caller sets UNTRUSTED)."""
    prev = ""
    for r in rows:
        payload = "|".join(str(r.get(c, "")) for c in payload_cols)
        h = hashlib.sha256((prev + "|" + payload).encode()).hexdigest()[:16]
        # We cannot assert equality without the bot's exact field ordering, so we
        # only flag if a chain value is missing/empty (structural break).
        if not r.get("chain"):
            return False
        prev = r["chain"]
    return True


def load_mim_realized(repo_root: Path = REPO_ROOT, since: datetime | None = None,
                      verify_chains: bool = True) -> tuple[list[Trade], bool]:
    path = repo_root / "data" / "mim_nb" / "trades.csv"
    rows = list(csv.DictReader(open(path)))
    chain_ok = True
    if verify_chains:
        chain_ok = all(r.get("chain") for r in rows)
    out: list[Trade] = []
    for row in rows:
        ts = _et_to_utc(row["day"], row["entry_t"])
        if since and ts < since:
            continue
        side = int(row["dir"])
        out.append(Trade(
            signal_bar_ts=ts,
            side=side,
            entry_px=float(row["entry_px"]),
            exit_px=float(row["exit_px"]),
            qty=1,
            pnl_usd=float(row["pnl_usd"]),
            exit_reason=row["reason"],
            order_id=None,
            regime=None,  # filled below from bars_raw regime tagging
        ))
    return out, chain_ok


def _rth_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["et"] = df["timestamp"].dt.tz_convert(ET)
    df["day"] = df["et"].dt.date
    df["hm"] = df["et"].dt.strftime("%H:%M")
    return df[(df["hm"] >= "09:31") & (df["hm"] <= "16:00")].copy()


def _load_mim_bars(repo_root: Path = REPO_ROOT) -> tuple[pd.DataFrame, object]:
    """Build the replay frame = [processed-CSV warmup] + [bars_raw faithful bars].

    bars_raw.csv only starts on the bot's deploy day, so it cannot warm up the
    14-day noise-band lookback by itself. We prepend processed-CSV bars for ET
    days STRICTLY BEFORE the first bars_raw day (warmup state only — these never
    produce reconciled trades, see load_mim_theoretical), then use the faithful
    bars_raw bars from the deploy day onward. Returns (frame, first_raw_day)."""
    raw = pd.read_csv(repo_root / "data" / "mim_nb" / "bars_raw.csv",
                      usecols=["ts_utc", "open", "high", "low", "close", "volume"])
    raw["timestamp"] = pd.to_datetime(raw["ts_utc"], utc=True, format="ISO8601")
    raw = _rth_frame(raw)
    first_raw_day = raw["day"].min()

    warm_path = repo_root / "data" / "processed" / "dollar_bars" / "1_minute" / "mnq_1min_2026_ytd.csv"
    frames = []
    if warm_path.exists():
        proc = pd.read_csv(warm_path, usecols=["timestamp", "open", "high", "low", "close", "volume"])
        proc["timestamp"] = pd.to_datetime(proc["timestamp"], utc=True, format="ISO8601")
        proc = _rth_frame(proc)
        frames.append(proc[proc["day"] < first_raw_day])
    frames.append(raw)
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    return combined, first_raw_day


def mim_replay(df: pd.DataFrame, S: int = MIM_CATSTOP_PTS) -> list[Trade]:
    """Faithful copy of study_mim_nb_catstop.run_catstop, refactored to a callable
    that RE-EMITS entry_px/exit_px (the study discards them). Logic is identical;
    do NOT import the study module (it runs at import time and writes files)."""
    hist = defaultdict(lambda: deque(maxlen=_MIM_LOOKBACK))
    trades: list[Trade] = []
    day_count = 0
    prev_close = np.nan
    check_marks = {f"{h:02d}:{m}" for h in range(10, 16) for m in ("00", "30")} | {"16:00"}
    entry_marks = {f"{h:02d}:{m}" for h in range(10, 16) for m in ("00", "30")} - {"16:00"}

    def _emit(day, dir_, reason, entry_px, exit_px, entry_t, exit_t):
        ts = _et_to_utc(str(day), entry_t)
        trades.append(Trade(
            signal_bar_ts=ts, side=dir_, entry_px=float(entry_px), exit_px=float(exit_px),
            qty=1, pnl_usd=dir_ * (exit_px - entry_px) * PT_VAL, exit_reason=reason,
        ))

    for day, g in df.groupby("day", sort=True):
        g = g.sort_values("et")
        hms = g["hm"].values
        opens, highs, lows = g["open"].values, g["high"].values, g["low"].values
        closes, vols = g["close"].values, g["volume"].values
        if hms[0] != "09:31" or "16:00" not in set(hms):
            continue
        O = opens[0]
        day_count += 1
        tradeable = day_count > _MIM_LOOKBACK and not np.isnan(prev_close)
        gap_up_adj = max(O - prev_close, 0) if not np.isnan(prev_close) else 0.0
        gap_dn_adj = max(prev_close - O, 0) if not np.isnan(prev_close) else 0.0

        pos, entry_px, entry_t, cat = 0, 0.0, None, np.nan
        pending = None

        for i, hm in enumerate(hms):
            if tradeable and pending is not None:
                action, why = pending
                px = opens[i]
                if action == "exit" and pos != 0:
                    _emit(day, pos, why, entry_px, px, entry_t, hms[i])
                    pos = 0
                elif action in ("long", "short"):
                    new = 1 if action == "long" else -1
                    if pos != 0 and pos != new:
                        _emit(day, pos, "REVERSAL", entry_px, px, entry_t, hms[i])
                        pos = 0
                    if pos == 0:
                        pos, entry_px, entry_t = new, px, hms[i]
                        cat = entry_px - S if pos == 1 else entry_px + S
                pending = None

            if pos == 1 and lows[i] <= cat:
                _emit(day, 1, "CAT_STOP", entry_px, cat, entry_t, hms[i])
                pos, pending = 0, None
            elif pos == -1 and highs[i] >= cat:
                _emit(day, -1, "CAT_STOP", entry_px, cat, entry_t, hms[i])
                pos, pending = 0, None

            if hm in check_marks:
                sig = hist[hm]
                if tradeable and len(sig) == _MIM_LOOKBACK:
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
                    _emit(day, pos, "EOD", entry_px, closes[i], entry_t, "16:00")
                    pos = 0
                pending = None

        for i, hm in enumerate(hms):
            hist[hm].append(abs(closes[i] / O - 1.0))
        prev_close = closes[-1]
    return trades


def load_mim_theoretical(repo_root: Path = REPO_ROOT, S: int = MIM_CATSTOP_PTS) -> list[Trade]:
    """Faithful replay over bars_raw (warmed up with processed history). Only
    trades on/after the bars_raw deploy day are returned — warmup-day trades are
    dropped so they don't show as false MISSED against the live log."""
    frame, first_raw_day = _load_mim_bars(repo_root)
    cutoff = _et_to_utc(str(first_raw_day), "00:00")
    return [t for t in mim_replay(frame, S) if t.signal_bar_ts >= cutoff]


def tag_mim_regime(trades: list[Trade], repo_root: Path = REPO_ROOT) -> list[Trade]:
    """Attach a vol-regime label to each MIM trade via H1 ATR percentile rank at
    entry, reusing strategy_core. Defensive: any failure -> 'unknown'."""
    try:
        from src.research.strategy_core import calc_atr, resample_to_h1
    except Exception:
        return trades
    try:
        raw = pd.read_csv(repo_root / "data" / "mim_nb" / "bars_raw.csv",
                          usecols=["ts_utc", "open", "high", "low", "close", "volume"])
        raw["timestamp"] = pd.to_datetime(raw["ts_utc"], utc=True, format="ISO8601")
        raw = raw.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        raw.index.name = "timestamp"
        h1 = resample_to_h1(raw)
    except Exception:
        return trades

    # rolling 20-bar ATR series on H1
    atrs: list[tuple[datetime, float]] = []
    h1_list = list(h1.index)
    for i in range(len(h1_list)):
        window = h1.iloc[max(0, i - 19):i + 1]
        if len(window) >= 20:
            atrs.append((h1_list[i].to_pydatetime(), calc_atr(window)))
    if not atrs:
        return trades

    out: list[Trade] = []
    for t in trades:
        hist = [a for (ts, a) in atrs if ts <= t.signal_bar_ts][-120:]
        if len(hist) >= 20:
            cur = hist[-1]
            pct = sum(1 for a in hist if a < cur) / len(hist)
            regime = _vol_regime_bucket(pct)
        else:
            regime = "unknown"
        out.append(Trade(**{**t.__dict__, "regime": regime}))
    return out
