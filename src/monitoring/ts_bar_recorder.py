"""Independent 1-minute bar recorder for GAP-1 (gap-fade): the witness.

Why this exists (census 2026-09-13, _bmad-output/diagnostics_gap_fade_census_20260913/):
gap-fade's live outcomes reproduced 26/26 against its sealed engine, but the only
record of the bars it saw was YANK's shadow logger, which went dark 2026-09-02..04 and
missed the 2026-08-05 open. Those gaps made decision parity read "broken by the
letter". TradeStation also revises its 1-minute history after the fact (party-mode
finding 2026-08-06: 145 of 390 bars differed a day later), so bars re-fetched later
are not the bars the bot decided on. The witness has to record bars as they close.

Design (boring on purpose):
- A separate process. It shares nothing with gap_fade_live.py except the fleet's auth
  pattern (TradeStationAuthV3.from_file + authenticate + start_auto_refresh), so a
  recorder fault cannot touch trading and a bot fault cannot silence the witness.
- Polls /v3/marketdata/barcharts once a minute (5 s after the minute) per symbol,
  keeps only BarStatus == "Closed" bars, and appends each new minute ONCE.
- Append-only, hash-chained CSVs in tools/verify_chain.py's format (every column
  except `chain`, header order, GENESIS seed):
    data/gap_fade/bars/<SYMBOL>.csv            one row per closed minute, first sighting
    data/gap_fade/bars/<SYMBOL>_revisions.csv  a later fetch disagreed with a bar
  Verify: .venv/bin/python tools/verify_chain.py --file data/gap_fade/bars/MNQZ26.csv
- bar_ts is TradeStation's TimeStamp verbatim: the bar's CLOSE time, UTC (Friday's
  last bar is 21:00Z). gap_fade_live.py and data/processed/ use the same convention.
- Honest about outages: after a gap it backfills from TradeStation, but each row
  carries `lag_s` (fetch time minus bar time) and `live` = 1 only when lag_s <= 180.
  Backfilled rows are the venue's current history, not the bars as they closed.
- Never raises out of the poll loop on API/network errors; logs and retries next
  minute. Exits 1 only when it cannot initialize (bad config, no token), so systemd
  restarts it.

Config (environment):
  RECORDER_SYMBOLS  comma list, e.g. "MNQU26,MNQZ26" — record both contracts across a
                    roll so the witness is already on the new one before the bot is.
  RECORDER_DIR      default data/gap_fade/bars (gitignored, live-appended)
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import logging
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Callable, Optional, Union

import httpx

BASE_DIR = Path(__file__).resolve().parents[2]
TS_BARS_BASE = "https://api.tradestation.com/v3/marketdata/barcharts"
GENESIS = "GENESIS"
BAR_FIELDS = [
    "bar_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "fetched_at",
    "lag_s",
    "live",
]
REV_FIELDS = [
    "bar_ts",
    "field",
    "first_seen",
    "now_seen",
    "first_fetched_at",
    "fetched_at",
]
OHLCV = (
    ("open", "Open"),
    ("high", "High"),
    ("low", "Low"),
    ("close", "Close"),
    ("volume", "TotalVolume"),
)
LIVE_LAG_S = 180
MIN_BARSBACK, MAX_BARSBACK = 5, 1000
KEEP_RECENT = 2000  # recorded minutes held in memory for revision checks (~33 h)
POLL_OFFSET_S = 5  # poll 5 s after each minute boundary

logger = logging.getLogger("ts_bar_recorder")


Row = dict[str, Any]
Counts = dict[str, int]


def chain_next(head: str, row: Row, fields: list[str]) -> str:
    """Next chain value, identical to tools/verify_chain.py and gap_fade_live."""
    payload = "|".join(str(row.get(k, "")) for k in fields)
    return hashlib.sha256((head + "|" + payload).encode()).hexdigest()[:16]


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


class ChainedAppender:
    """Append-only chained CSV. Resumes the chain from the last row on disk."""

    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.fields = list(fields)
        self.head = GENESIS
        self.rows: list[Row] = []
        if path.exists() and path.stat().st_size > 0:
            with path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames != self.fields + ["chain"]:
                    want = self.fields + ["chain"]
                    raise RuntimeError(
                        f"{path} has header {reader.fieldnames}, expected {want}"
                    )
                for row in reader:
                    self.rows.append(row)
                    self.head = row.get("chain") or self.head
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self.fields + ["chain"]).writeheader()

    def append(self, row: Row) -> Row:
        out = {k: row.get(k, "") for k in self.fields}
        out["chain"] = chain_next(self.head, out, self.fields)
        with self.path.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.fields + ["chain"]).writerow(out)
            fh.flush()
            os.fsync(fh.fileno())
        self.head = out["chain"]
        return out


class BarRecorder:
    """Recording logic for one symbol: first sighting wins, disagreements are logged."""

    def __init__(self, symbol: str, out_dir: Path, keep: int = KEEP_RECENT):
        self.symbol = symbol
        self.keep = keep
        self.bars = ChainedAppender(out_dir / f"{symbol}.csv", BAR_FIELDS)
        self.revs = ChainedAppender(out_dir / f"{symbol}_revisions.csv", REV_FIELDS)
        self.recent: dict[str, Row] = {r["bar_ts"]: r for r in self.bars.rows[-keep:]}
        self.last_ts: Optional[datetime] = (
            parse_ts(self.bars.rows[-1]["bar_ts"]) if self.bars.rows else None
        )
        # when a fetch last proved we hold everything the venue has (see ingest)
        self.caught_up_at: Optional[datetime] = None
        self.last_seen: dict[tuple[str, str], str] = {}
        for r in self.revs.rows:
            self.last_seen[(r["bar_ts"], r["field"])] = r["now_seen"]
        self.bars.rows = []  # the appender no longer needs the full history
        self.revs.rows = []

    def barsback(self, now: datetime) -> int:
        """Bars to request: enough to cover everything since we were last caught up.

        Measured from the last caught-up poll, not the last bar: over a weekend the
        last bar is days old, and sizing from it would pull 1,000 bars a minute."""
        ref = self.last_ts
        if ref is None:
            return MAX_BARSBACK
        if self.caught_up_at is not None and self.caught_up_at > ref:
            ref = self.caught_up_at
        minutes = int((now - ref).total_seconds() // 60) + 2
        return max(MIN_BARSBACK, min(MAX_BARSBACK, minutes))

    def ingest(self, bars: list[Row], fetched_at: datetime) -> Counts:
        """Record newly closed minutes; log revisions of minutes already recorded."""
        counts = {"appended": 0, "revisions": 0, "skipped_open": 0, "backfilled": 0}
        prior = self.last_ts
        oldest = min((parse_ts(b["TimeStamp"]) for b in bars), default=None)
        closed = []
        for b in bars:
            status = b.get("BarStatus")
            ts = parse_ts(b["TimeStamp"])
            if status is not None and status != "Closed":
                counts["skipped_open"] += 1
                continue
            if status is None and ts > fetched_at - timedelta(seconds=60):
                # no status field: only trust bars a minute old
                counts["skipped_open"] += 1
                continue
            closed.append((ts, b))
        closed.sort(key=lambda x: x[0])
        for ts, b in closed:
            key = b["TimeStamp"]
            if self.last_ts is None or ts > self.last_ts:
                lag = int((fetched_at - ts).total_seconds())
                row = {
                    "bar_ts": key,
                    **{f: b.get(src, "") for f, src in OHLCV},
                    "fetched_at": fetched_at.isoformat(),
                    "lag_s": lag,
                    "live": 1 if lag <= LIVE_LAG_S else 0,
                }
                self.recent[key] = self.bars.append(row)
                self.last_ts = ts
                counts["appended"] += 1
                counts["backfilled"] += 0 if lag <= LIVE_LAG_S else 1
            elif key in self.recent:
                first = self.recent[key]
                for f, src in OHLCV:
                    now_v = b.get(src, "")
                    if _same(first.get(f, ""), now_v):
                        self.last_seen.pop((key, f), None)  # back to first sighting
                    else:
                        if self.last_seen.get((key, f)) == str(now_v):
                            continue  # this exact disagreement is already on record
                        self.revs.append(
                            {
                                "bar_ts": key,
                                "field": f,
                                "first_seen": first.get(f, ""),
                                "now_seen": now_v,
                                "first_fetched_at": first.get("fetched_at", ""),
                                "fetched_at": fetched_at.isoformat(),
                            }
                        )
                        self.last_seen[(key, f)] = str(now_v)
                        counts["revisions"] += 1
        # The response is the venue's newest N bars, contiguous. If it reaches back to
        # the minute we already held, nothing between is missing: we are caught up.
        if prior is None or (oldest is not None and oldest <= prior):
            self.caught_up_at = fetched_at
        elif oldest is not None:
            logger.warning("%s: hole %s..%s not bridged", self.symbol, prior, oldest)
        if len(self.recent) > self.keep:
            for k in sorted(self.recent, key=parse_ts)[: len(self.recent) - self.keep]:
                del self.recent[k]
        return counts


def _same(a: Any, b: Any) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


async def fetch_bars(
    http: httpx.AsyncClient, auth: Any, symbol: str, barsback: int
) -> Optional[list[Row]]:
    """One barcharts request, gap-fade's endpoint and auth. None on any failure."""
    try:
        token = await auth.authenticate()
        r = await http.get(
            f"{TS_BARS_BASE}/{symbol}",
            params={"interval": 1, "unit": "Minute", "barsback": barsback},
            headers={"Authorization": f"Bearer {token}"},
        )
    except Exception as exc:  # network, auth — never kill the loop
        logger.warning("%s: fetch failed (%s)", symbol, exc)
        return None
    if r.status_code != 200:
        logger.warning("%s: bars API HTTP %s", symbol, r.status_code)
        return None
    bars: list[Row] = r.json().get("Bars", [])
    return bars


async def poll_once(
    recorders: dict[str, BarRecorder],
    http: httpx.AsyncClient,
    auth: Any,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Union[Counts, str]]:
    summary: dict[str, Union[Counts, str]] = {}
    for symbol, rec in recorders.items():
        t = now()
        bars = await fetch_bars(http, auth, symbol, rec.barsback(t))
        if bars is None:
            summary[symbol] = "fetch_failed"
            continue
        try:
            summary[symbol] = rec.ingest(bars, t)
        except Exception as exc:  # malformed payload — log, keep the witness alive
            logger.error("%s: ingest failed (%s)", symbol, exc)
            summary[symbol] = "ingest_failed"
    return summary


def seconds_to_next_poll(t: datetime) -> float:
    nxt = t.replace(second=0, microsecond=0) + timedelta(
        minutes=1, seconds=POLL_OFFSET_S
    )
    return max(1.0, (nxt - t).total_seconds())


async def run(
    symbols: list[str],
    out_dir: Path,
    auth: Any,
    http: httpx.AsyncClient,
    stop: asyncio.Event,
) -> None:
    recorders = {s: BarRecorder(s, out_dir) for s in symbols}
    for s, r in recorders.items():
        logger.info("recording %s -> %s (last recorded %s)", s, r.bars.path, r.last_ts)
    while not stop.is_set():
        summary = await poll_once(recorders, http, auth)
        logger.info("poll %s", summary)
        try:
            await asyncio.wait_for(
                stop.wait(), timeout=seconds_to_next_poll(datetime.now(timezone.utc))
            )
        except asyncio.TimeoutError:
            pass


def config_from_env(
    env: Mapping[str, str] = os.environ,
) -> tuple[list[str], Path]:
    symbols = [
        s.strip() for s in env.get("RECORDER_SYMBOLS", "").split(",") if s.strip()
    ]
    if not symbols:
        raise SystemExit(
            "RECORDER_SYMBOLS is empty — set e.g. RECORDER_SYMBOLS=MNQU26,MNQZ26"
        )
    out_dir = Path(
        env.get("RECORDER_DIR", str(BASE_DIR / "data" / "gap_fade" / "bars"))
    )
    return symbols, out_dir


async def _main_async() -> int:
    sys.path.insert(0, str(BASE_DIR))
    from src.data.auth_v3 import TradeStationAuthV3

    symbols, out_dir = config_from_env()
    auth = TradeStationAuthV3.from_file(".access_token")
    await auth.authenticate()
    await auth.start_auto_refresh()  # tokens expire ~20 min (lesson d4c0c39)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    async with httpx.AsyncClient(timeout=30) as http:
        await run(symbols, out_dir, auth, http, stop)
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    try:
        return asyncio.run(_main_async())
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("recorder failed to start: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
