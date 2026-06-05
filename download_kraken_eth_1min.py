#!/usr/bin/env python3
"""Download 1.5 years of 1-minute PF_ETHUSD candles from Kraken Futures.

Output: data/kraken/PF_ETHUSD_1min.csv
Columns: timestamp,open,high,low,close,volume

Resumes automatically if interrupted — skips days whose window end is already
covered by the CSV. A final dedup+sort step runs atomically via tmp→rename.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

SYMBOL = "PF_ETHUSD"
INTERVAL = "1m"
URL = f"https://futures.kraken.com/api/charts/v1/trade/{SYMBOL}/{INTERVAL}"
OUTPUT = Path("data/kraken/PF_ETHUSD_1min.csv")
DAYS = 548
RATE_SLEEP = 0.2
MAX_RETRIES = 3
RETRY_SLEEP = 5.0
LOG_EVERY = 30


def _load_resume_ts() -> tuple[int | None, bool]:
    if not OUTPUT.exists() or OUTPUT.stat().st_size == 0:
        return None, False
    try:
        df = pd.read_csv(OUTPUT, usecols=["timestamp"])
        if df.empty:
            return None, False
        latest = pd.to_datetime(df["timestamp"], utc=True).max()
        ts = int(latest.timestamp())
        logger.info(f"Resuming — latest row in CSV: {latest.isoformat()}")
        return ts, True
    except Exception as exc:
        logger.warning(f"Could not parse existing CSV for resume: {exc}")
        return None, False


async def _fetch_day(client: httpx.AsyncClient, from_ts: int, to_ts: int) -> list[dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.get(
                URL, params={"from": from_ts, "to": to_ts}, timeout=20.0
            )
            if resp.status_code != 200:
                logger.warning(
                    f"HTTP {resp.status_code} from={from_ts} "
                    f"(attempt {attempt}/{MAX_RETRIES}): {resp.text[:120]}"
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_SLEEP)
                continue

            candles = resp.json().get("candles", [])
            rows = []
            skipped = 0
            for c in candles:
                try:
                    close = float(c.get("close", 0))
                    if close == 0:
                        skipped += 1
                        continue
                    ts_ms = c["time"]
                    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                    rows.append({
                        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        "open": float(c.get("open", close)),
                        "high": float(c.get("high", close)),
                        "low": float(c.get("low", close)),
                        "close": close,
                        "volume": float(c.get("volume", 0)),
                    })
                except (KeyError, ValueError, TypeError):
                    skipped += 1
                    continue
            if skipped:
                logger.debug(f"Skipped {skipped} malformed candles for from={from_ts}")
            return rows

        except httpx.RequestError as exc:
            logger.warning(
                f"Request error from={from_ts} (attempt {attempt}/{MAX_RETRIES}): {exc}"
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_SLEEP)

    logger.error(f"All {MAX_RETRIES} attempts failed for from={from_ts} — skipping day")
    return []


def _sort_and_dedup() -> int:
    logger.info("Deduplicating and sorting CSV by timestamp ascending...")
    df = pd.read_csv(OUTPUT)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    tmp = OUTPUT.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(OUTPUT)
    return len(df)


async def download() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    resume_ts, file_valid = _load_resume_ts()

    if not file_valid and OUTPUT.exists() and OUTPUT.stat().st_size > 0:
        bak = OUTPUT.with_suffix(".csv.bak")
        OUTPUT.rename(bak)
        logger.warning(f"Corrupt CSV backed up to {bak} — starting fresh download")
        resume_ts = None

    now = datetime.now(timezone.utc)
    today_midnight = int(
        datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp()
    )
    start = time.perf_counter()
    total_rows = 0
    days_fetched = 0
    days_skipped = 0

    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    write_header = not OUTPUT.exists() or OUTPUT.stat().st_size == 0

    async with httpx.AsyncClient() as client:
        for day_offset in range(DAYS):
            to_ts = today_midnight - day_offset * 86400
            from_ts = to_ts - 86400

            if resume_ts is not None and to_ts <= resume_ts:
                days_skipped += 1
                continue

            rows = await _fetch_day(client, from_ts, to_ts)

            if rows:
                df = pd.DataFrame(rows, columns=cols)
                df.to_csv(OUTPUT, mode="a", header=write_header, index=False)
                write_header = False
                total_rows += len(rows)
                days_fetched += 1

                if days_fetched % LOG_EVERY == 0:
                    day_date = datetime.fromtimestamp(from_ts, tz=timezone.utc).date()
                    elapsed = time.perf_counter() - start
                    logger.info(
                        f"Progress: {days_fetched} days | {total_rows:,} rows | "
                        f"at: {day_date} | elapsed: {elapsed:.0f}s"
                    )
            else:
                day_date = datetime.fromtimestamp(from_ts, tz=timezone.utc).date()
                logger.debug(f"No candles for {day_date}")

            await asyncio.sleep(RATE_SLEEP)

    if OUTPUT.exists() and OUTPUT.stat().st_size > 0:
        final_rows = _sort_and_dedup()
    else:
        final_rows = 0

    elapsed = time.perf_counter() - start
    logger.info("=" * 60)
    logger.info(f"Done | {final_rows:,} total rows | {days_fetched} days fetched "
                f"| {days_skipped} skipped | {elapsed:.1f}s")
    logger.info(f"Output: {OUTPUT}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(download())
