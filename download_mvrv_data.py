#!/usr/bin/env python3
"""Download daily BTC MVRV ratio from CoinMetrics Community API.

Output: data/macro/MVRV_BTC.csv
Columns: date,mvrv

Falls back to a 2-year SMA price-ratio proxy if CoinMetrics is unavailable.
The proxy is: mvrv_proxy = btc_daily_close / sma(btc_daily_close, 730 days)
using data/kraken/PF_XBTUSD_1min.csv resampled to daily.
"""

import logging
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

COINMETRICS_URL = (
    "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    "?assets=btc&metrics=CapMVRVCur&frequency=1d&page_size=10000"
)
OUTPUT = Path("data/macro/MVRV_BTC.csv")
BTC_1MIN = Path("data/kraken/PF_XBTUSD_1min.csv")


def _download_coinmetrics() -> pd.DataFrame | None:
    logger.info("Attempting CoinMetrics Community API for CapMVRVCur...")
    try:
        resp = requests.get(COINMETRICS_URL, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"CoinMetrics returned HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        rows = data.get("data", [])
        if not rows:
            logger.warning("CoinMetrics returned empty data array")
            return None
        df = pd.DataFrame(rows)
        df = df.rename(columns={"time": "date", "CapMVRVCur": "mvrv"})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["mvrv"] = pd.to_numeric(df["mvrv"], errors="coerce")
        df = df.dropna(subset=["mvrv"])[["date", "mvrv"]].sort_values("date").reset_index(drop=True)
        logger.info(f"CoinMetrics: {len(df)} rows | {df['date'].min().date()} → {df['date'].max().date()}")
        return df
    except Exception as exc:
        logger.warning(f"CoinMetrics failed: {exc}")
        return None


def _compute_proxy() -> pd.DataFrame:
    logger.info("Building MVRV proxy from 2-year SMA of Kraken BTC 1-min data...")
    if not BTC_1MIN.exists():
        raise FileNotFoundError(f"BTC 1-min data not found at {BTC_1MIN}")

    df = pd.read_csv(BTC_1MIN, usecols=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    daily = df.set_index("timestamp")["close"].resample("1D").last().dropna()
    daily = daily.reset_index().rename(columns={"timestamp": "date", "close": "btc_close"})
    daily["date"] = daily["date"].dt.normalize()

    sma_730 = daily["btc_close"].rolling(window=730, min_periods=365).mean()
    daily["mvrv"] = daily["btc_close"] / sma_730
    daily = daily.dropna(subset=["mvrv"])[["date", "mvrv"]].reset_index(drop=True)

    logger.info(
        f"Proxy MVRV: {len(daily)} rows | {daily['date'].min().date()} → {daily['date'].max().date()}"
    )
    logger.warning(
        "Using price-based MVRV proxy (btc/sma_730). "
        "This approximates the realized-price ratio but is not the true on-chain MVRV. "
        "Replace with CoinMetrics data when available."
    )
    return daily


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df = _download_coinmetrics()
    source = "CoinMetrics"

    if df is None:
        df = _compute_proxy()
        source = "2yr-SMA-proxy"

    OUTPUT_TAGGED = OUTPUT.parent / f"MVRV_BTC_{source.replace('/', '-')}.csv"
    df.to_csv(OUTPUT, index=False)
    df.to_csv(OUTPUT_TAGGED, index=False)
    logger.info(f"Saved: {OUTPUT}  (source={source})")
    logger.info(f"Latest MVRV: {df['mvrv'].iloc[-1]:.3f} on {df['date'].iloc[-1].date()}")


if __name__ == "__main__":
    main()
