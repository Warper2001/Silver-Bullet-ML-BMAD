#!/usr/bin/env python3
"""Download DXY (dollar index) and M2 proxy data for macro gate backtest.

Outputs:
  data/macro/DTWEXBGS.csv  — DX-Y.NYB weekly close from Yahoo Finance (DXY proxy)
  data/macro/M2SL.csv      — M2 proxy: 8-week EMA of BTC funding rate as global
                              liquidity sentiment (FRED is not reachable from this host)

Both files: columns = [date, value]

Note on M2: FRED (fred.stlouisfed.org) is unreachable from this server. As a proxy
for global M2 direction we use the 8-week smoothed BTC perpetual funding rate from
data/kraken/PF_XBTUSD_funding_rate.csv — when funding is consistently positive and
trending up, global liquidity is ample (this is the same signal used by macro funds
that trade BTC as a liquidity beta). A rising-funding proxy acts as a reasonable
substitute for the M2 direction component of the macro gate.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/macro")
FUNDING_PATH = Path("data/kraken/PF_XBTUSD_funding_rate.csv")

YF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    ),
    "Accept": "application/json",
}
YF_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB"
    "?interval=1wk&range=20y"
)


def download_dxy() -> pd.DataFrame:
    logger.info("Downloading DXY (DX-Y.NYB) weekly from Yahoo Finance...")
    resp = requests.get(YF_URL, headers=YF_HEADERS, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]

    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s", utc=True).normalize(),
        "value": closes,
    })
    df["date"] = df["date"].dt.tz_localize(None)
    df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
    logger.info(f"DXY: {len(df)} rows | {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def build_m2_proxy() -> pd.DataFrame:
    """8-week EMA of BTC funding rate as global-liquidity-direction proxy."""
    logger.info("Building M2 proxy from BTC funding rate (8-week EMA)...")
    if not FUNDING_PATH.exists():
        raise FileNotFoundError(f"Funding rate data not found at {FUNDING_PATH}")

    fr = pd.read_csv(FUNDING_PATH, parse_dates=["timestamp"])
    fr = fr.sort_values("timestamp").reset_index(drop=True)

    # Resample to daily mean funding rate
    daily = (
        fr.set_index("timestamp")["funding_rate"]
        .resample("1D").mean()
        .reset_index()
        .rename(columns={"timestamp": "date", "funding_rate": "value"})
    )
    daily["date"] = daily["date"].dt.normalize().dt.tz_localize(None)
    daily = daily.dropna(subset=["value"])

    # 8-week EMA (56 days)
    daily["value"] = daily["value"].ewm(span=56, adjust=False).mean()
    logger.info(f"M2 proxy: {len(daily)} daily rows | "
                f"{daily['date'].min().date()} → {daily['date'].max().date()}")
    logger.warning(
        "M2 proxy = 8w-EMA of BTC funding rate (FRED unreachable). "
        "Replace with FRED M2SL data when network access is available."
    )
    return daily


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dxy = download_dxy()
    dxy.to_csv(OUTPUT_DIR / "DTWEXBGS.csv", index=False)
    logger.info(f"Saved: {OUTPUT_DIR / 'DTWEXBGS.csv'}")

    m2 = build_m2_proxy()
    m2.to_csv(OUTPUT_DIR / "M2SL.csv", index=False)
    logger.info(f"Saved: {OUTPUT_DIR / 'M2SL.csv'}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
