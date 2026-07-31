import os
import asyncio
import pandas as pd
import numpy as np
import httpx
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

base_dir = Path("/root/Silver-Bullet-ML-BMAD")
sys.path.insert(0, str(base_dir))
from src.data.auth_v3 import TradeStationAuthV3

TS_URL = "https://api.tradestation.com/v3/marketdata/barcharts/{symbol}?interval=1&unit=Minute&firstdate={start_date}"
KRAKEN_URL = "https://futures.kraken.com/api/charts/v1/trade/PF_XBTUSD/1m?from={start_ts}"

async def fetch_ts_data(symbol, start_time):
    auth = TradeStationAuthV3.from_file(str(base_dir / '.access_token'))
    token = await auth.authenticate()
    
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    # TradeStation requires 'YYYY-MM-DDTHH:mm:ssZ'
    formatted_start = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    url = TS_URL.format(symbol=symbol, start_date=formatted_start)
    
    print(f"Fetching {symbol} from TradeStation since {formatted_start}...")
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"Error fetching {symbol}: {resp.status_code} - {resp.text}")
            return pd.DataFrame()
            
        bars = resp.json().get("Bars", [])
        if not bars:
            print(f"No new bars found for {symbol}.")
            return pd.DataFrame()
            
        data = []
        for b in bars:
            ts = datetime.fromisoformat(b["TimeStamp"].replace('Z', '+00:00'))
            high = float(b["High"])
            low = float(b["Low"])
            close = float(b["Close"])
            vol = int(b["TotalVolume"])
            
            # Reconstruct 'notional'
            point_mult = 2.0 if "MNQ" in symbol else 5.0
            notional = max(((high + low) / 2) * vol * point_mult, 0.01)
            
            data.append({
                "timestamp": ts,
                "open": float(b["Open"]),
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
                "notional": notional
            })
            
        df = pd.DataFrame(data)
        return df

async def fetch_kraken_data(start_time):
    # Kraken uses unix timestamp in seconds
    start_ts = int(start_time.timestamp())
    url = KRAKEN_URL.format(start_ts=start_ts)
    
    print(f"Fetching PF_XBTUSD from Kraken since {start_time}...")
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            print(f"Error fetching Kraken data: {resp.status_code}")
            return pd.DataFrame()
            
        data = resp.json()
        candles = data.get("candles", [])
        if not candles:
            print("No new bars found for PF_XBTUSD.")
            return pd.DataFrame()
            
        parsed = []
        for c in candles:
            # c = {"time": 1700000000000, "open": "...", ...}
            ts = datetime.fromtimestamp(c["time"] / 1000.0, tz=timezone.utc)
            parsed.append({
                "timestamp": ts,
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c["volume"])
            })
            
        return pd.DataFrame(parsed)

async def update_csv(df_new, csv_path):
    if df_new.empty:
        return
        
    print(f"Updating {os.path.basename(csv_path)}...")
    df_old = pd.read_csv(csv_path)
    df_old["timestamp"] = pd.to_datetime(df_old["timestamp"])
    
    # Merge and drop duplicates
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    
    # Sort and drop duplicates by timestamp, keeping the last (newest)
    df_combined.sort_values("timestamp", inplace=True)
    df_combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    
    # Ensure correct format
    df_combined["timestamp"] = df_combined["timestamp"].apply(lambda x: x.isoformat())
    
    df_combined.to_csv(csv_path, index=False)
    print(f"Added {len(df_combined) - len(df_old)} new bars. Total rows: {len(df_combined)}")

async def main():
    print("Starting EOD Data Pipeline...")
    
    targets = [
        ("MNQM26", base_dir / "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv", fetch_ts_data),
        ("MESM26", base_dir / "data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv", fetch_ts_data),
        ("PF_XBTUSD", base_dir / "data/kraken/PF_XBTUSD_1min.csv", fetch_kraken_data)
    ]
    
    for symbol, path, fetch_func in targets:
        # Get the last timestamp in the CSV
        if path.exists():
            df_old = pd.read_csv(path)
            last_ts_str = df_old["timestamp"].iloc[-1]
            last_ts = datetime.fromisoformat(last_ts_str).replace(tzinfo=timezone.utc)
        else:
            print(f"File {path} not found. Skipping.")
            continue
            
        # Add a slight overlap safety buffer (1 hour)
        start_time = last_ts - timedelta(hours=1)
        
        if fetch_func == fetch_ts_data:
            df_new = await fetch_func(symbol, start_time)
        else:
            df_new = await fetch_func(start_time)
            
        await update_csv(df_new, path)

    print("EOD Data Pipeline Complete.")

if __name__ == "__main__":
    asyncio.run(main())
