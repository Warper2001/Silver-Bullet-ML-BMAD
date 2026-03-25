# Exporting MNQ Data from TradingView

**Last Updated:** 2026-03-22
**Your Subscription:** TradingView with CME data access

---

## ✅ Available Export Methods

### Method 1: TradingView Pine Script Export (Recommended)
**Best for:** Automated exports, custom date ranges

#### Step-by-Step:

1. **Open TradingView Chart**
   - Go to tradingview.com
   - Search for `MNQM26` or your desired MNQ contract
   - Set chart to desired timeframe (1min, 5min, etc.)

2. **Create Export Script**
   - Click "Pine Editor" at bottom
   - Paste the export script (see below)
   - Click "Add to Chart"

3. **Export Data**
   - Script will show data in "Pine Window" panel
   - Copy the data
   - Save to CSV file

**Export Script:**
```pine
// MNQ Data Export Script
//@version=5
indicator("MNQ Data Export", overlay=true, max_lines_count=500)

// Configuration
var table infoTable = table.new(position.top_right, 2, 1, bgcolor=color.white, border_width=1)

// Get OHLCV data
barData = str.format("{0},{1},{2},{3},{4},{5}",
     str.tostring(time),
     str.tostring(open),
     str.tostring(high),
     str.tostring(low),
     str.tostring(close),
     str.tostring(volume)
     )

// Display in table (last 100 bars)
if barstate.islast
    for i = 0 to 99
        if bar_index - i >= 0
            historicalData = str.format("{0},{1},{2},{3},{4},{5}",
                 str.tostring(time[bar_index - i]),
                 str.tostring(open[bar_index - i]),
                 str.tostring(high[bar_index - i]),
                 str.tostring(low[bar_index - i]),
                 str.tostring(close[bar_index - i]),
                 str.tostring(volume[bar_index - i])
                 )
            table.cell(infoTable, 0, i, historicalData, text_color=color.black)
```

**Limitations:**
- 500-1000 bars max per export
- Requires manual copy-paste
- Need to export multiple times for full history

---

### Method 2: TradingView Webhook (Advanced)
**Best for:** Automated, large datasets

1. **Use Pine Script with `alert.condition`**
2. **Send data to webhook**
3. **Python script receives and saves to CSV**

**Pine Script with Webhook:**
```pine
//@version=5
indicator("MNQ Webhook Export", overlay=false)

// Webhook URL (replace with your server)
webhook_url = "https://your-server.com/webhook"

// Bar data
bar_json = str.format("{{\"timestamp\":\"{0}\",\"open\":{1},\"high\":{2},\"low\":{3},\"close\":{4},\"volume\":{5}}",
     str.format("{0}", year * 10000 + month * 100 + dayofmonth),
     open, high, low, close, volume
     )

// Trigger on each bar
alertcondition(barstate.isconfirmed, "Bar Data", bar_json)
```

**Python Webhook Receiver:**
```python
from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def receive_tradingview_data():
    data = request.json
    # Save to CSV
    # Append to file
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(port=8080)
```

---

### Method 3: Third-Party Tools
**Best for:** Bulk historical data

#### A. **TradingView Data Exporter** (Browser Extension)
- Search for "TradingView Data Exporter" browser extensions
- Some support CME data export
- Export directly to CSV

#### B. **Screener Export Tools**
- Tools like `tvdatadump` (GitHub)
- Command-line utilities for TradingView data
- May require Python scripting

#### C. **Paid Tools**
- **ExportData.io** - TradingView export service
- **Tweeten** - Enhanced TradingView features
- Various Chrome extensions

---

### Method 4: TradingView API (If Available)
**Check if you have access:**

1. **TradingView Broker API**
   - Available if you have linked broker
   - May provide historical data access
   - Check your account settings

2. **TradingView Widgets API**
   - Limited data access
   - Mostly for display purposes
   - Not recommended for bulk export

---

## 🎯 Recommended Workflow

### For Initial Setup (2+ years of data):

1. **Use Pine Script Export (Method 1)**
   - Export 500-1000 bars at a time
   - Change chart date range
   - Repeat until full history captured

2. **Or Use Automated Script (Method 2)**
   - Set up webhook receiver
   - Let it run during market hours
   - Accumulates data automatically

### For Ongoing Updates:

1. **Set up Webhook (Method 2)**
   - Auto-saves new bars as they come in
   - Runs continuously during market hours
   - No manual intervention needed

---

## 📊 TradingView Data Format

TradingView Pine Script exports in this format:
```
timestamp,open,high,low,close,volume
1648000000000,24200.00,24250.00,24180.00,24220.00,1523
```

**Notes:**
- Timestamp is in Unix milliseconds
- Prices are decimal
- Volume is contract count

---

## 🔄 Conversion Process

### Step 1: Export from TradingView
- Use one of the methods above
- Save to `mnq_tradingview_export.csv`

### Step 2: Convert to HDF5
```bash
.venv/bin/python convert_csv_to_hdf5.py \
    --input mnq_tradingview_export.csv \
    --output data/historical/mnq/MNQM26.h5 \
    --symbol MNQM26
```

### Step 3: Verify Data
```bash
# Quick check
.venv/bin/python -c "
import h5py
with h5py.File('data/historical/mnq/MNQM26.h5', 'r') as f:
    data = f['historical_bars']
    print(f'Total bars: {len(data)}')
    print(f'First bar: {data[0]}')
    print(f'Last bar: {data[-1]}')
"
```

---

## 🚀 Quick Start Script

I can create an automated TradingView export script for you. It would:
1. Generate the Pine Script for you
2. Set up a webhook receiver in Python
3. Auto-convert received data to HDF5
4. Save to correct directory

**Would you like me to create this automation?**

---

## ⚠️ Important Notes

### TradingView Limitations:
- **Export limits:** Pine Script has output limits (500-1000 bars)
- **Rate limits:** Don't export too frequently
- **TOS compliance:** Ensure exports comply with TradingStation TOS
- **Personal use only:** Exported data is for personal analysis

### Data Quality:
- **Real-time data:** TradingView provides real-time CME data
- **Historical depth:** Depends on your subscription tier
- **Backfilling:** May need to export multiple date ranges

### Subscription Requirements:
- **CME Data:** You have this ✓
- **Export capability:** Check your TradingView plan limits
- **API access:** May need Pro/Pro+ for some features

---

## 📞 Next Steps

**Option A: Manual Export (Quick)**
1. I'll provide a ready-to-use Pine Script
2. You export data from TradingView
3. Convert with existing script

**Option B: Automated Export (Setup)**
1. I'll create a webhook receiver
2. Provide Pine Script with alerts
3. Auto-convert and save data

**Option C: Batch Export (Large Dataset)**
1. Multiple exports for different date ranges
2. Merge and convert all data
3. Build complete historical dataset

---

**Which option would you prefer?** I can create the necessary scripts and walk you through the setup.

---

**Sources:**
- [TradingView Pine Script Docs](https://www.tradingview.com/pine-script-docs/)
- [TradingView Alert Documentation](https://www.tradingview.com/support/solutions/alerts/)
- [CME Group Market Data](https://www.cmegroup.com/market-data/)
