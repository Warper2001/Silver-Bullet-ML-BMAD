#!/bin/bash
# TradingView Export Setup Script
#
# This script sets up the TradingView webhook receiver and provides
# instructions for configuring TradingView.

set -e

echo "========================================================================"
echo "TradingView → MNQ Data Export Setup"
echo "========================================================================"
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Creating..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
fi

# Install Flask if not present
echo ""
echo "Checking dependencies..."
.venv/bin/pip install flask 2>&1 | grep -i "already satisfied" || echo "✅ Flask installed/updated"

# Create data directory
mkdir -p data/historical/mnq
echo "✅ Data directory created: data/historical/mnq"

# Get local IP address
LOCAL_IP=$(hostname -I | awk '{print $1}')
[ -z "$LOCAL_IP" ] && LOCAL_IP="localhost"

echo ""
echo "========================================================================"
echo "SETUP COMPLETE"
echo "========================================================================"
echo ""
echo "📋 NEXT STEPS:"
echo ""
echo "1. Start the webhook receiver:"
echo "   .venv/bin/python integrations/tradingview_webhook_receiver.py"
echo ""
echo "2. Open TradingView:"
echo "   - Go to tradingview.com"
echo "   - Search for 'MNQM26' (or your desired contract)"
echo "   - Open the Pine Editor (bottom panel)"
echo ""
echo "3. Add the Pine Script:"
echo "   - Copy the script from: integrations/tradingview_pine_script.txt"
echo "   - Paste it into Pine Editor"
echo "   - Click 'Add to Chart'"
echo ""
echo "4. Configure the script:"
echo "   - Click on indicator settings (gear icon)"
echo "   - Update 'Webhook URL' to: http://$LOCAL_IP:8080/webhook/tradingview"
echo "   - Set 'Symbol Name' to your contract (e.g., MNQM26)"
echo ""
echo "5. Create an Alert:"
echo "   - Click 'Alert' button (top toolbar)"
echo "   - Condition: MNQ Export Webhook"
echo "   - Once Per Bar Close: ✅ YES (important!)"
echo "   - Webhook URL: http://$LOCAL_IP:8080/webhook/tradingview"
echo "   - Click 'Create'"
echo ""
echo "6. Data will start flowing!"
echo "   - Check terminal for confirmation"
echo "   - Auto-saves every 500 bars"
echo "   - Manual save: curl -X POST http://$LOCAL_IP:8080/save"
echo ""
echo "========================================================================"
echo "WEBHOOK URL TO USE:"
echo "   http://$LOCAL_IP:8080/webhook/tradingview"
echo "========================================================================"
echo ""
echo "Starting webhook receiver..."
echo ""

# Start the receiver
.venv/bin/python integrations/tradingview_webhook_receiver.py
