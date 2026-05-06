#!/bin/bash
###############################################################################
# Auto-start TIER 1 paper trading at market open (6 PM CDT Sunday)
###############################################################################

echo "=========================================="
echo "TIER 1 Paper Trading Startup"
echo "=========================================="
echo "Time: $(date)"
echo "Waiting for futures market open (6:00 PM CDT)..."
echo ""

# Check if already running
if [ -f "data/state/tier1_paper_trading.pid" ]; then
    PID=$(cat data/state/tier1_paper_trading.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  Paper trading already running (PID: $PID)"
        echo "Stopping existing instance..."
        ./deploy_tier1_realtime.sh stop
        sleep 2
    fi
fi

# Start paper trading
echo "🚀 Starting TIER 1 paper trading system..."
./deploy_tier1_realtime.sh start

# Verify OAuth token refresh is working
echo ""
echo "⏳ Waiting 30 seconds to verify token refresh..."
sleep 30

# Check if process is still running
if [ -f "data/state/tier1_paper_trading.pid" ]; then
    PID=$(cat data/state/tier1_paper_trading.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Paper trading started successfully (PID: $PID)"
        echo ""
        echo "📊 Monitor with:"
        echo "  ./deploy_tier1_realtime.sh status"
        echo "  ./deploy_tier1_realtime.sh logs"
        echo ""
        echo "🔄 OAuth token refresh: EVERY 10 MINUTES (CRITICAL)"
        echo "⚠️  If token refresh stops, API access will be blocked!"
    else
        echo "❌ ERROR: Paper trading failed to start"
        echo "Check logs: logs/tier1_paper_rest.log"
        exit 1
    fi
else
    echo "❌ ERROR: PID file not found"
    echo "Check logs: logs/tier1_paper_rest.log"
    exit 1
fi

echo ""
echo "=========================================="
echo "Startup Complete - Time: $(date)"
echo "=========================================="
