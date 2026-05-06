#!/bin/bash
###############################################################################
# Monitor OAuth token refresh - CRITICAL for API access
###############################################################################

echo "=========================================="
echo "OAuth Token Refresh Monitor"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Check if paper trading is running
if [ ! -f "data/state/tier1_paper_trading.pid" ]; then
    echo "❌ Paper trading not running (no PID file)"
    exit 1
fi

PID=$(cat data/state/tier1_paper_trading.pid)
if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ Paper trading process not found (PID: $PID)"
    exit 1
fi

echo "✅ Paper trading running (PID: $PID)"
echo ""

# Check recent OAuth refresh activity
echo "🔄 Recent OAuth Token Refresh Activity:"
echo "----------------------------------------"

# Check last 10 refresh attempts
tail -100 logs/tier1_paper_rest.log | grep -i "token refresh" | tail -5

echo ""
echo "⏰ Expected: Token refresh every 10 minutes"
echo "⚠️  WARNING: If refresh stops, API access will be BLOCKED!"
echo ""
echo "Next expected refresh: ~10 minutes from last successful refresh"
echo ""

# Check for any errors
ERRORS=$(tail -50 logs/tier1_paper_rest.log | grep -i "error\|401\|403" | wc -l)
if [ $ERRORS -gt 0 ]; then
    echo "⚠️  Found $ERRORS recent error(s) in logs:"
    echo "----------------------------------------"
    tail -50 logs/tier1_paper_rest.log | grep -i "error\|401\|403" | tail -3
else
    echo "✅ No recent errors detected"
fi

echo ""
echo "=========================================="
