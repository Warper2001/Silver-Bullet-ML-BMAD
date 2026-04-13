#!/bin/bash
###############################################################################
# Hybrid Regime-Aware Trading System - Live Monitor
###############################################################################

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

clear
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}HYBRID TRADING SYSTEM MONITOR${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

while true; do
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}HYBRID TRADING SYSTEM MONITOR${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Status Check: $(date '+%Y-%m-%d %H:%M:%S UTC')${NC}"
    echo ""
    
    # Check if process is running
    if pgrep -f "start_paper_trading.py" > /dev/null; then
        echo -e "${GREEN}✅ Process: RUNNING${NC}"
        PID=$(pgrep -f "start_paper_trading.py")
        echo -e "   PID: $PID"
    else
        echo -e "${RED}❌ Process: NOT RUNNING${NC}"
    fi
    echo ""
    
    # Signal Count
    SIGNAL_COUNT=$(grep -c "Signal #" logs/hybrid_trading.log 2>/dev/null || echo "0")
    echo -e "${BLUE}📊 Signals Generated Today:${NC} $SIGNAL_COUNT"
    echo ""
    
    # Current Price
    echo -e "${BLUE}💰 Current MNQ Price:${NC}"
    tail -10 logs/paper_trading.log | grep "Bar created" | tail -1 || echo "   No recent data"
    echo ""
    
    # Recent Regime Detection
    echo -e "${BLUE}🎯 Recent Regime Detection:${NC}"
    grep "Regime=" logs/hybrid_trading.log 2>/dev/null | tail -5 || echo "   No regime data yet"
    echo ""
    
    # System Health
    echo -e "${BLUE}🏥 System Health:${NC}"
    
    # Check for errors in last 5 minutes
    ERRORS=$(tail -1000 logs/hybrid_trading.log 2>/dev/null | grep -c "ERROR" || echo "0")
    if [ "$ERRORS" -eq 0 ]; then
        echo -e "   ${GREEN}✅ No errors in recent logs${NC}"
    else
        echo -e "   ${RED}❌ $ERRORS errors detected${NC}"
    fi
    
    # Check if processing
    PROCESSING=$(tail -100 logs/hybrid_trading.log 2>/dev/null | grep -c "Predicted regimes" || echo "0")
    echo -e "   ${GREEN}✅ Processing: $PROCESSING regime predictions${NC}"
    echo ""
    
    # Today's Performance
    echo -e "${BLUE}📈 Today's Performance:${NC}"
    TODAY=$(date +%Y-%m-%d)
    TODAY_SIGNALS=$(grep "Signal #" logs/hybrid_trading.log 2>/dev/null | grep -c "$TODAY" || echo "0")
    echo -e "   Signals: $TODAY_SIGNALS"
    echo -e "   Target: 3-4 per day"
    
    if [ "$TODAY_SIGNALS" -ge 3 ] && [ "$TODAY_SIGNALS" -le 6 ]; then
        echo -e "   ${GREEN}✅ On track${NC}"
    elif [ "$TODAY_SIGNALS" -gt 6 ]; then
        echo -e "   ${YELLOW}⚠️  Above target (may need adjustment)${NC}"
    else
        echo -e "   ${YELLOW}⏳ Waiting for signals...${NC}"
    fi
    echo ""
    
    # Recent Log Activity
    echo -e "${BLUE}📝 Recent Activity (Last 5):${NC}"
    tail -100 logs/hybrid_trading.log 2>/dev/null | grep -E "(Bar created|Predicted regimes|Signal #)" | tail -5 || echo "   No recent activity"
    echo ""
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    sleep 10
done
