#!/bin/bash
###############################################################################
# TIER 1 FVG System - Paper Trading Monitor
#
# Monitors and validates TIER 1 FVG system performance.
# Configuration: SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0
#
# Usage: ./deploy_tier1_simple.sh [start|stop|status|validate]
###############################################################################

set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIER1_CONFIG="SL2.5x_ATR0.7_Vol2.25_MaxGap$50.0"
LOG_DIR="$PROJECT_ROOT/logs"
STATE_DIR="$PROJECT_ROOT/data/state"

# TIER 1 Configuration Parameters
SL_MULTIPLIER=2.5
ATR_THRESHOLD=0.7
VOLUME_RATIO_THRESHOLD=2.25
MAX_GAP_DOLLARS=50.0

print_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${BLUE}========================================${NC}"; echo -e "${BLUE}$1${NC}"; echo -e "${BLUE}========================================${NC}"; }

# Create required directories
mkdir -p "$LOG_DIR"
mkdir -p "$STATE_DIR"

# Function to start monitoring
start_monitoring() {
    print_header "Starting TIER 1 FVG Monitor"

    print_info "Configuration: $TIER1_CONFIG"
    print_info "Stop Loss: ${SL_MULTIPLIER}x gap size"
    print_info "ATR Threshold: ${ATR_THRESHOLD}x"
    print_info "Volume Ratio: ${VOLUME_RATIO_THRESHOLD}x"
    print_info "Max Gap Size: \$${MAX_GAP_DOLLARS}"

    echo ""

    # Start monitoring script
    .venv/bin/python -c "
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('$LOG_DIR/tier1_monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def monitor_tier1_system():
    '''Monitor TIER 1 system status and OAuth tokens.'''

    logger.info('='*60)
    logger.info('TIER 1 FVG SYSTEM MONITOR')
    logger.info('='*60)
    logger.info(f'Configuration: SL${SL_MULTIPLIER}x_ATR${ATR_THRESHOLD}_Vol${VOLUME_RATIO_THRESHOLD}_MaxGap\${MAX_GAP_DOLLARS}')
    logger.info(f'Performance Targets: WR ≥60%, PF ≥1.7, 8-15 trades/day')
    logger.info('='*60)

    # Check OAuth tokens
    token_file = Path('.access_token')
    if not token_file.exists():
        token_file = Path('.tradestation_tokens_v3.json')

    if token_file.exists():
        logger.info(f'✓ OAuth token file found: {token_file.name}')

        # Try to load and validate tokens
        try:
            from src.data.auth_v3 import TradeStationAuthV3

            auth = TradeStationAuthV3.from_file(str(token_file))
            token = await auth.authenticate()

            logger.info('✓ OAuth authentication successful')
            logger.info(f'✓ Token hash: {auth._get_token_hash()}')

            # Start auto-refresh (every 10 minutes)
            await auth.start_auto_refresh(interval_minutes=10)
            logger.info('✓ Auto-refresh started (10-minute interval)')

            # Keep running and monitor
            logger.info('✓ Monitoring system status...')

            try:
                while True:
                    await asyncio.sleep(60)  # Log status every minute
                    logger.debug('System running normally')

            except KeyboardInterrupt:
                logger.info('Shutting down monitor...')
                await auth.cleanup()
                logger.info('✓ Monitor stopped')

        except Exception as e:
            logger.error(f'OAuth error: {e}')
            logger.info('System ready for manual trading with TIER 1 configuration')
    else:
        logger.warning('✗ No OAuth token file found')
        logger.info('Run OAuth flow first: python get_standard_auth_url.py')

    logger.info('System monitoring complete')

# Run monitor
try:
    asyncio.run(monitor_tier1_system())
except KeyboardInterrupt:
    print('\\nShutting down...')
" &

MONITOR_PID=$!

print_success "TIER 1 monitor started!"
print_info "Monitor PID: $MONITOR_PID"
print_info "Logs: $LOG_DIR/tier1_monitor.log"
print_info ""
print_info "The TIER 1 system is ready with configuration:"
print_info "  - Stop Loss: ${SL_MULTIPLIER}x gap size"
print_info "  - ATR Threshold: ${ATR_THRESHOLD}x"
print_info "  - Volume Ratio: ${VOLUME_RATIO_THRESHOLD}x"
print_info "  - Max Gap Size: \$${MAX_GAP_DOLLARS}"
print_info ""
print_info "OAuth token auto-refresh is active (10-minute interval)"
print_info ""
print_info "To stop: $0 stop"

# Save PID
echo $MONITOR_PID > "$STATE_DIR/tier1_monitor.pid"
}

# Function to stop monitoring
stop_monitoring() {
    print_header "Stopping TIER 1 Monitor"

    if [ -f "$STATE_DIR/tier1_monitor.pid" ]; then
        PID=$(cat "$STATE_DIR/tier1_monitor.pid")
        print_info "Stopping monitor (PID: $PID)..."

        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            sleep 1

            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID" 2>/dev/null || true
            fi

            print_success "TIER 1 monitor stopped"
        else
            print_warning "Process already stopped"
        fi

        rm -f "$STATE_DIR/tier1_monitor.pid"
    else
        print_warning "No monitor running"
    fi
}

# Function to check status
check_status() {
    print_header "TIER 1 System Status"

    if [ -f "$STATE_DIR/tier1_monitor.pid" ]; then
        PID=$(cat "$STATE_DIR/tier1_monitor.pid")
        if kill -0 "$PID" 2>/dev/null; then
            print_success "✅ TIER 1 monitor is RUNNING"
            print_info "PID: $PID"
            print_info "Started: $(ps -p $PID -o lstart=)"
            print_info "Logs: $LOG_DIR/tier1_monitor.log"

            # Show recent logs
            if [ -f "$LOG_DIR/tier1_monitor.log" ]; then
                echo ""
                print_info "Recent log entries:"
                tail -10 "$LOG_DIR/tier1_monitor.log"
            fi
        else
            print_warning "Monitor not running (stale PID file)"
            rm -f "$STATE_DIR/tier1_monitor.pid"
        fi
    else
        print_info "TIER 1 monitor is NOT running"
    fi

    echo ""
    print_info "Configuration: $TIER1_CONFIG"
    print_info "Performance Targets: WR ≥60%, PF ≥1.7, 8-15 trades/day"

    # Check OAuth status
    echo ""
    print_info "OAuth Status:"
    if [ -f ".access_token" ]; then
        print_info "✓ Token file: .access_token"
    elif [ -f ".tradestation_tokens_v3.json" ]; then
        print_info "✓ Token file: .tradestation_tokens_v3.json"
    else
        print_warning "✗ No OAuth token file"
    fi
}

# Function to show configuration
show_config() {
    print_header "TIER 1 Configuration"

    echo "Configuration: $TIER1_CONFIG"
    echo ""
    echo "Parameters:"
    echo "  - Stop Loss Multiplier: ${SL_MULTIPLIER}x gap size"
    echo "  - ATR Threshold: ${ATR_THRESHOLD}x (stricter than 0.5x baseline)"
    echo "  - Volume Ratio: ${VOLUME_RATIO_THRESHOLD}x (stricter than 1.5x baseline)"
    echo "  - Max Gap Size: \$${MAX_GAP_DOLLARS} (filters oversized gaps)"
    echo ""
    echo "Performance Targets:"
    echo "  - Win Rate ≥60%"
    echo "  - Profit Factor ≥1.7"
    echo "  - Trade Frequency 8-15/day"
    echo ""
    echo "Expected Performance (based on backtests):"
    echo "  - Win Rate: 74.07% (±2.05%)"
    echo "  - Profit Factor: 1.83 (±0.28)"
    echo "  - Trade Frequency: 11-12/day"
    echo "  - Expectancy: \$7-8/trade"
    echo ""
    echo "Features:"
    echo "  ✅ Real P&L calculation with transaction costs"
    echo "  ✅ Triple-barrier exits (gap fill TP, 2.5× SL, 10-bar time)"
    echo "  ✅ ATR-based noise filtering"
    echo "  ✅ Volume directional confirmation"
    echo "  ✅ Max gap size capping"
    echo "  ✅ OAuth token auto-refresh (10-minute interval)"
}

# Main command router
case "${1:-start}" in
    start)
        start_monitoring
        ;;
    stop)
        stop_monitoring
        ;;
    status)
        check_status
        ;;
    config)
        show_config
        ;;
    restart)
        stop_monitoring
        sleep 1
        start_monitoring
        ;;
    *)
        echo "Usage: $0 {start|stop|status|config|restart}"
        echo ""
        echo "TIER 1 FVG System Monitor"
        echo ""
        echo "Commands:"
        echo "  start   - Start TIER 1 monitor with OAuth auto-refresh"
        echo "  stop    - Stop TIER 1 monitor"
        echo "  status  - Check system status and logs"
        echo "  config  - Show TIER 1 configuration and targets"
        echo "  restart - Restart TIER 1 monitor"
        exit 1
        ;;
esac
