#!/bin/bash
###############################################################################
# Hybrid Regime-Aware Trading System - Quick Start Script
#
# Usage: ./start_hybrid_paper_trading.sh
###############################################################################

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Hybrid Regime-Aware Trading System${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Check prerequisites
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

# Check Python virtual environment
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found"
    exit 1
fi

# Check access token
if [ ! -f ".access_token" ]; then
    echo "❌ No access token found"
    echo "Please run OAuth flow first:"
    echo "  .venv/bin/python get_standard_auth_url.py"
    echo "  .venv/bin/python exchange_token_simple.py <code>"
    exit 1
fi

# Check HMM model
if [ ! -d "models/hmm/regime_model" ]; then
    echo "❌ HMM model not found"
    exit 1
fi

# Check regime-specific models
if [ ! -f "models/xgboost/regime_aware_real_labels/xgboost_regime_0_real_labels.joblib" ]; then
    echo "❌ Regime 0 model not found"
    exit 1
fi

if [ ! -f "models/xgboost/regime_aware_real_labels/xgboost_regime_2_real_labels.joblib" ]; then
    echo "❌ Regime 2 model not found"
    exit 1
fi

if [ ! -f "models/xgboost/regime_aware_real_labels/xgboost_generic_real_labels.joblib" ]; then
    echo "❌ Generic model not found"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites met${NC}"
echo ""

# Step 2: Show configuration
echo -e "${YELLOW}Step 2: Configuration${NC}"
echo "  Probability Threshold: 40%"
echo "  Min Bars Between Trades: 30 (2.5 hours)"
echo "  Triple-Barrier Exits:"
echo "    - Take Profit: 0.3%"
echo "    - Stop Loss: 0.2%"
echo "    - Time Stop: 30 minutes"
echo "  Expected Performance:"
echo "    - Trades per day: 3.92"
echo "    - Win rate: 51.80%"
echo "    - Sharpe ratio: 0.74"
echo ""

# Step 3: Start system
echo -e "${YELLOW}Step 3: Starting hybrid system...${NC}"
echo ""

.venv/bin/python start_paper_trading.py
