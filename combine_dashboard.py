import streamlit as st
import asyncio
import httpx
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Streamlit config MUST be first
st.set_page_config(page_title="50k Combine Tracker", page_icon="🎯", layout="wide")

# Setup paths
base_dir = Path("/root/Silver-Bullet-ML-BMAD")
sys.path.insert(0, str(base_dir))

# Mock or import TradeStation Auth
try:
    from src.data.auth_v3 import TradeStationAuthV3
except ImportError:
    st.error("TradeStationAuthV3 not found. Make sure you are running from the repo root.")
    st.stop()

try:
    from src.research.projectx_auth import ProjectXAuth
    from src.research.projectx_client import ProjectXClient
    USE_PROJECTX = True
except ImportError:
    USE_PROJECTX = False

SIM_ACCOUNT_ID = "SIM2797251F"
BASE_EQUITY = 101527.70  # The starting balance for this SIM account iteration
PROFIT_TARGET = 3000.00
TRAILING_DD_LIMIT = 2000.00
DAILY_LOSS_LIMIT = 1000.00

# CSS for better metric styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        color: #A0A0A0;
    }
    .positive { color: #00FF00; }
    .negative { color: #FF4136; }
    .warning { color: #FF851B; }
</style>
""", unsafe_allow_html=True)

# Fetch functions
@st.cache_data(ttl=15) # Cache token for 15 seconds to avoid spamming auth
def get_auth_token():
    # We must run asyncio in Streamlit synchronously
    async def fetch():
        auth = TradeStationAuthV3.from_file(str(base_dir / '.access_token'))
        token = await auth.authenticate()
        
        px_token = None
        if USE_PROJECTX:
            try:
                px_auth = ProjectXAuth.from_file(str(base_dir / '.projectx_api_key'))
                px_token = await px_auth.authenticate()
            except Exception as e:
                st.warning(f"ProjectX Auth failed: {e}")
        return token, px_token
    return asyncio.run(fetch())

def get_account_data(tokens):
    ts_token, px_token = tokens
    async def fetch():
        ts_headers = {"Authorization": f"Bearer {ts_token}", "Accept": "application/json"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            bal_resp = await client.get(f"https://sim-api.tradestation.com/v3/brokerage/accounts/{SIM_ACCOUNT_ID}/balances", headers=ts_headers)
            pos_resp = await client.get(f"https://sim-api.tradestation.com/v3/brokerage/accounts/{SIM_ACCOUNT_ID}/positions", headers=ts_headers)
            
            px_bal = {}
            if px_token:
                px_headers = {"Authorization": f"Bearer {px_token}", "Accept": "application/json"}
                # ProjectX doesn't have an open /Account/balances endpoint, 
                # so we can't fetch equity directly. Let's just catch the 404 without breaking.
                px_resp = await client.get("https://api.topstepx.com/api/Account/balances", headers=px_headers)
                if px_resp.status_code == 200:
                    px_bal = px_resp.json()
                else:
                    px_bal = []
            
            return bal_resp.json(), pos_resp.json(), px_bal
    return asyncio.run(fetch())

def main():
    st.title("🎯 BMAD 50k Combine Tracker")
    st.markdown("Real-time monitoring of S26 Crypto and Stat Arb bots.")
    
    try:
        tokens = get_auth_token()
        bal_data, pos_data, px_bal = get_account_data(tokens)
    except Exception as e:
        st.error(f"Error fetching data from TradeStation API: {e}")
        st.stop()
        
    balances = bal_data.get("Balances", [{}])[0]
    ts_equity = float(balances.get("Equity", BASE_EQUITY))
    ts_todays_pnl = float(balances.get("TodaysProfitLoss", 0.0))
    open_positions = pos_data.get("Positions", [])
    
    # -----------------------------------------
    # TRADESTATION SIM METRICS
    # -----------------------------------------
    ts_net_pnl = ts_equity - BASE_EQUITY
    
    if 'ts_peak_equity' not in st.session_state:
        st.session_state.ts_peak_equity = max(BASE_EQUITY, ts_equity)
    else:
        st.session_state.ts_peak_equity = max(st.session_state.ts_peak_equity, ts_equity)
        
    ts_current_dd = st.session_state.ts_peak_equity - ts_equity
    ts_dd_remaining = TRAILING_DD_LIMIT - ts_current_dd

    st.subheader("🟠 TradeStation SIM (Execution Engine)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Account Equity</div>
            <div class="metric-value">${ts_equity:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        color = "positive" if ts_todays_pnl >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Today's P&L</div>
            <div class="metric-value {color}">${ts_todays_pnl:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        color = "positive" if ts_net_pnl >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Net P&L</div>
            <div class="metric-value {color}">${ts_net_pnl:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        color = "negative" if ts_dd_remaining < 500 else "positive"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Est. Drawdown Buffer</div>
            <div class="metric-value {color}">${ts_dd_remaining:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------
    # TOPSTEPX METRICS
    # -----------------------------------------
    if px_bal and isinstance(px_bal, list) and len(px_bal) > 0:
        px_equity = float(px_bal[0].get("balance", BASE_EQUITY))
        px_todays_pnl = float(px_bal[0].get("dailyPnl", 0.0))
        px_net_pnl = px_equity - BASE_EQUITY
        
        if 'px_peak_equity' not in st.session_state:
            st.session_state.px_peak_equity = max(BASE_EQUITY, px_equity)
        else:
            st.session_state.px_peak_equity = max(st.session_state.px_peak_equity, px_equity)
            
        px_current_dd = st.session_state.px_peak_equity - px_equity
        px_dd_remaining = TRAILING_DD_LIMIT - px_current_dd
        px_target_remaining = max(0, PROFIT_TARGET - px_net_pnl)

        st.subheader("🟢 TopstepX (Official Prop Firm Ledger)")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Account Equity</div>
                <div class="metric-value">${px_equity:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            color = "positive" if px_todays_pnl >= 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Today's P&L</div>
                <div class="metric-value {color}">${px_todays_pnl:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            color = "warning" if px_target_remaining > 0 else "positive"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Target Remaining</div>
                <div class="metric-value {color}">${px_target_remaining:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c4:
            color = "negative" if px_dd_remaining < 500 else "positive"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Drawdown Buffer</div>
                <div class="metric-value {color}">${px_dd_remaining:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Progress Bar to Target
        progress = max(0.0, min(1.0, px_net_pnl / PROFIT_TARGET)) if PROFIT_TARGET > 0 else 0
        st.progress(progress)
        st.caption(f"{progress*100:.1f}% to Funding (TopstepX Ledger)")
    else:
        st.info("TopstepX API not available. Cannot show official combine ledger.")

    st.divider()
    
    # Open Positions
    st.subheader(f"Open Positions ({len(open_positions)})")
    if open_positions:
        for pos in open_positions:
            qty = pos.get('Quantity', '0')
            sym = pos.get('Symbol', 'Unknown')
            pnl = float(pos.get('TotalCost', 0)) # simplified
            unrealized = float(pos.get('UnrealizedProfitLoss', 0))
            st.write(f"**{sym}**: {qty} contracts | Unrealized P&L: ${unrealized:.2f}")
    else:
        st.info("Bots are currently flat. Waiting for high-probability setups.")
        
    # Auto-refresh mechanism
    st_autorefresh = st.empty()
    st_autorefresh.button("🔄 Refresh Data")

if __name__ == "__main__":
    main()
