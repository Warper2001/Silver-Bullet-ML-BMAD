"""Main Streamlit application for Silver-Bullet-ML-BMAD dashboard."""

import streamlit as st
from datetime import datetime

from dashboard.navigation import render_page
from dashboard.shared_state import get_system_status

# Auto-refresh interval in seconds
REFRESH_INTERVAL = 2


def main() -> None:
    """Main Streamlit application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Silver-Bullet-ML-BMAD",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state for auto-refresh
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = 0

    # Header
    st.title("Silver-Bullet-ML-BMAD Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # System Status Banner
    status = get_system_status()
    status_emoji = "🟢" if status == "RUNNING" else "🟡" if status == "HALTED" else "🔴"
    st.info(f"{status_emoji} System Status: {status}")

    # Navigation
    page = st.sidebar.radio(
        "Navigate to:",
        ["Overview", "Positions", "Signals", "Charts", "Drift Monitoring", "Settings", "Logs"],
        label_visibility="collapsed",
    )

    # Render selected page
    render_page(page)

    # Auto-refresh mechanism
    import time

    current_time = time.time()
    if current_time - st.session_state["last_refresh"] >= REFRESH_INTERVAL:
        st.session_state["last_refresh"] = current_time
        time.sleep(0.1)  # Small delay to prevent tight rerun loop
        st.rerun()


if __name__ == "__main__":
    main()
