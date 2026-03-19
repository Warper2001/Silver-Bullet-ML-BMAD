"""Page routing logic for Streamlit dashboard."""

import logging
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared_state import (
    Direction,
    MarkerType,
    calculate_data_age,
    calculate_time_remaining,
    exit_position,
    filter_signals,
    format_ml_probability_bar,
    format_position_pnl,
    format_resource_usage,
    format_signal_status,
    get_account_metrics,
    get_alerts_summary,
    get_dollar_bars,
    get_default_config,
    get_open_positions,
    get_pattern_overlays,
    get_resource_color,
    get_silver_bullet_signals,
    get_system_config,
    get_system_health,
    get_trade_markers,
    is_data_stale,
    save_system_config,
    validate_config,
)

logger = logging.getLogger(__name__)


def render_overview():
    """Render overview page with account metrics and health indicators."""
    st.header("📊 Account Overview")

    try:
        # Fetch metrics
        metrics = get_account_metrics()

        # Check for stale data
        age_seconds = (datetime.now() - metrics.last_update).total_seconds()
        if age_seconds > 30:
            st.warning(f"⚠️ Data is {age_seconds:.0f} seconds old")

        # Top row: Equity, Daily P&L, Drawdown
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Account Equity",
                value=f"${metrics.equity:,.2f}",
                delta=f"{metrics.daily_change_pct:+.2f}% ({metrics.daily_change_usd:+,.2f})",
                delta_color="normal"
            )

        with col2:
            pnl_color = "normal" if metrics.daily_pnl >= 0 else "inverse"
            st.metric(
                label="Daily P&L",
                value=f"${metrics.daily_pnl:+,.2f}",
                delta="" if metrics.daily_pnl == 0 else ("↑ Profit" if metrics.daily_pnl > 0 else "↓ Loss"),
                delta_color=pnl_color
            )

        with col3:
            drawdown_pct = (metrics.daily_drawdown / metrics.daily_loss_limit) * 100
            st.metric(
                label="Daily Drawdown",
                value=f"${metrics.daily_drawdown:,.2f}",
                delta=f"of ${metrics.daily_loss_limit:,.2f} limit"
            )
            st.progress(min(drawdown_pct / 100, 1.0))

        # Bottom row: Positions, Trades, Uptime
        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric(
                label="Open Positions",
                value=f"{metrics.open_positions_count} positions",
                delta=f"{metrics.open_contracts} contracts"
            )

        with col5:
            st.metric(
                label="Today's Trades",
                value=f"{metrics.trade_count} trades",
                delta=f"{metrics.win_rate:.1f}% win rate"
            )

        with col6:
            st.metric(
                label="System Uptime",
                value=metrics.system_uptime,
                delta=metrics.last_update.strftime("%H:%M:%S EST")
            )

    except FileNotFoundError as e:
        st.error(f"Shared state not found: {e}")
        st.info("Ensure the trading system is running.")
        logger.error(f"Shared state not found: {e}", exc_info=True)
        return
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")
        logger.error(f"Error loading account metrics: {e}", exc_info=True)
        return

    # Story 8.7: System Health Section
    st.markdown("---")
    st.subheader("🏥 System Health")

    try:
        # Fetch health data
        health = get_system_health()
        alerts = get_alerts_summary()

        # Health indicators row
        hcol1, hcol2, hcol3, hcol4 = st.columns(4)

        with hcol1:
            # API Connection Status
            api_status = "✅ Connected" if health.api_status.connected else "❌ Disconnected"
            api_delta = f"Latency: {health.api_status.ping_latency_ms:.0f}ms"
            st.metric(
                label="API Status",
                value=api_status,
                delta=api_delta
            )

        with hcol2:
            # Data Freshness
            staleness_seconds = calculate_data_age(health.api_status.last_ping_time)
            is_stale = is_data_stale(health.api_status.last_ping_time)
            data_delta = "Fresh" if not is_stale else "⚠️ Stale"
            st.metric(
                label="Data Freshness",
                value=f"{staleness_seconds}s ago",
                delta=data_delta
            )

        with hcol3:
            # Active Alerts
            alert_emoji = "✅" if alerts['count'] == 0 else "⚠️"
            st.metric(
                label="Active Alerts",
                value=f"{alerts['count']} alerts",
                delta=alert_emoji
            )

        with hcol4:
            # System Uptime (from health monitoring)
            st.metric(
                label="System Uptime",
                value=health.system_uptime,
                delta=f"Restarted: {health.last_restart_time.strftime('%H:%M:%S')}"
            )

        # Resource Usage Section
        st.markdown("---")
        st.subheader("💻 Resource Usage")

        resources = health.resources

        rcol1, rcol2, rcol3 = st.columns(3)

        with rcol1:
            # CPU Usage
            cpu_color, cpu_emoji = format_resource_usage(resources.cpu_percent)
            st.metric(
                label="CPU Usage",
                value=f"{resources.cpu_percent:.1f}%",
                delta=cpu_emoji
            )
            st.progress(resources.cpu_percent / 100)

        with rcol2:
            # Memory Usage
            mem_color, mem_emoji = format_resource_usage(resources.memory_percent)
            st.metric(
                label="Memory Usage",
                value=f"{resources.memory_percent:.1f}%",
                delta=mem_emoji
            )
            st.progress(resources.memory_percent / 100)

        with rcol3:
            # Disk Usage
            disk_color, disk_emoji = format_resource_usage(resources.disk_percent)
            st.metric(
                label="Disk Usage",
                value=f"{resources.disk_percent:.1f}%",
                delta=disk_emoji
            )
            st.progress(resources.disk_percent / 100)

        # Pipeline Status Section
        st.markdown("---")
        st.subheader("⚙️ Pipeline Status")

        pcol1, pcol2, pcol3, pcol4 = st.columns(4)

        with pcol1:
            data_flow = health.data_flow_status
            data_flow_icon = "✅" if data_flow.is_healthy else "❌"
            data_flow_delta = f"Error: {data_flow.error_count}" if data_flow.error_count > 0 else "OK"
            st.metric(
                label="Data Flow",
                value=data_flow_icon,
                delta=data_flow_delta
            )

        with pcol2:
            signal_detection = health.signal_detection_status
            signal_icon = "✅" if signal_detection.is_healthy else "❌"
            signal_delta = "OK" if signal_detection.is_healthy else "Failed"
            st.metric(
                label="Signal Detection",
                value=signal_icon,
                delta=signal_delta
            )

        with pcol3:
            ml_prediction = health.ml_prediction_status
            ml_icon = "✅" if ml_prediction.is_healthy else "❌"
            ml_delta = "OK" if ml_prediction.is_healthy else "Failed"
            st.metric(
                label="ML Prediction",
                value=ml_icon,
                delta=ml_delta
            )

        with pcol4:
            execution = health.execution_status
            exec_icon = "✅" if execution.is_healthy else "❌"
            exec_delta = "OK" if execution.is_healthy else "Failed"
            st.metric(
                label="Execution",
                value=exec_icon,
                delta=exec_delta
            )

        # Detailed Metrics Modal (Expander)
        with st.expander("📊 Detailed Health Metrics"):
            st.markdown("### API Connection Details")
            api_col1, api_col2, api_col3 = st.columns(3)
            with api_col1:
                st.metric("Status", "✅ Connected" if health.api_status.connected else "❌ Disconnected")
            with api_col2:
                st.metric("Latency", f"{health.api_status.ping_latency_ms:.1f} ms")
            with api_col3:
                staleness = calculate_data_age(health.api_status.last_ping_time)
                st.metric("Last Ping", f"{staleness}s ago")

            st.markdown("---")
            st.markdown("### Pipeline Component Details")

            # Pipeline details table
            pipeline_data = {
                "Component": [
                    "Data Flow",
                    "Signal Detection",
                    "ML Prediction",
                    "Execution"
                ],
                "Status": [
                    "✅ Healthy" if health.data_flow_status.is_healthy else "❌ Failed",
                    "✅ Healthy" if health.signal_detection_status.is_healthy else "❌ Failed",
                    "✅ Healthy" if health.ml_prediction_status.is_healthy else "❌ Failed",
                    "✅ Healthy" if health.execution_status.is_healthy else "❌ Failed"
                ],
                "Last Execution": [
                    health.data_flow_status.last_execution_time.strftime("%H:%M:%S"),
                    health.signal_detection_status.last_execution_time.strftime("%H:%M:%S"),
                    health.ml_prediction_status.last_execution_time.strftime("%H:%M:%S"),
                    health.execution_status.last_execution_time.strftime("%H:%M:%S")
                ],
                "Error Count": [
                    health.data_flow_status.error_count,
                    health.signal_detection_status.error_count,
                    health.ml_prediction_status.error_count,
                    health.execution_status.error_count
                ]
            }

            st.dataframe(pipeline_data, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Failed to load health indicators: {e}")
        logger.error(f"Error loading health indicators: {e}", exc_info=True)
        return


def render_positions():
    """Render positions page with open positions table."""
    st.header("💼 Open Positions")

    try:
        # Fetch positions
        positions = get_open_positions()

        if not positions:
            st.info("No open positions")
            return

        # Convert to DataFrame for display
        df = positions_to_dataframe(positions)

        # Display table with all required columns
        st.dataframe(
            df,
            column_order=[
                "Signal ID",
                "Direction",
                "Entry Price",
                "Current Price",
                "P&L ($)",
                "P&L (%)",
                "Upper Barrier",
                "Lower Barrier",
                "Vertical Barrier",
                "Time Remaining",
                "Confidence",
                "ML Probability",
            ],
            hide_index=True,
        )

        # Manual exit section
        st.divider()
        st.subheader("Manual Exit")

        # Add exit buttons for each position
        for position in positions:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{position.signal_id}** - {position.direction.value} @ ${position.entry_price:,.2f}")
            with col2:
                if st.button("Exit", key=f"exit_{position.signal_id}"):
                    st.session_state[f"show_exit_confirm_{position.signal_id}"] = True

            # Show password confirmation if exit button clicked
            if st.session_state.get(f"show_exit_confirm_{position.signal_id}", False):
                with st.expander(f"Confirm Exit for {position.signal_id}", expanded=True):
                    password = st.text_input(
                        "Enter password to confirm:",
                        type="password",
                        key=f"password_{position.signal_id}"
                    )

                    col_confirm, col_cancel = st.columns(2)
                    with col_confirm:
                        if st.button("Confirm Exit", key=f"confirm_{position.signal_id}"):
                            if exit_position(position.signal_id, password):
                                st.success(f"Position {position.signal_id} exited successfully")
                                st.session_state[f"show_exit_confirm_{position.signal_id}"] = False
                                st.rerun()
                            else:
                                st.error("Invalid password or exit failed")
                    with col_cancel:
                        if st.button("Cancel", key=f"cancel_{position.signal_id}"):
                            st.session_state[f"show_exit_confirm_{position.signal_id}"] = False
                            st.rerun()

    except FileNotFoundError as e:
        st.error(f"Shared state not found: {e}")
        st.info("Ensure the trading system is running.")
        logger.error(f"Shared state not found: {e}", exc_info=True)
        return
    except Exception as e:
        st.error(f"Failed to load positions: {e}")
        logger.error(f"Error loading positions: {e}", exc_info=True)
        return


def positions_to_dataframe(positions):
    """Convert positions list to formatted DataFrame.

    Args:
        positions: List of OpenPosition objects

    Returns:
        pandas.DataFrame with formatted columns
    """
    data = []
    for pos in positions:
        data.append(
            {
                "Signal ID": pos.signal_id,
                "Direction": (
                    "📈 LONG" if pos.direction == Direction.LONG else "📉 SHORT"
                ),
                "Entry Price": f"${pos.entry_price:,.2f}",
                "Current Price": f"${pos.current_price:,.2f}",
                "P&L ($)": format_position_pnl(pos.pnl_usd),
                "P&L (%)": f"{pos.pnl_pct:+.2f}%",
                "Upper Barrier": f"${pos.barriers.upper_barrier:,.2f}",
                "Lower Barrier": f"${pos.barriers.lower_barrier:,.2f}",
                "Vertical Barrier": pos.barriers.vertical_barrier.strftime("%H:%M:%S"),
                "Time Remaining": calculate_time_remaining(pos.barriers.vertical_barrier),
                "Confidence": "★" * pos.confidence,
                "ML Probability": pos.ml_probability,
            }
        )

    return pd.DataFrame(data)


def render_signals():
    """Render signals page with live signals table."""
    st.header("🎯 Live Signals")

    try:
        # Initialize filter session state if not exists
        if 'signal_filter_status' not in st.session_state:
            st.session_state.signal_filter_status = "All"
        if 'signal_filter_direction' not in st.session_state:
            st.session_state.signal_filter_direction = "All"
        if 'signal_filter_confidence' not in st.session_state:
            st.session_state.signal_filter_confidence = "All"

        # Add filter controls
        col1, col2, col3 = st.columns(3)

        with col1:
            st.selectbox(
                "Status Filter:",
                options=["All", "Filtered", "Executed", "Rejected"],
                key="signal_filter_status"
            )

        with col2:
            st.selectbox(
                "Direction Filter:",
                options=["All", "LONG", "SHORT"],
                key="signal_filter_direction"
            )

        with col3:
            st.selectbox(
                "Confidence Filter:",
                options=["All", "5★", "4★+", "3★+"],
                key="signal_filter_confidence"
            )

        st.divider()

        # Fetch signals
        signals = get_silver_bullet_signals()

        # Apply filters
        filtered_signals = filter_signals(
            signals,
            status=st.session_state.signal_filter_status,
            direction=st.session_state.signal_filter_direction,
            confidence=st.session_state.signal_filter_confidence
        )

        if not filtered_signals:
            st.info("No signals yet today")
            return

        # Convert to DataFrame for display
        df = signals_to_dataframe(filtered_signals)

        # Display table
        st.dataframe(df, hide_index=True)

    except FileNotFoundError as e:
        st.error(f"Shared state not found: {e}")
        st.info("Ensure the trading system is running.")
        logger.error(f"Shared state not found: {e}", exc_info=True)
        return
    except Exception as e:
        st.error(f"Failed to load signals: {e}")
        logger.error(f"Error loading signals: {e}", exc_info=True)
        return


def signals_to_dataframe(signals):
    """Convert signals list to formatted DataFrame.

    Args:
        signals: List of SilverBulletSignal objects

    Returns:
        pandas.DataFrame with formatted columns
    """
    data = []
    for signal in signals:
        data.append({
            "Timestamp": signal.timestamp.strftime("%H:%M:%S"),
            "Direction": (
                "📈 LONG" if signal.direction == Direction.LONG else "📉 SHORT"
            ),
            "Confidence": "★" * signal.confidence,
            "ML Probability": format_ml_probability_bar(signal.ml_probability),
            "MSS Present": "✓" if signal.mss_present else "✗",
            "FVG Present": "✓" if signal.fvg_present else "✗",
            "Sweep Present": "✓" if signal.sweep_present else "✗",
            "Time Window": signal.time_window,
            "Status": format_signal_status(signal.status)
        })

    return pd.DataFrame(data)


def render_charts():
    """Render charts page with Dollar Bar chart and pattern overlays."""
    st.header("📈 Dollar Bar Charts")

    try:
        # Initialize time range session state
        if 'chart_time_range' not in st.session_state:
            st.session_state.chart_time_range = "today"

        # Time range selector and refresh controls
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            time_range = st.selectbox(
                "Time Range:",
                options=["hour", "today", "week"],
                index=["hour", "today", "week"].index(
                    st.session_state.chart_time_range
                ),
                key="chart_time_range_selector"
            )
            st.session_state.chart_time_range = time_range

        with col2:
            st.write("")  # Spacer

        with col3:
            if st.button("🔄 Refresh"):
                st.rerun()

        st.divider()

        # Fetch chart data with loading indicator
        with st.spinner("Loading chart data..."):
            df = get_dollar_bars(time_range)
            markers, fvg_zones = get_pattern_overlays(time_range)
            trades = get_trade_markers(time_range)

        # Handle empty data
        if df.empty:
            st.warning("No Dollar Bar data available for selected time range")
            return

        # Create Plotly figure
        fig = go.Figure()

        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Dollar Bars",
            increasing_line_color='#00FF00',  # Green
            decreasing_line_color='#FF0000',  # Red
        ))

        # Add MSS markers
        mss_bullish = [m for m in markers if m.marker_type == MarkerType.MSS_BULLISH]
        mss_bearish = [m for m in markers if m.marker_type == MarkerType.MSS_BEARISH]

        if mss_bullish:
            fig.add_trace(go.Scatter(
                x=[m.timestamp for m in mss_bullish],
                y=[m.price for m in mss_bullish],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='#00FF00',
                    line=dict(width=2)
                ),
                name='MSS Bullish',
                hovertemplate='<b>MSS Bullish</b><br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
            ))

        if mss_bearish:
            fig.add_trace(go.Scatter(
                x=[m.timestamp for m in mss_bearish],
                y=[m.price for m in mss_bearish],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='#FF0000',
                    line=dict(width=2)
                ),
                name='MSS Bearish',
                hovertemplate='<b>MSS Bearish</b><br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
            ))

        # Add FVG zones as shaded rectangles
        for zone in fvg_zones:
            fill_color = "rgba(0, 255, 0, 0.2)" if zone.direction == "bullish" else "rgba(255, 0, 0, 0.2)"
            fig.add_shape(
                type="rect",
                x0=zone.start_time,
                y0=zone.bottom_price,
                x1=zone.end_time,
                y1=zone.top_price,
                fillcolor=fill_color,
                line=dict(width=0),
                layer="below"
            )

        # Add liquidity sweep markers
        sweeps = [m for m in markers if m.marker_type == MarkerType.SWEEP]
        if sweeps:
            fig.add_trace(go.Scatter(
                x=[s.timestamp for s in sweeps],
                y=[s.price for s in sweeps],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=20,
                    color='#9467BD',  # Purple
                    line=dict(width=2)
                ),
                name='Liquidity Sweep',
                hovertemplate='<b>Sweep</b><br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
            ))

        # Add trade markers
        entries = [t for t in trades if t.trade_type == "entry"]
        exits_profit = [t for t in trades if t.trade_type == "exit" and t.pnl_usd and t.pnl_usd > 0]
        exits_loss = [t for t in trades if t.trade_type == "exit" and t.pnl_usd and t.pnl_usd <= 0]

        if entries:
            fig.add_trace(go.Scatter(
                x=[e.timestamp for e in entries],
                y=[e.price for e in entries],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=12,
                    color='#0080FF',  # Blue
                    line=dict(width=2)
                ),
                name='Entry',
                hovertemplate='<b>Entry</b><br>Signal: %{text}<br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>',
                text=[e.signal_id for e in entries]
            ))

        if exits_profit:
            fig.add_trace(go.Scatter(
                x=[e.timestamp for e in exits_profit],
                y=[e.price for e in exits_profit],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=15,
                    color='#00FF00',  # Green
                    line=dict(width=3)
                ),
                name='Exit (Profit)',
                hovertemplate=(
                    '<b>Exit (Profit)</b><br>P&L: $%{marker.customdata[0]:.2f}<br>'
                    'Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
                ),
                customdata=[[e.pnl_usd] for e in exits_profit]
            ))

        if exits_loss:
            fig.add_trace(go.Scatter(
                x=[e.timestamp for e in exits_loss],
                y=[e.price for e in exits_loss],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=15,
                    color='#FF0000',  # Red
                    line=dict(width=3)
                ),
                name='Exit (Loss)',
                hovertemplate=(
                    '<b>Exit (Loss)</b><br>P&L: $%{marker.customdata[0]:.2f}<br>'
                    'Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
                ),
                customdata=[[e.pnl_usd] for e in exits_loss]
            ))

        # Update layout with legend
        fig.update_layout(
            title=f"Dollar Bar Chart - {time_range.capitalize()}",
            xaxis_title="Time",
            yaxis_title="Price",
            hovermode='x unified',
            height=600,
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.2,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(0,0,0,0.1)'
            )
        )

        # Display chart
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError as e:
        st.error(f"Data not found: {e}")
        st.info("Ensure the trading system is running.")
        logger.error(f"Data not found: {e}", exc_info=True)
        return
    except Exception as e:
        st.error(f"Failed to load chart: {e}")
        logger.error(f"Error loading chart: {e}", exc_info=True)
        return


def render_settings():
    """Render settings page with configuration forms."""
    st.header("⚙️ System Configuration")

    try:
        # Get current and default configuration
        current_config = get_system_config()
        default_config = get_default_config()

        # Risk Limits Section
        st.subheader("Risk Limits")
        with st.expander("Risk Management Settings", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                daily_loss = st.number_input(
                    "Daily Loss Limit ($)",
                    min_value=0,
                    max_value=10000,
                    value=int(current_config.risk_limits.daily_loss_limit),
                    help="Maximum daily loss in USD"
                )

                max_drawdown = st.number_input(
                    "Maximum Drawdown (%)",
                    min_value=0,
                    max_value=100,
                    value=int(current_config.risk_limits.max_drawdown_pct),
                    help="Maximum drawdown percentage"
                )

                per_trade_risk = st.number_input(
                    "Per-Trade Risk (%)",
                    min_value=0,
                    max_value=100,
                    value=float(current_config.risk_limits.per_trade_risk_pct),
                    help="Risk per trade as percentage of account"
                )

                max_position = st.number_input(
                    "Max Position (contracts)",
                    min_value=1,
                    max_value=10,
                    value=current_config.risk_limits.max_position_contracts,
                    help="Maximum number of contracts per position"
                )

            with col2:
                st.write("**Default Values:**")
                st.write(f"Daily Loss: ${default_config.risk_limits.daily_loss_limit:.2f}")
                st.write(f"Max Drawdown: {default_config.risk_limits.max_drawdown_pct:.1f}%")
                st.write(f"Per-Trade Risk: {default_config.risk_limits.per_trade_risk_pct:.1f}%")
                st.write(f"Max Position: {default_config.risk_limits.max_position_contracts} contracts")

        # Time Windows Section
        st.subheader("Time Windows")
        with st.expander("Trading Time Windows", expanded=True):
            # London AM
            st.write("**London AM (02:00-05:00 EST)**")
            london_col1, london_col2 = st.columns(2)

            with london_col1:
                london_enabled = st.checkbox(
                    "Enable London AM",
                    value=current_config.london_am.enabled
                )

            with london_col2:
                st.write(f"Default: {'Enabled' if default_config.london_am.enabled else 'Disabled'}")

            if london_enabled:
                london_start = st.time_input(
                    "Start Time",
                    value=datetime.strptime(current_config.london_am.start_time, "%H:%M").time()
                )
                london_end = st.time_input(
                    "End Time",
                    value=datetime.strptime(current_config.london_am.end_time, "%H:%M").time()
                )

            st.markdown("---")

            # NY AM
            st.write("**NY AM (09:30-11:00 EST)**")
            ny_am_col1, ny_am_col2 = st.columns(2)

            with ny_am_col1:
                ny_am_enabled = st.checkbox(
                    "Enable NY AM",
                    value=current_config.ny_am.enabled
                )

            with ny_am_col2:
                st.write(f"Default: {'Enabled' if default_config.ny_am.enabled else 'Disabled'}")

            if ny_am_enabled:
                ny_am_start = st.time_input(
                    "Start Time",
                    value=datetime.strptime(current_config.ny_am.start_time, "%H:%M").time()
                )
                ny_am_end = st.time_input(
                    "End Time",
                    value=datetime.strptime(current_config.ny_am.end_time, "%H:%M").time()
                )

            st.markdown("---")

            # NY PM
            st.write("**NY PM (13:30-15:30 EST)**")
            ny_pm_col1, ny_pm_col2 = st.columns(2)

            with ny_pm_col1:
                ny_pm_enabled = st.checkbox(
                    "Enable NY PM",
                    value=current_config.ny_pm.enabled
                )

            with ny_pm_col2:
                st.write(f"Default: {'Enabled' if default_config.ny_pm.enabled else 'Disabled'}")

            if ny_pm_enabled:
                ny_pm_start = st.time_input(
                    "Start Time",
                    value=datetime.strptime(current_config.ny_pm.start_time, "%H:%M").time()
                )
                ny_pm_end = st.time_input(
                    "End Time",
                    value=datetime.strptime(current_config.ny_pm.end_time, "%H:%M").time()
                )

        # ML Threshold Section
        st.subheader("ML Configuration")
        with st.expander("Machine Learning Settings", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                ml_threshold = st.slider(
                    "Minimum Probability Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(current_config.ml_config.min_probability),
                    step=0.05,
                    help="Signals below this probability will be filtered"
                )

            with col2:
                st.write(f"Default: {default_config.ml_config.min_probability:.2f}")

        # Save/Reset Buttons
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Changes", type="primary"):
                # Validate inputs before prompting for password
                from dashboard.shared_state import (
                    SystemConfig,
                    RiskLimits,
                    TimeWindow,
                    MLConfig,
                )

                # Build new configuration
                new_config = SystemConfig(
                    risk_limits=RiskLimits(
                        daily_loss_limit=float(daily_loss),
                        max_drawdown_pct=float(max_drawdown),
                        per_trade_risk_pct=float(per_trade_risk),
                        max_position_contracts=int(max_position)
                    ),
                    london_am=TimeWindow(
                        enabled=london_enabled,
                        start_time=london_start.strftime("%H:%M") if london_enabled else "02:00",
                        end_time=london_end.strftime("%H:%M") if london_enabled else "05:00"
                    ),
                    ny_am=TimeWindow(
                        enabled=ny_am_enabled,
                        start_time=ny_am_start.strftime("%H:%M") if ny_am_enabled else "09:30",
                        end_time=ny_am_end.strftime("%H:%M") if ny_am_enabled else "11:00"
                    ),
                    ny_pm=TimeWindow(
                        enabled=ny_pm_enabled,
                        start_time=ny_pm_start.strftime("%H:%M") if ny_pm_enabled else "13:30",
                        end_time=ny_pm_end.strftime("%H:%M") if ny_pm_enabled else "15:30"
                    ),
                    ml_config=MLConfig(
                        min_probability=float(ml_threshold)
                    )
                )

                # Validate configuration
                is_valid, error_msg = validate_config(new_config)
                if not is_valid:
                    st.error(f"❌ Validation Error: {error_msg}")
                else:
                    # Prompt for password
                    password = st.text_input(
                        "Enter password to confirm changes",
                        type="password",
                        key="settings_password"
                    )

                    if password:
                        if save_system_config(new_config, password):
                            st.success("✅ Configuration saved successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Incorrect password or save failed")

        with col2:
            if st.button("🔄 Reset to Defaults"):
                st.info("ℹ️ Click to reset all values to defaults")
                if st.checkbox("Confirm reset to defaults", key="confirm_reset"):
                    # Prompt for password before reset
                    reset_password = st.text_input(
                        "Enter password to confirm reset",
                        type="password",
                        key="reset_password"
                    )

                    if reset_password:
                        from dashboard.shared_state import (
                            SystemConfig,
                            RiskLimits,
                            TimeWindow,
                            MLConfig,
                        )

                        # Reset to default configuration
                        default_config = get_default_config()

                        if save_system_config(default_config, reset_password):
                            st.success("✅ Configuration reset to defaults!")
                            st.rerun()
                        else:
                            st.error("❌ Incorrect password or reset failed")

    except Exception as e:
        st.error(f"Failed to load settings: {e}")
        logger.error(f"Error loading settings: {e}", exc_info=True)
        return


def render_logs():
    """Render logs page (placeholder)."""
    st.header("📋 System Logs")
    st.info("Logs page coming in Story 8.7")


def render_page(page: str):
    """Route to selected page."""
    page_renderers = {
        "Overview": render_overview,
        "Positions": render_positions,
        "Signals": render_signals,
        "Charts": render_charts,
        "Settings": render_settings,
        "Logs": render_logs,
    }

    renderer = page_renderers.get(page, render_overview)
    renderer()
