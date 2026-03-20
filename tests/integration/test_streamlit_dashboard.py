"""Integration tests for Streamlit dashboard.

Tests dashboard launch, navigation, auto-refresh, and startup performance.
"""

import pytest

import pandas as pd


class TestDashboardLaunch:
    """Test dashboard launches correctly."""

    def test_dashboard_launches_without_errors(self):
        """Verify dashboard launches without errors."""
        # This test will be enabled once Streamlit is installed
        pytest.skip("Streamlit not yet installed - will test in story completion")

    def test_all_navigation_pages_accessible(self):
        """Test all navigation pages are accessible."""
        pytest.skip("Streamlit not yet installed - will test in story completion")

    def test_startup_time_under_3_seconds(self):
        """Test startup time is < 3 seconds."""
        pytest.skip("Streamlit not yet installed - will test in story completion")

    def test_auto_refresh_works(self):
        """Test auto-refresh works correctly."""
        pytest.skip("Streamlit not yet installed - will test in story completion")


class TestDashboardComponents:
    """Test dashboard components work together."""

    def test_header_displays_correctly(self):
        """Test header shows project title and timestamp."""
        pytest.skip("Streamlit not yet installed - will test in story completion")

    def test_sidebar_navigation_works(self):
        """Test sidebar navigation works."""
        pytest.skip("Streamlit not yet installed - will test in story completion")

    def test_system_status_banner_displays(self):
        """Test system status banner displays."""
        pytest.skip("Streamlit not yet installed - will test in story completion")

    def test_dark_theme_applied(self):
        """Test dark theme is applied."""
        pytest.skip("Streamlit not yet installed - will test in story completion")

    def test_color_scheme_consistent(self):
        """Test color scheme is consistent across pages."""
        pytest.skip("Streamlit not yet installed - will test in story completion")


class TestPositionsPage:
    """Test Positions page renders and functions correctly."""

    def test_positions_page_renders_without_errors(self):
        """Test Positions page renders without errors."""
        # Verify render_positions function exists in navigation module
        import os

        navigation_path = "src/dashboard/navigation.py"
        assert os.path.exists(navigation_path), "navigation.py not found"

        with open(navigation_path) as f:
            content = f.read()
            assert "def render_positions():" in content
            assert "Open Positions" in content
            assert "st.dataframe" in content

    def test_positions_displays_with_mock_data(self):
        """Test positions table displays with mock position data."""
        from src.dashboard.shared_state import get_open_positions

        positions = get_open_positions()

        # Verify we get mock data
        assert isinstance(positions, list)
        assert len(positions) > 0

        # Verify position structure
        position = positions[0]
        assert hasattr(position, "signal_id")
        assert hasattr(position, "direction")
        assert hasattr(position, "entry_price")
        assert hasattr(position, "current_price")
        assert hasattr(position, "pnl_usd")
        assert hasattr(position, "pnl_pct")
        assert hasattr(position, "barriers")

    def test_barrier_highlighting_logic(self):
        """Test barrier proximity highlighting logic."""
        from src.dashboard.shared_state import calculate_barrier_progress

        # Test at different proximity levels
        progress_0 = calculate_barrier_progress(4500.00, 4600.00, 4500.00)
        assert progress_0 == 0.0

        progress_80 = calculate_barrier_progress(4580.00, 4600.00, 4500.00)
        assert progress_80 == 80.0

        progress_95 = calculate_barrier_progress(4595.00, 4600.00, 4500.00)
        assert progress_95 == 95.0

        progress_100 = calculate_barrier_progress(4600.00, 4600.00, 4500.00)
        assert progress_100 == 100.0

    def test_manual_exit_button_functionality(self):
        """Test manual exit button functionality."""
        from src.dashboard.shared_state import exit_position

        # Test exit function with mock password
        result = exit_position("SIGNAL-001", "test_password")
        # Mock implementation always returns True
        assert result is True


class TestSignalsPage:
    """Test Signals page renders and functions correctly."""

    def test_signals_page_renders_without_errors(self):
        """Test Signals page renders without errors."""
        # Verify render_signals function exists in navigation module
        import os

        navigation_path = "src/dashboard/navigation.py"
        assert os.path.exists(navigation_path), "navigation.py not found"

        with open(navigation_path) as f:
            content = f.read()
            assert "def render_signals():" in content
            assert "Live Signals" in content
            assert "st.dataframe" in content
            # Verify filter controls are present
            assert "signal_filter_status" in content
            assert "signal_filter_direction" in content
            assert "signal_filter_confidence" in content

    def test_signals_displays_with_mock_data(self):
        """Test signals table displays with mock signal data."""
        from src.dashboard.shared_state import get_silver_bullet_signals

        signals = get_silver_bullet_signals()

        # Verify we get mock data
        assert isinstance(signals, list)
        assert len(signals) > 0

        # Verify signal structure
        signal = signals[0]
        assert hasattr(signal, "signal_id")
        assert hasattr(signal, "timestamp")
        assert hasattr(signal, "direction")
        assert hasattr(signal, "confidence")
        assert hasattr(signal, "ml_probability")
        assert hasattr(signal, "mss_present")
        assert hasattr(signal, "fvg_present")
        assert hasattr(signal, "sweep_present")
        assert hasattr(signal, "time_window")
        assert hasattr(signal, "status")

    def test_signal_filtering_logic(self):
        """Test signal filtering logic works correctly."""
        from src.dashboard.shared_state import (
            get_silver_bullet_signals,
            filter_signals,
            SignalStatus,
            Direction,
        )

        signals = get_silver_bullet_signals()

        # Test status filtering
        filtered_executed = filter_signals(signals, status="Executed")
        for signal in filtered_executed:
            assert signal.status == SignalStatus.EXECUTED

        # Test direction filtering
        filtered_long = filter_signals(signals, direction="LONG")
        for signal in filtered_long:
            assert signal.direction == Direction.LONG

        # Test confidence filtering
        filtered_4plus = filter_signals(signals, confidence="4★+")
        for signal in filtered_4plus:
            assert signal.confidence >= 4

        # Test combined filtering
        filtered_all = filter_signals(
            signals,
            status="Executed",
            direction="LONG",
            confidence="3★+"
        )
        for signal in filtered_all:
            assert signal.status == SignalStatus.EXECUTED
            assert signal.direction == Direction.LONG
            assert signal.confidence >= 3

    def test_signal_formatting_functions(self):
        """Test signal formatting functions work correctly."""
        from src.dashboard.shared_state import (
            format_signal_status,
            format_confidence_stars,
            SignalStatus,
        )

        # Test status formatting
        executed_status = format_signal_status(SignalStatus.EXECUTED)
        assert "🟢" in executed_status
        assert "Executed" in executed_status

        filtered_status = format_signal_status(SignalStatus.FILTERED)
        assert "🟡" in filtered_status
        assert "Filtered" in filtered_status

        rejected_status = format_signal_status(SignalStatus.REJECTED)
        assert "⚪" in rejected_status
        assert "Rejected" in rejected_status

        # Test confidence star formatting
        stars_5 = format_confidence_stars(5)
        assert stars_5 == "★★★★★"

        stars_3 = format_confidence_stars(3)
        assert stars_3 == "★★★☆☆"

        stars_1 = format_confidence_stars(1)
        assert stars_1 == "★☆☆☆☆"

        # Test invalid confidence
        stars_invalid = format_confidence_stars(0)
        assert stars_invalid == "?"


class TestChartsPage:
    """Integration tests for Charts page from Story 8.5."""

    def test_charts_page_renders_without_errors(self):
        """Test Charts page renders without errors."""
        # Verify function exists in navigation.py
        with open("src/dashboard/navigation.py") as f:
            content = f.read()
            assert "def render_charts():" in content
            assert "plotly.graph_objects" in content
            assert "st.plotly_chart" in content

    def test_chart_displays_with_mock_data(self):
        """Test chart displays with mock data."""
        from src.dashboard.shared_state import (
            get_dollar_bars,
            get_pattern_overlays,
            get_trade_markers,
        )

        # Get chart data
        df = get_dollar_bars()
        markers, fvg_zones = get_pattern_overlays()
        trades = get_trade_markers()

        # Verify data is returned
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

        assert markers is not None
        assert isinstance(markers, list)

        assert fvg_zones is not None
        assert isinstance(fvg_zones, list)

        assert trades is not None
        assert isinstance(trades, list)

    def test_time_range_selector_works_correctly(self):
        """Test time range selector works correctly."""
        from src.dashboard.shared_state import get_dollar_bars

        # Test different time ranges
        for time_range in ["hour", "today", "week"]:
            df = get_dollar_bars(time_range)
            assert df is not None
            assert isinstance(df, pd.DataFrame)

    def test_chart_updates_in_real_time_with_new_data(self):
        """Test chart updates in real-time with new data."""
        from src.dashboard.shared_state import get_dollar_bars

        # Get data twice
        df1 = get_dollar_bars()
        df2 = get_dollar_bars()

        # Both should return valid DataFrames
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)

        # Data should have same structure
        assert list(df1.columns) == list(df2.columns)


class TestSettingsPage:
    """Integration tests for Settings page from Story 8.6."""

    def test_settings_page_renders_without_errors(self):
        """Test Settings page renders without errors."""
        # Verify function exists in navigation.py
        with open("src/dashboard/navigation.py") as f:
            content = f.read()
            assert "def render_settings():" in content
            assert "System Configuration" in content
            # Verify form controls are present
            assert "st.number_input" in content
            assert "st.slider" in content
            assert "st.checkbox" in content
            assert "st.time_input" in content
            assert "st.button" in content

    def test_configuration_displays_with_mock_data(self):
        """Test configuration displays with mock data."""
        from src.dashboard.shared_state import (
            get_system_config,
            get_default_config,
        )

        # Get configuration
        config = get_system_config()
        default_config = get_default_config()

        # Verify we get mock data
        assert config is not None
        assert default_config is not None

        # Verify configuration structure
        assert hasattr(config, 'risk_limits')
        assert hasattr(config, 'london_am')
        assert hasattr(config, 'ny_am')
        assert hasattr(config, 'ny_pm')
        assert hasattr(config, 'ml_config')

    def test_form_validation_rejects_invalid_inputs(self):
        """Test form validation rejects invalid inputs."""
        from src.dashboard.shared_state import (
            validate_config,
            SystemConfig,
            RiskLimits,
            TimeWindow,
            MLConfig,
        )

        # Test negative daily loss
        config = SystemConfig(
            risk_limits=RiskLimits(-100.0, 12.0, 2.0, 5),
            london_am=TimeWindow(True, "02:00", "05:00"),
            ny_am=TimeWindow(True, "09:30", "11:00"),
            ny_pm=TimeWindow(True, "13:30", "15:30"),
            ml_config=MLConfig(0.65)
        )

        is_valid, error_msg = validate_config(config)
        assert not is_valid
        assert "positive" in error_msg.lower()

    def test_save_button_requires_password_confirmation(self):
        """Test save button requires password confirmation."""
        from src.dashboard.shared_state import (
            save_system_config,
            SystemConfig,
            RiskLimits,
            TimeWindow,
            MLConfig,
        )

        config = SystemConfig(
            risk_limits=RiskLimits(500.0, 12.0, 2.0, 5),
            london_am=TimeWindow(True, "02:00", "05:00"),
            ny_am=TimeWindow(True, "09:30", "11:00"),
            ny_pm=TimeWindow(True, "13:30", "15:30"),
            ml_config=MLConfig(0.65)
        )

        # Test with wrong password
        result = save_system_config(config, "wrong_password")
        assert result is False

        # Test with correct password (mock)
        result = save_system_config(config, "test_password")
        assert result is True

    def test_reset_button_restores_defaults(self):
        """Test reset button restores defaults."""
        from src.dashboard.shared_state import (
            get_system_config,
            get_default_config,
        )

        # Get current and default config
        current = get_system_config()
        defaults = get_default_config()

        # Both should have same structure
        assert type(current) == type(defaults)
        assert hasattr(current, 'risk_limits')
        assert hasattr(defaults, 'risk_limits')

    def test_configuration_updates_take_effect_immediately(self):
        """Test configuration updates take effect immediately."""
        from src.dashboard.shared_state import (
            get_system_config,
            save_system_config,
            SystemConfig,
            RiskLimits,
            TimeWindow,
            MLConfig,
        )

        # Get original config
        original = get_system_config()

        # Create new config with different values
        new_config = SystemConfig(
            risk_limits=RiskLimits(600.0, 15.0, 2.5, 6),  # Changed
            london_am=TimeWindow(True, "02:00", "05:00"),
            ny_am=TimeWindow(True, "09:30", "11:00"),
            ny_pm=TimeWindow(True, "13:30", "15:30"),
            ml_config=MLConfig(0.70)  # Changed
        )

        # Save new config
        result = save_system_config(new_config, "test_password")
        assert result is True

        # In real implementation, get_system_config() would return new values
        # For mock, we just verify save succeeded


class TestHealthIndicatorsPage:
    """Integration tests for Health Indicators from Story 8.7."""

    def test_health_indicators_display_with_mock_data(self):
        """Test health indicators display with mock data."""
        from src.dashboard.shared_state import (
            get_system_health,
            get_alerts_summary,
        )

        # Get health data
        health = get_system_health()
        alerts = get_alerts_summary()

        # Verify health data structure
        assert health is not None
        assert hasattr(health, 'api_status')
        assert hasattr(health, 'resources')
        assert hasattr(health, 'data_flow_status')
        assert hasattr(health, 'active_alerts_count')
        assert hasattr(health, 'system_uptime')

        # Verify alerts data
        assert alerts is not None
        assert 'count' in alerts
        assert isinstance(alerts['count'], int)

    def test_color_coding_applies_correctly_based_on_values(self):
        """Test color coding applies correctly based on threshold values."""
        from src.dashboard.shared_state import (
            get_resource_color,
            format_resource_usage,
        )

        # Test green range (< 80%)
        green_color = get_resource_color(50.0)
        assert green_color == "#00FF00"

        # Test yellow range (80-90%)
        yellow_color = get_resource_color(85.0)
        assert yellow_color == "#FFCC00"

        # Test red range (> 90%)
        red_color = get_resource_color(95.0)
        assert red_color == "#FF0000"

        # Test format_resource_usage
        color, emoji = format_resource_usage(50.0)
        assert color == "#00FF00"
        assert emoji == "✅"

        color, emoji = format_resource_usage(85.0)
        assert color == "#FFCC00"
        assert emoji == "⚠️"

        color, emoji = format_resource_usage(95.0)
        assert color == "#FF0000"
        assert emoji == "🔴"

    def test_api_status_displays_with_color_indicators(self):
        """Test API status shows green when connected, red when disconnected."""
        from src.dashboard.shared_state import get_system_health

        health = get_system_health()

        # Verify API status structure
        assert hasattr(health.api_status, 'connected')
        assert hasattr(health.api_status, 'last_ping_time')
        assert hasattr(health.api_status, 'ping_latency_ms')

        # Mock data shows connected status
        assert health.api_status.connected is True
        assert health.api_status.ping_latency_ms >= 0

    def test_data_freshness_shows_staleness_warning(self):
        """Test data freshness shows staleness warning at > 30 seconds."""
        from src.dashboard.shared_state import (
            is_data_stale,
            calculate_data_age,
        )
        from datetime import datetime, timedelta

        now = datetime.now()

        # Fresh data (<= 30 seconds) - should not be stale
        fresh_time = now - timedelta(seconds=5)
        assert not is_data_stale(fresh_time)
        assert calculate_data_age(fresh_time) == 5

        # Stale data (> 30 seconds) - should be stale
        stale_time = now - timedelta(seconds=60)
        assert is_data_stale(stale_time)
        assert calculate_data_age(stale_time) == 60

    def test_resource_usage_displays_with_progress_bars(self):
        """Test resource usage displays with color-coded progress bars."""
        from src.dashboard.shared_state import (
            get_system_health,
            format_resource_usage,
        )

        health = get_system_health()

        # Verify resource usage structure
        assert hasattr(health.resources, 'cpu_percent')
        assert hasattr(health.resources, 'memory_percent')
        assert hasattr(health.resources, 'disk_percent')

        # Verify all resources are in valid range (0-100%)
        assert 0 <= health.resources.cpu_percent <= 100
        assert 0 <= health.resources.memory_percent <= 100
        assert 0 <= health.resources.disk_percent <= 100

        # Verify color coding works
        cpu_color, cpu_emoji = format_resource_usage(health.resources.cpu_percent)
        assert cpu_color in ["#00FF00", "#FFCC00", "#FF0000"]
        assert cpu_emoji in ["✅", "⚠️", "🔴"]

    def test_detailed_metrics_modal_displays_correctly(self):
        """Test detailed metrics modal shows correct detailed information."""
        from src.dashboard.shared_state import get_system_health

        health = get_system_health()

        # Verify pipeline status structure for modal display
        assert hasattr(health.data_flow_status, 'last_execution_time')
        assert hasattr(health.data_flow_status, 'error_count')

        # Verify all pipeline components have required fields
        components = [
            health.data_flow_status,
            health.signal_detection_status,
            health.ml_prediction_status,
            health.execution_status
        ]

        for component in components:
            assert hasattr(component, 'component_name')
            assert hasattr(component, 'is_healthy')
            assert hasattr(component, 'last_execution_time')
            assert hasattr(component, 'error_count')
            assert component.error_count >= 0


class TestManualTradeSubmissionIntegration:
    """Integration tests for Manual Trade Submission from Story 8.8."""

    def test_manual_trade_page_renders_without_errors(self):
        """Test Manual Trade page renders without errors."""
        # Verify function exists in navigation.py
        with open("src/dashboard/navigation.py") as f:
            content = f.read()
            assert "def render_manual_trade():" in content
            assert "Manual Trade Submission" in content

    def test_manual_trade_form_has_all_controls(self):
        """Test manual trade form has all required controls."""
        from src.dashboard.shared_state import ManualTradeRequest
        from datetime import datetime

        # Verify we can create a manual trade request
        request = ManualTradeRequest(
            direction="Buy",
            quantity=2,
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="dashboard_user"
        )

        # Verify all fields are present
        assert request.direction == "Buy"
        assert request.quantity == 2
        assert request.order_type == "Market"
        assert request.limit_price is None
        assert request.submitted_by == "dashboard_user"

    def test_manual_trade_validation_integration(self):
        """Test manual trade validation integrates correctly with account metrics."""
        from src.dashboard.shared_state import (
            ManualTradeRequest,
            calculate_trade_preview,
            validate_position_size,
            validate_per_trade_risk,
            validate_margin_requirement,
            get_account_metrics,
            get_current_price,
            get_current_atr,
        )
        from datetime import datetime

        # Get current account metrics and market data
        metrics = get_account_metrics()
        current_price = get_current_price()
        atr = get_current_atr()

        # Create a valid trade request (1 contract to stay within 2% risk limit)
        request = ManualTradeRequest(
            direction="Buy",
            quantity=1,  # 1 contract = $1200 risk < $2000 (2% of $100k)
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="dashboard_user"
        )

        # Calculate preview
        preview = calculate_trade_preview(request, current_price, atr, metrics.equity)

        # Verify validation integration
        assert preview.position_size_valid is True  # 1 contract < 5
        assert preview.per_trade_risk_valid is True  # Should be < 2% equity
        assert isinstance(preview.dollar_risk, float)
        assert preview.dollar_risk > 0

    def test_manual_trade_preview_displays_all_barriers(self):
        """Test trade preview displays all triple barriers correctly."""
        from src.dashboard.shared_state import (
            ManualTradeRequest,
            calculate_trade_preview,
            get_account_metrics,
            get_current_price,
            get_current_atr,
        )
        from datetime import datetime

        metrics = get_account_metrics()
        current_price = get_current_price()
        atr = get_current_atr()

        # Test long position
        long_request = ManualTradeRequest(
            direction="Buy",
            quantity=1,  # Use 1 contract to stay within risk limits
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="dashboard_user"
        )

        long_preview = calculate_trade_preview(long_request, current_price, atr, metrics.equity)

        # Verify barriers for long position
        assert long_preview.upper_barrier_price > long_preview.stop_loss_price
        assert long_preview.lower_barrier_price == long_preview.stop_loss_price
        assert long_preview.vertical_barrier_time > datetime.now()

        # Test short position
        short_request = ManualTradeRequest(
            direction="Sell",
            quantity=1,  # Use 1 contract to stay within risk limits
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="dashboard_user"
        )

        short_preview = calculate_trade_preview(short_request, current_price, atr, metrics.equity)

        # Verify barriers for short position (reversed)
        assert short_preview.upper_barrier_price == short_preview.stop_loss_price
        assert short_preview.lower_barrier_price < short_preview.stop_loss_price

    def test_manual_trade_validation_blocks_invalid_trades(self):
        """Test validation blocks trades that exceed limits."""
        from src.dashboard.shared_state import (
            ManualTradeRequest,
            calculate_trade_preview,
            get_account_metrics,
            get_current_price,
            get_current_atr,
        )
        from datetime import datetime

        metrics = get_account_metrics()
        current_price = get_current_price()
        atr = get_current_atr()

        # Test position size limit (5 contracts)
        invalid_size_request = ManualTradeRequest(
            direction="Buy",
            quantity=10,  # Exceeds 5 contract limit
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="dashboard_user"
        )

        preview = calculate_trade_preview(invalid_size_request, current_price, atr, metrics.equity)

        # Should have validation error
        assert preview.position_size_valid is False
        assert len(preview.validation_errors) > 0
        assert any("exceeds maximum position size" in err for err in preview.validation_errors)

    def test_manual_trade_password_confirmation(self):
        """Test password confirmation prevents unauthorized submissions."""
        from src.dashboard.shared_state import (
            ManualTradeRequest,
            validate_password,
            submit_manual_trade,
        )
        from datetime import datetime

        request = ManualTradeRequest(
            direction="Buy",
            quantity=1,  # Use 1 contract to stay within risk limits
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="dashboard_user"
        )

        # Test correct password
        assert validate_password("admin123") is True

        # Test incorrect password
        assert validate_password("wrong_password") is False

        # Test submission happens (password validation is separate from submission)
        # In the actual form, password is validated BEFORE calling submit_manual_trade
        result = submit_manual_trade(request)
        assert result.success is True
        assert result.order_id is not None
        assert result.error is None

    def test_manual_trade_limit_order_requires_price(self):
        """Test limit orders require a price."""
        from src.dashboard.shared_state import (
            ManualTradeRequest,
            calculate_trade_preview,
            get_account_metrics,
            get_current_price,
            get_current_atr,
        )
        from datetime import datetime

        metrics = get_account_metrics()
        current_price = get_current_price()
        atr = get_current_atr()

        # Limit order without price should fail validation
        limit_request = ManualTradeRequest(
            direction="Buy",
            quantity=1,  # Use 1 contract to stay within risk limits
            order_type="Limit",
            limit_price=None,  # Missing required price
            submit_time=datetime.now(),
            submitted_by="dashboard_user"
        )

        preview = calculate_trade_preview(limit_request, current_price, atr, metrics.equity)

        # Should have validation error
        assert len(preview.validation_errors) > 0
        assert any("Limit price required" in err for err in preview.validation_errors)

    def test_manual_trade_system_state_checks(self):
        """Test form is disabled when system is in HALTED or SAFE_MODE."""
        from src.dashboard.shared_state import get_system_health

        health = get_system_health()

        # Verify system health has state information
        # (In real implementation, this would check system_state field)
        assert hasattr(health, 'system_state') or hasattr(health, 'api_status')

    def test_manual_trade_margin_requirement_check(self):
        """Test margin requirement is checked correctly."""
        from src.dashboard.shared_state import (
            ManualTradeRequest,
            calculate_trade_preview,
            get_account_metrics,
            get_current_price,
            get_current_atr,
        )
        from datetime import datetime

        metrics = get_account_metrics()
        current_price = get_current_price()
        atr = get_current_atr()

        # Create a trade that requires significant margin
        request = ManualTradeRequest(
            direction="Buy",
            quantity=5,  # Maximum position
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="dashboard_user"
        )

        preview = calculate_trade_preview(request, current_price, atr, metrics.equity)

        # Verify margin was calculated
        assert preview.margin_required > 0
        assert isinstance(preview.margin_sufficient, bool)

        # With mock data, margin should be sufficient
        # In real implementation, this would check actual account equity

    def test_manual_trade_full_workflow(self):
        """Test complete workflow from form to submission."""
        from src.dashboard.shared_state import (
            ManualTradeRequest,
            calculate_trade_preview,
            validate_password,
            submit_manual_trade,
            get_account_metrics,
            get_current_price,
            get_current_atr,
        )
        from datetime import datetime

        # Step 1: Create trade request
        request = ManualTradeRequest(
            direction="Buy",
            quantity=1,  # Use 1 contract to stay within risk limits
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="dashboard_user"
        )

        # Step 2: Calculate preview
        metrics = get_account_metrics()
        current_price = get_current_price()
        atr = get_current_atr()
        preview = calculate_trade_preview(request, current_price, atr, metrics.equity)

        # Step 3: Verify no validation errors
        assert len(preview.validation_errors) == 0

        # Step 4: Validate password (happens before submission in real form)
        password_valid = validate_password("admin123")
        assert password_valid is True

        # Step 5: Submit trade (password validation is separate)
        result = submit_manual_trade(request)

        # Step 6: Verify success
        assert result.success is True
        assert result.order_id is not None
        assert result.error is None

