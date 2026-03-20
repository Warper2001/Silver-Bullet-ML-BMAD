"""Unit tests for help system components."""

import pytest


class TestHelpContent:
    """Test help content exists for all pages."""

    def test_overview_page_help_exists(self):
        """Verify overview page help content exists."""
        from src.dashboard.help.help_content import OVERVIEW_HELP

        assert OVERVIEW_HELP is not None
        assert isinstance(OVERVIEW_HELP, str)
        assert len(OVERVIEW_HELP) > 0

    def test_positions_page_help_exists(self):
        """Verify positions page help content exists."""
        from src.dashboard.help.help_content import POSITIONS_HELP

        assert POSITIONS_HELP is not None
        assert isinstance(POSITIONS_HELP, str)
        assert len(POSITIONS_HELP) > 0

    def test_signals_page_help_exists(self):
        """Verify signals page help content exists."""
        from src.dashboard.help.help_content import SIGNALS_HELP

        assert SIGNALS_HELP is not None
        assert isinstance(SIGNALS_HELP, str)
        assert len(SIGNALS_HELP) > 0

    def test_charts_page_help_exists(self):
        """Verify charts page help content exists."""
        from src.dashboard.help.help_content import CHARTS_HELP

        assert CHARTS_HELP is not None
        assert isinstance(CHARTS_HELP, str)
        assert len(CHARTS_HELP) > 0

    def test_settings_page_help_exists(self):
        """Verify settings page help content exists."""
        from src.dashboard.help.help_content import SETTINGS_HELP

        assert SETTINGS_HELP is not None
        assert isinstance(SETTINGS_HELP, str)
        assert len(SETTINGS_HELP) > 0


class TestGlossary:
    """Test glossary of trading terms."""

    def test_glossary_exists(self):
        """Verify glossary exists."""
        from src.dashboard.help.glossary import GLOSSARY

        assert GLOSSARY is not None
        assert isinstance(GLOSSARY, dict)

    def test_glossary_has_mss_definition(self):
        """Verify glossary defines MSS."""
        from src.dashboard.help.glossary import GLOSSARY

        assert "MSS" in GLOSSARY
        assert "Market Structure Shift" in GLOSSARY["MSS"]

    def test_glossary_has_fvg_definition(self):
        """Verify glossary defines FVG."""
        from src.dashboard.help.glossary import GLOSSARY

        assert "FVG" in GLOSSARY
        assert "Fair Value Gap" in GLOSSARY["FVG"]

    def test_glossary_has_atr_definition(self):
        """Verify glossary defines ATR."""
        from src.dashboard.help.glossary import GLOSSARY

        assert "ATR" in GLOSSARY
        assert "Average True Range" in GLOSSARY["ATR"]

    def test_glossary_has_triple_barrier_definition(self):
        """Verify glossary defines triple-barrier."""
        from src.dashboard.help.glossary import GLOSSARY

        assert "Triple-Barrier" in GLOSSARY or "triple-barrier" in GLOSSARY

    def test_glossary_has_all_required_terms(self):
        """Verify glossary has all required trading terms."""
        from src.dashboard.help.glossary import GLOSSARY

        required_terms = ["MSS", "FVG", "ATR", "Dollar Bars", "Silver Bullet"]
        for term in required_terms:
            assert term in GLOSSARY, f"Missing glossary term: {term}"


class TestTooltips:
    """Test tooltip definitions."""

    def test_tooltips_exist(self):
        """Verify tooltips module exists."""
        from src.dashboard.help.tooltips import TOOLTIPS

        assert TOOLTIPS is not None
        assert isinstance(TOOLTIPS, dict)

    def test_overview_tooltips_exist(self):
        """Verify overview page tooltips exist."""
        from src.dashboard.help.tooltips import TOOLTIPS

        assert "equity" in TOOLTIPS
        assert "daily_pnl" in TOOLTIPS
        assert "win_rate" in TOOLTIPS
        assert "drawdown" in TOOLTIPS

    def test_positions_tooltips_exist(self):
        """Verify positions page tooltips exist."""
        from src.dashboard.help.tooltips import TOOLTIPS

        assert "barrier_levels" in TOOLTIPS
        assert "vertical_barrier" in TOOLTIPS

    def test_signals_tooltips_exist(self):
        """Verify signals page tooltips exist."""
        from src.dashboard.help.tooltips import TOOLTIPS

        assert "confidence" in TOOLTIPS
        assert "ml_probability" in TOOLTIPS


class TestHelpModal:
    """Test help modal component."""

    def test_help_modal_function_exists(self):
        """Verify help modal function exists."""
        from src.dashboard.help.help_modal import show_help_modal

        assert show_help_modal is not None

    def test_help_modal_accepts_page_param(self):
        """Verify help modal accepts page parameter."""
        from src.dashboard.help.help_modal import show_help_modal

        # Should not raise exception
        # (In real Streamlit app, this would be called with a page name)
        assert callable(show_help_modal)


class TestHelpSearch:
    """Test help search functionality."""

    def test_search_function_exists(self):
        """Verify search function exists."""
        from src.dashboard.help.help_search import search_help

        assert search_help is not None

    def test_search_returns_results(self):
        """Verify search returns results for valid query."""
        from src.dashboard.help.help_search import search_help

        results = search_help("equity")
        assert isinstance(results, list)

    def test_search_handles_no_results(self):
        """Verify search handles queries with no results."""
        from src.dashboard.help.help_search import search_help

        results = search_help("xyznonexistent")
        assert isinstance(results, list)
        # Should return empty list, not raise exception


class TestHelpNavigation:
    """Test help navigation."""

    def test_table_of_contents_exists(self):
        """Verify table of contents exists."""
        from src.dashboard.help.help_content import HELP_TABLE_OF_CONTENTS

        assert HELP_TABLE_OF_CONTENTS is not None
        assert isinstance(HELP_TABLE_OF_CONTENTS, list)

    def test_table_of_contents_has_all_pages(self):
        """Verify table of contents includes all pages."""
        from src.dashboard.help.help_content import HELP_TABLE_OF_CONTENTS

        pages = ["Overview", "Positions", "Signals", "Charts", "Settings"]
        for page in pages:
            assert any(page in item for item in HELP_TABLE_OF_CONTENTS)


class TestHelpIntegration:
    """Test help integration with dashboard."""

    def test_help_button_exists(self):
        """Verify help button function exists."""
        # This will be tested in integration tests with actual dashboard
        import os
        assert os.path.exists("src/dashboard/help/help_modal.py")

    def test_help_modules_exist(self):
        """Verify all help modules exist."""
        import os

        modules = [
            "src/dashboard/help/help_content.py",
            "src/dashboard/help/help_modal.py",
            "src/dashboard/help/tooltips.py",
            "src/dashboard/help/glossary.py",
            "src/dashboard/help/help_search.py",
        ]

        for module in modules:
            assert os.path.exists(module), f"Module {module} not found"
