"""Integration tests for help system."""

import pytest


class TestHelpSystemIntegration:
    """Test help system integration with dashboard."""

    def test_help_content_retrieval(self):
        """Test help content can be retrieved for all pages."""
        from src.dashboard.help.help_content import get_help_content

        pages = ["Overview", "Positions", "Signals", "Charts", "Settings"]

        for page in pages:
            content = get_help_content(page)
            assert content is not None
            assert len(content) > 0
            assert "Help content not found" not in content

    def test_glossary_search_functionality(self):
        """Test glossary search works correctly."""
        from src.dashboard.help.glossary import search_glossary

        # Search for MSS
        results = search_glossary("MSS")
        assert len(results) > 0
        assert "MSS" in results or "Market Structure Shift" in results

        # Search for FVG
        results = search_glossary("FVG")
        assert len(results) > 0

        # Search for non-existent term
        results = search_glossary("xyznonexistent")
        assert len(results) == 0

    def test_help_search_across_content(self):
        """Test help search works across all content."""
        from src.dashboard.help.help_search import search_help

        # Search for "equity"
        results = search_help("equity")
        assert len(results) > 0
        assert any(result["page"] == "Overview" for result in results)

        # Search for "barrier"
        results = search_help("barrier")
        assert len(results) > 0
        assert any(result["page"] == "Positions" for result in results)

    def test_tooltip_retrieval(self):
        """Test tooltips can be retrieved."""
        from src.dashboard.help.tooltips import get_tooltip

        # Test existing tooltips
        equity_tooltip = get_tooltip("equity")
        assert equity_tooltip is not None
        assert len(equity_tooltip) > 0

        # Test non-existent tooltip
        missing_tooltip = get_tooltip("nonexistent")
        assert missing_tooltip == ""

    def test_glossary_term_retrieval(self):
        """Test glossary terms can be retrieved."""
        from src.dashboard.help.glossary import get_glossary_term

        # Test existing terms
        mss_def = get_glossary_term("MSS")
        assert mss_def is not None
        assert "Market Structure Shift" in mss_def

        # Test non-existent term
        missing_def = get_glossary_term("nonexistent")
        assert "not found" in missing_def.lower()

    def test_help_modal_components(self):
        """Test help modal has all required components."""
        import os

        # Verify help modal module exists
        assert os.path.exists("src/dashboard/help/help_modal.py")

        # Verify it has required functions
        from src.dashboard.help import help_modal

        assert hasattr(help_modal, "show_help_modal")
        assert hasattr(help_modal, "show_help_button")
        assert hasattr(help_modal, "show_tooltip")


class TestHelpContentQuality:
    """Test quality and completeness of help content."""

    def test_overview_help_covers_all_metrics(self):
        """Verify Overview help covers all account metrics."""
        from src.dashboard.help.help_content import OVERVIEW_HELP

        required_terms = [
            "Account Equity",
            "Daily P&L",
            "Win Rate",
            "Drawdown",
            "Open Positions",
        ]

        for term in required_terms:
            assert term in OVERVIEW_HELP

    def test_positions_help_covers_barriers(self):
        """Verify Positions help covers barrier levels."""
        from src.dashboard.help.help_content import POSITIONS_HELP

        required_terms = [
            "Upper Barrier",
            "Lower Barrier",
            "Vertical Barrier",
            "Manual Exit",
        ]

        for term in required_terms:
            assert term in POSITIONS_HELP

    def test_signals_help_covers_signals(self):
        """Verify Signals help covers signal concepts."""
        from src.dashboard.help.help_content import SIGNALS_HELP

        required_terms = [
            "Confidence",
            "ML Probability",
            "Time Window",
            "Signal Status",
        ]

        for term in required_terms:
            assert term in SIGNALS_HELP

    def test_glossary_comprehensive(self):
        """Verify glossary is comprehensive."""
        from src.dashboard.help.glossary import GLOSSARY

        required_terms = [
            "MSS",
            "FVG",
            "ATR",
            "Triple-Barrier",
            "Dollar Bars",
            "Silver Bullet",
            "Confidence Score",
            "ML Probability",
        ]

        for term in required_terms:
            assert term in GLOSSARY or "triple-barrier" in GLOSSARY


class TestHelpSystemUsability:
    """Test help system usability."""

    def test_help_content_is_readable(self):
        """Verify help content is well-formatted and readable."""
        from src.dashboard.help.help_content import get_help_content

        for page in ["Overview", "Positions", "Signals"]:
            content = get_help_content(page)
            # Should have markdown headers
            assert "#" in content  # Markdown headers
            # Should have bullet points or numbered lists
            assert ("-" in content or "*" in content or "1." in content)

    def test_glossary_definitions_are_clear(self):
        """Verify glossary definitions are clear and understandable."""
        from src.dashboard.help.glossary import GLOSSARY

        for term, definition in GLOSSARY.items():
            # Should have markdown bold formatting
            assert "**" in definition
            # Should not be empty
            assert len(definition.strip()) > 20

    def test_search_provides_context(self):
        """Verify search results provide helpful context."""
        from src.dashboard.help.help_search import search_help

        results = search_help("barrier")

        if len(results) > 0:
            # Each result should have page, section, and preview
            for result in results[:3]:  # Check first 3 results
                assert "page" in result
                assert "section" in result
                assert "preview" in result
                assert len(result["preview"]) > 0


class TestHelpSystemModules:
    """Test help system module structure."""

    def test_all_help_modules_exist(self):
        """Verify all help system modules exist."""
        import os

        modules = [
            "src/dashboard/help/__init__.py",
            "src/dashboard/help/help_content.py",
            "src/dashboard/help/help_modal.py",
            "src/dashboard/help/tooltips.py",
            "src/dashboard/help/glossary.py",
            "src/dashboard/help/help_search.py",
        ]

        for module in modules:
            assert os.path.exists(module), f"Module {module} not found"

    def test_help_modules_importable(self):
        """Verify all help modules can be imported."""
        import importlib

        modules = [
            "src.dashboard.help.help_content",
            "src.dashboard.help.help_modal",
            "src.dashboard.help.tooltips",
            "src.dashboard.help.glossary",
            "src.dashboard.help.help_search",
        ]

        for module in modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                pytest.fail(f"Failed to import {module}: {e}")
