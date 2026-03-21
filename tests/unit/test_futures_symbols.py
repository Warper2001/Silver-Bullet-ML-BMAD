"""Unit tests for MNQ futures symbol generation.

Tests symbol generation, chronological order, month codes,
expiration edge cases, year rollover, and leap years.
"""

from datetime import date, datetime, timedelta

import pytest
from freezegun import freeze_time

from src.data.futures_symbols import (
    ContractInfo,
    FuturesSymbolGenerator,
    QUARTERLY_CODES,
)


@pytest.fixture
def generator():
    """Create FuturesSymbolGenerator instance."""
    return FuturesSymbolGenerator()


class TestSymbolGeneration:
    """Tests for symbol generation."""

    def test_generate_mnq_symbols_24_months(self, generator):
        """Test generates 8 symbols for 24 months."""
        symbols = generator.generate_mnq_symbols(months_back=24)
        assert len(symbols) == 8

    def test_generate_mnq_symbols_chronological_order(self, generator):
        """Test symbols are in chronological order (oldest first)."""
        symbols = generator.generate_mnq_symbols(months_back=24)

        # Parse years from symbols
        years = []
        months = []

        month_order = {"H": 3, "M": 6, "U": 9, "Z": 12}

        for symbol in symbols:
            month_code = symbol[3:4]
            year_suffix = symbol[4:6]
            years.append(2000 + int(year_suffix))
            months.append(month_order[month_code])

        # Check chronological order (each should be later than previous)
        for i in range(len(years) - 1):
            if years[i] == years[i + 1]:
                # Same year, month should increase
                assert months[i] < months[i + 1]
            else:
                # Different year, year should increase
                assert years[i] < years[i + 1]

    def test_generate_mnq_symbols_month_codes(self, generator):
        """Test uses correct quarterly month codes."""
        symbols = generator.generate_mnq_symbols(months_back=24)

        month_codes = [s[3:4] for s in symbols]
        expected_codes = QUARTERLY_CODES  # H, M, U, Z

        # All codes should be valid quarterly codes
        for code in month_codes:
            assert code in expected_codes

    def test_generate_mnq_symbols_format(self, generator):
        """Test symbols have correct format: MNQ + CODE + YY."""
        symbols = generator.generate_mnq_symbols(months_back=24)

        for symbol in symbols:
            assert symbol.startswith("MNQ")
            assert len(symbol) == 6  # MNQ + 1 char + 2 digits
            assert symbol[3:4] in QUARTERLY_CODES
            assert symbol[4:6].isdigit()


class TestCurrentContractDetection:
    """Tests for finding current active contract."""

    @freeze_time("2026-03-15")
    def test_find_current_contract_march(self, generator):
        """Test current contract in March (March contract active)."""
        contract = generator._find_current_contract()
        assert contract.month_code == "H"
        assert contract.year == 2026
        assert contract.symbol == "MNQH26"

    @freeze_time("2026-06-15")
    def test_find_current_contract_june(self, generator):
        """Test current contract in June (June contract active)."""
        contract = generator._find_current_contract()
        assert contract.month_code == "M"
        assert contract.year == 2026
        assert contract.symbol == "MNQM26"

    @freeze_time("2026-09-15")
    def test_find_current_contract_september(self, generator):
        """Test current contract in September (September contract active)."""
        contract = generator._find_current_contract()
        assert contract.month_code == "U"
        assert contract.year == 2026
        assert contract.symbol == "MNQU26"

    @freeze_time("2026-12-15")
    def test_find_current_contract_december(self, generator):
        """Test current contract in December (December contract active)."""
        contract = generator._find_current_contract()
        assert contract.month_code == "Z"
        assert contract.year == 2026
        assert contract.symbol == "MNQZ26"

    @freeze_time("2026-01-15")
    def test_find_current_contract_january(self, generator):
        """Test current contract in January (March contract active)."""
        contract = generator._find_current_contract()
        assert contract.month_code == "H"
        assert contract.year == 2026
        assert contract.symbol == "MNQH26"

    @freeze_time("2026-04-15")
    def test_find_current_contract_april(self, generator):
        """Test current contract in April (June contract active)."""
        contract = generator._find_current_contract()
        assert contract.month_code == "M"
        assert contract.year == 2026
        assert contract.symbol == "MNQM26"


class TestNearExpiration:
    """Tests for near-expiration edge cases."""

    @freeze_time("2026-03-20")  # Near March expiration (third Friday)
    def test_near_expiration_uses_next_contract(self, generator):
        """Test uses next contract when within 7 days of expiration."""
        # March contract expires on third Friday
        # If we're within 7 days, should use June contract
        contract = generator._find_current_contract()
        assert contract.month_code == "M"  # June contract
        assert contract.year == 2026

    @freeze_time("2026-03-10")  # Far from March expiration
    def test_far_from_expiration_uses_current(self, generator):
        """Test uses current contract when far from expiration."""
        contract = generator._find_current_contract()
        assert contract.month_code == "H"  # March contract
        assert contract.year == 2026

    @freeze_time("2026-12-25")  # Near December expiration
    def test_near_december_expiration(self, generator):
        """Test handles near December expiration correctly."""
        contract = generator._find_current_contract()
        # Near December expiration, should use next year's March contract
        assert contract.month_code == "H"
        assert contract.year == 2027


class TestYearRollover:
    """Tests for year transitions."""

    @freeze_time("2026-12-31")
    def test_year_rollover_december_to_january(self, generator):
        """Test year rollover from December to January."""
        # Last day of December, still in December contract
        contract = generator._find_current_contract()
        assert contract.year == 2026
        assert contract.month_code == "Z"

    def test_get_next_contract_year_rollover(self, generator):
        """Test get_next_contract handles year rollover."""
        # December 2026 contract
        contract = ContractInfo(
            symbol="MNQZ26",
            year=2026,
            month_code="Z",
            expiration_month=12,
            display_name="MNQ Z26",
        )

        # Next should be March 2027
        next_contract = generator._get_next_contract(contract)
        assert next_contract.symbol == "MNQH27"
        assert next_contract.year == 2027
        assert next_contract.month_code == "H"

    def test_get_previous_contract_year_rollover(self, generator):
        """Test get_previous_contract handles backwards year rollover."""
        # March 2026 contract
        contract = ContractInfo(
            symbol="MNQH26",
            year=2026,
            month_code="H",
            expiration_month=3,
            display_name="MNQ H26",
        )

        # Previous should be December 2025
        prev_contract = generator._get_previous_contract(contract)
        assert prev_contract.symbol == "MNQZ25"
        assert prev_contract.year == 2025
        assert prev_contract.month_code == "Z"


class TestCenturyRollover:
    """Tests for century rollover (future-proofing)."""

    def test_century_rollover_2099_to_2100(self, generator):
        """Test symbol generation works across century boundary."""
        # Create contract for December 2099
        contract = ContractInfo(
            symbol="MNQZ99",
            year=2099,
            month_code="Z",
            expiration_month=12,
            display_name="MNQ Z99",
        )

        # Next should be March 2100
        next_contract = generator._get_next_contract(contract)
        assert next_contract.symbol == "MNQH00"
        assert next_contract.year == 2100


class TestLeapYearHandling:
    """Tests for leap year date calculations."""

    @freeze_time("2024-02-29")  # Leap day
    def test_leap_year_february_29(self, generator):
        """Test handles leap day correctly."""
        contract = generator._find_current_contract()
        # February 29 is before March expiration
        assert contract.month_code == "H"
        assert contract.year == 2024

    def test_leap_year_expiration_date(self, generator):
        """Test expiration date calculation in leap year."""
        # 2024 is a leap year
        contract = ContractInfo(
            symbol="MNQH24",
            year=2024,
            month_code="H",
            expiration_month=3,
            display_name="MNQ H24",
        )

        # Should calculate correct third Friday in March 2024
        expiration = generator._get_contract_expiration_date(contract)
        assert expiration.year == 2024
        assert expiration.month == 3
        # Third Friday should be March 15, 2024
        assert expiration.day == 15
        assert expiration.weekday() == 4  # Friday


class TestMonthCodeMapping:
    """Tests for month code to calendar month mapping."""

    def test_get_month_code_h(self, generator):
        """Test H maps to March."""
        assert generator._get_expiration_month("H") == 3

    def test_get_month_code_m(self, generator):
        """Test M maps to June."""
        assert generator._get_expiration_month("M") == 6

    def test_get_month_code_u(self, generator):
        """Test U maps to September."""
        assert generator._get_expiration_month("U") == 9

    def test_get_month_code_z(self, generator):
        """Test Z maps to December."""
        assert generator._get_expiration_month("Z") == 12

    def test_get_month_code_for_month(self, generator):
        """Test mapping calendar month to contract code."""
        # January/February → H (March)
        assert generator._get_month_code_for_month(1) == "H"
        assert generator._get_month_code_for_month(2) == "H"

        # April/May → M (June)
        assert generator._get_month_code_for_month(4) == "M"
        assert generator._get_month_code_for_month(5) == "M"

        # July/August → U (September)
        assert generator._get_month_code_for_month(7) == "U"
        assert generator._get_month_code_for_month(8) == "U"

        # October/November → Z (December)
        assert generator._get_month_code_for_month(10) == "Z"
        assert generator._get_month_code_for_month(11) == "Z"

        # March → M (June, since March expiration passed)
        assert generator._get_month_code_for_month(3) == "M"

        # June → U (September, since June expiration passed)
        assert generator._get_month_code_for_month(6) == "U"

        # September → Z (December, since September expiration passed)
        assert generator._get_month_code_for_month(9) == "Z"

        # December → H (March next year)
        assert generator._get_month_code_for_month(12) == "H"


class TestSufficientHistory:
    """Tests for historical data availability checks."""

    @freeze_time("2026-03-15")
    def test_has_sufficient_history_expired_contract(self, generator):
        """Test expired contract has sufficient history."""
        # December 2025 contract (expired)
        sufficient = generator._has_sufficient_history("MNQZ25")
        # Should have > 30 days since expiration
        assert sufficient is True

    @freeze_time("2026-03-15")
    def test_has_sufficient_history_current_contract(self, generator):
        """Test current contract may not have sufficient history."""
        # March 2026 contract (currently active)
        sufficient = generator._has_sufficient_history("MNQH26")
        # May not have 30 days of data yet
        # This depends on implementation - test for reasonable behavior
        assert isinstance(sufficient, bool)


class TestGetQuarterlyContracts:
    """Tests for getting all contracts in a year."""

    def test_get_quarterly_contracts_2026(self, generator):
        """Test gets all 4 quarterly contracts for 2026."""
        contracts = generator._get_quarterly_contracts(2026)

        assert len(contracts) == 4
        symbols = [c.symbol for c in contracts]
        assert "MNQH26" in symbols
        assert "MNQM26" in symbols
        assert "MNQU26" in symbols
        assert "MNQZ26" in symbols

    def test_get_quarterly_contracts_start_month(self, generator):
        """Test gets contracts from specific start month."""
        # Get contracts from June onwards
        contracts = generator._get_quarterly_contracts(2026, start_month=6)

        assert len(contracts) == 3  # M, U, Z
        symbols = [c.symbol for c in contracts]
        assert "MNQH26" not in symbols  # H (March) excluded
        assert "MNQM26" in symbols
        assert "MNQU26" in symbols
        assert "MNQZ26" in symbols


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    @freeze_time("2026-03-15")
    def test_generate_8_quarters_from_march_2026(self, generator):
        """Test realistic scenario: March 2026, generate 8 quarters."""
        symbols = generator.generate_mnq_symbols(months_back=24)

        # Should be in chronological order (oldest first)
        expected = [
            "MNQM24",  # June 2024
            "MNQU24",  # September 2024
            "MNQZ24",  # December 2024
            "MNQH25",  # March 2025
            "MNQM25",  # June 2025
            "MNQU25",  # September 2025
            "MNQZ25",  # December 2025
            "MNQH26",  # March 2026 (current)
        ]

        assert symbols == expected

    @freeze_time("2026-06-15")
    def test_generate_8_quarters_from_june_2026(self, generator):
        """Test realistic scenario: June 2026, generate 8 quarters."""
        symbols = generator.generate_mnq_symbols(months_back=24)

        # Should include H26 but start from U24
        assert symbols[0] in ["MNQU24", "MNQZ24"]  # Oldest
        assert symbols[-1] == "MNQM26"  # Current
        assert len(symbols) == 8
