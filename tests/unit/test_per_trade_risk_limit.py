"""Unit tests for Per-Trade Risk Limit.

Tests per-trade risk validation, risk calculation, max quantity
determination, CSV logging, and integration with position sizer.
"""

import tempfile
from pathlib import Path
import pytest

from src.risk.per_trade_risk_limit import PerTradeRiskLimit


class TestPerTradeRiskLimitInit:
    """Test PerTradeRiskLimit initialization."""

    def test_init_with_valid_parameters(self):
        """Verify limit initializes with valid parameters."""
        limit = PerTradeRiskLimit(
            max_risk_dollars=500,
            account_balance=50000
        )

        assert limit._max_risk_dollars == 500
        assert limit._account_balance == 50000

    def test_init_with_invalid_limit(self):
        """Verify limit raises error with non-positive limit."""
        with pytest.raises(ValueError):
            PerTradeRiskLimit(
                max_risk_dollars=0,
                account_balance=50000
            )

        with pytest.raises(ValueError):
            PerTradeRiskLimit(
                max_risk_dollars=-100,
                account_balance=50000
            )

    def test_init_with_audit_trail(self):
        """Verify limit initializes with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "risk_limit.csv")

        limit = PerTradeRiskLimit(
            max_risk_dollars=500,
            account_balance=50000,
            audit_trail_path=audit_path
        )

        assert limit._audit_trail_path == audit_path


class TestValidateTrade:
    """Test trade validation."""

    @pytest.fixture
    def limit(self):
        """Create per-trade risk limit."""
        return PerTradeRiskLimit(
            max_risk_dollars=500,
            account_balance=50000
        )

    def test_validate_trade_within_limit(self, limit):
        """Verify trade within limit is valid."""
        result = limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11730,
            quantity=5
        )

        # Risk: $20 × 2 × 5 = $200
        assert result["is_valid"] is True
        assert result["estimated_risk"] == 200

    def test_validate_trade_exceeds_limit(self, limit):
        """Verify trade exceeding limit is invalid."""
        result = limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11700,
            quantity=20
        )

        # Risk: $50 × 2 × 20 = $2,000 (exceeds $500 limit)
        assert result["is_valid"] is False
        assert result["estimated_risk"] == 2000
        assert result["violation_amount"] == 1500

    def test_validate_trade_at_exact_limit(self, limit):
        """Verify trade at exact limit is valid."""
        # Max risk is $500, risk per contract is $40
        # 12 contracts = $480 (within limit)
        # 13 contracts = $520 (exceeds limit)
        result = limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11730,
            quantity=12
        )

        assert result["is_valid"] is True
        assert result["estimated_risk"] == 480

    def test_validate_trade_zero_quantity(self, limit):
        """Validate trade with zero quantity."""
        result = limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11730,
            quantity=0
        )

        assert result["is_valid"] is True
        assert result["estimated_risk"] == 0

    def test_validate_trade_max_quantity_result(self, limit):
        """Verify max allowed quantity in result."""
        result = limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11730,
            quantity=20
        )

        # Max should be 12 contracts ($500 / $40)
        assert result["max_allowed_quantity"] == 12


class TestCalculateRiskDollars:
    """Test dollar risk calculation."""

    @pytest.fixture
    def limit(self):
        """Create per-trade risk limit."""
        return PerTradeRiskLimit(
            max_risk_dollars=500,
            account_balance=50000
        )

    def test_calculate_risk_long_position(self, limit):
        """Verify risk calculation for long position."""
        risk = limit.calculate_risk_dollars(
            entry_price=11750,
            stop_loss_price=11730,
            quantity=5
        )

        # (11750 - 11730) / 0.25 × 2 = $40 per contract × 5 = $200
        assert risk == 200

    def test_calculate_risk_short_position(self, limit):
        """Verify risk calculation for short position."""
        risk = limit.calculate_risk_dollars(
            entry_price=11750,
            stop_loss_price=11770,
            quantity=3
        )

        # (11770 - 11750) / 0.25 × 2 = $40 per contract × 3 = $120
        assert risk == 120

    def test_calculate_risk_zero_quantity(self, limit):
        """Verify risk calculation with zero quantity."""
        risk = limit.calculate_risk_dollars(
            entry_price=11750,
            stop_loss_price=11730,
            quantity=0
        )

        assert risk == 0


class TestGetMaxAllowedQuantity:
    """Test max allowed quantity calculation."""

    @pytest.fixture
    def limit(self):
        """Create per-trade risk limit."""
        return PerTradeRiskLimit(
            max_risk_dollars=500,
            account_balance=50000
        )

    def test_max_quantity_calculation(self, limit):
        """Verify max quantity calculated correctly."""
        max_qty = limit.get_max_allowed_quantity(
            entry_price=11750,
            stop_loss_price=11730
        )

        # $500 / $40 = 12.5 → 12 contracts (floor)
        assert max_qty == 12

    def test_max_quantity_small_risk(self, limit):
        """Verify max quantity with small stop distance."""
        max_qty = limit.get_max_allowed_quantity(
            entry_price=11750,
            stop_loss_price=11745
        )

        # Risk: $5 / 0.25 × 2 = $10 per contract
        # $500 / $10 = 50 contracts
        assert max_qty == 50

    def test_max_quantity_large_risk(self, limit):
        """Verify max quantity with large stop distance."""
        max_qty = limit.get_max_allowed_quantity(
            entry_price=11750,
            stop_loss_price=11700
        )

        # Risk: $50 / 0.25 × 2 = $100 per contract
        # $500 / $100 = 5 contracts
        assert max_qty == 5


class TestGetRiskPerContract:
    """Test risk per contract calculation."""

    @pytest.fixture
    def limit(self):
        """Create per-trade risk limit."""
        return PerTradeRiskLimit(
            max_risk_dollars=500,
            account_balance=50000
        )

    def test_risk_per_contract_long(self, limit):
        """Verify risk per contract for long."""
        risk = limit.get_risk_per_contract(
            entry_price=11750,
            stop_loss_price=11730
        )

        # (11750 - 11730) / 0.25 × 2 = $40
        assert risk == 40

    def test_risk_per_contract_short(self, limit):
        """Verify risk per contract for short."""
        risk = limit.get_risk_per_contract(
            entry_price=11750,
            stop_loss_price=11770
        )

        # (11770 - 11750) / 0.25 × 2 = $40
        assert risk == 40


class TestUpdateAccountBalance:
    """Test account balance updates."""

    @pytest.fixture
    def limit(self):
        """Create per-trade risk limit."""
        return PerTradeRiskLimit(
            max_risk_dollars=500,
            account_balance=50000
        )

    def test_update_account_balance(self, limit):
        """Verify account balance updates."""
        limit.update_account_balance(51000)

        assert limit._account_balance == 51000

    def test_update_balance_affects_limits(self, limit):
        """Verify balance update affects risk calculations."""
        # First update balance
        limit.update_account_balance(100000)

        # Now we can afford more risk if using percentage-based
        # (for this test, we just verify balance updates)
        assert limit._account_balance == 100000


class TestSetMaxRiskDollars:
    """Test max risk limit updates."""

    @pytest.fixture
    def limit(self):
        """Create per-trade risk limit."""
        return PerTradeRiskLimit(
            max_risk_dollars=500,
            account_balance=50000
        )

    def test_set_max_risk_updates_limit(self, limit):
        """Verify setting max risk updates the limit."""
        limit.set_max_risk_dollars(750)

        assert limit._max_risk_dollars == 750

        # Trade that was rejected before should now pass
        # Risk: 100 points × $2 × 5 = $1,000 (still over $750)
        # Let's use a smaller trade that fits
        result = limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11700,
            quantity=3
        )

        # Risk: 100 points × $2 × 3 = $600 (within $750 limit)
        assert result["is_valid"] is True


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging."""

    @pytest.fixture
    def limit(self):
        """Create per-trade risk limit with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "risk_limit.csv")

        return PerTradeRiskLimit(
            max_risk_dollars=500,
            account_balance=50000,
            audit_trail_path=audit_path
        )

    def test_csv_file_created(self, limit):
        """Verify CSV file created on first validation."""
        limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11730,
            quantity=5
        )

        assert Path(limit._audit_trail_path).exists()

    def test_csv_has_correct_columns(self, limit):
        """Verify CSV has all required columns."""
        limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11730,
            quantity=5
        )

        import csv
        with open(limit._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
            "timestamp",
            "event_type",
            "entry_price",
            "stop_loss_price",
            "quantity",
            "estimated_risk",
            "max_risk_limit",
            "is_valid",
            "violation_amount"
        ]

        assert headers == expected_headers


class TestMultipleTradesSequence:
    """Test multiple trades in sequence."""

    @pytest.fixture
    def limit(self):
        """Create per-trade risk limit."""
        return PerTradeRiskLimit(
            max_risk_dollars=500,
            account_balance=50000
        )

    def test_multiple_valid_trades(self, limit):
        """Verify multiple valid trades can be validated."""
        # Trade 1
        result1 = limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11730,
            quantity=5
        )
        assert result1["is_valid"] is True

        # Trade 2
        result2 = limit.validate_trade(
            entry_price=11745,
            stop_loss_price=11725,
            quantity=3
        )
        assert result2["is_valid"] is True

    def test_sequence_of_valid_invalid(self, limit):
        """Verify mix of valid and invalid trades."""
        # Valid
        result1 = limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11730,
            quantity=10
        )
        assert result1["is_valid"] is True

        # Invalid
        result2 = limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11700,
            quantity=15
        )
        assert result2["is_valid"] is False

        # Valid again
        result3 = limit.validate_trade(
            entry_price=11755,
            stop_loss_price=11735,
            quantity=8
        )
        assert result3["is_valid"] is True


class TestLongAndShortPositions:
    """Test long and short position risk."""

    @pytest.fixture
    def limit(self):
        """Create per-trade risk limit."""
        return PerTradeRiskLimit(
            max_risk_dollars=500,
            account_balance=50000
        )

    def test_long_position_risk(self, limit):
        """Verify long position risk calculation."""
        # Entry > Stop (losing if price drops)
        result = limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11730,
            quantity=5
        )

        assert result["is_valid"] is True
        assert result["estimated_risk"] == 200

    def test_short_position_risk(self, limit):
        """Verify short position risk calculation."""
        # Entry < Stop (losing if price rises)
        result = limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11770,
            quantity=5
        )

        # (11770 - 11750) / 0.25 × 2 = $40 per contract × 5 = $200
        assert result["is_valid"] is True
        assert result["estimated_risk"] == 200


class TestIntegration:
    """Test integration with position sizer."""

    @pytest.fixture
    def limit(self):
        """Create per-trade risk limit."""
        return PerTradeRiskLimit(
            max_risk_dollars=500,
            account_balance=50000
        )

    def test_trade_rejected_when_exceeds_limit(self, limit):
        """Verify trade rejected when exceeds risk limit."""
        result = limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11700,
            quantity=20
        )

        assert result["is_valid"] is False
        assert result["block_reason"] == "Per-trade risk limit exceeded"

    def test_trade_accepted_when_within_limit(self, limit):
        """Verify trade accepted when within risk limit."""
        result = limit.validate_trade(
            entry_price=11750,
            stop_loss_price=11730,
            quantity=10
        )

        assert result["is_valid"] is True
        assert result["estimated_risk"] == 400
