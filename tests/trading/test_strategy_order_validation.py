"""StrategyOrder validation tests.

Tests the lightweight StrategyOrder frozen dataclass used by strategies.
Verifies:
- Quantity/price must be positive
- Reason field rejects numeric data
- Reason field has length limit
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cryptoscreener.trading.contracts import OrderSide
from cryptoscreener.trading.strategy.base import StrategyOrder


class TestStrategyOrderBasicValidation:
    """Test basic StrategyOrder validation."""

    def test_valid_order(self) -> None:
        """Valid order creates successfully."""
        order = StrategyOrder(
            side=OrderSide.BUY,
            price=Decimal("42000.50"),
            quantity=Decimal("0.001"),
            reason="enter_long",
        )
        assert order.side == OrderSide.BUY
        assert order.price == Decimal("42000.50")
        assert order.quantity == Decimal("0.001")
        assert order.reason == "enter_long"

    def test_rejects_zero_quantity(self) -> None:
        """Zero quantity is rejected."""
        with pytest.raises(ValueError, match="quantity must be positive"):
            StrategyOrder(
                side=OrderSide.BUY,
                price=Decimal("42000"),
                quantity=Decimal("0"),
            )

    def test_rejects_negative_quantity(self) -> None:
        """Negative quantity is rejected."""
        with pytest.raises(ValueError, match="quantity must be positive"):
            StrategyOrder(
                side=OrderSide.SELL,
                price=Decimal("42000"),
                quantity=Decimal("-0.001"),
            )

    def test_rejects_zero_price(self) -> None:
        """Zero price is rejected."""
        with pytest.raises(ValueError, match="price must be positive"):
            StrategyOrder(
                side=OrderSide.BUY,
                price=Decimal("0"),
                quantity=Decimal("0.001"),
            )

    def test_rejects_negative_price(self) -> None:
        """Negative price is rejected."""
        with pytest.raises(ValueError, match="price must be positive"):
            StrategyOrder(
                side=OrderSide.SELL,
                price=Decimal("-42000"),
                quantity=Decimal("0.001"),
            )


class TestStrategyOrderReasonValidation:
    """Test reason field validation (no numbers constraint)."""

    def test_reason_rejects_digits(self) -> None:
        """Reason field rejects strings containing digits."""
        with pytest.raises(ValueError, match="reason must not contain numbers"):
            StrategyOrder(
                side=OrderSide.BUY,
                price=Decimal("42000"),
                quantity=Decimal("0.001"),
                reason="entry at 0.5%",
            )

    def test_reason_rejects_embedded_numbers(self) -> None:
        """Reason field rejects any embedded numeric data."""
        with pytest.raises(ValueError, match="reason must not contain numbers"):
            StrategyOrder(
                side=OrderSide.SELL,
                price=Decimal("42000"),
                quantity=Decimal("0.001"),
                reason="tp_level_3",
            )

    def test_reason_rejects_price_references(self) -> None:
        """Reason field rejects price/percentage references."""
        with pytest.raises(ValueError, match="reason must not contain numbers"):
            StrategyOrder(
                side=OrderSide.BUY,
                price=Decimal("42000"),
                quantity=Decimal("0.001"),
                reason="sl_at_41500",
            )

    def test_reason_accepts_text_only(self) -> None:
        """Reason field accepts pure text."""
        order = StrategyOrder(
            side=OrderSide.BUY,
            price=Decimal("42000"),
            quantity=Decimal("0.001"),
            reason="close_long",
        )
        assert order.reason == "close_long"

    def test_reason_accepts_empty(self) -> None:
        """Reason field accepts empty string (default)."""
        order = StrategyOrder(
            side=OrderSide.BUY,
            price=Decimal("42000"),
            quantity=Decimal("0.001"),
        )
        assert order.reason == ""

    def test_reason_accepts_common_patterns(self) -> None:
        """Reason field accepts common strategy reason patterns."""
        valid_reasons = [
            "enter_long",
            "enter_short",
            "close_long",
            "close_short",
            "take_profit",
            "stop_loss",
            "rebalance",
            "exit_all",
        ]
        for reason in valid_reasons:
            order = StrategyOrder(
                side=OrderSide.BUY,
                price=Decimal("42000"),
                quantity=Decimal("0.001"),
                reason=reason,
            )
            assert order.reason == reason

    def test_reason_rejects_too_long(self) -> None:
        """Reason field rejects strings > 64 chars."""
        with pytest.raises(ValueError, match="reason must be <= 64 chars"):
            StrategyOrder(
                side=OrderSide.BUY,
                price=Decimal("42000"),
                quantity=Decimal("0.001"),
                reason="a" * 65,
            )

    def test_reason_accepts_max_length(self) -> None:
        """Reason field accepts 64-char string."""
        order = StrategyOrder(
            side=OrderSide.BUY,
            price=Decimal("42000"),
            quantity=Decimal("0.001"),
            reason="a" * 64,
        )
        assert len(order.reason) == 64


class TestStrategyOrderImmutability:
    """Test that StrategyOrder is immutable (frozen dataclass)."""

    def test_frozen(self) -> None:
        """StrategyOrder is frozen (immutable)."""
        order = StrategyOrder(
            side=OrderSide.BUY,
            price=Decimal("42000"),
            quantity=Decimal("0.001"),
        )
        with pytest.raises(AttributeError):
            order.price = Decimal("43000")  # type: ignore[misc]
