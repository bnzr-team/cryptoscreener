"""Tests for CROSS fill model.

Verifies the Phase 1 fill model behavior:
- BUY order filled if trade price <= bid price
- SELL order filled if trade price >= ask price
- Fill at our limit price (optimistic)
"""

from __future__ import annotations

from decimal import Decimal

from cryptoscreener.trading.contracts import OrderSide
from cryptoscreener.trading.sim.fill_model import (
    CrossFillModel,
    FillResult,
    MarketTick,
    OrderState,
)


class TestCrossFillModelBuy:
    """Test CROSS fill model for BUY orders."""

    def test_buy_filled_when_trade_at_limit(self) -> None:
        """BUY order filled when trade exactly at limit price."""
        model = CrossFillModel()
        order = OrderState(
            side=OrderSide.BUY,
            price=Decimal("42000"),
            quantity=Decimal("0.1"),
            placed_ts=1706140800000,
        )
        tick = MarketTick(
            ts=1706140800100,
            trade_price=Decimal("42000"),
        )

        result = model.check_fill(order, tick)

        assert result.filled is True
        assert result.fill_price == Decimal("42000")
        assert result.fill_qty == Decimal("0.1")

    def test_buy_filled_when_trade_below_limit(self) -> None:
        """BUY order filled when trade price below limit."""
        model = CrossFillModel()
        order = OrderState(
            side=OrderSide.BUY,
            price=Decimal("42000"),
            quantity=Decimal("0.1"),
            placed_ts=1706140800000,
        )
        tick = MarketTick(
            ts=1706140800100,
            trade_price=Decimal("41990"),  # Below our bid
        )

        result = model.check_fill(order, tick)

        assert result.filled is True
        assert result.fill_price == Decimal("42000")  # Fill at our limit (optimistic)
        assert result.fill_qty == Decimal("0.1")

    def test_buy_not_filled_when_trade_above_limit(self) -> None:
        """BUY order NOT filled when trade price above limit."""
        model = CrossFillModel()
        order = OrderState(
            side=OrderSide.BUY,
            price=Decimal("42000"),
            quantity=Decimal("0.1"),
            placed_ts=1706140800000,
        )
        tick = MarketTick(
            ts=1706140800100,
            trade_price=Decimal("42001"),  # Above our bid
        )

        result = model.check_fill(order, tick)

        assert result.filled is False
        assert result.fill_price is None
        assert result.fill_qty is None


class TestCrossFillModelSell:
    """Test CROSS fill model for SELL orders."""

    def test_sell_filled_when_trade_at_limit(self) -> None:
        """SELL order filled when trade exactly at limit price."""
        model = CrossFillModel()
        order = OrderState(
            side=OrderSide.SELL,
            price=Decimal("42000"),
            quantity=Decimal("0.1"),
            placed_ts=1706140800000,
        )
        tick = MarketTick(
            ts=1706140800100,
            trade_price=Decimal("42000"),
        )

        result = model.check_fill(order, tick)

        assert result.filled is True
        assert result.fill_price == Decimal("42000")
        assert result.fill_qty == Decimal("0.1")

    def test_sell_filled_when_trade_above_limit(self) -> None:
        """SELL order filled when trade price above limit."""
        model = CrossFillModel()
        order = OrderState(
            side=OrderSide.SELL,
            price=Decimal("42000"),
            quantity=Decimal("0.1"),
            placed_ts=1706140800000,
        )
        tick = MarketTick(
            ts=1706140800100,
            trade_price=Decimal("42010"),  # Above our ask
        )

        result = model.check_fill(order, tick)

        assert result.filled is True
        assert result.fill_price == Decimal("42000")  # Fill at our limit (optimistic)
        assert result.fill_qty == Decimal("0.1")

    def test_sell_not_filled_when_trade_below_limit(self) -> None:
        """SELL order NOT filled when trade price below limit."""
        model = CrossFillModel()
        order = OrderState(
            side=OrderSide.SELL,
            price=Decimal("42000"),
            quantity=Decimal("0.1"),
            placed_ts=1706140800000,
        )
        tick = MarketTick(
            ts=1706140800100,
            trade_price=Decimal("41999"),  # Below our ask
        )

        result = model.check_fill(order, tick)

        assert result.filled is False
        assert result.fill_price is None
        assert result.fill_qty is None


class TestCrossFillModelNoTradePrice:
    """Test CROSS fill model when no trade price available."""

    def test_no_fill_without_trade_price(self) -> None:
        """No fill when tick has no trade price (book update only)."""
        model = CrossFillModel()
        order = OrderState(
            side=OrderSide.BUY,
            price=Decimal("42000"),
            quantity=Decimal("0.1"),
            placed_ts=1706140800000,
        )
        tick = MarketTick(
            ts=1706140800100,
            trade_price=None,  # No trade, just book update
            bid_price=Decimal("41999"),
            ask_price=Decimal("42001"),
        )

        result = model.check_fill(order, tick)

        assert result.filled is False


class TestFillResultImmutability:
    """Test FillResult dataclass properties."""

    def test_fill_result_is_frozen(self) -> None:
        """FillResult should be immutable (frozen dataclass)."""
        result = FillResult(
            filled=True,
            fill_price=Decimal("42000"),
            fill_qty=Decimal("0.1"),
        )

        # Attempting to modify should raise
        import dataclasses

        assert dataclasses.is_dataclass(result)
        # Frozen dataclass raises FrozenInstanceError on assignment
        try:
            result.filled = False  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except dataclasses.FrozenInstanceError:
            pass  # Expected


class TestOrderStateImmutability:
    """Test OrderState dataclass properties."""

    def test_order_state_is_frozen(self) -> None:
        """OrderState should be immutable (frozen dataclass)."""
        order = OrderState(
            side=OrderSide.BUY,
            price=Decimal("42000"),
            quantity=Decimal("0.1"),
            placed_ts=1706140800000,
        )

        import dataclasses

        assert dataclasses.is_dataclass(order)
        try:
            order.price = Decimal("43000")  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except dataclasses.FrozenInstanceError:
            pass  # Expected
