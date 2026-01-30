"""StrategyDecision contract roundtrip tests.

Verifies:
- JSON roundtrip preserves all fields
- Decimal precision is maintained
- Nested StrategyDecisionOrder objects serialize correctly
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from cryptoscreener.trading.contracts import (
    OrderSide,
    PositionSide,
    StrategyDecision,
    StrategyDecisionOrder,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "trading_contracts"


class TestStrategyDecisionRoundtrip:
    """Test StrategyDecision JSON roundtrip."""

    def test_roundtrip_with_orders(self) -> None:
        """StrategyDecision with orders survives JSON roundtrip."""
        order = StrategyDecisionOrder(
            session_id="sim_test123",
            side=OrderSide.SELL,
            price=Decimal("42004.20"),
            quantity=Decimal("0.001"),
            reason="close_long",
        )
        original = StrategyDecision(
            session_id="sim_test123",
            ts=1706140800500,
            tick_seq=42,
            bid=Decimal("42000.00"),
            ask=Decimal("42001.00"),
            mid=Decimal("42000.50"),
            last_trade_price=Decimal("42000.25"),
            position_qty=Decimal("0.001"),
            position_side=PositionSide.LONG,
            unrealized_pnl=Decimal("0.50"),
            realized_pnl=Decimal("2.30"),
            pending_order_count=0,
            orders=[order],
            symbol="BTCUSDT",
        )

        json_str = original.model_dump_json()
        restored = StrategyDecision.model_validate_json(json_str)

        assert original == restored
        assert restored.orders[0].side == OrderSide.SELL
        assert restored.orders[0].price == Decimal("42004.20")

    def test_roundtrip_no_orders(self) -> None:
        """StrategyDecision without orders survives JSON roundtrip."""
        original = StrategyDecision(
            session_id="sim_test123",
            ts=1706140800000,
            tick_seq=0,
            bid=Decimal("42000.00"),
            ask=Decimal("42001.00"),
            mid=Decimal("42000.50"),
            last_trade_price=Decimal("0"),
            position_qty=Decimal("0"),
            position_side=PositionSide.FLAT,
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            pending_order_count=0,
            orders=[],
            symbol="BTCUSDT",
        )

        json_str = original.model_dump_json()
        restored = StrategyDecision.model_validate_json(json_str)

        assert original == restored
        assert len(restored.orders) == 0

    def test_decimal_precision_preserved(self) -> None:
        """Decimal precision is preserved through roundtrip."""
        original = StrategyDecision(
            session_id="sim_test123",
            ts=1706140800000,
            tick_seq=0,
            bid=Decimal("42000.123456789"),
            ask=Decimal("42001.987654321"),
            mid=Decimal("42001.055555555"),
            last_trade_price=Decimal("42000.111111111"),
            position_qty=Decimal("0.00000001"),
            position_side=PositionSide.LONG,
            unrealized_pnl=Decimal("0.00000001"),
            realized_pnl=Decimal("-0.00000001"),
            pending_order_count=0,
            orders=[],
            symbol="BTCUSDT",
        )

        json_str = original.model_dump_json()
        restored = StrategyDecision.model_validate_json(json_str)

        assert restored.bid == Decimal("42000.123456789")
        assert restored.ask == Decimal("42001.987654321")
        assert restored.position_qty == Decimal("0.00000001")


class TestStrategyDecisionValidation:
    """Test StrategyDecision validation."""

    def test_requires_session_id(self) -> None:
        """session_id is required."""
        with pytest.raises(ValidationError):
            StrategyDecision(
                session_id="",  # Empty string should fail
                ts=1706140800000,
                tick_seq=0,
                bid=Decimal("42000"),
                ask=Decimal("42001"),
                mid=Decimal("42000.50"),
                last_trade_price=Decimal("0"),
                position_qty=Decimal("0"),
                position_side=PositionSide.FLAT,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                pending_order_count=0,
                orders=[],
                symbol="BTCUSDT",
            )

    def test_tick_seq_non_negative(self) -> None:
        """tick_seq must be non-negative."""
        with pytest.raises(ValidationError):
            StrategyDecision(
                session_id="sim_test123",
                ts=1706140800000,
                tick_seq=-1,  # Negative should fail
                bid=Decimal("42000"),
                ask=Decimal("42001"),
                mid=Decimal("42000.50"),
                last_trade_price=Decimal("0"),
                position_qty=Decimal("0"),
                position_side=PositionSide.FLAT,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                pending_order_count=0,
                orders=[],
                symbol="BTCUSDT",
            )

    def test_enum_validation(self) -> None:
        """Invalid enum values are rejected."""
        with pytest.raises(ValidationError):
            StrategyDecision(
                session_id="sim_test123",
                ts=1706140800000,
                tick_seq=0,
                bid=Decimal("42000"),
                ask=Decimal("42001"),
                mid=Decimal("42000.50"),
                last_trade_price=Decimal("0"),
                position_qty=Decimal("0"),
                position_side="INVALID",  # type: ignore[arg-type]
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                pending_order_count=0,
                orders=[],
                symbol="BTCUSDT",
            )


class TestStrategyDecisionOrderValidation:
    """Test StrategyDecisionOrder validation."""

    def test_order_roundtrip(self) -> None:
        """StrategyDecisionOrder survives JSON roundtrip."""
        original = StrategyDecisionOrder(
            session_id="sim_test123",
            side=OrderSide.BUY,
            price=Decimal("41995.80"),
            quantity=Decimal("0.002"),
            reason="enter_long",
        )

        json_str = original.model_dump_json()
        restored = StrategyDecisionOrder.model_validate_json(json_str)

        assert original == restored

    def test_order_requires_session_id(self) -> None:
        """Order requires session_id."""
        with pytest.raises(ValidationError):
            StrategyDecisionOrder(
                session_id="",
                side=OrderSide.BUY,
                price=Decimal("42000"),
                quantity=Decimal("0.001"),
            )


class TestStrategyDecisionFixtures:
    """Test loading fixtures."""

    def test_fixture_with_orders_loads(self) -> None:
        """Fixture with orders loads correctly."""
        fixture_path = FIXTURES_DIR / "strategy_decision_with_orders.json"
        with open(fixture_path) as f:
            data = json.load(f)

        decision = StrategyDecision.model_validate(data)

        assert decision.session_id == "sim_abc123def456"
        assert decision.tick_seq == 42
        assert len(decision.orders) == 1
        assert decision.orders[0].side == OrderSide.SELL

    def test_fixture_no_orders_loads(self) -> None:
        """Fixture without orders loads correctly."""
        fixture_path = FIXTURES_DIR / "strategy_decision_no_orders.json"
        with open(fixture_path) as f:
            data = json.load(f)

        decision = StrategyDecision.model_validate(data)

        assert decision.session_id == "sim_abc123def456"
        assert decision.tick_seq == 0
        assert len(decision.orders) == 0
        assert decision.position_side == PositionSide.FLAT


class TestStrategyDecisionProperties:
    """Test StrategyDecision properties."""

    def test_dedupe_key(self) -> None:
        """dedupe_key is (session_id, tick_seq)."""
        decision = StrategyDecision(
            session_id="sim_test123",
            ts=1706140800000,
            tick_seq=42,
            bid=Decimal("42000"),
            ask=Decimal("42001"),
            mid=Decimal("42000.50"),
            last_trade_price=Decimal("0"),
            position_qty=Decimal("0"),
            position_side=PositionSide.FLAT,
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            pending_order_count=0,
            orders=[],
            symbol="BTCUSDT",
        )

        assert decision.dedupe_key == ("sim_test123", 42)

    def test_order_count(self) -> None:
        """order_count returns correct count."""
        order = StrategyDecisionOrder(
            session_id="sim_test123",
            side=OrderSide.BUY,
            price=Decimal("42000"),
            quantity=Decimal("0.001"),
        )
        decision = StrategyDecision(
            session_id="sim_test123",
            ts=1706140800000,
            tick_seq=0,
            bid=Decimal("42000"),
            ask=Decimal("42001"),
            mid=Decimal("42000.50"),
            last_trade_price=Decimal("0"),
            position_qty=Decimal("0"),
            position_side=PositionSide.FLAT,
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            pending_order_count=0,
            orders=[order, order],
            symbol="BTCUSDT",
        )

        assert decision.order_count == 2

    def test_has_orders(self) -> None:
        """has_orders returns correct value."""
        decision_empty = StrategyDecision(
            session_id="sim_test123",
            ts=1706140800000,
            tick_seq=0,
            bid=Decimal("42000"),
            ask=Decimal("42001"),
            mid=Decimal("42000.50"),
            last_trade_price=Decimal("0"),
            position_qty=Decimal("0"),
            position_side=PositionSide.FLAT,
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            pending_order_count=0,
            orders=[],
            symbol="BTCUSDT",
        )
        assert not decision_empty.has_orders

        order = StrategyDecisionOrder(
            session_id="sim_test123",
            side=OrderSide.BUY,
            price=Decimal("42000"),
            quantity=Decimal("0.001"),
        )
        decision_with = StrategyDecision(
            session_id="sim_test123",
            ts=1706140800000,
            tick_seq=0,
            bid=Decimal("42000"),
            ask=Decimal("42001"),
            mid=Decimal("42000.50"),
            last_trade_price=Decimal("0"),
            position_qty=Decimal("0"),
            position_side=PositionSide.FLAT,
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            pending_order_count=0,
            orders=[order],
            symbol="BTCUSDT",
        )
        assert decision_with.has_orders
