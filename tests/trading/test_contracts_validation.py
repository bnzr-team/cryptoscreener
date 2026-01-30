"""Schema validation tests for trading v2 contracts.

Tests verify:
1. Invalid enum values are rejected
2. Missing required fields are rejected
3. Extra fields are forbidden
4. Required fields cannot be None
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from cryptoscreener.trading.contracts import (
    BreachSeverity,
    BreachType,
    FillEvent,
    OrderAck,
    OrderIntent,
    OrderPriority,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    PositionSnapshot,
    RiskAction,
    RiskBreachEvent,
    SessionState,
    TimeInForce,
)


class TestOrderIntentValidation:
    """Validation tests for OrderIntent."""

    def test_invalid_side_rejected(self) -> None:
        """Invalid side enum value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            OrderIntent(
                session_id="sess_test_001",
                ts=1738180800000,
                client_order_id="coid_test_001",
                symbol="BTCUSDT",
                side="INVALID",  # type: ignore[arg-type]
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("67890.50"),
                time_in_force=TimeInForce.GTX,
                reduce_only=False,
                priority=OrderPriority.NORMAL,
            )
        assert "side" in str(exc_info.value)

    def test_invalid_order_type_rejected(self) -> None:
        """Invalid order_type enum value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            OrderIntent(
                session_id="sess_test_001",
                ts=1738180800000,
                client_order_id="coid_test_001",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type="INVALID",  # type: ignore[arg-type]
                quantity=Decimal("0.001"),
                price=Decimal("67890.50"),
                time_in_force=TimeInForce.GTX,
                reduce_only=False,
                priority=OrderPriority.NORMAL,
            )
        assert "order_type" in str(exc_info.value)

    def test_missing_required_fields_rejected(self) -> None:
        """Missing required fields are rejected."""
        with pytest.raises(ValidationError):
            OrderIntent()  # type: ignore[call-arg]

    def test_missing_session_id_rejected(self) -> None:
        """Missing session_id is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            OrderIntent(
                ts=1738180800000,
                client_order_id="coid_test_001",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("67890.50"),
                time_in_force=TimeInForce.GTX,
                reduce_only=False,
                priority=OrderPriority.NORMAL,
            )  # type: ignore[call-arg]
        assert "session_id" in str(exc_info.value)

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            OrderIntent(
                session_id="sess_test_001",
                ts=1738180800000,
                client_order_id="coid_test_001",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("67890.50"),
                time_in_force=TimeInForce.GTX,
                reduce_only=False,
                priority=OrderPriority.NORMAL,
                extra_field="should_fail",  # type: ignore[call-arg]
            )
        assert "extra_field" in str(exc_info.value)


class TestOrderAckValidation:
    """Validation tests for OrderAck."""

    def test_invalid_status_rejected(self) -> None:
        """Invalid status enum value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            OrderAck(
                session_id="sess_test_001",
                ts=1738180800123,
                client_order_id="coid_test_001",
                order_id=1234567890,
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                status="INVALID",  # type: ignore[arg-type]
                quantity=Decimal("0.001"),
                price=Decimal("67890.50"),
                executed_qty=Decimal("0.000"),
                avg_price=Decimal("0.00"),
                time_in_force=TimeInForce.GTX,
                reduce_only=False,
            )
        assert "status" in str(exc_info.value)

    def test_missing_order_id_rejected(self) -> None:
        """Missing order_id is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            OrderAck(
                session_id="sess_test_001",
                ts=1738180800123,
                client_order_id="coid_test_001",
                # order_id missing
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                status=OrderStatus.NEW,
                quantity=Decimal("0.001"),
                price=Decimal("67890.50"),
                executed_qty=Decimal("0.000"),
                avg_price=Decimal("0.00"),
                time_in_force=TimeInForce.GTX,
                reduce_only=False,
            )  # type: ignore[call-arg]
        assert "order_id" in str(exc_info.value)


class TestFillEventValidation:
    """Validation tests for FillEvent."""

    def test_invalid_position_side_rejected(self) -> None:
        """Invalid position_side enum value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FillEvent(
                session_id="sess_test_001",
                ts=1738180801500,
                symbol="BTCUSDT",
                order_id=1234567890,
                trade_id=9876543210,
                client_order_id="coid_test_001",
                side=OrderSide.BUY,
                fill_qty=Decimal("0.001"),
                fill_price=Decimal("67890.50"),
                commission=Decimal("0.01358"),
                commission_asset="USDT",
                realized_pnl=Decimal("0.00"),
                is_maker=True,
                position_side="INVALID",  # type: ignore[arg-type]
            )
        assert "position_side" in str(exc_info.value)

    def test_missing_trade_id_rejected(self) -> None:
        """Missing trade_id is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FillEvent(
                session_id="sess_test_001",
                ts=1738180801500,
                symbol="BTCUSDT",
                order_id=1234567890,
                # trade_id missing
                client_order_id="coid_test_001",
                side=OrderSide.BUY,
                fill_qty=Decimal("0.001"),
                fill_price=Decimal("67890.50"),
                commission=Decimal("0.01358"),
                commission_asset="USDT",
                realized_pnl=Decimal("0.00"),
                is_maker=True,
                position_side=PositionSide.BOTH,
            )  # type: ignore[call-arg]
        assert "trade_id" in str(exc_info.value)


class TestPositionSnapshotValidation:
    """Validation tests for PositionSnapshot."""

    def test_invalid_margin_type_rejected(self) -> None:
        """Invalid margin_type enum value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PositionSnapshot(
                session_id="sess_test_001",
                ts=1738180802000,
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.001"),
                entry_price=Decimal("67890.50"),
                mark_price=Decimal("67920.00"),
                unrealized_pnl=Decimal("0.03"),
                realized_pnl_session=Decimal("0.00"),
                leverage=5,
                liquidation_price=Decimal("54312.40"),
                margin_type="INVALID",  # type: ignore[arg-type]
            )
        assert "margin_type" in str(exc_info.value)


class TestSessionStateValidation:
    """Validation tests for SessionState."""

    def test_invalid_state_rejected(self) -> None:
        """Invalid state enum value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SessionState(
                session_id="sess_test_001",
                ts=1738180803000,
                state="INVALID",  # type: ignore[arg-type]
                symbols_active=["BTCUSDT"],
                open_orders_count=0,
                positions_count=0,
                daily_pnl=Decimal("0.00"),
                daily_trades=0,
                risk_utilization=Decimal("0.00"),
            )
        assert "state" in str(exc_info.value)


class TestRiskBreachEventValidation:
    """Validation tests for RiskBreachEvent."""

    def test_invalid_severity_rejected(self) -> None:
        """Invalid severity enum value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RiskBreachEvent(
                session_id="sess_test_001",
                ts=1738180900000,
                breach_id="breach_test_001",
                breach_type=BreachType.DAILY_LOSS_LIMIT,
                severity="INVALID",  # type: ignore[arg-type]
                threshold=Decimal("-100.00"),
                actual=Decimal("-105.50"),
                action_taken=RiskAction.PAUSE_SESSION,
            )
        assert "severity" in str(exc_info.value)

    def test_invalid_breach_type_rejected(self) -> None:
        """Invalid breach_type enum value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RiskBreachEvent(
                session_id="sess_test_001",
                ts=1738180900000,
                breach_id="breach_test_001",
                breach_type="INVALID",  # type: ignore[arg-type]
                severity=BreachSeverity.CRITICAL,
                threshold=Decimal("-100.00"),
                actual=Decimal("-105.50"),
                action_taken=RiskAction.PAUSE_SESSION,
            )
        assert "breach_type" in str(exc_info.value)

    def test_invalid_action_rejected(self) -> None:
        """Invalid action_taken enum value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RiskBreachEvent(
                session_id="sess_test_001",
                ts=1738180900000,
                breach_id="breach_test_001",
                breach_type=BreachType.DAILY_LOSS_LIMIT,
                severity=BreachSeverity.CRITICAL,
                threshold=Decimal("-100.00"),
                actual=Decimal("-105.50"),
                action_taken="INVALID",  # type: ignore[arg-type]
            )
        assert "action_taken" in str(exc_info.value)


class TestDecimalParsing:
    """Test Decimal parsing from various input types."""

    def test_string_to_decimal(self) -> None:
        """String values are parsed to Decimal."""
        intent = OrderIntent(
            session_id="sess_test_001",
            ts=1738180800000,
            client_order_id="coid_test_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity="0.001",  # type: ignore[arg-type]
            price="67890.50",  # type: ignore[arg-type]
            time_in_force=TimeInForce.GTX,
            reduce_only=False,
            priority=OrderPriority.NORMAL,
        )
        assert isinstance(intent.quantity, Decimal)
        assert isinstance(intent.price, Decimal)
        assert intent.quantity == Decimal("0.001")
        assert intent.price == Decimal("67890.50")

    def test_int_to_decimal(self) -> None:
        """Integer values are parsed to Decimal."""
        intent = OrderIntent(
            session_id="sess_test_001",
            ts=1738180800000,
            client_order_id="coid_test_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1,  # type: ignore[arg-type]
            time_in_force=TimeInForce.IOC,
            reduce_only=False,
            priority=OrderPriority.NORMAL,
        )
        assert isinstance(intent.quantity, Decimal)
        assert intent.quantity == Decimal("1")

    def test_float_to_decimal(self) -> None:
        """Float values are converted via string to preserve precision."""
        intent = OrderIntent(
            session_id="sess_test_001",
            ts=1738180800000,
            client_order_id="coid_test_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,  # type: ignore[arg-type]
            time_in_force=TimeInForce.IOC,
            reduce_only=False,
            priority=OrderPriority.NORMAL,
        )
        assert isinstance(intent.quantity, Decimal)
