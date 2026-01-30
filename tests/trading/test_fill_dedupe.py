"""FillEvent dedupe logic tests.

Tests verify:
1. Dedupe key is (symbol, order_id, trade_id)
2. Duplicate fills can be detected using dedupe_key
3. Same symbol+order but different trade_id are not duplicates
"""

from __future__ import annotations

from decimal import Decimal

from cryptoscreener.trading.contracts import (
    BreachSeverity,
    BreachType,
    FillEvent,
    MarginType,
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
    SessionStateEnum,
    TimeInForce,
)


class TestFillDedupeKey:
    """Test FillEvent dedupe_key property."""

    def test_dedupe_key_format(self) -> None:
        """Dedupe key is (symbol, order_id, trade_id)."""
        fill = FillEvent(
            session_id="sess_test_001",
            ts=1738180801500,
            symbol="BTCUSDT",
            order_id=123,
            trade_id=456,
            client_order_id="coid_test_001",
            side=OrderSide.BUY,
            fill_qty=Decimal("0.001"),
            fill_price=Decimal("67890.50"),
            commission=Decimal("0.01358"),
            commission_asset="USDT",
            realized_pnl=Decimal("0.00"),
            is_maker=True,
            position_side=PositionSide.BOTH,
        )
        assert fill.dedupe_key == ("BTCUSDT", 123, 456)

    def test_duplicate_fills_same_key(self) -> None:
        """Two fills with same dedupe key are duplicates."""
        fill1 = FillEvent(
            session_id="sess_test_001",
            ts=1738180801500,
            symbol="BTCUSDT",
            order_id=123,
            trade_id=456,
            client_order_id="coid_test_001",
            side=OrderSide.BUY,
            fill_qty=Decimal("0.001"),
            fill_price=Decimal("67890.50"),
            commission=Decimal("0.01358"),
            commission_asset="USDT",
            realized_pnl=Decimal("0.00"),
            is_maker=True,
            position_side=PositionSide.BOTH,
        )
        fill2 = FillEvent(
            session_id="sess_test_001",
            ts=1738180801501,  # Different timestamp
            symbol="BTCUSDT",
            order_id=123,
            trade_id=456,  # Same trade_id = duplicate
            client_order_id="coid_test_001",
            side=OrderSide.BUY,
            fill_qty=Decimal("0.001"),
            fill_price=Decimal("67890.50"),
            commission=Decimal("0.01358"),
            commission_asset="USDT",
            realized_pnl=Decimal("0.00"),
            is_maker=True,
            position_side=PositionSide.BOTH,
        )
        assert fill1.dedupe_key == fill2.dedupe_key

    def test_different_trade_id_not_duplicate(self) -> None:
        """Two fills with different trade_id are not duplicates."""
        fill1 = FillEvent(
            session_id="sess_test_001",
            ts=1738180801500,
            symbol="BTCUSDT",
            order_id=123,
            trade_id=456,
            client_order_id="coid_test_001",
            side=OrderSide.BUY,
            fill_qty=Decimal("0.001"),
            fill_price=Decimal("67890.50"),
            commission=Decimal("0.01358"),
            commission_asset="USDT",
            realized_pnl=Decimal("0.00"),
            is_maker=True,
            position_side=PositionSide.BOTH,
        )
        fill2 = FillEvent(
            session_id="sess_test_001",
            ts=1738180801600,
            symbol="BTCUSDT",
            order_id=123,  # Same order
            trade_id=457,  # Different trade_id (partial fill)
            client_order_id="coid_test_001",
            side=OrderSide.BUY,
            fill_qty=Decimal("0.002"),
            fill_price=Decimal("67890.60"),
            commission=Decimal("0.02716"),
            commission_asset="USDT",
            realized_pnl=Decimal("0.00"),
            is_maker=True,
            position_side=PositionSide.BOTH,
        )
        assert fill1.dedupe_key != fill2.dedupe_key

    def test_different_symbol_not_duplicate(self) -> None:
        """Two fills with different symbol are not duplicates."""
        fill1 = FillEvent(
            session_id="sess_test_001",
            ts=1738180801500,
            symbol="BTCUSDT",
            order_id=123,
            trade_id=456,
            client_order_id="coid_test_001",
            side=OrderSide.BUY,
            fill_qty=Decimal("0.001"),
            fill_price=Decimal("67890.50"),
            commission=Decimal("0.01358"),
            commission_asset="USDT",
            realized_pnl=Decimal("0.00"),
            is_maker=True,
            position_side=PositionSide.BOTH,
        )
        fill2 = FillEvent(
            session_id="sess_test_001",
            ts=1738180801500,
            symbol="ETHUSDT",  # Different symbol
            order_id=123,
            trade_id=456,
            client_order_id="coid_test_002",
            side=OrderSide.BUY,
            fill_qty=Decimal("0.1"),
            fill_price=Decimal("3450.25"),
            commission=Decimal("0.0690"),
            commission_asset="USDT",
            realized_pnl=Decimal("0.00"),
            is_maker=True,
            position_side=PositionSide.BOTH,
        )
        assert fill1.dedupe_key != fill2.dedupe_key


class TestDedupeWithSet:
    """Test dedupe using Python set."""

    def test_set_based_deduplication(self) -> None:
        """Dedupe key can be used with set for deduplication."""
        fill1 = FillEvent(
            session_id="sess_test_001",
            ts=1738180801500,
            symbol="BTCUSDT",
            order_id=123,
            trade_id=456,
            client_order_id="coid_test_001",
            side=OrderSide.BUY,
            fill_qty=Decimal("0.001"),
            fill_price=Decimal("67890.50"),
            commission=Decimal("0.01358"),
            commission_asset="USDT",
            realized_pnl=Decimal("0.00"),
            is_maker=True,
            position_side=PositionSide.BOTH,
        )
        fill2 = FillEvent(
            session_id="sess_test_001",
            ts=1738180801501,
            symbol="BTCUSDT",
            order_id=123,
            trade_id=456,  # Duplicate
            client_order_id="coid_test_001",
            side=OrderSide.BUY,
            fill_qty=Decimal("0.001"),
            fill_price=Decimal("67890.50"),
            commission=Decimal("0.01358"),
            commission_asset="USDT",
            realized_pnl=Decimal("0.00"),
            is_maker=True,
            position_side=PositionSide.BOTH,
        )
        fill3 = FillEvent(
            session_id="sess_test_001",
            ts=1738180801600,
            symbol="BTCUSDT",
            order_id=123,
            trade_id=457,  # Not duplicate
            client_order_id="coid_test_001",
            side=OrderSide.BUY,
            fill_qty=Decimal("0.002"),
            fill_price=Decimal("67890.60"),
            commission=Decimal("0.02716"),
            commission_asset="USDT",
            realized_pnl=Decimal("0.00"),
            is_maker=True,
            position_side=PositionSide.BOTH,
        )

        dedupe_set: set[tuple[str, int, int]] = set()
        dedupe_set.add(fill1.dedupe_key)
        dedupe_set.add(fill2.dedupe_key)  # Should not increase size
        dedupe_set.add(fill3.dedupe_key)  # Should increase size

        assert len(dedupe_set) == 2

    def test_dict_based_deduplication(self) -> None:
        """Dedupe key can be used with dict for deduplication with tracking."""
        fills: list[FillEvent] = []
        seen: dict[tuple[str, int, int], FillEvent] = {}

        fill1 = FillEvent(
            session_id="sess_test_001",
            ts=1738180801500,
            symbol="BTCUSDT",
            order_id=123,
            trade_id=456,
            client_order_id="coid_test_001",
            side=OrderSide.BUY,
            fill_qty=Decimal("0.001"),
            fill_price=Decimal("67890.50"),
            commission=Decimal("0.01358"),
            commission_asset="USDT",
            realized_pnl=Decimal("0.00"),
            is_maker=True,
            position_side=PositionSide.BOTH,
        )
        fill2_duplicate = FillEvent(
            session_id="sess_test_001",
            ts=1738180801501,
            symbol="BTCUSDT",
            order_id=123,
            trade_id=456,  # Duplicate
            client_order_id="coid_test_001",
            side=OrderSide.BUY,
            fill_qty=Decimal("0.001"),
            fill_price=Decimal("67890.50"),
            commission=Decimal("0.01358"),
            commission_asset="USDT",
            realized_pnl=Decimal("0.00"),
            is_maker=True,
            position_side=PositionSide.BOTH,
        )

        # Process fill1
        if fill1.dedupe_key not in seen:
            seen[fill1.dedupe_key] = fill1
            fills.append(fill1)

        # Process fill2 (duplicate)
        if fill2_duplicate.dedupe_key not in seen:
            seen[fill2_duplicate.dedupe_key] = fill2_duplicate
            fills.append(fill2_duplicate)

        assert len(fills) == 1
        assert fills[0] is fill1


class TestOtherContractDedupeKeys:
    """Test dedupe keys on other contracts."""

    def test_order_intent_dedupe_key(self) -> None:
        """OrderIntent dedupe key is (session_id, client_order_id)."""
        intent = OrderIntent(
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
        )
        assert intent.dedupe_key == ("sess_test_001", "coid_test_001")

    def test_order_ack_dedupe_key(self) -> None:
        """OrderAck dedupe key is (session_id, order_id)."""
        ack = OrderAck(
            session_id="sess_test_001",
            ts=1738180800123,
            client_order_id="coid_test_001",
            order_id=1234567890,
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
        )
        assert ack.dedupe_key == ("sess_test_001", 1234567890)

    def test_position_snapshot_dedupe_key(self) -> None:
        """PositionSnapshot dedupe key is (session_id, symbol, ts)."""
        pos = PositionSnapshot(
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
            margin_type=MarginType.ISOLATED,
        )
        assert pos.dedupe_key == ("sess_test_001", "BTCUSDT", 1738180802000)

    def test_session_state_dedupe_key(self) -> None:
        """SessionState dedupe key is (session_id, ts)."""
        state = SessionState(
            session_id="sess_test_001",
            ts=1738180803000,
            state=SessionStateEnum.ACTIVE,
            symbols_active=["BTCUSDT"],
            open_orders_count=0,
            positions_count=0,
            daily_pnl=Decimal("0.00"),
            daily_trades=0,
            risk_utilization=Decimal("0.00"),
        )
        assert state.dedupe_key == ("sess_test_001", 1738180803000)

    def test_risk_breach_dedupe_key(self) -> None:
        """RiskBreachEvent dedupe key is (session_id, breach_id)."""
        breach = RiskBreachEvent(
            session_id="sess_test_001",
            ts=1738180900000,
            breach_id="breach_test_001",
            breach_type=BreachType.DAILY_LOSS_LIMIT,
            severity=BreachSeverity.CRITICAL,
            threshold=Decimal("-100.00"),
            actual=Decimal("-105.50"),
            action_taken=RiskAction.PAUSE_SESSION,
        )
        assert breach.dedupe_key == ("sess_test_001", "breach_test_001")
