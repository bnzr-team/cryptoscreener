"""Roundtrip serialization tests for trading v2 contracts.

Tests verify:
1. Object -> JSON -> Object equality (Decimal-safe)
2. Decimal precision preserved through roundtrip
3. All contracts can be loaded from fixtures
4. Determinism (two serializations produce identical output)
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

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

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "trading_contracts"


class TestOrderIntentRoundtrip:
    """Roundtrip tests for OrderIntent."""

    def test_roundtrip_json(self) -> None:
        """Object -> JSON -> Object equality."""
        original = OrderIntent(
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
        json_str = original.model_dump_json()
        restored = OrderIntent.model_validate_json(json_str)
        assert original == restored

    def test_decimal_precision(self) -> None:
        """Decimal fields preserve precision through roundtrip."""
        original = OrderIntent(
            session_id="sess_test_001",
            ts=1738180800000,
            client_order_id="coid_test_002",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.123456789"),
            price=Decimal("67890.123456789"),
            time_in_force=TimeInForce.GTX,
            reduce_only=False,
            priority=OrderPriority.NORMAL,
        )
        json_str = original.model_dump_json()
        restored = OrderIntent.model_validate_json(json_str)
        assert restored.quantity == Decimal("0.123456789")
        assert restored.price == Decimal("67890.123456789")

    def test_load_from_fixture(self) -> None:
        """Load from fixture file."""
        fixture_path = FIXTURES_DIR / "order_intent_valid.json"
        with open(fixture_path) as f:
            data = json.load(f)
        obj = OrderIntent.model_validate(data)
        assert obj.symbol == "BTCUSDT"
        assert obj.side == OrderSide.BUY

    def test_determinism(self) -> None:
        """Two serializations produce identical output."""
        obj = OrderIntent(
            session_id="sess_test_001",
            ts=1738180800000,
            client_order_id="coid_test_003",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("67890.50"),
            time_in_force=TimeInForce.GTX,
            reduce_only=False,
            priority=OrderPriority.NORMAL,
        )
        json1 = obj.model_dump_json()
        json2 = obj.model_dump_json()
        assert json1 == json2
        assert (
            hashlib.sha256(json1.encode()).hexdigest()
            == hashlib.sha256(json2.encode()).hexdigest()
        )


class TestOrderAckRoundtrip:
    """Roundtrip tests for OrderAck."""

    def test_roundtrip_json(self) -> None:
        """Object -> JSON -> Object equality."""
        original = OrderAck(
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
        json_str = original.model_dump_json()
        restored = OrderAck.model_validate_json(json_str)
        assert original == restored

    def test_load_from_fixture_valid(self) -> None:
        """Load valid order ack from fixture."""
        fixture_path = FIXTURES_DIR / "order_ack_valid.json"
        with open(fixture_path) as f:
            data = json.load(f)
        obj = OrderAck.model_validate(data)
        assert obj.status == OrderStatus.NEW
        assert obj.error_code is None

    def test_load_from_fixture_rejected(self) -> None:
        """Load rejected order ack from fixture."""
        fixture_path = FIXTURES_DIR / "order_ack_rejected.json"
        with open(fixture_path) as f:
            data = json.load(f)
        obj = OrderAck.model_validate(data)
        assert obj.status == OrderStatus.REJECTED
        assert obj.error_code == -2022
        assert obj.is_rejected


class TestFillEventRoundtrip:
    """Roundtrip tests for FillEvent."""

    def test_roundtrip_json(self) -> None:
        """Object -> JSON -> Object equality."""
        original = FillEvent(
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
            position_side=PositionSide.BOTH,
        )
        json_str = original.model_dump_json()
        restored = FillEvent.model_validate_json(json_str)
        assert original == restored

    def test_load_from_fixture_maker(self) -> None:
        """Load maker fill from fixture."""
        fixture_path = FIXTURES_DIR / "fill_event_maker.json"
        with open(fixture_path) as f:
            data = json.load(f)
        obj = FillEvent.model_validate(data)
        assert obj.is_maker is True
        assert obj.symbol == "BTCUSDT"

    def test_load_from_fixture_taker(self) -> None:
        """Load taker fill from fixture."""
        fixture_path = FIXTURES_DIR / "fill_event_taker.json"
        with open(fixture_path) as f:
            data = json.load(f)
        obj = FillEvent.model_validate(data)
        assert obj.is_maker is False
        assert obj.symbol == "ETHUSDT"

    def test_notional_property(self) -> None:
        """Test notional calculation."""
        fill = FillEvent(
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
            position_side=PositionSide.BOTH,
        )
        assert fill.notional == Decimal("67.89050")


class TestPositionSnapshotRoundtrip:
    """Roundtrip tests for PositionSnapshot."""

    def test_roundtrip_json(self) -> None:
        """Object -> JSON -> Object equality."""
        original = PositionSnapshot(
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
        json_str = original.model_dump_json()
        restored = PositionSnapshot.model_validate_json(json_str)
        assert original == restored

    def test_load_from_fixture_long(self) -> None:
        """Load long position from fixture."""
        fixture_path = FIXTURES_DIR / "position_snapshot_long.json"
        with open(fixture_path) as f:
            data = json.load(f)
        obj = PositionSnapshot.model_validate(data)
        assert obj.side == PositionSide.LONG
        assert obj.leverage == 5

    def test_load_from_fixture_flat(self) -> None:
        """Load flat position from fixture."""
        fixture_path = FIXTURES_DIR / "position_snapshot_flat.json"
        with open(fixture_path) as f:
            data = json.load(f)
        obj = PositionSnapshot.model_validate(data)
        assert obj.side == PositionSide.FLAT
        assert obj.is_flat


class TestSessionStateRoundtrip:
    """Roundtrip tests for SessionState."""

    def test_roundtrip_json(self) -> None:
        """Object -> JSON -> Object equality."""
        original = SessionState(
            session_id="sess_test_001",
            ts=1738180803000,
            state=SessionStateEnum.ACTIVE,
            prev_state=SessionStateEnum.READY,
            reason="startup_complete",
            symbols_active=["BTCUSDT", "ETHUSDT"],
            open_orders_count=6,
            positions_count=0,
            daily_pnl=Decimal("0.00"),
            daily_trades=0,
            risk_utilization=Decimal("0.15"),
        )
        json_str = original.model_dump_json()
        restored = SessionState.model_validate_json(json_str)
        assert original == restored

    def test_load_from_fixture_active(self) -> None:
        """Load active session from fixture."""
        fixture_path = FIXTURES_DIR / "session_state_active.json"
        with open(fixture_path) as f:
            data = json.load(f)
        obj = SessionState.model_validate(data)
        assert obj.state == SessionStateEnum.ACTIVE
        assert obj.is_active

    def test_load_from_fixture_killed(self) -> None:
        """Load killed session from fixture."""
        fixture_path = FIXTURES_DIR / "session_state_killed.json"
        with open(fixture_path) as f:
            data = json.load(f)
        obj = SessionState.model_validate(data)
        assert obj.state == SessionStateEnum.KILLED
        assert obj.is_stopped


class TestRiskBreachEventRoundtrip:
    """Roundtrip tests for RiskBreachEvent."""

    def test_roundtrip_json(self) -> None:
        """Object -> JSON -> Object equality."""
        original = RiskBreachEvent(
            session_id="sess_test_001",
            ts=1738180900000,
            breach_id="breach_test_001",
            breach_type=BreachType.DAILY_LOSS_LIMIT,
            severity=BreachSeverity.CRITICAL,
            threshold=Decimal("-100.00"),
            actual=Decimal("-105.50"),
            action_taken=RiskAction.PAUSE_SESSION,
            details="Daily loss limit exceeded.",
        )
        json_str = original.model_dump_json()
        restored = RiskBreachEvent.model_validate_json(json_str)
        assert original == restored

    def test_load_from_fixture_warning(self) -> None:
        """Load warning breach from fixture."""
        fixture_path = FIXTURES_DIR / "risk_breach_warning.json"
        with open(fixture_path) as f:
            data = json.load(f)
        obj = RiskBreachEvent.model_validate(data)
        assert obj.severity == BreachSeverity.WARNING
        assert not obj.is_fatal

    def test_load_from_fixture_fatal(self) -> None:
        """Load fatal breach from fixture."""
        fixture_path = FIXTURES_DIR / "risk_breach_fatal.json"
        with open(fixture_path) as f:
            data = json.load(f)
        obj = RiskBreachEvent.model_validate(data)
        assert obj.severity == BreachSeverity.FATAL
        assert obj.is_fatal
        assert obj.requires_action


class TestFixtureIntegrity:
    """Test fixture file integrity using checksums."""

    def test_all_fixtures_valid(self) -> None:
        """All fixtures can be loaded without validation errors."""
        manifest_path = FIXTURES_DIR / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        contract_map: dict[str, Any] = {
            "OrderIntent": OrderIntent,
            "OrderAck": OrderAck,
            "FillEvent": FillEvent,
            "PositionSnapshot": PositionSnapshot,
            "SessionState": SessionState,
            "RiskBreachEvent": RiskBreachEvent,
        }

        for filename, info in manifest["fixtures"].items():
            fixture_path = FIXTURES_DIR / filename
            with open(fixture_path) as f:
                data = json.load(f)

            contract_cls = contract_map[info["contract"]]
            obj = contract_cls.model_validate(data)
            assert obj is not None

    def test_checksums_match(self) -> None:
        """Fixture checksums match manifest."""
        manifest_path = FIXTURES_DIR / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        for filename, info in manifest["fixtures"].items():
            fixture_path = FIXTURES_DIR / filename
            with open(fixture_path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            assert actual_hash == info["sha256"], f"Checksum mismatch for {filename}"
