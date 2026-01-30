"""Trading v2 data contracts.

This module provides Pydantic models for all v2 trading contracts
as specified in docs/trading/03_CONTRACTS.md.

All contracts follow these invariants:
- schema_version and session_id are required on every model
- Money/price/qty fields use Decimal (not float)
- Strict enums for status/side/etc.
- No extra fields allowed (extra='forbid')
"""

from cryptoscreener.trading.contracts.fill_event import FillEvent
from cryptoscreener.trading.contracts.order_ack import OrderAck
from cryptoscreener.trading.contracts.order_intent import OrderIntent
from cryptoscreener.trading.contracts.position_snapshot import PositionSnapshot
from cryptoscreener.trading.contracts.risk_breach_event import RiskBreachEvent
from cryptoscreener.trading.contracts.session_state import SessionState
from cryptoscreener.trading.contracts.strategy_decision import (
    StrategyDecision,
    StrategyDecisionOrder,
)
from cryptoscreener.trading.contracts.types import (
    BreachSeverity,
    BreachType,
    MarginType,
    OrderPriority,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    RiskAction,
    SessionStateEnum,
    TimeInForce,
)

__all__ = [
    "BreachSeverity",
    "BreachType",
    "FillEvent",
    "MarginType",
    "OrderAck",
    "OrderIntent",
    "OrderPriority",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PositionSide",
    "PositionSnapshot",
    "RiskAction",
    "RiskBreachEvent",
    "SessionState",
    "SessionStateEnum",
    "StrategyDecision",
    "StrategyDecisionOrder",
    "TimeInForce",
]
