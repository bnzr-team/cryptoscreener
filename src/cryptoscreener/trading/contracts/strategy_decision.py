"""StrategyDecision contract.

Journaled output from strategy on_tick() for replay and audit.
Producer: Strategy.on_tick() via ScenarioRunner
Consumer: Journal, Replay verifier, Audit log
"""

from __future__ import annotations

from decimal import Decimal  # noqa: TC003 - used at runtime in validators
from typing import Annotated

from pydantic import Field, field_validator

from cryptoscreener.trading.contracts.base import (
    TradingContractBase,
    parse_decimal,
)
from cryptoscreener.trading.contracts.types import (  # noqa: TC001 - used at runtime in Pydantic
    OrderSide,
    PositionSide,
)


class StrategyDecisionOrder(TradingContractBase):
    """An order intent within a StrategyDecision.

    Represents a single order that the strategy wants to place.
    """

    side: OrderSide = Field(description="BUY or SELL")
    price: Annotated[Decimal, Field(description="Limit price")] = Field()
    quantity: Annotated[Decimal, Field(description="Order quantity")] = Field()
    reason: str = Field(default="", description="Human-readable reason for order")

    @field_validator("price", "quantity", mode="before")
    @classmethod
    def parse_decimals(cls, v: object) -> Decimal:
        """Parse decimal fields."""
        return parse_decimal(v)


class StrategyDecision(TradingContractBase):
    """Journaled output from strategy on_tick().

    Records the strategy's decision for a given tick, including:
    - The context snapshot at decision time
    - List of orders the strategy wants to place
    - Timing information

    This contract enables:
    - Deterministic replay verification
    - Audit trail of strategy decisions
    - Debugging strategy behavior
    """

    # Tick identification
    ts: int = Field(description="Tick timestamp (ms)")
    tick_seq: int = Field(ge=0, description="Tick sequence number within session")

    # Context snapshot at decision time
    bid: Annotated[Decimal, Field(description="Best bid at decision time")] = Field()
    ask: Annotated[Decimal, Field(description="Best ask at decision time")] = Field()
    mid: Annotated[Decimal, Field(description="Mid price at decision time")] = Field()
    last_trade_price: Annotated[Decimal, Field(description="Last trade price")] = Field()

    # Position state
    position_qty: Annotated[Decimal, Field(description="Signed position qty")] = Field()
    position_side: PositionSide = Field(description="LONG, SHORT, or FLAT")
    unrealized_pnl: Annotated[Decimal, Field(description="Unrealized PnL")] = Field()
    realized_pnl: Annotated[Decimal, Field(description="Realized PnL")] = Field()

    # Order state
    pending_order_count: int = Field(ge=0, description="Pending orders before decision")

    # Strategy output
    orders: list[StrategyDecisionOrder] = Field(
        default_factory=list,
        description="Orders the strategy wants to place",
    )

    # Symbol
    symbol: str = Field(description="Trading symbol")

    @field_validator(
        "bid",
        "ask",
        "mid",
        "last_trade_price",
        "position_qty",
        "unrealized_pnl",
        "realized_pnl",
        mode="before",
    )
    @classmethod
    def parse_decimals(cls, v: object) -> Decimal:
        """Parse decimal fields."""
        return parse_decimal(v)

    @property
    def dedupe_key(self) -> tuple[str, int]:
        """Dedupe key: (session_id, tick_seq)."""
        return (self.session_id, self.tick_seq)

    @property
    def order_count(self) -> int:
        """Number of orders in this decision."""
        return len(self.orders)

    @property
    def has_orders(self) -> bool:
        """Whether this decision has any orders."""
        return len(self.orders) > 0
