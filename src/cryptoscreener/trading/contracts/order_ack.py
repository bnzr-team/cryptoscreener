"""OrderAck contract.

Acknowledgement from exchange after order submission.
Producer: OrderGovernor (from Binance REST response)
Consumer: OrderManager, Journal, Reconciler

Design Decision: This is the CANONICAL contract using Decimal types.
Raw exchange responses should be converted to this format.
"""

from __future__ import annotations

from decimal import Decimal  # noqa: TC003 - used at runtime in validators
from typing import Annotated

from pydantic import Field, field_validator

from cryptoscreener.trading.contracts.base import (
    TradingContractBase,
    parse_decimal,
)
from cryptoscreener.trading.contracts.types import (
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


class OrderAck(TradingContractBase):
    """Acknowledgement from exchange after order submission.

    All monetary values use Decimal (canonical, not raw float from exchange).
    """

    ts: int = Field(description="Ack receipt timestamp (ms)")
    client_order_id: str = Field(description="From OrderIntent")
    order_id: int = Field(description="Exchange-assigned order ID")
    symbol: str = Field(description="Trading pair")
    side: OrderSide = Field(description="BUY or SELL")
    order_type: OrderType = Field(description="Order type")
    status: OrderStatus = Field(description="Order status from exchange")
    quantity: Annotated[Decimal, Field(description="Original quantity")] = Field()
    price: Annotated[Decimal | None, Field(description="Limit price")] = None
    executed_qty: Annotated[Decimal, Field(description="Filled quantity so far")] = Field()
    avg_price: Annotated[Decimal, Field(description="Average fill price")] = Field()
    time_in_force: TimeInForce = Field(description="TIF from request")
    reduce_only: bool = Field(description="From request")
    error_code: int | None = Field(
        default=None,
        description="Binance error code if rejected",
    )
    error_msg: str | None = Field(
        default=None,
        description="Binance error message if rejected",
    )

    @field_validator("quantity", "executed_qty", "avg_price", mode="before")
    @classmethod
    def parse_decimal_fields(cls, v: object) -> Decimal:
        """Parse Decimal fields."""
        return parse_decimal(v)

    @field_validator("price", mode="before")
    @classmethod
    def parse_price(cls, v: object) -> Decimal | None:
        """Parse price to Decimal if provided."""
        if v is None:
            return None
        return parse_decimal(v)

    @property
    def is_rejected(self) -> bool:
        """Check if order was rejected."""
        return self.status == OrderStatus.REJECTED

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def dedupe_key(self) -> tuple[str, int]:
        """Dedupe key for OrderAck: (session_id, order_id)."""
        return (self.session_id, self.order_id)
