"""FillEvent contract.

Notification of a trade fill (partial or complete).
Producer: UserDataStream WS (ORDER_TRADE_UPDATE)
Consumer: PositionTracker, Journal, PnL Calculator

CRITICAL: Dedupe key is (symbol, order_id, trade_id).
"""

from __future__ import annotations

from decimal import Decimal  # noqa: TC003 - used at runtime in validators
from typing import Annotated

from pydantic import Field, field_validator

from cryptoscreener.trading.contracts.base import (
    TradingContractBase,
    parse_decimal,
)
from cryptoscreener.trading.contracts.types import (  # noqa: TC001 - used at runtime
    OrderSide,
    PositionSide,
)


class FillEvent(TradingContractBase):
    """Notification of a trade fill.

    CRITICAL: Dedupe on (symbol, order_id, trade_id).
    """

    ts: int = Field(description="Fill timestamp (ms)")
    symbol: str = Field(description="Trading pair")
    order_id: int = Field(description="Exchange order ID")
    trade_id: int = Field(description="Exchange trade ID (unique per fill)")
    client_order_id: str = Field(description="Client order ID")
    side: OrderSide = Field(description="BUY or SELL")
    fill_qty: Annotated[Decimal, Field(description="Filled quantity this event")] = Field()
    fill_price: Annotated[Decimal, Field(description="Fill price")] = Field()
    commission: Annotated[Decimal, Field(description="Commission amount")] = Field()
    commission_asset: str = Field(description="Commission currency (e.g., USDT)")
    realized_pnl: Annotated[Decimal, Field(description="Realized PnL for this fill")] = Field()
    is_maker: bool = Field(description="True if maker fill")
    position_side: PositionSide = Field(description="BOTH, LONG, or SHORT")

    @field_validator(
        "fill_qty",
        "fill_price",
        "commission",
        "realized_pnl",
        mode="before",
    )
    @classmethod
    def parse_decimal_fields(cls, v: object) -> Decimal:
        """Parse Decimal fields."""
        return parse_decimal(v)

    @property
    def dedupe_key(self) -> tuple[str, int, int]:
        """Dedupe key: (symbol, order_id, trade_id).

        CRITICAL: Use this key to suppress duplicate fill events.
        """
        return (self.symbol, self.order_id, self.trade_id)

    @property
    def notional(self) -> Decimal:
        """Calculate fill notional value."""
        return self.fill_qty * self.fill_price
