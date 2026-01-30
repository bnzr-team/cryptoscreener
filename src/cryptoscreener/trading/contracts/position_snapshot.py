"""PositionSnapshot contract.

Point-in-time snapshot of position state per symbol.
Producer: PositionTracker
Consumer: RiskManager, Dashboard, Journal
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import Field, field_validator

from cryptoscreener.trading.contracts.base import (
    TradingContractBase,
    parse_decimal,
)
from cryptoscreener.trading.contracts.types import (
    MarginType,
    PositionSide,
)


class PositionSnapshot(TradingContractBase):
    """Point-in-time snapshot of position state."""

    ts: int = Field(description="Snapshot timestamp (ms)")
    symbol: str = Field(description="Trading pair")
    side: PositionSide = Field(description="LONG, SHORT, or FLAT")
    quantity: Annotated[Decimal, Field(description="Position size (absolute)")] = Field()
    entry_price: Annotated[Decimal, Field(description="Average entry price")] = Field()
    mark_price: Annotated[Decimal, Field(description="Current mark price")] = Field()
    unrealized_pnl: Annotated[Decimal, Field(description="Unrealized PnL (USD)")] = Field()
    realized_pnl_session: Annotated[Decimal, Field(description="Realized PnL this session")] = (
        Field()
    )
    leverage: int = Field(description="Position leverage")
    liquidation_price: Annotated[Decimal, Field(description="Estimated liquidation price")] = (
        Field()
    )
    margin_type: MarginType = Field(description="ISOLATED or CROSS")

    @field_validator(
        "quantity",
        "entry_price",
        "mark_price",
        "unrealized_pnl",
        "realized_pnl_session",
        "liquidation_price",
        mode="before",
    )
    @classmethod
    def parse_decimal_fields(cls, v: object) -> Decimal:
        """Parse Decimal fields."""
        return parse_decimal(v)

    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.side == PositionSide.FLAT or self.quantity == Decimal("0")

    @property
    def notional(self) -> Decimal:
        """Calculate position notional value."""
        return self.quantity * self.mark_price

    @property
    def dedupe_key(self) -> tuple[str, str, int]:
        """Dedupe key: (session_id, symbol, ts)."""
        return (self.session_id, self.symbol, self.ts)
