"""PolicyContext - Read-only context for policy evaluation.

Provides market state, position state, and timing information needed
for policy rule evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptoscreener.trading.contracts import PositionSide
    from cryptoscreener.trading.strategy import StrategyContext


@dataclass(frozen=True)
class PolicyContext:
    """Read-only context for policy evaluation.

    All fields are immutable. Computed from StrategyContext.
    """

    # Timestamps
    ts: int
    last_book_ts: int
    last_trade_ts: int

    # Market state
    bid: Decimal
    ask: Decimal
    mid: Decimal
    spread_bps: Decimal

    # Position state
    position_qty: Decimal
    position_side: PositionSide
    entry_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal

    # Symbol
    symbol: str

    # Config limits
    max_position_qty: Decimal

    @property
    def total_pnl(self) -> Decimal:
        """Total session PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def book_age_ms(self) -> int:
        """Age of book data in milliseconds."""
        return self.ts - self.last_book_ts

    @property
    def trade_age_ms(self) -> int:
        """Age of trade data in milliseconds."""
        return self.ts - self.last_trade_ts

    @property
    def inventory_ratio(self) -> Decimal:
        """Position as fraction of max allowed (0.0 to 1.0+)."""
        if self.max_position_qty == 0:
            return Decimal("0")
        return abs(self.position_qty) / self.max_position_qty

    @classmethod
    def from_strategy_context(cls, ctx: StrategyContext) -> PolicyContext:
        """Create PolicyContext from StrategyContext.

        Args:
            ctx: Strategy context with market and position state.

        Returns:
            Immutable policy context.
        """
        return cls(
            ts=ctx.ts,
            last_book_ts=ctx.last_book_ts,
            last_trade_ts=ctx.last_trade_ts,
            bid=ctx.bid,
            ask=ctx.ask,
            mid=ctx.mid,
            spread_bps=ctx.spread_bps,
            position_qty=ctx.position_qty,
            position_side=ctx.position_side,
            entry_price=ctx.entry_price,
            unrealized_pnl=ctx.unrealized_pnl,
            realized_pnl=ctx.realized_pnl,
            symbol=ctx.symbol,
            max_position_qty=ctx.max_position_qty,
        )
