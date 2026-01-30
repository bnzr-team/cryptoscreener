"""Strategy plugin interface base classes.

Defines the Strategy Protocol and StrategyContext for trading strategies.
See DEC-042 for design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from cryptoscreener.trading.contracts import (  # noqa: TC001 - used at runtime in dataclass fields
    OrderSide,
    PositionSide,
)


@dataclass(frozen=True)
class StrategyOrder:
    """Lightweight order request from strategy.

    This is a minimal struct for strategy outputs before
    they are converted to full OrderIntent contracts by the simulator.
    Intentionally separate from contracts.OrderIntent to avoid SSOT confusion.
    """

    side: OrderSide
    price: Decimal
    quantity: Decimal
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate order request."""
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")


@dataclass(frozen=True)
class StrategyContext:
    """Read-only context passed to strategy on each tick.

    All fields are immutable (frozen dataclass) to prevent
    strategy from accidentally modifying simulator state.
    """

    # Timestamps
    ts: int  # Current event timestamp (ms)

    # Market state
    bid: Decimal  # Best bid price
    ask: Decimal  # Best ask price
    last_trade_price: Decimal  # Last trade price
    last_book_ts: int  # Last book update timestamp (ms)
    last_trade_ts: int  # Last trade timestamp (ms)

    # Position state
    position_qty: Decimal  # Signed position quantity (+ long, - short)
    position_side: PositionSide  # LONG, SHORT, or FLAT
    entry_price: Decimal  # Average entry price
    unrealized_pnl: Decimal  # Current unrealized PnL
    realized_pnl: Decimal  # Session realized PnL

    # Order state
    pending_order_count: int  # Number of pending orders

    # Config
    symbol: str  # Trading symbol
    max_position_qty: Decimal  # Maximum allowed position

    @property
    def mid(self) -> Decimal:
        """Calculate mid price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return Decimal("0")

    @property
    def spread_bps(self) -> Decimal:
        """Calculate spread in basis points."""
        if self.mid > 0:
            spread = self.ask - self.bid
            return (spread / self.mid) * Decimal("10000")
        return Decimal("0")

    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.position_qty == 0


class Strategy(Protocol):
    """Protocol defining the strategy interface.

    Strategies must implement on_tick() which receives a read-only
    context and returns a list of order intents.

    Example:
        class MyStrategy:
            def on_tick(self, ctx: StrategyContext) -> list[StrategyOrder]:
                if ctx.is_flat:
                    return [StrategyOrder(OrderSide.BUY, ctx.bid, Decimal("0.001"))]
                return []
    """

    def on_tick(self, ctx: StrategyContext) -> list[StrategyOrder]:
        """Process a tick and return order intents.

        Args:
            ctx: Read-only context with market and position state.

        Returns:
            List of OrderIntent to submit (may be empty).
        """
        ...
