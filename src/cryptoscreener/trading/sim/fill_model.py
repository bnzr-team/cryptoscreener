"""Fill models for the simulator.

Phase 1: CROSS model only - optimistic fills at limit price when price crosses.
Phase 2 (future): QUEUE_LITE with queue position modeling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal  # noqa: TC003 - used at runtime in dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptoscreener.trading.contracts import OrderSide


@dataclass(frozen=True)
class FillResult:
    """Result of a fill check.

    Attributes:
        filled: Whether the order was filled.
        fill_price: Price at which order was filled (if filled).
        fill_qty: Quantity filled (if filled).
    """

    filled: bool
    fill_price: Decimal | None = None
    fill_qty: Decimal | None = None


@dataclass(frozen=True)
class OrderState:
    """State of a pending order.

    Attributes:
        side: BUY or SELL.
        price: Limit price.
        quantity: Order quantity.
        placed_ts: Timestamp when order was placed.
    """

    side: OrderSide
    price: Decimal
    quantity: Decimal
    placed_ts: int


@dataclass(frozen=True)
class MarketTick:
    """Market data tick for fill evaluation.

    Attributes:
        ts: Timestamp of the tick.
        trade_price: Last trade price (if trade event).
        bid_price: Best bid price (if book event).
        ask_price: Best ask price (if book event).
    """

    ts: int
    trade_price: Decimal | None = None
    bid_price: Decimal | None = None
    ask_price: Decimal | None = None


class BaseFillModel(ABC):
    """Abstract base class for fill models."""

    @abstractmethod
    def check_fill(self, order: OrderState, tick: MarketTick) -> FillResult:
        """Check if an order should be filled given market data.

        Args:
            order: The pending order to check.
            tick: Current market data tick.

        Returns:
            FillResult indicating if and how the order was filled.
        """
        ...


class CrossFillModel(BaseFillModel):
    """Phase 1 CROSS fill model.

    Simple fill logic:
    - BID (buy) order filled if any trade price <= bid_price
    - ASK (sell) order filled if any trade price >= ask_price
    - Fill occurs at our limit price (optimistic Phase 1)

    Does NOT model:
    - Queue position
    - Partial fills
    - Latency jitter
    - Adverse selection
    """

    def check_fill(self, order: OrderState, tick: MarketTick) -> FillResult:
        """Check if order crosses with market.

        Args:
            order: The pending order.
            tick: Market data tick (trade or book update).

        Returns:
            FillResult - filled at limit price if crossed, unfilled otherwise.
        """
        from cryptoscreener.trading.contracts import OrderSide

        # Only process if we have trade price
        if tick.trade_price is None:
            return FillResult(filled=False)

        trade_price = tick.trade_price

        # BUY order: filled if trade price <= our bid price
        # (someone sold at or below our price)
        if order.side == OrderSide.BUY and trade_price <= order.price:
            return FillResult(
                filled=True,
                fill_price=order.price,  # Optimistic: fill at our limit
                fill_qty=order.quantity,
            )

        # SELL order: filled if trade price >= our ask price
        # (someone bought at or above our price)
        if order.side == OrderSide.SELL and trade_price >= order.price:
            return FillResult(
                filled=True,
                fill_price=order.price,  # Optimistic: fill at our limit
                fill_qty=order.quantity,
            )

        return FillResult(filled=False)
