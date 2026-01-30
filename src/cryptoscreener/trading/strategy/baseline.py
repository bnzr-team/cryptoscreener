"""Baseline market-making strategy.

This is the reference strategy extracted from the simulator's simple_mm_strategy.
It provides a minimal market-making approach suitable for testing and backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from cryptoscreener.trading.contracts import OrderSide
from cryptoscreener.trading.strategy.base import StrategyContext, StrategyOrder


@dataclass(frozen=True)
class BaselineStrategyConfig:
    """Configuration for BaselineStrategy."""

    spread_bps: Decimal = Decimal("10")  # Spread in basis points
    order_qty: Decimal = Decimal("0.001")  # Order quantity
    max_position: Decimal = Decimal("0.01")  # Maximum position size


class BaselineStrategy:
    """Simple market-making strategy for testing.

    Places limit orders on both sides with a configurable spread.
    Respects max position limits. Prioritizes closing positions
    to complete round trips.

    This strategy is deterministic given the same context sequence.
    """

    def __init__(self, config: BaselineStrategyConfig | None = None) -> None:
        """Initialize strategy.

        Args:
            config: Strategy configuration. Uses defaults if not provided.
        """
        self.config = config or BaselineStrategyConfig()

    def on_tick(self, ctx: StrategyContext) -> list[StrategyOrder]:
        """Process a tick and return order intents.

        Strategy logic:
        1. If we have a position, prioritize closing it (for round trips)
        2. If flat, place orders on both sides to enter

        Args:
            ctx: Read-only context with market and position state.

        Returns:
            List of StrategyOrder to submit.
        """
        orders: list[StrategyOrder] = []

        # Skip if no valid quotes
        if ctx.bid <= 0 or ctx.ask <= 0:
            return orders

        # Calculate our prices
        spread_frac = self.config.spread_bps / Decimal("10000")
        our_bid = ctx.mid * (1 - spread_frac)
        our_ask = ctx.mid * (1 + spread_frac)

        current_pos = ctx.position_qty
        order_qty = self.config.order_qty
        max_pos = self.config.max_position

        # Strategy: If we have a position, prioritize closing it
        # If flat, place orders on both sides to enter
        if current_pos > 0:
            # Long position: place SELL to close
            if ctx.pending_order_count == 0:
                close_qty = min(order_qty, current_pos)
                orders.append(
                    StrategyOrder(
                        side=OrderSide.SELL,
                        price=our_ask,
                        quantity=close_qty,
                        reason="close_long",
                    )
                )
        elif current_pos < 0:
            # Short position: place BUY to close
            if ctx.pending_order_count == 0:
                close_qty = min(order_qty, abs(current_pos))
                orders.append(
                    StrategyOrder(
                        side=OrderSide.BUY,
                        price=our_bid,
                        quantity=close_qty,
                        reason="close_short",
                    )
                )
        else:
            # Flat: place both sides to enter
            if ctx.pending_order_count < 2:
                if current_pos + order_qty <= max_pos:
                    orders.append(
                        StrategyOrder(
                            side=OrderSide.BUY,
                            price=our_bid,
                            quantity=order_qty,
                            reason="enter_long",
                        )
                    )
                if current_pos - order_qty >= -max_pos:
                    orders.append(
                        StrategyOrder(
                            side=OrderSide.SELL,
                            price=our_ask,
                            quantity=order_qty,
                            reason="enter_short",
                        )
                    )

        return orders
