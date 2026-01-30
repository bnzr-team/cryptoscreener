"""Trading strategy plugin interface.

This module provides the strategy abstraction layer for DEC-042:
- Strategy Protocol/ABC: on_tick(ctx) -> list[StrategyOrder]
- StrategyContext: read-only view of market and position state
- StrategyDecision: journaled output for replay determinism
"""

from cryptoscreener.trading.strategy.base import Strategy, StrategyContext, StrategyOrder
from cryptoscreener.trading.strategy.baseline import BaselineStrategy

__all__ = [
    "BaselineStrategy",
    "Strategy",
    "StrategyContext",
    "StrategyOrder",
]
