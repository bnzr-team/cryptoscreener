"""PolicyEngineStrategy - Strategy wrapper with policy rule enforcement.

Wraps BaselineStrategy and applies policy rules to filter/modify orders.
Implements the DEC-044 pattern: Policy evaluates, then modifies base strategy output.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from cryptoscreener.trading.contracts import OrderSide
from cryptoscreener.trading.policy import (
    PolicyContext,
    PolicyEngine,
    PolicyParams,
)
from cryptoscreener.trading.strategy.base import StrategyOrder
from cryptoscreener.trading.strategy.baseline import BaselineStrategy

if TYPE_CHECKING:
    from cryptoscreener.trading.policy import PolicyOutput
    from cryptoscreener.trading.policy.inputs import PolicyInputsProvider
    from cryptoscreener.trading.strategy.base import StrategyContext
    from cryptoscreener.trading.strategy.baseline import BaselineStrategyConfig


class PolicyEngineStrategy:
    """Strategy that applies policy rules to BaselineStrategy output.

    Flow:
    1. Get PolicyInputs from provider
    2. Evaluate PolicyEngine to get PolicyOutput
    3. Handle FORCE_CLOSE / kill scenarios
    4. Get orders from BaselineStrategy
    5. Apply policy filters (SUPPRESS_ENTRY, SUPPRESS_ALL)
    6. Apply parameter overrides (MODIFY_PARAMS for spread)
    """

    def __init__(
        self,
        inputs_provider: PolicyInputsProvider,
        *,
        base_config: BaselineStrategyConfig | None = None,
        policy_params: PolicyParams | None = None,
    ) -> None:
        """Initialize strategy with policy engine and inputs provider.

        Args:
            inputs_provider: Provider for ML inputs.
            base_config: Configuration for base strategy.
            policy_params: Policy parameters.
        """
        self._base = BaselineStrategy(base_config)
        self._engine = PolicyEngine(policy_params)
        self._provider = inputs_provider
        self._params = policy_params or PolicyParams()
        self._killed = False

    @property
    def is_killed(self) -> bool:
        """Check if session has been killed by policy."""
        return self._killed

    def on_tick(self, ctx: StrategyContext) -> list[StrategyOrder]:
        """Process tick with policy enforcement.

        Args:
            ctx: Strategy context with market and position state.

        Returns:
            List of orders (may be empty, filtered, or modified by policy).
        """
        # If already killed, emit no orders
        if self._killed:
            return []

        # Get policy inputs
        inputs = self._provider.get_inputs(ctx)

        # Build policy context
        policy_ctx = PolicyContext.from_strategy_context(ctx)

        # Evaluate policy
        policy_output = self._engine.evaluate(policy_ctx, inputs)

        # Handle kill
        if policy_output.kill:
            self._killed = True
            return self._build_close_order(ctx, policy_output)

        # Handle force close (without kill)
        if policy_output.force_close:
            return self._build_close_order(ctx, policy_output)

        # Handle suppress all
        if policy_output.should_suppress_all:
            return []

        # Get base strategy orders
        base_orders = self._base.on_tick(ctx)

        # Apply policy modifications
        return self._apply_policy(ctx, base_orders, policy_output)

    def _build_close_order(
        self,
        ctx: StrategyContext,
        policy_output: PolicyOutput,
    ) -> list[StrategyOrder]:
        """Build aggressive close order for force close / kill.

        Args:
            ctx: Strategy context.
            policy_output: Policy output with reason codes.

        Returns:
            List with single close order, or empty if flat.
        """
        if ctx.position_qty == 0:
            return []

        # Determine close side and price
        if ctx.position_qty > 0:
            # Long position: sell at bid (aggressive)
            side = OrderSide.SELL
            price = ctx.bid
        else:
            # Short position: buy at ask (aggressive)
            side = OrderSide.BUY
            price = ctx.ask

        # Use first reason code or default
        reason = policy_output.reason_codes[0] if policy_output.reason_codes else "force_close"

        return [
            StrategyOrder(
                side=side,
                price=price,
                quantity=abs(ctx.position_qty),
                reason=reason,
            )
        ]

    def _apply_policy(
        self,
        ctx: StrategyContext,
        orders: list[StrategyOrder],
        policy_output: PolicyOutput,
    ) -> list[StrategyOrder]:
        """Apply policy filters and modifications to orders.

        Args:
            ctx: Strategy context.
            orders: Orders from base strategy.
            policy_output: Policy output with patterns and overrides.

        Returns:
            Modified/filtered orders.
        """
        result: list[StrategyOrder] = []

        for order in orders:
            # Check if this is an entry order (increases position)
            is_entry = self._is_entry_order(ctx, order)

            # SUPPRESS_ENTRY: filter entry orders, allow exit orders
            if policy_output.should_suppress_entry and is_entry:
                continue

            # MODIFY_PARAMS: adjust spread on order price
            if policy_output.has_spread_override:
                order = self._apply_spread_override(ctx, order, policy_output)

            result.append(order)

        return result

    def _is_entry_order(self, ctx: StrategyContext, order: StrategyOrder) -> bool:
        """Check if order would increase position size.

        Args:
            ctx: Strategy context with current position.
            order: Order to check.

        Returns:
            True if order increases |position|, False if reduces.
        """
        current_pos = ctx.position_qty

        if order.side == OrderSide.BUY:
            # Buy increases position if flat or long
            return current_pos >= 0
        else:
            # Sell increases position if flat or short
            return current_pos <= 0

    def _apply_spread_override(
        self,
        ctx: StrategyContext,
        order: StrategyOrder,
        policy_output: PolicyOutput,
    ) -> StrategyOrder:
        """Adjust order price based on spread multiplier.

        Args:
            ctx: Strategy context with mid price.
            order: Original order.
            policy_output: Policy output with spread multiplier.

        Returns:
            Order with adjusted price.
        """
        spread_mult = policy_output.spread_mult
        base_spread_frac = self._base.config.spread_bps / Decimal("10000")
        new_spread_frac = base_spread_frac * spread_mult

        if order.side == OrderSide.BUY:
            # Widen bid down
            new_price = ctx.mid * (1 - new_spread_frac)
        else:
            # Widen ask up
            new_price = ctx.mid * (1 + new_spread_frac)

        # Determine reason suffix based on policy
        reason_suffix = ""
        if "toxic_widen" in policy_output.reason_codes:
            reason_suffix = "_toxic"
        elif "vol_spread_adjust" in policy_output.reason_codes:
            reason_suffix = "_vol"

        return StrategyOrder(
            side=order.side,
            price=new_price,
            quantity=order.quantity,
            reason=order.reason + reason_suffix if reason_suffix else order.reason,
        )
