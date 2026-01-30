"""ConstantPolicyInputsProvider - Returns constant inputs for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptoscreener.trading.policy.inputs import PolicyInputs
    from cryptoscreener.trading.strategy import StrategyContext


class ConstantPolicyInputsProvider:
    """Provides constant PolicyInputs regardless of context.

    Useful for testing with known, deterministic inputs.
    """

    def __init__(self, inputs: PolicyInputs) -> None:
        """Initialize with constant inputs.

        Args:
            inputs: The constant inputs to return for all calls.
        """
        self._inputs = inputs

    def get_inputs(self, ctx: StrategyContext) -> PolicyInputs:
        """Get constant policy inputs.

        Args:
            ctx: Strategy context (ignored).

        Returns:
            The constant inputs provided at construction.
        """
        return self._inputs
