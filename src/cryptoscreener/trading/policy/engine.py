"""PolicyEngine - Evaluates policy rules and produces PolicyOutput.

Implements the policy rules from DEC-043 (05_ML_POLICY_LIBRARY.md).
Rules are evaluated in priority order with higher priority rules overriding lower.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptoscreener.trading.policy.output import PolicyOutput, PolicyPattern
from cryptoscreener.trading.policy.params import PolicyParams

if TYPE_CHECKING:
    from decimal import Decimal

    from cryptoscreener.trading.policy.context import PolicyContext
    from cryptoscreener.trading.policy.inputs import PolicyInputs


class PolicyEngine:
    """Evaluates policy rules to produce PolicyOutput.

    Implements MVP rules from DEC-043:
    - POL-002: SUPPRESS_ENTRY on low in-play
    - POL-004: MODIFY_PARAMS (spread widen) on toxicity
    - POL-005: SUPPRESS_ALL on high toxicity
    - POL-012: SUPPRESS_ALL on hard inventory limit
    - POL-013: SUPPRESS_ALL on stale book data
    - POL-019: FORCE_CLOSE + kill on max session loss

    Rules are evaluated in priority order (DEC-043 Rule Priority Matrix).
    """

    def __init__(self, params: PolicyParams | None = None) -> None:
        """Initialize engine with parameters.

        Args:
            params: Policy parameters. Uses defaults if not provided.
        """
        self.params = params or PolicyParams()

    def evaluate(
        self,
        ctx: PolicyContext,
        inputs: PolicyInputs,
    ) -> PolicyOutput:
        """Evaluate all policy rules and return combined output.

        Rules are evaluated in priority order (highest first).
        Higher priority rules can override lower priority actions.

        Args:
            ctx: Policy context with market and position state.
            inputs: ML model inputs.

        Returns:
            Combined policy output from all active rules.
        """
        patterns: set[PolicyPattern] = set()
        reason_codes: list[str] = []
        param_overrides: dict[str, Decimal] = {}
        force_close = False
        kill = False

        # Priority 1: Kill switch (POL-019, POL-020)
        if self._check_kill_max_loss(ctx):
            patterns.add(PolicyPattern.FORCE_CLOSE)
            reason_codes.append("kill_max_loss")
            force_close = True
            kill = True
            # Early return - kill overrides everything
            return PolicyOutput.with_patterns(
                patterns=patterns,
                reason_codes=reason_codes,
                force_close=force_close,
                kill=kill,
            )

        # Priority 2: Hard inventory limit (POL-012)
        if self._check_hard_limit(ctx):
            patterns.add(PolicyPattern.SUPPRESS_ALL)
            reason_codes.append("hard_limit_block")
            # Continue to check other rules but entry is blocked

        # Priority 3: Staleness pause (POL-013)
        if self._check_stale_book(ctx):
            patterns.add(PolicyPattern.SUPPRESS_ALL)
            reason_codes.append("stale_book_pause")

        # Priority 4: Toxic disable (POL-005)
        if self._check_toxic_disable(inputs):
            patterns.add(PolicyPattern.SUPPRESS_ALL)
            reason_codes.append("toxic_disable")

        # If SUPPRESS_ALL is active, no need to check lower priority rules
        if PolicyPattern.SUPPRESS_ALL in patterns:
            return PolicyOutput.with_patterns(
                patterns=patterns,
                reason_codes=reason_codes,
                param_overrides=param_overrides,
            )

        # Priority 6: Toxic widen (POL-004)
        if self._check_toxic_widen(inputs):
            patterns.add(PolicyPattern.MODIFY_PARAMS)
            reason_codes.append("toxic_widen")
            param_overrides["spread_mult"] = self.params.toxic_spread_mult

        # Priority 8: Low in-play pause (POL-002)
        if self._check_low_inplay(inputs):
            patterns.add(PolicyPattern.SUPPRESS_ENTRY)
            reason_codes.append("low_inplay_pause")

        return PolicyOutput.with_patterns(
            patterns=patterns,
            reason_codes=reason_codes,
            param_overrides=param_overrides,
            force_close=force_close,
            kill=kill,
        )

    def _check_kill_max_loss(self, ctx: PolicyContext) -> bool:
        """POL-019: Check if max session loss is breached.

        Precondition: (realized_pnl + unrealized_pnl) < -max_session_loss
        """
        return ctx.total_pnl < -self.params.max_session_loss

    def _check_hard_limit(self, ctx: PolicyContext) -> bool:
        """POL-012: Check if position exceeds hard limit.

        Precondition: abs(position_qty) >= inventory_hard_limit
        """
        return abs(ctx.position_qty) >= self.params.inventory_hard_limit

    def _check_stale_book(self, ctx: PolicyContext) -> bool:
        """POL-013: Check if book data is stale.

        Precondition: (ts - last_book_ts) > stale_quote_ms
        """
        return ctx.book_age_ms > self.params.stale_quote_ms

    def _check_toxic_disable(self, inputs: PolicyInputs) -> bool:
        """POL-005: Check if toxicity is severe enough to disable quoting.

        Precondition: p_toxic >= toxicity_disable_threshold
        """
        return inputs.p_toxic >= self.params.toxicity_disable_threshold

    def _check_toxic_widen(self, inputs: PolicyInputs) -> bool:
        """POL-004: Check if toxicity warrants spread widening.

        Precondition: p_toxic >= toxicity_widen_threshold
        """
        return inputs.p_toxic >= self.params.toxicity_widen_threshold

    def _check_low_inplay(self, inputs: PolicyInputs) -> bool:
        """POL-002: Check if in-play probability is too low.

        Precondition: p_inplay_2m < inplay_exit_prob
        """
        return inputs.p_inplay_2m < self.params.inplay_exit_prob
