"""FixturePolicyInputsProvider - Returns inputs based on fixture name.

Maps each test fixture to appropriate ML input values per DEC-044 plan.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from cryptoscreener.trading.policy.inputs import PolicyInputs

if TYPE_CHECKING:
    from cryptoscreener.trading.strategy import StrategyContext

# Fixture-to-inputs mapping per DEC-044 plan
# These values are chosen to exercise specific policies per 07_SIM_EXPECTATIONS.md
FIXTURE_INPUTS: dict[str, PolicyInputs] = {
    # monotonic_up: Low in-play (trend), UP trend, normal vol
    # Should trigger: POL-002 (suppress entry), POL-007 (trend skew)
    "monotonic_up": PolicyInputs(
        p_inplay_2m=Decimal("0.3"),
        p_toxic=Decimal("0.1"),
        regime_vol="NORMAL",
        regime_trend="UP",
        natr_14_5m=Decimal("15"),
    ),
    # mean_reverting: High in-play (range-bound), neutral, normal vol
    # Should trigger: POL-001 (trading enabled), normal operation
    "mean_reverting": PolicyInputs(
        p_inplay_2m=Decimal("0.8"),
        p_toxic=Decimal("0.1"),
        regime_vol="NORMAL",
        regime_trend="NEUTRAL",
        natr_14_5m=Decimal("12"),
    ),
    # Alias for mean_reverting_range.jsonl
    "mean_reverting_range": PolicyInputs(
        p_inplay_2m=Decimal("0.8"),
        p_toxic=Decimal("0.1"),
        regime_vol="NORMAL",
        regime_trend="NEUTRAL",
        natr_14_5m=Decimal("12"),
    ),
    # flash_crash: Low in-play, HIGH toxicity, HIGH vol, DOWN trend
    # Should trigger: POL-004/005 (toxic), POL-003 (vol), POL-019 (kill)
    "flash_crash": PolicyInputs(
        p_inplay_2m=Decimal("0.2"),
        p_toxic=Decimal("0.8"),
        regime_vol="HIGH",
        regime_trend="DOWN",
        natr_14_5m=Decimal("50"),
    ),
    # ws_gap: Medium in-play, low toxic, normal
    # Book staleness checked via timestamps, not ML inputs
    "ws_gap": PolicyInputs(
        p_inplay_2m=Decimal("0.5"),
        p_toxic=Decimal("0.2"),
        regime_vol="NORMAL",
        regime_trend="NEUTRAL",
        natr_14_5m=Decimal("10"),
    ),
}

# Default inputs for unknown fixtures
DEFAULT_INPUTS = PolicyInputs(
    p_inplay_2m=Decimal("0.5"),
    p_toxic=Decimal("0.2"),
    regime_vol="NORMAL",
    regime_trend="NEUTRAL",
    natr_14_5m=Decimal("15"),
)


class FixturePolicyInputsProvider:
    """Provides PolicyInputs based on fixture name.

    Uses predefined mappings to return appropriate ML inputs
    for each test fixture, ensuring deterministic behavior.
    """

    def __init__(self, fixture_name: str) -> None:
        """Initialize with fixture name.

        Args:
            fixture_name: Name of the fixture (without .jsonl extension).
        """
        # Strip extension if present
        name = fixture_name.replace(".jsonl", "")
        self._inputs = FIXTURE_INPUTS.get(name, DEFAULT_INPUTS)
        self._fixture_name = name

    def get_inputs(self, ctx: StrategyContext) -> PolicyInputs:
        """Get policy inputs for this fixture.

        Args:
            ctx: Strategy context (timestamp may be used for dynamic stubs).

        Returns:
            Fixture-specific policy inputs.
        """
        return self._inputs

    @property
    def fixture_name(self) -> str:
        """Get the fixture name."""
        return self._fixture_name
