"""PolicyInputs - ML model outputs for policy evaluation.

Defines the inputs that come from ML models (or stubs for testing).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from cryptoscreener.trading.strategy import StrategyContext

RegimeVol = Literal["LOW", "NORMAL", "HIGH"]
RegimeTrend = Literal["UP", "DOWN", "NEUTRAL"]


@dataclass(frozen=True)
class PolicyInputs:
    """ML model outputs for policy evaluation.

    All fields are immutable. Values come from ML models or test stubs.
    """

    # In-play probability (0.0 to 1.0)
    p_inplay_2m: Decimal

    # Toxicity probability (0.0 to 1.0)
    p_toxic: Decimal

    # Volatility regime
    regime_vol: RegimeVol

    # Trend regime
    regime_trend: RegimeTrend

    # Normalized ATR (basis points)
    natr_14_5m: Decimal

    def __post_init__(self) -> None:
        """Validate inputs are in expected ranges."""
        if not (Decimal("0") <= self.p_inplay_2m <= Decimal("1")):
            raise ValueError(f"p_inplay_2m must be 0-1, got {self.p_inplay_2m}")
        if not (Decimal("0") <= self.p_toxic <= Decimal("1")):
            raise ValueError(f"p_toxic must be 0-1, got {self.p_toxic}")
        if self.regime_vol not in ("LOW", "NORMAL", "HIGH"):
            raise ValueError(f"regime_vol must be LOW/NORMAL/HIGH, got {self.regime_vol}")
        if self.regime_trend not in ("UP", "DOWN", "NEUTRAL"):
            raise ValueError(f"regime_trend must be UP/DOWN/NEUTRAL, got {self.regime_trend}")


class PolicyInputsProvider(Protocol):
    """Protocol for providing PolicyInputs.

    Implementations may provide:
    - Constant values (for testing)
    - Fixture-specific values
    - ML model predictions (future)
    """

    def get_inputs(self, ctx: StrategyContext) -> PolicyInputs:
        """Get policy inputs for the given context.

        Args:
            ctx: Strategy context with market state.

        Returns:
            Policy inputs (ML model outputs or stubs).
        """
        ...
