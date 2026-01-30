"""PolicyParams - Named configuration parameters from DEC-043.

All numeric thresholds are defined here, not hardcoded in rule logic.
This enables config-first design per DEC-043.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class PolicyParams:
    """Named configuration parameters from DEC-043.

    All thresholds are explicit named parameters.
    No magic numbers in policy rule logic.
    """

    # In-Play thresholds (POL-001, POL-002)
    inplay_enter_prob: Decimal = Decimal("0.6")
    inplay_exit_prob: Decimal = Decimal("0.4")
    inplay_cooldown_ms: int = 5000

    # Toxicity thresholds (POL-004, POL-005, POL-006)
    toxicity_widen_threshold: Decimal = Decimal("0.5")
    toxicity_disable_threshold: Decimal = Decimal("0.8")
    toxicity_exit_threshold: Decimal = Decimal("0.3")
    toxic_spread_mult: Decimal = Decimal("2.0")
    toxic_cooldown_ms: int = 2000

    # Volatility thresholds (POL-003)
    vol_regime_high_threshold: Decimal = Decimal("30")  # natr_14_5m threshold
    high_vol_spread_mult: Decimal = Decimal("1.5")

    # Trend thresholds (POL-007, POL-009)
    trend_confidence_min: Decimal = Decimal("0.6")
    trend_skew_bps_max: Decimal = Decimal("20")

    # Inventory thresholds (POL-008, POL-010, POL-012)
    inventory_soft_limit: Decimal = Decimal("0.005")  # Qty triggering unwind
    inventory_hard_limit: Decimal = Decimal("0.01")  # Max position (hard block)
    inventory_skew_start: Decimal = Decimal("0.002")  # Qty triggering skew

    # PnL thresholds (POL-011, POL-019, POL-020)
    max_session_loss: Decimal = Decimal("100")  # USD
    max_drawdown: Decimal = Decimal("50")  # USD
    pnl_unwind_threshold: Decimal = Decimal("30")  # USD loss to trigger unwind

    # Staleness thresholds (POL-013, POL-014, POL-015)
    stale_quote_ms: int = 5000
    stale_trade_ms: int = 10000
    ws_reconnect_grace_ms: int = 3000

    # Rate limits (POL-016, POL-017, POL-018)
    rate_limit_buffer: int = 20
    fill_cooldown_ms: int = 500
    max_orders_per_window: int = 50

    # Spread limits
    spread_bps_min: Decimal = Decimal("5")
    spread_bps_default: Decimal = Decimal("10")
    spread_bps_max: Decimal = Decimal("100")

    def __post_init__(self) -> None:
        """Validate parameter relationships."""
        if self.inplay_exit_prob >= self.inplay_enter_prob:
            raise ValueError(
                f"inplay_exit_prob ({self.inplay_exit_prob}) must be < "
                f"inplay_enter_prob ({self.inplay_enter_prob}) for hysteresis"
            )
        if self.toxicity_exit_threshold >= self.toxicity_widen_threshold:
            raise ValueError(
                f"toxicity_exit_threshold ({self.toxicity_exit_threshold}) must be < "
                f"toxicity_widen_threshold ({self.toxicity_widen_threshold}) for hysteresis"
            )
        if self.inventory_soft_limit >= self.inventory_hard_limit:
            raise ValueError(
                f"inventory_soft_limit ({self.inventory_soft_limit}) must be < "
                f"inventory_hard_limit ({self.inventory_hard_limit})"
            )
