"""
Cost calculator for trading cost estimation.

Implements:
- spread_bps = (ask - bid) / mid * 10000
- fees_bps = configurable per profile
- impact_bps(Q) = estimated price impact for clip size Q

Per COST_MODEL_SPEC.md and LABELS_SPEC.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class Profile(str, Enum):
    """Execution profile type."""

    A = "A"  # Maker-ish (lower fees, requires fill probability)
    B = "B"  # Taker-ish (higher fees/impact, higher completion probability)


@dataclass(frozen=True)
class CostModelConfig:
    """Configuration for cost model.

    Attributes:
        fees_bps_a: Fees in bps for Profile A (maker-ish). Default: 2.0 bps.
        fees_bps_b: Fees in bps for Profile B (taker-ish). Default: 4.0 bps.
        clip_k_scalping: Clip size multiplier for scalping style. Default: 0.01.
        clip_k_intraday: Clip size multiplier for intraday style. Default: 0.03.
        impact_clip_max_bps: Maximum impact before clipping. Default: 100.0 bps.
        depth_levels: Number of orderbook levels to consider for impact. Default: 10.
    """

    fees_bps_a: float = 2.0
    fees_bps_b: float = 4.0
    clip_k_scalping: float = 0.01
    clip_k_intraday: float = 0.03
    impact_clip_max_bps: float = 100.0
    depth_levels: int = 10


@dataclass(frozen=True)
class ExecutionCosts:
    """Computed execution costs.

    Attributes:
        spread_bps: Spread cost in basis points.
        fees_bps: Fee cost in basis points.
        impact_bps: Price impact cost in basis points.
        total_bps: Total cost (spread + fees + impact).
        clip_size_usd: Clip size used for impact calculation.
    """

    spread_bps: float
    fees_bps: float
    impact_bps: float
    total_bps: float
    clip_size_usd: float


@dataclass
class OrderbookLevel:
    """Single orderbook level.

    Attributes:
        price: Price at this level.
        qty: Quantity available at this price.
    """

    price: float
    qty: float


@dataclass
class OrderbookSnapshot:
    """Orderbook snapshot for impact calculation.

    Attributes:
        bids: List of bid levels (price descending).
        asks: List of ask levels (price ascending).
        mid: Mid price.
    """

    bids: list[OrderbookLevel] = field(default_factory=list)
    asks: list[OrderbookLevel] = field(default_factory=list)
    mid: float = 0.0


def compute_spread_bps(bid: float, ask: float) -> float:
    """Compute spread in basis points.

    Args:
        bid: Best bid price.
        ask: Best ask price.

    Returns:
        Spread in basis points. Returns 0.0 if mid is zero or negative.

    Example:
        >>> compute_spread_bps(100.0, 100.1)
        10.0  # (0.1 / 100.05) * 10000
    """
    if bid <= 0 or ask <= 0:
        return 0.0

    mid = (bid + ask) / 2
    if mid <= 0:
        return 0.0

    spread = ask - bid
    return (spread / mid) * 10000


def compute_impact_bps(
    orderbook: OrderbookSnapshot,
    clip_size_usd: float,
    side: str = "buy",
    max_bps: float = 100.0,
) -> float:
    """Compute price impact in basis points for a given clip size.

    Walks through orderbook levels to estimate the average execution price,
    then computes slippage from mid price.

    Args:
        orderbook: Orderbook snapshot with bids and asks.
        clip_size_usd: Size to execute in USD.
        side: "buy" or "sell".
        max_bps: Maximum impact before clipping.

    Returns:
        Estimated price impact in basis points.

    Example:
        >>> book = OrderbookSnapshot(
        ...     bids=[OrderbookLevel(99.9, 10), OrderbookLevel(99.8, 10)],
        ...     asks=[OrderbookLevel(100.1, 10), OrderbookLevel(100.2, 10)],
        ...     mid=100.0,
        ... )
        >>> compute_impact_bps(book, 500, "buy")  # Walk through asks
    """
    if clip_size_usd <= 0 or orderbook.mid <= 0:
        return 0.0

    levels = orderbook.asks if side == "buy" else orderbook.bids

    if not levels:
        return max_bps  # No liquidity, return max

    remaining_usd = clip_size_usd
    total_qty = 0.0
    total_cost = 0.0

    for level in levels:
        level_value_usd = level.price * level.qty
        if level_value_usd >= remaining_usd:
            # Partial fill at this level
            qty_filled = remaining_usd / level.price
            total_qty += qty_filled
            total_cost += qty_filled * level.price
            remaining_usd = 0
            break
        else:
            # Full fill at this level
            total_qty += level.qty
            total_cost += level.qty * level.price
            remaining_usd -= level_value_usd

    if total_qty <= 0:
        return max_bps

    avg_price = total_cost / total_qty

    # Compute slippage from mid
    if side == "buy":
        slippage = avg_price - orderbook.mid
    else:
        slippage = orderbook.mid - avg_price

    impact_bps = (slippage / orderbook.mid) * 10000

    # Clip to reasonable range
    return min(max(impact_bps, 0.0), max_bps)


class CostCalculator:
    """Calculator for execution costs.

    Computes total trading costs including spread, fees, and market impact.
    Per COST_MODEL_SPEC.md: cost_bps = spread_bps + fees_bps + impact_bps(Q)
    """

    def __init__(self, config: CostModelConfig | None = None) -> None:
        """Initialize cost calculator.

        Args:
            config: Cost model configuration. Uses defaults if not provided.
        """
        self._config = config or CostModelConfig()

    def get_fees_bps(self, profile: Profile) -> float:
        """Get fees in bps for a profile.

        Args:
            profile: Execution profile (A or B).

        Returns:
            Fees in basis points.
        """
        if profile == Profile.A:
            return self._config.fees_bps_a
        return self._config.fees_bps_b

    def compute_clip_size_usd(
        self,
        usd_volume_60s: float,
        style: str = "scalping",
    ) -> float:
        """Compute clip size in USD based on recent volume.

        Per LABELS_SPEC.md: Q_usd = k * usd_volume_60s

        Args:
            usd_volume_60s: USD volume over last 60 seconds.
            style: "scalping" (k=0.01) or "intraday" (k=0.03).

        Returns:
            Clip size in USD.
        """
        if style == "scalping":
            k = self._config.clip_k_scalping
        else:
            k = self._config.clip_k_intraday

        return k * usd_volume_60s

    def compute_costs(
        self,
        bid: float,
        ask: float,
        profile: Profile,
        orderbook: OrderbookSnapshot | None = None,
        usd_volume_60s: float = 0.0,
        style: str = "scalping",
    ) -> ExecutionCosts:
        """Compute total execution costs.

        Args:
            bid: Best bid price.
            ask: Best ask price.
            profile: Execution profile (A or B).
            orderbook: Optional orderbook snapshot for impact calculation.
            usd_volume_60s: USD volume over last 60 seconds (for clip size).
            style: Trading style for clip size calculation.

        Returns:
            ExecutionCosts with all components.
        """
        spread_bps = compute_spread_bps(bid, ask)
        fees_bps = self.get_fees_bps(profile)

        # Compute impact if orderbook is provided
        clip_size_usd = 0.0
        impact_bps = 0.0

        if orderbook is not None and usd_volume_60s > 0:
            clip_size_usd = self.compute_clip_size_usd(usd_volume_60s, style)
            impact_bps = compute_impact_bps(
                orderbook,
                clip_size_usd,
                side="buy",  # Conservative: use buy side for impact
                max_bps=self._config.impact_clip_max_bps,
            )

        total_bps = spread_bps + fees_bps + impact_bps

        return ExecutionCosts(
            spread_bps=spread_bps,
            fees_bps=fees_bps,
            impact_bps=impact_bps,
            total_bps=total_bps,
            clip_size_usd=clip_size_usd,
        )

    def compute_costs_both_profiles(
        self,
        bid: float,
        ask: float,
        orderbook: OrderbookSnapshot | None = None,
        usd_volume_60s: float = 0.0,
        style: str = "scalping",
    ) -> dict[Profile, ExecutionCosts]:
        """Compute costs for both profiles.

        Args:
            bid: Best bid price.
            ask: Best ask price.
            orderbook: Optional orderbook snapshot for impact calculation.
            usd_volume_60s: USD volume over last 60 seconds.
            style: Trading style for clip size calculation.

        Returns:
            Dict mapping Profile to ExecutionCosts.
        """
        return {
            profile: self.compute_costs(
                bid=bid,
                ask=ask,
                profile=profile,
                orderbook=orderbook,
                usd_volume_60s=usd_volume_60s,
                style=style,
            )
            for profile in Profile
        }
