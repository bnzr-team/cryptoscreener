"""
Label builder for ML ground truth generation.

Implements labeling logic per LABELS_SPEC.md:

For each (symbol, time t, horizon H, profile):
1. Compute spread_bps(t) and fees_bps(profile)
2. Determine clip size: Q_usd = k * usd_volume_60s(t)
3. Estimate impact_bps(profile, t, Q_usd)
4. cost_bps = spread_bps + fees_bps + impact_bps
5. Compute MFE_bps(H) = max favorable excursion
6. net_edge_bps(H) = MFE_bps(H) - cost_bps
7. I_tradeable(H) = 1 if (net_edge_bps >= X_bps) AND gates_pass

Toxicity labels:
- y_toxic = 1 if price moves against by > Y_bps within τ after entry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from cryptoscreener.cost_model import CostCalculator, CostModelConfig, ExecutionCosts
from cryptoscreener.cost_model.calculator import OrderbookSnapshot, Profile


class Horizon(str, Enum):
    """Prediction horizon."""

    H_30S = "30s"
    H_2M = "2m"
    H_5M = "5m"


# Horizon durations in milliseconds
HORIZON_MS: dict[Horizon, int] = {
    Horizon.H_30S: 30_000,
    Horizon.H_2M: 120_000,
    Horizon.H_5M: 300_000,
}


@dataclass(frozen=True)
class ToxicityConfig:
    """Configuration for toxicity label generation.

    Attributes:
        tau_ms: Time window to check for adverse movement (default: 30s).
        threshold_bps: Adverse movement threshold in bps (default: 10 bps).
    """

    tau_ms: int = 30_000
    threshold_bps: float = 10.0


@dataclass(frozen=True)
class LabelBuilderConfig:
    """Configuration for label builder.

    Attributes:
        x_bps_30s_a: Min net edge for tradeability at 30s, Profile A.
        x_bps_30s_b: Min net edge for tradeability at 30s, Profile B.
        x_bps_2m_a: Min net edge for tradeability at 2m, Profile A.
        x_bps_2m_b: Min net edge for tradeability at 2m, Profile B.
        x_bps_5m_a: Min net edge for tradeability at 5m, Profile A.
        x_bps_5m_b: Min net edge for tradeability at 5m, Profile B.
        spread_max_bps: Maximum spread gate threshold.
        impact_max_bps: Maximum impact gate threshold.
        toxicity: Toxicity label configuration.
        cost_model: Cost model configuration.
    """

    # Minimum net edge thresholds by horizon and profile
    x_bps_30s_a: float = 5.0
    x_bps_30s_b: float = 8.0
    x_bps_2m_a: float = 10.0
    x_bps_2m_b: float = 15.0
    x_bps_5m_a: float = 15.0
    x_bps_5m_b: float = 20.0

    # Gate thresholds
    spread_max_bps: float = 10.0
    impact_max_bps: float = 20.0

    # Toxicity config
    toxicity: ToxicityConfig = field(default_factory=ToxicityConfig)

    # Cost model config
    cost_model: CostModelConfig = field(default_factory=CostModelConfig)

    def get_x_bps(self, horizon: Horizon, profile: Profile) -> float:
        """Get minimum net edge threshold for horizon/profile.

        Args:
            horizon: Prediction horizon.
            profile: Execution profile.

        Returns:
            Threshold in basis points.
        """
        key = f"x_bps_{horizon.value}_{profile.value.lower()}"
        return getattr(self, key, 10.0)


@dataclass(frozen=True)
class TradeabilityLabel:
    """Tradeability label for a single (horizon, profile) combination.

    Attributes:
        horizon: Prediction horizon.
        profile: Execution profile.
        i_tradeable: 1 if tradeable, 0 otherwise.
        mfe_bps: Maximum favorable excursion in bps.
        cost_bps: Total execution cost in bps.
        net_edge_bps: Net edge (mfe_bps - cost_bps).
        gates_passed: Whether all gates passed.
        gate_failures: List of failed gate names.
    """

    horizon: Horizon
    profile: Profile
    i_tradeable: int
    mfe_bps: float
    cost_bps: float
    net_edge_bps: float
    gates_passed: bool
    gate_failures: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LabelRow:
    """Complete label row for a (symbol, timestamp) pair.

    Attributes:
        ts: Timestamp in milliseconds.
        symbol: Trading pair symbol.
        tradeability: Dict of (horizon, profile) -> TradeabilityLabel.
        y_toxic: Toxicity label (1 = toxic, 0 = not toxic).
        severity_toxic_bps: Severity of adverse movement in bps.
        mid_price: Mid price at timestamp.
        spread_bps: Spread in basis points.
    """

    ts: int
    symbol: str
    tradeability: dict[tuple[Horizon, Profile], TradeabilityLabel]
    y_toxic: int
    severity_toxic_bps: float
    mid_price: float
    spread_bps: float


@dataclass
class PricePoint:
    """Price point for MFE/MAE calculation.

    Attributes:
        ts: Timestamp in milliseconds.
        mid: Mid price.
    """

    ts: int
    mid: float


class LabelBuilder:
    """Builder for ML training labels.

    Generates ground truth labels for:
    - I_tradeable(H) for each horizon and profile
    - p_toxic (toxicity)
    """

    def __init__(self, config: LabelBuilderConfig | None = None) -> None:
        """Initialize label builder.

        Args:
            config: Label builder configuration. Uses defaults if not provided.
        """
        self._config = config or LabelBuilderConfig()
        self._cost_calculator = CostCalculator(self._config.cost_model)

    def compute_mfe_bps(
        self,
        entry_price: float,
        future_prices: Sequence[PricePoint],
        horizon: Horizon,
        entry_ts: int,
    ) -> float:
        """Compute Maximum Favorable Excursion in basis points.

        MFE = max favorable move within horizon window.

        Args:
            entry_price: Entry price.
            future_prices: Sequence of future price points.
            horizon: Prediction horizon.
            entry_ts: Entry timestamp in milliseconds.

        Returns:
            MFE in basis points.
        """
        if entry_price <= 0 or not future_prices:
            return 0.0

        horizon_ms = HORIZON_MS[horizon]
        end_ts = entry_ts + horizon_ms

        max_price = entry_price
        for point in future_prices:
            if point.ts > end_ts:
                break
            if point.mid > max_price:
                max_price = point.mid

        mfe_bps = ((max_price - entry_price) / entry_price) * 10000
        return max(mfe_bps, 0.0)

    def compute_mae_bps(
        self,
        entry_price: float,
        future_prices: Sequence[PricePoint],
        horizon: Horizon,
        entry_ts: int,
    ) -> float:
        """Compute Maximum Adverse Excursion in basis points.

        MAE = max adverse move within horizon window.

        Args:
            entry_price: Entry price.
            future_prices: Sequence of future price points.
            horizon: Prediction horizon.
            entry_ts: Entry timestamp in milliseconds.

        Returns:
            MAE in basis points (positive value).
        """
        if entry_price <= 0 or not future_prices:
            return 0.0

        horizon_ms = HORIZON_MS[horizon]
        end_ts = entry_ts + horizon_ms

        min_price = entry_price
        for point in future_prices:
            if point.ts > end_ts:
                break
            if point.mid < min_price:
                min_price = point.mid

        mae_bps = ((entry_price - min_price) / entry_price) * 10000
        return max(mae_bps, 0.0)

    def check_gates(
        self,
        costs: ExecutionCosts,
    ) -> tuple[bool, list[str]]:
        """Check trading gates.

        Args:
            costs: Computed execution costs.

        Returns:
            Tuple of (gates_passed, list of failed gate names).
        """
        failures: list[str] = []

        if costs.spread_bps > self._config.spread_max_bps:
            failures.append("SPREAD_GATE")

        if costs.impact_bps > self._config.impact_max_bps:
            failures.append("IMPACT_GATE")

        return (len(failures) == 0, failures)

    def compute_tradeability_label(
        self,
        horizon: Horizon,
        profile: Profile,
        entry_price: float,
        future_prices: Sequence[PricePoint],
        entry_ts: int,
        costs: ExecutionCosts,
    ) -> TradeabilityLabel:
        """Compute tradeability label for a single (horizon, profile).

        Args:
            horizon: Prediction horizon.
            profile: Execution profile.
            entry_price: Entry price.
            future_prices: Sequence of future price points.
            entry_ts: Entry timestamp.
            costs: Computed execution costs for this profile.

        Returns:
            TradeabilityLabel with all components.
        """
        mfe_bps = self.compute_mfe_bps(entry_price, future_prices, horizon, entry_ts)
        cost_bps = costs.total_bps
        net_edge_bps = mfe_bps - cost_bps

        gates_passed, gate_failures = self.check_gates(costs)

        x_bps = self._config.get_x_bps(horizon, profile)
        i_tradeable = 1 if (net_edge_bps >= x_bps and gates_passed) else 0

        return TradeabilityLabel(
            horizon=horizon,
            profile=profile,
            i_tradeable=i_tradeable,
            mfe_bps=mfe_bps,
            cost_bps=cost_bps,
            net_edge_bps=net_edge_bps,
            gates_passed=gates_passed,
            gate_failures=gate_failures,
        )

    def compute_toxicity_label(
        self,
        entry_price: float,
        future_prices: Sequence[PricePoint],
        entry_ts: int,
    ) -> tuple[int, float]:
        """Compute toxicity label.

        An event is "toxic" if price moves against entry by more than
        threshold_bps within tau_ms after entry.

        Args:
            entry_price: Entry price.
            future_prices: Sequence of future price points.
            entry_ts: Entry timestamp.

        Returns:
            Tuple of (y_toxic: 0 or 1, severity_bps: float).
        """
        if entry_price <= 0 or not future_prices:
            return (0, 0.0)

        tau_ms = self._config.toxicity.tau_ms
        threshold_bps = self._config.toxicity.threshold_bps
        end_ts = entry_ts + tau_ms

        max_adverse_bps = 0.0
        for point in future_prices:
            if point.ts > end_ts:
                break

            # Adverse move = price drop (for long entry)
            adverse_bps = ((entry_price - point.mid) / entry_price) * 10000
            if adverse_bps > max_adverse_bps:
                max_adverse_bps = adverse_bps

        y_toxic = 1 if max_adverse_bps > threshold_bps else 0
        return (y_toxic, max_adverse_bps)

    def build_label_row(
        self,
        ts: int,
        symbol: str,
        bid: float,
        ask: float,
        future_prices: Sequence[PricePoint],
        orderbook: OrderbookSnapshot | None = None,
        usd_volume_60s: float = 0.0,
        style: str = "scalping",
    ) -> LabelRow:
        """Build complete label row for a (symbol, timestamp) pair.

        Args:
            ts: Timestamp in milliseconds.
            symbol: Trading pair symbol.
            bid: Best bid price.
            ask: Best ask price.
            future_prices: Sequence of future price points.
            orderbook: Optional orderbook snapshot.
            usd_volume_60s: USD volume over last 60 seconds.
            style: Trading style for clip size calculation.

        Returns:
            LabelRow with all labels.
        """
        mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        spread_bps = ((ask - bid) / mid_price * 10000) if mid_price > 0 else 0.0

        # Compute costs for both profiles
        costs_by_profile = self._cost_calculator.compute_costs_both_profiles(
            bid=bid,
            ask=ask,
            orderbook=orderbook,
            usd_volume_60s=usd_volume_60s,
            style=style,
        )

        # Compute tradeability labels for all (horizon, profile) combinations
        tradeability: dict[tuple[Horizon, Profile], TradeabilityLabel] = {}
        for horizon in Horizon:
            for profile in Profile:
                label = self.compute_tradeability_label(
                    horizon=horizon,
                    profile=profile,
                    entry_price=mid_price,
                    future_prices=future_prices,
                    entry_ts=ts,
                    costs=costs_by_profile[profile],
                )
                tradeability[(horizon, profile)] = label

        # Compute toxicity label
        y_toxic, severity_toxic_bps = self.compute_toxicity_label(
            entry_price=mid_price,
            future_prices=future_prices,
            entry_ts=ts,
        )

        return LabelRow(
            ts=ts,
            symbol=symbol,
            tradeability=tradeability,
            y_toxic=y_toxic,
            severity_toxic_bps=severity_toxic_bps,
            mid_price=mid_price,
            spread_bps=spread_bps,
        )

    def label_row_to_flat_dict(self, row: LabelRow) -> dict:
        """Convert LabelRow to flat dictionary for DataFrame/parquet.

        Args:
            row: LabelRow to convert.

        Returns:
            Flat dictionary with all fields.
        """
        result: dict = {
            "ts": row.ts,
            "symbol": row.symbol,
            "mid_price": row.mid_price,
            "spread_bps": row.spread_bps,
            "y_toxic": row.y_toxic,
            "severity_toxic_bps": row.severity_toxic_bps,
        }

        # Flatten tradeability labels
        for (horizon, profile), label in row.tradeability.items():
            prefix = f"{horizon.value}_{profile.value.lower()}"
            result[f"i_tradeable_{prefix}"] = label.i_tradeable
            result[f"mfe_bps_{prefix}"] = label.mfe_bps
            result[f"cost_bps_{prefix}"] = label.cost_bps
            result[f"net_edge_bps_{prefix}"] = label.net_edge_bps
            result[f"gates_passed_{prefix}"] = int(label.gates_passed)

        return result
