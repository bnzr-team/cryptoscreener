"""Baseline model runner using heuristics."""

from __future__ import annotations

from cryptoscreener.contracts.events import (
    ExecutionProfile,
    Features,
    FeatureSnapshot,
    PredictionSnapshot,
    PredictionStatus,
    ReasonCode,
    RegimeTrend,
    RegimeVol,
)
from cryptoscreener.model_runner.base import ModelRunner, ModelRunnerConfig


class BaselineRunner(ModelRunner):
    """
    Baseline model runner using heuristic rules.

    Computes predictions based on feature thresholds and regime classification.
    Designed for deterministic, reproducible predictions without ML.

    Heuristics:
    - P(inplay) based on spread, book imbalance, and flow imbalance
    - P(toxic) based on impact and flow imbalance extremes
    - Expected utility from spread and imbalance signals
    - Status from thresholds on p_inplay and p_toxic
    """

    def __init__(self, config: ModelRunnerConfig | None = None) -> None:
        """Initialize baseline runner."""
        super().__init__(config)

    def predict(self, snapshot: FeatureSnapshot) -> PredictionSnapshot:
        """
        Generate prediction using heuristic rules.

        Args:
            snapshot: FeatureSnapshot with features for a symbol.

        Returns:
            PredictionSnapshot with heuristic-based prediction.
        """
        features = snapshot.features

        # Check data health first
        if self._has_data_issues(snapshot):
            return self._make_data_issue_prediction(snapshot)

        # Compute base probabilities from features
        p_inplay_base = self._compute_p_inplay_base(features)

        # Adjust for regime
        regime_mult = self._get_regime_multiplier(features)
        p_inplay_30s = min(1.0, p_inplay_base * 0.8 * regime_mult)
        p_inplay_2m = min(1.0, p_inplay_base * regime_mult)
        p_inplay_5m = min(1.0, p_inplay_base * 1.1 * regime_mult)

        # Compute toxicity probability
        p_toxic = self._compute_p_toxic(features)

        # Compute expected utility
        expected_utility = self._compute_expected_utility(features, p_inplay_2m, p_toxic)

        # Check PRD critical gates for TRADEABLE
        gate_failures = self._check_critical_gates(features)

        # Determine status (with gate enforcement)
        status = self._determine_status(p_inplay_2m, p_toxic, gate_failures)

        # Build reason codes (including gate failures)
        reasons = self._build_reasons(features, p_inplay_2m, p_toxic, gate_failures)

        # Record metrics
        self._metrics.record_prediction(snapshot.symbol, status.value)

        return PredictionSnapshot(
            ts=snapshot.ts,
            symbol=snapshot.symbol,
            profile=ExecutionProfile(self._config.default_profile),
            p_inplay_30s=round(p_inplay_30s, 4),
            p_inplay_2m=round(p_inplay_2m, 4),
            p_inplay_5m=round(p_inplay_5m, 4),
            expected_utility_bps_2m=round(expected_utility, 2),
            p_toxic=round(p_toxic, 4),
            status=status,
            reasons=reasons,
            model_version=self._config.model_version,
            calibration_version=self._config.calibration_version,
            data_health=snapshot.data_health,
        )

    def _has_data_issues(self, snapshot: FeatureSnapshot) -> bool:
        """Check if snapshot has data quality issues."""
        health = snapshot.data_health

        # Stale data thresholds (5 seconds)
        if health.stale_book_ms > 5000:
            return True
        if health.stale_trades_ms > 30000:  # Trades can be stale longer
            return True
        return bool(health.missing_streams)

    def _make_data_issue_prediction(self, snapshot: FeatureSnapshot) -> PredictionSnapshot:
        """Create prediction for data issue case."""
        self._metrics.record_prediction(snapshot.symbol, "DATA_ISSUE")

        return PredictionSnapshot(
            ts=snapshot.ts,
            symbol=snapshot.symbol,
            profile=ExecutionProfile(self._config.default_profile),
            p_inplay_30s=0.0,
            p_inplay_2m=0.0,
            p_inplay_5m=0.0,
            expected_utility_bps_2m=0.0,
            p_toxic=0.0,
            status=PredictionStatus.DATA_ISSUE,
            reasons=[
                ReasonCode(
                    code="RC_DATA_STALE",
                    value=float(snapshot.data_health.stale_book_ms),
                    unit="ms",
                    evidence=f"Book data stale for {snapshot.data_health.stale_book_ms}ms",
                )
            ],
            model_version=self._config.model_version,
            calibration_version=self._config.calibration_version,
            data_health=snapshot.data_health,
        )

    def _compute_p_inplay_base(self, features: Features) -> float:
        """
        Compute base P(inplay) from features.

        Higher probability when:
        - Spread is tight (good liquidity)
        - Book imbalance is strong (directional pressure)
        - Flow imbalance confirms direction
        """
        # Spread factor: tighter spread = higher probability
        # Normalize: 1 bps = 1.0, 10 bps = 0.1
        spread_factor = max(0.1, min(1.0, 1.0 / max(features.spread_bps, 0.1)))

        # Imbalance factor: stronger imbalance = higher probability
        imbalance_strength = abs(features.book_imbalance)
        flow_strength = abs(features.flow_imbalance)

        # Concordance bonus: imbalances in same direction
        concordant = (features.book_imbalance * features.flow_imbalance) > 0
        concordance_mult = 1.2 if concordant else 0.8

        # Base probability
        p_base = (
            0.3 * spread_factor + 0.35 * imbalance_strength + 0.35 * flow_strength
        ) * concordance_mult

        return min(1.0, max(0.0, p_base))

    def _get_regime_multiplier(self, features: Features) -> float:
        """Get regime-based multiplier for probability."""
        # High vol + trend = good for prediction
        if features.regime_vol == RegimeVol.HIGH and features.regime_trend == RegimeTrend.TREND:
            return 1.3

        # High vol + chop = bad (noise)
        if features.regime_vol == RegimeVol.HIGH and features.regime_trend == RegimeTrend.CHOP:
            return 0.7

        # Low vol + trend = moderate
        if features.regime_vol == RegimeVol.LOW and features.regime_trend == RegimeTrend.TREND:
            return 1.0

        # Low vol + chop = worst
        return 0.5

    def _compute_p_toxic(self, features: Features) -> float:
        """
        Compute toxicity probability.

        High toxicity when:
        - High impact (large order would move price significantly)
        - Extreme flow imbalance (one-sided flow, potential informed trading)
        """
        # Impact factor: higher impact = more toxic
        impact_factor = min(1.0, features.impact_bps_q / 50.0)  # 50 bps = max toxic

        # Flow extreme factor
        flow_extreme = abs(features.flow_imbalance) ** 2  # Quadratic for extremes

        # Combine
        p_toxic = 0.4 * impact_factor + 0.6 * flow_extreme

        return min(1.0, max(0.0, p_toxic))

    def _compute_expected_utility(
        self,
        features: Features,
        p_inplay: float,
        p_toxic: float,
    ) -> float:
        """
        Compute expected utility in basis points.

        Utility = P(inplay) * expected_gain - P(toxic) * expected_loss
        """
        # Expected gain when in-play (based on volatility)
        expected_gain_bps = features.natr_14_5m * 100  # NATR to bps

        # Expected loss from spread and impact
        expected_loss_bps = features.spread_bps + features.impact_bps_q

        # Net expected utility
        utility = p_inplay * expected_gain_bps * (1 - p_toxic) - p_toxic * expected_loss_bps

        return utility

    def _check_critical_gates(self, features: Features) -> list[str]:
        """
        Check PRD critical gates for TRADEABLE status.

        Returns list of failed gate codes (empty if all pass).
        These are HARD gates - failing any blocks TRADEABLE.
        """
        failures: list[str] = []

        # Spread gate: spread must be <= spread_max_bps
        if features.spread_bps > self._config.spread_max_bps:
            failures.append("GATE_SPREAD")

        # Impact gate: impact must be <= impact_max_bps
        if features.impact_bps_q > self._config.impact_max_bps:
            failures.append("GATE_IMPACT")

        return failures

    def _determine_status(
        self, p_inplay: float, p_toxic: float, gate_failures: list[str] | None = None
    ) -> PredictionStatus:
        """
        Determine prediction status from probabilities and gate checks.

        PRD Critical Gates are enforced BEFORE status assignment:
        - If any gate fails, TRADEABLE is blocked (downgrade to WATCH)
        """
        gate_failures = gate_failures or []

        # High toxicity = TRAP (checked first, regardless of gates)
        if p_toxic >= self._config.toxic_threshold:
            return PredictionStatus.TRAP

        # Check if would be TRADEABLE based on p_inplay
        if p_inplay >= self._config.tradeable_threshold:
            # HARD GATE CHECK: any gate failure blocks TRADEABLE
            if gate_failures:
                # Downgrade to WATCH (gates failed)
                return PredictionStatus.WATCH
            return PredictionStatus.TRADEABLE

        # Moderate inplay = WATCH
        if p_inplay >= self._config.watch_threshold:
            return PredictionStatus.WATCH

        # Low probability = DEAD
        return PredictionStatus.DEAD

    def _build_reasons(
        self,
        features: Features,
        p_inplay: float,
        p_toxic: float,
        gate_failures: list[str] | None = None,
    ) -> list[ReasonCode]:
        """Build reason codes explaining the prediction."""
        reasons: list[ReasonCode] = []
        gate_failures = gate_failures or []

        # Gate failure reasons (these explain why TRADEABLE was blocked)
        if "GATE_SPREAD" in gate_failures:
            reasons.append(
                ReasonCode(
                    code="RC_GATE_SPREAD_FAIL",
                    value=round(features.spread_bps, 2),
                    unit="bps",
                    evidence=f"Spread {features.spread_bps:.1f} bps exceeds max {self._config.spread_max_bps:.1f} bps",
                )
            )

        if "GATE_IMPACT" in gate_failures:
            reasons.append(
                ReasonCode(
                    code="RC_GATE_IMPACT_FAIL",
                    value=round(features.impact_bps_q, 2),
                    unit="bps",
                    evidence=f"Impact {features.impact_bps_q:.1f} bps exceeds max {self._config.impact_max_bps:.1f} bps",
                )
            )

        # Flow imbalance reason
        if abs(features.flow_imbalance) > 0.3:
            direction = "buy" if features.flow_imbalance > 0 else "sell"
            reasons.append(
                ReasonCode(
                    code="RC_FLOW_SURGE",
                    value=round(features.flow_imbalance, 3),
                    unit="ratio",
                    evidence=f"Strong {direction} flow imbalance",
                )
            )

        # Book imbalance reason
        if abs(features.book_imbalance) > 0.3:
            side = "bid" if features.book_imbalance > 0 else "ask"
            reasons.append(
                ReasonCode(
                    code="RC_BOOK_PRESSURE",
                    value=round(features.book_imbalance, 3),
                    unit="ratio",
                    evidence=f"Strong {side}-side book pressure",
                )
            )

        # Spread reason
        if features.spread_bps < 2:
            reasons.append(
                ReasonCode(
                    code="RC_TIGHT_SPREAD",
                    value=round(features.spread_bps, 2),
                    unit="bps",
                    evidence="Tight spread indicates good liquidity",
                )
            )
        elif features.spread_bps > 10:
            reasons.append(
                ReasonCode(
                    code="RC_WIDE_SPREAD",
                    value=round(features.spread_bps, 2),
                    unit="bps",
                    evidence="Wide spread indicates poor liquidity",
                )
            )

        # Toxicity warning
        if p_toxic > 0.5:
            reasons.append(
                ReasonCode(
                    code="RC_TOXIC_RISK",
                    value=round(p_toxic, 3),
                    unit="prob",
                    evidence="Elevated toxic flow risk",
                )
            )

        # Regime reason
        if features.regime_vol == RegimeVol.HIGH:
            reasons.append(
                ReasonCode(
                    code="RC_HIGH_VOL",
                    value=round(features.natr_14_5m, 4),
                    unit="natr",
                    evidence="High volatility regime",
                )
            )

        return reasons
