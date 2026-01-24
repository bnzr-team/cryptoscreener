"""Scorer for computing ranking scores from predictions.

Implements PRD Section 8 scoring formula:
    score = p_inplay * clamp(expected_utility_bps, 0, Umax) * (1 - alpha*p_toxic)

Where p_inplay is aggregated from multiple horizons with configurable weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptoscreener.contracts.events import PredictionSnapshot


@dataclass
class ScorerConfig:
    """Configuration for Scorer.

    Attributes:
        w_30s: Weight for p_inplay_30s in aggregation.
        w_2m: Weight for p_inplay_2m in aggregation.
        w_5m: Weight for p_inplay_5m in aggregation.
        utility_max_bps: Maximum expected utility for clamping (Umax).
        toxic_alpha: Penalty coefficient for p_toxic (alpha).
    """

    # Horizon weights (must sum to 1.0)
    w_30s: float = 0.2
    w_2m: float = 0.5
    w_5m: float = 0.3

    # Utility clamping
    utility_max_bps: float = 50.0

    # Toxicity penalty
    toxic_alpha: float = 0.5


class Scorer:
    """Compute ranking scores from PredictionSnapshots.

    Implements the utility-aware scoring formula from PRD Section 8.3:
        score = p_inplay * utility_factor * toxicity_penalty

    The score is normalized to [0, 1] for consistent ranking.
    """

    def __init__(self, config: ScorerConfig | None = None) -> None:
        """Initialize scorer with configuration.

        Args:
            config: Scorer configuration. Uses defaults if not provided.
        """
        self._config = config or ScorerConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration values."""
        weights_sum = self._config.w_30s + self._config.w_2m + self._config.w_5m
        if abs(weights_sum - 1.0) > 1e-6:
            msg = f"Horizon weights must sum to 1.0, got {weights_sum}"
            raise ValueError(msg)

        if self._config.utility_max_bps <= 0:
            msg = f"utility_max_bps must be positive, got {self._config.utility_max_bps}"
            raise ValueError(msg)

        if not 0 <= self._config.toxic_alpha <= 1:
            msg = f"toxic_alpha must be in [0, 1], got {self._config.toxic_alpha}"
            raise ValueError(msg)

    @property
    def config(self) -> ScorerConfig:
        """Get scorer configuration."""
        return self._config

    def compute_p_inplay(self, prediction: PredictionSnapshot) -> float:
        """Compute aggregated p_inplay from multiple horizons.

        Args:
            prediction: PredictionSnapshot with horizon probabilities.

        Returns:
            Weighted average of p_inplay across horizons.
        """
        return (
            self._config.w_30s * prediction.p_inplay_30s
            + self._config.w_2m * prediction.p_inplay_2m
            + self._config.w_5m * prediction.p_inplay_5m
        )

    def compute_utility_factor(self, prediction: PredictionSnapshot) -> float:
        """Compute utility factor from expected utility.

        Clamps utility to [0, utility_max_bps] and normalizes to [0, 1].

        Args:
            prediction: PredictionSnapshot with expected utility.

        Returns:
            Normalized utility factor in [0, 1].
        """
        clamped = max(0.0, min(prediction.expected_utility_bps_2m, self._config.utility_max_bps))
        return clamped / self._config.utility_max_bps

    def compute_toxicity_penalty(self, prediction: PredictionSnapshot) -> float:
        """Compute toxicity penalty factor.

        Returns (1 - alpha * p_toxic) which reduces score for toxic conditions.

        Args:
            prediction: PredictionSnapshot with p_toxic.

        Returns:
            Penalty factor in [1-alpha, 1].
        """
        return 1.0 - self._config.toxic_alpha * prediction.p_toxic

    def score(self, prediction: PredictionSnapshot) -> float:
        """Compute ranking score for a prediction.

        Formula: score = p_inplay * utility_factor * toxicity_penalty

        The score is normalized to [0, 1] for consistent ranking.

        Args:
            prediction: PredictionSnapshot to score.

        Returns:
            Ranking score in [0, 1].
        """
        p_inplay = self.compute_p_inplay(prediction)
        utility_factor = self.compute_utility_factor(prediction)
        toxicity_penalty = self.compute_toxicity_penalty(prediction)

        raw_score = p_inplay * utility_factor * toxicity_penalty

        # Clamp to [0, 1] for safety
        return max(0.0, min(1.0, raw_score))

    def score_batch(self, predictions: list[PredictionSnapshot]) -> list[float]:
        """Compute scores for multiple predictions.

        Args:
            predictions: List of PredictionSnapshots.

        Returns:
            List of scores corresponding to each prediction.
        """
        return [self.score(p) for p in predictions]
