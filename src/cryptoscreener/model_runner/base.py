"""Base interface for model runners."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptoscreener.contracts.events import FeatureSnapshot, PredictionSnapshot


@dataclass
class ModelRunnerConfig:
    """Configuration for ModelRunner."""

    # Model version string (semver+gitsha format)
    model_version: str = "baseline-v1.0.0+0000000"

    # Calibration version string
    calibration_version: str = "cal-v1.0.0"

    # Default execution profile
    default_profile: str = "A"

    # P(toxic) threshold for status classification
    toxic_threshold: float = 0.7

    # P(inplay) threshold for TRADEABLE status
    tradeable_threshold: float = 0.6

    # P(inplay) threshold for WATCH status
    watch_threshold: float = 0.3

    # PRD Critical Gates for TRADEABLE status
    # Spread gate: max spread in bps to allow TRADEABLE
    spread_max_bps: float = 10.0

    # Impact gate: max impact in bps to allow TRADEABLE
    impact_max_bps: float = 20.0


@dataclass
class RunnerMetrics:
    """Metrics for model runner operations."""

    # Prediction counts
    predictions_made: int = 0
    predictions_tradeable: int = 0
    predictions_watch: int = 0
    predictions_trap: int = 0
    predictions_dead: int = 0
    predictions_data_issue: int = 0

    # Per-symbol counts
    predictions_per_symbol: dict[str, int] = field(default_factory=dict)

    def record_prediction(self, symbol: str, status: str) -> None:
        """Record a prediction."""
        self.predictions_made += 1
        self.predictions_per_symbol[symbol] = self.predictions_per_symbol.get(symbol, 0) + 1

        if status == "TRADEABLE":
            self.predictions_tradeable += 1
        elif status == "WATCH":
            self.predictions_watch += 1
        elif status == "TRAP":
            self.predictions_trap += 1
        elif status == "DEAD":
            self.predictions_dead += 1
        elif status == "DATA_ISSUE":
            self.predictions_data_issue += 1

    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions_made = 0
        self.predictions_tradeable = 0
        self.predictions_watch = 0
        self.predictions_trap = 0
        self.predictions_dead = 0
        self.predictions_data_issue = 0
        self.predictions_per_symbol.clear()


class ModelRunner(ABC):
    """
    Abstract base class for model runners.

    A model runner takes FeatureSnapshot and produces PredictionSnapshot.
    Implementations can use heuristics (baseline) or ML models.

    Usage:
        runner = BaselineRunner(config)
        prediction = runner.predict(feature_snapshot)
    """

    def __init__(self, config: ModelRunnerConfig | None = None) -> None:
        """
        Initialize model runner.

        Args:
            config: Runner configuration. Uses defaults if not provided.
        """
        self._config = config or ModelRunnerConfig()
        self._metrics = RunnerMetrics()

    @property
    def config(self) -> ModelRunnerConfig:
        """Get runner configuration."""
        return self._config

    @property
    def metrics(self) -> RunnerMetrics:
        """Get runner metrics."""
        return self._metrics

    @property
    def model_version(self) -> str:
        """Get model version string."""
        return self._config.model_version

    @property
    def calibration_version(self) -> str:
        """Get calibration version string."""
        return self._config.calibration_version

    @abstractmethod
    def predict(self, snapshot: FeatureSnapshot) -> PredictionSnapshot:
        """
        Generate prediction from feature snapshot.

        Args:
            snapshot: FeatureSnapshot with features for a symbol.

        Returns:
            PredictionSnapshot with prediction and reasons.
        """

    def predict_batch(self, snapshots: list[FeatureSnapshot]) -> list[PredictionSnapshot]:
        """
        Generate predictions for multiple snapshots.

        Args:
            snapshots: List of FeatureSnapshots.

        Returns:
            List of PredictionSnapshots.
        """
        return [self.predict(s) for s in snapshots]

    def reset_metrics(self) -> None:
        """Reset runner metrics."""
        self._metrics.reset()

    def compute_digest(self) -> str:
        """
        Compute a digest of the model configuration for replay verification.

        Returns:
            SHA256 hex digest of model config.
        """
        config_str = (
            f"{self._config.model_version}:"
            f"{self._config.calibration_version}:"
            f"{self._config.default_profile}:"
            f"{self._config.toxic_threshold}:"
            f"{self._config.tradeable_threshold}:"
            f"{self._config.watch_threshold}:"
            f"{self._config.spread_max_bps}:"
            f"{self._config.impact_max_bps}"
        )
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
