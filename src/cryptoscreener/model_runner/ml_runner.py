"""ML model runner with calibration support.

Runs ML inference with optional probability calibration.
Falls back to baseline runner when model or calibrator unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime in MLRunnerConfig
from typing import TYPE_CHECKING

from cryptoscreener.calibration import (
    CalibrationArtifact,
    load_calibration_artifact,
)
from cryptoscreener.contracts.events import (
    ExecutionProfile,
    FeatureSnapshot,
    PredictionSnapshot,
    PredictionStatus,
    ReasonCode,
)
from cryptoscreener.model_runner.base import ModelRunner, ModelRunnerConfig

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class ModelArtifactError(Exception):
    """Raised when model artifact is invalid or unavailable."""


class CalibrationArtifactError(Exception):
    """Raised when calibration artifact is invalid or unavailable."""


@dataclass
class MLRunnerConfig(ModelRunnerConfig):
    """Configuration for MLRunner.

    Extends ModelRunnerConfig with model and calibration paths.

    Attributes:
        model_path: Path to model artifact (pickle/joblib/onnx).
        calibration_path: Path to calibration artifact JSON.
        require_calibration: If True, fail if calibration unavailable.
        fallback_to_baseline: If True, use baseline runner on failure.
    """

    model_path: Path | None = None
    calibration_path: Path | None = None
    require_calibration: bool = True
    fallback_to_baseline: bool = True


class MLRunner(ModelRunner):
    """
    ML model runner with calibration support.

    Loads model and calibration artifacts, runs inference,
    and applies probability calibration.

    Usage:
        config = MLRunnerConfig(
            model_path=Path("models/v1.0.0.onnx"),
            calibration_path=Path("calibration/v1.0.0.json"),
        )
        runner = MLRunner(config)
        prediction = runner.predict(feature_snapshot)

    Fallback behavior:
        - If model unavailable and fallback_to_baseline=True: uses BaselineRunner
        - If calibration unavailable and require_calibration=False: uses raw probs
        - If calibration unavailable and require_calibration=True: raises error
    """

    def __init__(self, config: MLRunnerConfig | None = None) -> None:
        """Initialize ML runner.

        Args:
            config: Runner configuration with model/calibration paths.

        Raises:
            ModelArtifactError: If model required but unavailable.
            CalibrationArtifactError: If calibration required but unavailable.
        """
        # Use MLRunnerConfig defaults if none provided
        self._ml_config = config or MLRunnerConfig()
        super().__init__(self._ml_config)

        self._model: object | None = None
        self._calibration: CalibrationArtifact | None = None
        self._using_fallback = False

        # Try to load model
        if self._ml_config.model_path:
            try:
                self._model = self._load_model(self._ml_config.model_path)
                logger.info(f"Loaded model from {self._ml_config.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                if not self._ml_config.fallback_to_baseline:
                    raise ModelArtifactError(f"Model unavailable: {e}") from e
                self._using_fallback = True
                logger.info("Falling back to baseline runner")

        # Try to load calibration
        if self._ml_config.calibration_path:
            try:
                self._calibration = load_calibration_artifact(self._ml_config.calibration_path)
                logger.info(f"Loaded calibration from {self._ml_config.calibration_path}")
                # Update calibration_version from artifact
                self._config.calibration_version = (
                    f"cal-{self._calibration.metadata.schema_version}"
                    f"+{self._calibration.metadata.git_sha[:7]}"
                )
            except FileNotFoundError as e:
                logger.warning(f"Calibration artifact not found: {e}")
                if self._ml_config.require_calibration:
                    raise CalibrationArtifactError(
                        f"Calibration required but unavailable: {e}"
                    ) from e
            except Exception as e:
                logger.warning(f"Failed to load calibration: {e}")
                if self._ml_config.require_calibration:
                    raise CalibrationArtifactError(f"Calibration artifact invalid: {e}") from e

        # Create fallback runner if needed
        self._fallback_runner: ModelRunner | None = None
        if self._using_fallback or self._model is None:
            from cryptoscreener.model_runner.baseline import BaselineRunner

            self._fallback_runner = BaselineRunner(self._ml_config)
            self._using_fallback = True

    @property
    def is_using_fallback(self) -> bool:
        """Check if runner is using baseline fallback."""
        return self._using_fallback

    @property
    def has_calibration(self) -> bool:
        """Check if calibration is loaded."""
        return self._calibration is not None

    @property
    def calibration_heads(self) -> list[str]:
        """Get list of calibrated prediction heads."""
        if self._calibration is None:
            return []
        return list(self._calibration.calibrators.keys())

    def _load_model(self, path: Path) -> object:
        """Load model artifact.

        Supports pickle, joblib, and ONNX formats.

        Args:
            path: Path to model file.

        Returns:
            Loaded model object.

        Raises:
            ModelArtifactError: If model format unknown or load fails.
        """
        if not path.exists():
            raise ModelArtifactError(f"Model file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".pkl":
            import pickle

            with path.open("rb") as f:
                return pickle.load(f)

        elif suffix == ".joblib":
            import joblib  # type: ignore[import-not-found]

            return joblib.load(path)

        elif suffix == ".onnx":
            # ONNX runtime inference
            import onnxruntime as ort  # type: ignore[import-not-found]

            return ort.InferenceSession(str(path))

        else:
            raise ModelArtifactError(f"Unknown model format: {suffix}")

    def predict(self, snapshot: FeatureSnapshot) -> PredictionSnapshot:
        """Generate prediction from feature snapshot.

        If using fallback, delegates to BaselineRunner.
        Otherwise runs ML inference and applies calibration.

        Args:
            snapshot: FeatureSnapshot with features for a symbol.

        Returns:
            PredictionSnapshot with prediction and reasons.
        """
        # Use fallback if model unavailable
        if self._using_fallback and self._fallback_runner is not None:
            prediction = self._fallback_runner.predict(snapshot)
            self._metrics.record_prediction(snapshot.symbol, prediction.status.value)
            return prediction

        # Run ML inference
        return self._predict_ml(snapshot)

    def _predict_ml(self, snapshot: FeatureSnapshot) -> PredictionSnapshot:
        """Run ML inference with calibration.

        Args:
            snapshot: FeatureSnapshot with features.

        Returns:
            PredictionSnapshot with calibrated probabilities.
        """
        features = snapshot.features

        # Check data health first
        if self._has_data_issues(snapshot):
            return self._make_data_issue_prediction(snapshot)

        # Extract feature vector for model
        feature_vector = self._extract_features(snapshot)

        # Run model inference to get raw probabilities
        raw_probs = self._run_inference(feature_vector)

        # Apply calibration if available
        p_inplay_30s = self._calibrate("p_inplay_30s", raw_probs["p_inplay_30s"])
        p_inplay_2m = self._calibrate("p_inplay_2m", raw_probs["p_inplay_2m"])
        p_inplay_5m = self._calibrate("p_inplay_5m", raw_probs["p_inplay_5m"])
        p_toxic = self._calibrate("p_toxic", raw_probs["p_toxic"])

        # Compute expected utility (using calibrated probabilities)
        expected_utility = self._compute_expected_utility(features, p_inplay_2m, p_toxic)

        # Check critical gates
        gate_failures = self._check_critical_gates(features)

        # Determine status
        status = self._determine_status(p_inplay_2m, p_toxic, gate_failures)

        # Build reasons
        reasons = self._build_reasons(features, p_inplay_2m, p_toxic, gate_failures, raw_probs)

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

    def _calibrate(self, head_name: str, raw_prob: float) -> float:
        """Apply calibration to a probability.

        Args:
            head_name: Prediction head name.
            raw_prob: Raw probability from model.

        Returns:
            Calibrated probability, or raw if no calibrator.
        """
        if self._calibration is None:
            return raw_prob

        try:
            return self._calibration.transform(head_name, raw_prob)
        except KeyError:
            # No calibrator for this head, use raw
            logger.debug(f"No calibrator for {head_name}, using raw probability")
            return raw_prob

    def _extract_features(self, snapshot: FeatureSnapshot) -> list[float]:
        """Extract feature vector from snapshot.

        Args:
            snapshot: FeatureSnapshot with features.

        Returns:
            Feature vector for model input.
        """
        f = snapshot.features

        # Standard feature order for model
        return [
            f.spread_bps,
            f.mid,
            f.book_imbalance,
            f.flow_imbalance,
            f.natr_14_5m,
            f.impact_bps_q,
            1.0 if f.regime_vol.value == "high" else 0.0,
            1.0 if f.regime_trend.value == "trend" else 0.0,
        ]

    def _run_inference(self, feature_vector: list[float]) -> dict[str, float]:
        """Run model inference.

        Args:
            feature_vector: Input features.

        Returns:
            Dictionary of raw probabilities.
        """
        if self._model is None:
            # Should not reach here if fallback is working
            raise ModelArtifactError("Model not loaded")

        # Check model type and run appropriate inference
        model_type = type(self._model).__name__

        if model_type == "InferenceSession":
            # ONNX runtime
            import numpy as np  # type: ignore[import-not-found]

            input_name = self._model.get_inputs()[0].name  # type: ignore[attr-defined]
            inputs = {input_name: np.array([feature_vector], dtype=np.float32)}
            outputs = self._model.run(None, inputs)  # type: ignore[attr-defined]

            # Assuming model outputs [p_inplay_30s, p_inplay_2m, p_inplay_5m, p_toxic]
            probs = outputs[0][0]
            return {
                "p_inplay_30s": float(probs[0]),
                "p_inplay_2m": float(probs[1]),
                "p_inplay_5m": float(probs[2]),
                "p_toxic": float(probs[3]),
            }

        elif hasattr(self._model, "predict_proba"):
            # Scikit-learn style model
            import numpy as np

            X = np.array([feature_vector])
            # Assuming multi-output classifier
            probs = self._model.predict_proba(X)
            if isinstance(probs, list):
                # Multi-output: list of arrays
                return {
                    "p_inplay_30s": float(probs[0][0][1]),
                    "p_inplay_2m": float(probs[1][0][1]),
                    "p_inplay_5m": float(probs[2][0][1]),
                    "p_toxic": float(probs[3][0][1]),
                }
            else:
                # Single output
                return {
                    "p_inplay_30s": float(probs[0][1]),
                    "p_inplay_2m": float(probs[0][1]),
                    "p_inplay_5m": float(probs[0][1]),
                    "p_toxic": 0.0,
                }

        else:
            raise ModelArtifactError(f"Unknown model type: {model_type}")

    def _has_data_issues(self, snapshot: FeatureSnapshot) -> bool:
        """Check if snapshot has data quality issues."""
        health = snapshot.data_health
        if health.stale_book_ms > 5000:
            return True
        if health.stale_trades_ms > 30000:
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

    def _compute_expected_utility(
        self,
        features: object,
        p_inplay: float,
        p_toxic: float,
    ) -> float:
        """Compute expected utility in basis points."""
        # Access features attributes
        natr = getattr(features, "natr_14_5m", 0.01)
        spread = getattr(features, "spread_bps", 5.0)
        impact = getattr(features, "impact_bps_q", 10.0)

        expected_gain_bps = natr * 100
        expected_loss_bps = spread + impact
        utility = p_inplay * expected_gain_bps * (1 - p_toxic) - p_toxic * expected_loss_bps

        return utility

    def _check_critical_gates(self, features: object) -> list[str]:
        """Check PRD critical gates for TRADEABLE status."""
        failures: list[str] = []

        spread = getattr(features, "spread_bps", 0.0)
        impact = getattr(features, "impact_bps_q", 0.0)

        if spread > self._config.spread_max_bps:
            failures.append("GATE_SPREAD")

        if impact > self._config.impact_max_bps:
            failures.append("GATE_IMPACT")

        return failures

    def _determine_status(
        self, p_inplay: float, p_toxic: float, gate_failures: list[str] | None = None
    ) -> PredictionStatus:
        """Determine prediction status from probabilities and gate checks."""
        gate_failures = gate_failures or []

        if p_toxic >= self._config.toxic_threshold:
            return PredictionStatus.TRAP

        if p_inplay >= self._config.tradeable_threshold:
            if gate_failures:
                return PredictionStatus.WATCH
            return PredictionStatus.TRADEABLE

        if p_inplay >= self._config.watch_threshold:
            return PredictionStatus.WATCH

        return PredictionStatus.DEAD

    def _build_reasons(
        self,
        features: object,
        p_inplay: float,
        p_toxic: float,
        gate_failures: list[str] | None = None,
        raw_probs: dict[str, float] | None = None,
    ) -> list[ReasonCode]:
        """Build reason codes explaining the prediction."""
        reasons: list[ReasonCode] = []
        gate_failures = gate_failures or []

        spread = getattr(features, "spread_bps", 0.0)
        impact = getattr(features, "impact_bps_q", 0.0)
        flow_imbalance = getattr(features, "flow_imbalance", 0.0)
        book_imbalance = getattr(features, "book_imbalance", 0.0)

        # Gate failure reasons
        if "GATE_SPREAD" in gate_failures:
            reasons.append(
                ReasonCode(
                    code="RC_GATE_SPREAD_FAIL",
                    value=round(spread, 2),
                    unit="bps",
                    evidence=f"Spread {spread:.1f} bps exceeds max {self._config.spread_max_bps:.1f} bps",
                )
            )

        if "GATE_IMPACT" in gate_failures:
            reasons.append(
                ReasonCode(
                    code="RC_GATE_IMPACT_FAIL",
                    value=round(impact, 2),
                    unit="bps",
                    evidence=f"Impact {impact:.1f} bps exceeds max {self._config.impact_max_bps:.1f} bps",
                )
            )

        # Flow imbalance
        if abs(flow_imbalance) > 0.3:
            direction = "buy" if flow_imbalance > 0 else "sell"
            reasons.append(
                ReasonCode(
                    code="RC_FLOW_SURGE",
                    value=round(flow_imbalance, 3),
                    unit="ratio",
                    evidence=f"Strong {direction} flow imbalance",
                )
            )

        # Book imbalance
        if abs(book_imbalance) > 0.3:
            side = "bid" if book_imbalance > 0 else "ask"
            reasons.append(
                ReasonCode(
                    code="RC_BOOK_PRESSURE",
                    value=round(book_imbalance, 3),
                    unit="ratio",
                    evidence=f"Strong {side}-side book pressure",
                )
            )

        # Calibration info (if raw_probs available)
        if raw_probs and self._calibration:
            # Show calibration adjustment for p_inplay_2m
            raw_2m = raw_probs.get("p_inplay_2m", p_inplay)
            if abs(raw_2m - p_inplay) > 0.01:
                reasons.append(
                    ReasonCode(
                        code="RC_CALIBRATION_ADJ",
                        value=round(p_inplay - raw_2m, 3),
                        unit="delta",
                        evidence=f"Calibration adjusted p_inplay_2m: {raw_2m:.3f} → {p_inplay:.3f}",
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

        return reasons

    def predict_batch(self, snapshots: Sequence[FeatureSnapshot]) -> list[PredictionSnapshot]:
        """Generate predictions for multiple snapshots.

        Args:
            snapshots: List of FeatureSnapshots.

        Returns:
            List of PredictionSnapshots.
        """
        return [self.predict(s) for s in snapshots]
