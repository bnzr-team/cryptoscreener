"""ML trainer for multi-output classification (DEC-038).

Trains sklearn models for 4 prediction heads:
- p_inplay_30s, p_inplay_2m, p_inplay_5m, p_toxic

Uses MultiOutputClassifier wrapper for joint training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from cryptoscreener.backtest.metrics import (
    compute_auc,
    compute_brier_score,
    compute_ece,
    compute_pr_auc,
)
from cryptoscreener.training.feature_schema import (
    FEATURE_ORDER,
    PREDICTION_HEADS,
    get_label_column,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Raised when training fails."""


@dataclass
class TrainingConfig:
    """Training configuration.

    Attributes:
        seed: Random seed for reproducibility.
        val_ratio: Fraction of data for validation (by time).
        model_type: Model type: random_forest or logistic.
        profile: Label profile to use (a or b).
        n_estimators: Number of trees for RandomForest.
        max_depth: Max tree depth for RandomForest.
        min_samples_split: Min samples to split for RandomForest.
        min_samples_leaf: Min samples per leaf for RandomForest.
        C: Regularization strength for Logistic (inverse).
        max_iter: Max iterations for Logistic.
    """

    seed: int = 42
    val_ratio: float = 0.2
    model_type: Literal["random_forest", "logistic"] = "random_forest"
    profile: str = "a"

    # RandomForest hyperparameters
    n_estimators: int = 100
    max_depth: int | None = 10
    min_samples_split: int = 10
    min_samples_leaf: int = 5

    # Logistic hyperparameters
    C: float = 1.0
    max_iter: int = 1000

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if not 0 < self.val_ratio < 1:
            raise ValueError(f"val_ratio must be in (0, 1), got {self.val_ratio}")
        if self.model_type not in ("random_forest", "logistic"):
            raise ValueError(f"Unknown model_type: {self.model_type}")
        if self.profile not in ("a", "b"):
            raise ValueError(f"profile must be 'a' or 'b', got {self.profile}")


@dataclass
class HeadMetrics:
    """Evaluation metrics for a single prediction head.

    Attributes:
        head_name: Prediction head name.
        auc: Area Under ROC Curve.
        pr_auc: Area Under Precision-Recall Curve.
        brier: Brier score (lower is better).
        ece: Expected Calibration Error (lower is better).
        n_samples: Number of samples.
        n_positives: Number of positive labels.
    """

    head_name: str
    auc: float
    pr_auc: float
    brier: float
    ece: float
    n_samples: int
    n_positives: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "head_name": self.head_name,
            "auc": round(self.auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "brier": round(self.brier, 4),
            "ece": round(self.ece, 4),
            "n_samples": self.n_samples,
            "n_positives": self.n_positives,
        }


@dataclass
class TrainingResult:
    """Result of model training.

    Attributes:
        model: Trained sklearn model.
        metrics: Evaluation metrics per head.
        config: Training configuration used.
        feature_order: Feature names in order.
        head_order: Prediction head names in order.
    """

    model: Any  # sklearn MultiOutputClassifier
    metrics: dict[str, HeadMetrics]
    config: TrainingConfig
    feature_order: tuple[str, ...] = FEATURE_ORDER
    head_order: tuple[str, ...] = PREDICTION_HEADS


class Trainer:
    """Multi-output classifier trainer.

    Usage:
        config = TrainingConfig(seed=42)
        trainer = Trainer(config)

        X_train, y_train = trainer.prepare_data(train_rows)
        X_val, y_val = trainer.prepare_data(val_rows)

        result = trainer.train_and_evaluate(X_train, y_train, X_val, y_val)
        model = result.model
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        """Initialize trainer.

        Args:
            config: Training configuration. Uses defaults if not provided.
        """
        self._config = config or TrainingConfig()
        self._set_determinism()

    @property
    def config(self) -> TrainingConfig:
        """Get training configuration."""
        return self._config

    def _set_determinism(self) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(self._config.seed)
        logger.info(f"Set random seed to {self._config.seed}")

    def prepare_data(
        self,
        rows: Sequence[dict[str, Any]],
        feature_order: tuple[str, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and label matrix from data rows.

        Args:
            rows: Data rows with feature and label columns.
            feature_order: Feature names in order. Uses FEATURE_ORDER by default.

        Returns:
            Tuple of (X, y) where:
            - X: Feature matrix of shape (n_samples, 8)
            - y: Label matrix of shape (n_samples, 4)

        Raises:
            TrainingError: If required columns are missing.
        """
        if feature_order is None:
            feature_order = FEATURE_ORDER

        if not rows:
            raise TrainingError("No data rows provided")

        n_samples = len(rows)
        n_features = len(feature_order)
        n_heads = len(PREDICTION_HEADS)

        X = np.zeros((n_samples, n_features), dtype=np.float64)
        y = np.zeros((n_samples, n_heads), dtype=np.int32)

        # Get label column names
        label_cols = [get_label_column(h, self._config.profile) for h in PREDICTION_HEADS]

        for i, row in enumerate(rows):
            # Extract features
            for j, feat in enumerate(feature_order):
                if feat not in row:
                    raise TrainingError(f"Row {i}: missing feature column '{feat}'")
                X[i, j] = float(row[feat])

            # Extract labels
            for k, label_col in enumerate(label_cols):
                if label_col not in row:
                    raise TrainingError(f"Row {i}: missing label column '{label_col}'")
                y[i, k] = int(row[label_col])

        logger.info(f"Prepared data: {n_samples} samples, {n_features} features, {n_heads} heads")

        return X, y

    def create_model(self) -> Any:
        """Create sklearn model based on configuration.

        Returns:
            MultiOutputClassifier wrapping base estimator.
        """
        from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
        from sklearn.multioutput import MultiOutputClassifier  # type: ignore[import-untyped]

        if self._config.model_type == "random_forest":
            base_estimator = RandomForestClassifier(
                n_estimators=self._config.n_estimators,
                max_depth=self._config.max_depth,
                min_samples_split=self._config.min_samples_split,
                min_samples_leaf=self._config.min_samples_leaf,
                random_state=self._config.seed,
                n_jobs=-1,  # Use all cores
            )
            logger.info(
                f"Created RandomForestClassifier with {self._config.n_estimators} trees, "
                f"max_depth={self._config.max_depth}"
            )
        else:  # logistic
            base_estimator = LogisticRegression(
                C=self._config.C,
                max_iter=self._config.max_iter,
                random_state=self._config.seed,
            )
            logger.info(f"Created LogisticRegression with C={self._config.C}")

        return MultiOutputClassifier(base_estimator)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train multi-output classifier.

        Args:
            X_train: Training features of shape (n_samples, 8).
            y_train: Training labels of shape (n_samples, 4).

        Returns:
            Trained MultiOutputClassifier.

        Raises:
            TrainingError: If training fails.
        """
        if X_train.shape[1] != len(FEATURE_ORDER):
            raise TrainingError(f"Expected {len(FEATURE_ORDER)} features, got {X_train.shape[1]}")
        if y_train.shape[1] != len(PREDICTION_HEADS):
            raise TrainingError(f"Expected {len(PREDICTION_HEADS)} heads, got {y_train.shape[1]}")

        model = self.create_model()

        logger.info(f"Training on {X_train.shape[0]} samples...")
        model.fit(X_train, y_train)
        logger.info("Training complete")

        return model

    def predict_proba(
        self,
        model: Any,
        X: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Get probability predictions for each head.

        Args:
            model: Trained MultiOutputClassifier.
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Dict mapping head name to probability array.
        """
        # MultiOutputClassifier.predict_proba returns list of arrays
        proba_list = model.predict_proba(X)

        result: dict[str, np.ndarray] = {}
        for i, head in enumerate(PREDICTION_HEADS):
            # Each array is (n_samples, n_classes), we want P(class=1)
            probs = proba_list[i]
            if probs.ndim == 2:
                result[head] = probs[:, 1]  # Probability of class 1
            else:
                result[head] = probs  # Already 1D
        return result

    def evaluate(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, HeadMetrics]:
        """Evaluate model and compute metrics per head.

        Args:
            model: Trained MultiOutputClassifier.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Dict mapping head name to HeadMetrics.
        """
        proba_dict = self.predict_proba(model, X_val)

        metrics: dict[str, HeadMetrics] = {}
        for i, head in enumerate(PREDICTION_HEADS):
            y_true = y_val[:, i].tolist()
            y_probs = proba_dict[head].tolist()

            n_samples = len(y_true)
            n_positives = sum(y_true)

            # Compute metrics using existing functions
            auc = compute_auc(y_true, y_probs)
            pr_auc = compute_pr_auc(y_true, y_probs)
            brier = compute_brier_score(y_true, y_probs)
            cal_metrics = compute_ece(y_true, y_probs)

            metrics[head] = HeadMetrics(
                head_name=head,
                auc=auc,
                pr_auc=pr_auc,
                brier=brier,
                ece=cal_metrics.ece,
                n_samples=n_samples,
                n_positives=n_positives,
            )

            logger.info(
                f"{head}: AUC={auc:.4f}, PR-AUC={pr_auc:.4f}, "
                f"Brier={brier:.4f}, ECE={cal_metrics.ece:.4f} "
                f"({n_positives}/{n_samples} positives)"
            )

        return metrics

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainingResult:
        """Train model and evaluate on validation set.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            TrainingResult with model and metrics.
        """
        model = self.train(X_train, y_train)
        metrics = self.evaluate(model, X_val, y_val)

        return TrainingResult(
            model=model,
            metrics=metrics,
            config=self._config,
            feature_order=FEATURE_ORDER,
            head_order=PREDICTION_HEADS,
        )
