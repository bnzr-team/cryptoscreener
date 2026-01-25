"""
Platt scaling (sigmoid) calibration.

Fits a logistic regression to map raw probabilities to calibrated ones:
    p_calibrated = 1 / (1 + exp(A * logit(p_raw) + B))

This is the standard approach for calibrating binary classifiers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class PlattCalibrator:
    """Platt scaling calibrator with learned A, B parameters.

    Transforms probabilities via:
        logit = log(p / (1 - p))
        calibrated = sigmoid(A * logit + B)

    Attributes:
        a: Slope parameter.
        b: Intercept parameter.
        head_name: Name of the prediction head this calibrates.
    """

    a: float
    b: float
    head_name: str

    def transform(self, p_raw: float) -> float:
        """Apply Platt scaling to a single probability.

        Args:
            p_raw: Raw probability in (0, 1). Values at boundaries
                   are clamped to avoid log(0).

        Returns:
            Calibrated probability in [0, 1].
        """
        # Clamp to avoid log(0) or log(inf)
        eps = 1e-7
        p = max(eps, min(1 - eps, p_raw))

        # Logit transform
        logit = math.log(p / (1 - p))

        # Apply linear transform and sigmoid
        z = self.a * logit + self.b
        return 1.0 / (1.0 + math.exp(-z))

    def transform_batch(self, probs: Sequence[float]) -> list[float]:
        """Apply Platt scaling to a batch of probabilities.

        Args:
            probs: Raw probabilities.

        Returns:
            List of calibrated probabilities.
        """
        return [self.transform(p) for p in probs]

    def to_dict(self) -> dict[str, float | str]:
        """Serialize to dictionary."""
        return {
            "a": self.a,
            "b": self.b,
            "head_name": self.head_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | str]) -> PlattCalibrator:
        """Deserialize from dictionary."""
        return cls(
            a=float(data["a"]),
            b=float(data["b"]),
            head_name=str(data["head_name"]),
        )


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def _logit(p: float, eps: float = 1e-7) -> float:
    """Logit transform with clamping."""
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


class NegativeSlopeError(ValueError):
    """Raised when fitted calibrator has negative slope (a <= 0).

    A negative slope means the model's probabilities are anti-correlated
    with actual outcomes. This would invert rankings and is catastrophic
    for downstream ranker/alerter systems.

    Possible causes:
    - Model is fundamentally broken (predicts opposite of reality)
    - Training data is corrupted or mislabeled
    - Validation window is too small/noisy

    Recommended actions:
    - Investigate model quality
    - Check for data issues
    - Use larger validation window
    """

    def __init__(self, head_name: str, a: float, b: float) -> None:
        self.head_name = head_name
        self.a = a
        self.b = b
        super().__init__(
            f"Calibrator for '{head_name}' has negative slope (a={a:.4f}). "
            "This would invert rankings. Investigate model/data quality."
        )


def fit_platt(
    y_true: Sequence[int],
    p_raw: Sequence[float],
    head_name: str,
    max_iter: int = 100,
    lr: float = 0.1,
    tol: float = 1e-6,
    *,
    reject_negative_slope: bool = True,
) -> PlattCalibrator:
    """Fit Platt scaling parameters using gradient descent.

    Minimizes cross-entropy loss:
        L = -sum(y * log(p_cal) + (1-y) * log(1 - p_cal))

    Uses Platt's original target adjustment for better calibration:
        t_i = (y_i * N+ + 1) / (N+ + 2)  for positive class
        t_i = 1 / (N- + 2)               for negative class

    Args:
        y_true: Binary labels (0 or 1).
        p_raw: Raw probabilities from model.
        head_name: Name of prediction head.
        max_iter: Maximum iterations for optimization.
        lr: Learning rate.
        tol: Convergence tolerance.
        reject_negative_slope: If True (default), raise NegativeSlopeError
            when fitted a <= 0. This prevents ranking inversion.

    Returns:
        Fitted PlattCalibrator.

    Raises:
        ValueError: If inputs are invalid.
        NegativeSlopeError: If fitted slope is negative and reject_negative_slope=True.
    """
    if len(y_true) != len(p_raw):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, p_raw={len(p_raw)}")

    if len(y_true) < 2:
        raise ValueError("Need at least 2 samples for calibration")

    n = len(y_true)
    n_pos = sum(y_true)
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative samples for calibration")

    # Platt's target adjustment
    t_pos = (n_pos + 1) / (n_pos + 2)
    t_neg = 1 / (n_neg + 2)
    targets = [t_pos if y == 1 else t_neg for y in y_true]

    # Convert to logits
    logits = [_logit(p) for p in p_raw]

    # Initialize parameters (identity transform: a=1, b=0)
    a = 1.0
    b = 0.0

    prev_loss = float("inf")

    for _ in range(max_iter):
        # Forward pass
        preds = [_sigmoid(a * logit + b) for logit in logits]

        # Compute loss (cross-entropy)
        loss = 0.0
        for t, p in zip(targets, preds, strict=True):
            eps = 1e-7
            p = max(eps, min(1 - eps, p))
            loss -= t * math.log(p) + (1 - t) * math.log(1 - p)
        loss /= n

        # Check convergence
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

        # Gradient descent
        grad_a = 0.0
        grad_b = 0.0
        for logit, t, p in zip(logits, targets, preds, strict=True):
            error = p - t
            grad_a += error * logit
            grad_b += error

        grad_a /= n
        grad_b /= n

        a -= lr * grad_a
        b -= lr * grad_b

    # CRITICAL: Reject negative slope to prevent ranking inversion
    if reject_negative_slope and a <= 0:
        raise NegativeSlopeError(head_name, a, b)

    return PlattCalibrator(a=a, b=b, head_name=head_name)
