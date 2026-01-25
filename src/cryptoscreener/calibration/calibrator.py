"""
High-level calibration interface.

Provides a unified API for fitting and applying calibrators
to model predictions.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cryptoscreener.calibration.platt import PlattCalibrator


class CalibratorMethod(str, Enum):
    """Supported calibration methods."""

    PLATT = "platt"
    # ISOTONIC = "isotonic"  # Future: isotonic regression


def fit_calibrator(
    y_true: Sequence[int],
    p_raw: Sequence[float],
    head_name: str,
    method: CalibratorMethod = CalibratorMethod.PLATT,
    **kwargs: Any,
) -> PlattCalibrator:
    """Fit a calibrator to data.

    Args:
        y_true: Binary labels (0 or 1).
        p_raw: Raw probabilities from model.
        head_name: Name of prediction head.
        method: Calibration method to use.
        **kwargs: Additional arguments for calibrator (max_iter, lr, tol).

    Returns:
        Fitted calibrator.

    Raises:
        ValueError: If method is not supported.
    """
    if method == CalibratorMethod.PLATT:
        from cryptoscreener.calibration.platt import fit_platt

        return fit_platt(y_true, p_raw, head_name, **kwargs)
    else:
        raise ValueError(f"Unsupported calibration method: {method}")
