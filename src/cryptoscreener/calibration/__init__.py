"""
Probability calibration for ML model outputs.

Provides Platt scaling calibrators to ensure predicted probabilities
match observed frequencies.

Per PRD §11 Milestone 3: "Training pipeline skeleton"
"""

from cryptoscreener.calibration.artifact import (
    CalibrationArtifact,
    CalibrationMetadata,
    load_calibration_artifact,
    save_calibration_artifact,
)
from cryptoscreener.calibration.calibrator import (
    CalibratorMethod,
    fit_calibrator,
)
from cryptoscreener.calibration.platt import PlattCalibrator

__all__ = [
    "CalibrationArtifact",
    "CalibrationMetadata",
    "CalibratorMethod",
    "PlattCalibrator",
    "fit_calibrator",
    "load_calibration_artifact",
    "save_calibration_artifact",
]
