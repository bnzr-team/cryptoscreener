"""
Calibration artifact storage and metadata.

Provides serialization, hashing, and provenance tracking for
calibration models to ensure reproducibility.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING, Any

import orjson

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cryptoscreener.calibration.platt import PlattCalibrator


# Schema version for calibration artifacts
CALIBRATION_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class CalibrationMetadata:
    """Metadata for a calibration artifact.

    Attributes:
        schema_version: Version of the calibration schema.
        git_sha: Git commit SHA at calibration time.
        config_hash: SHA256 hash of calibration configuration.
        data_hash: SHA256 hash of calibration data (val set).
        calibration_timestamp: When calibration was performed.
        method: Calibration method used (e.g., "platt", "isotonic").
        heads: List of prediction heads calibrated.
        n_samples: Number of samples used for calibration.
        metrics_before: Brier/ECE before calibration per head.
        metrics_after: Brier/ECE after calibration per head.
    """

    schema_version: str
    git_sha: str
    config_hash: str
    data_hash: str
    calibration_timestamp: str
    method: str
    heads: list[str]
    n_samples: int
    metrics_before: dict[str, dict[str, float]] = field(default_factory=dict)
    metrics_after: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "schema_version": self.schema_version,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "calibration_timestamp": self.calibration_timestamp,
            "method": self.method,
            "heads": self.heads,
            "n_samples": self.n_samples,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationMetadata:
        """Deserialize from dictionary."""
        return cls(
            schema_version=data["schema_version"],
            git_sha=data["git_sha"],
            config_hash=data["config_hash"],
            data_hash=data["data_hash"],
            calibration_timestamp=data["calibration_timestamp"],
            method=data["method"],
            heads=data["heads"],
            n_samples=data["n_samples"],
            metrics_before=data.get("metrics_before", {}),
            metrics_after=data.get("metrics_after", {}),
        )


@dataclass
class CalibrationArtifact:
    """Complete calibration artifact with calibrators and metadata.

    Attributes:
        calibrators: Dictionary mapping head names to calibrators.
        metadata: Calibration metadata.
    """

    calibrators: dict[str, PlattCalibrator]
    metadata: CalibrationMetadata

    def transform(self, head_name: str, p_raw: float) -> float:
        """Apply calibration to a probability.

        Args:
            head_name: Name of prediction head.
            p_raw: Raw probability.

        Returns:
            Calibrated probability.

        Raises:
            KeyError: If head_name not in calibrators.
        """
        if head_name not in self.calibrators:
            raise KeyError(f"No calibrator for head: {head_name}")
        return self.calibrators[head_name].transform(p_raw)

    def transform_batch(self, head_name: str, probs: Sequence[float]) -> list[float]:
        """Apply calibration to a batch of probabilities.

        Args:
            head_name: Name of prediction head.
            probs: Raw probabilities.

        Returns:
            List of calibrated probabilities.
        """
        if head_name not in self.calibrators:
            raise KeyError(f"No calibrator for head: {head_name}")
        return self.calibrators[head_name].transform_batch(probs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "calibrators": {name: cal.to_dict() for name, cal in self.calibrators.items()},
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationArtifact:
        """Deserialize from dictionary."""
        from cryptoscreener.calibration.platt import PlattCalibrator

        calibrators = {
            name: PlattCalibrator.from_dict(cal_data)
            for name, cal_data in data["calibrators"].items()
        }
        metadata = CalibrationMetadata.from_dict(data["metadata"])
        return cls(calibrators=calibrators, metadata=metadata)


def _get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def _compute_data_hash(data: Sequence[dict[str, Any]]) -> str:
    """Compute SHA256 hash of data rows."""
    hasher = hashlib.sha256()
    for row in data:
        hasher.update(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
    return hasher.hexdigest()[:16]


def _compute_config_hash(config: dict[str, Any]) -> str:
    """Compute SHA256 hash of configuration."""
    data = orjson.dumps(config, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(data).hexdigest()[:16]


def save_calibration_artifact(
    artifact: CalibrationArtifact,
    output_path: Path,
) -> None:
    """Save calibration artifact to JSON file.

    Args:
        artifact: Calibration artifact to save.
        output_path: Path to output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(orjson.dumps(artifact.to_dict(), option=orjson.OPT_INDENT_2))


def load_calibration_artifact(input_path: Path) -> CalibrationArtifact:
    """Load calibration artifact from JSON file.

    Args:
        input_path: Path to input file.

    Returns:
        Loaded CalibrationArtifact.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file is invalid.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Calibration artifact not found: {input_path}")

    with input_path.open("rb") as f:
        data = orjson.loads(f.read())

    return CalibrationArtifact.from_dict(data)


def create_calibration_metadata(
    method: str,
    heads: list[str],
    n_samples: int,
    config: dict[str, Any],
    val_data: Sequence[dict[str, Any]],
    metrics_before: dict[str, dict[str, float]] | None = None,
    metrics_after: dict[str, dict[str, float]] | None = None,
) -> CalibrationMetadata:
    """Create calibration metadata with hashes and provenance.

    Args:
        method: Calibration method (e.g., "platt").
        heads: List of head names being calibrated.
        n_samples: Number of samples in calibration set.
        config: Calibration configuration dictionary.
        val_data: Validation data used for calibration.
        metrics_before: Optional metrics before calibration.
        metrics_after: Optional metrics after calibration.

    Returns:
        CalibrationMetadata with computed hashes.
    """
    return CalibrationMetadata(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        git_sha=_get_git_sha(),
        config_hash=_compute_config_hash(config),
        data_hash=_compute_data_hash(val_data),
        calibration_timestamp=datetime.now(UTC).isoformat(),
        method=method,
        heads=heads,
        n_samples=n_samples,
        metrics_before=metrics_before or {},
        metrics_after=metrics_after or {},
    )
