"""Feature schema for training (DEC-038).

Defines canonical feature ordering that must match MLRunner._extract_features().
Provides versioning and validation for feature compatibility.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# Feature schema version - increment on breaking changes
FEATURE_SCHEMA_VERSION = "1.0.0"

# Canonical feature order matching MLRunner._extract_features()
# This is the SSOT for feature ordering across training and inference
FEATURE_ORDER: tuple[str, ...] = (
    "spread_bps",
    "mid",
    "book_imbalance",
    "flow_imbalance",
    "natr_14_5m",
    "impact_bps_q",
    "regime_vol_binary",  # 1 if high, 0 otherwise
    "regime_trend_binary",  # 1 if trend, 0 otherwise
)

# Label columns for each prediction head
HEAD_TO_LABEL: dict[str, str] = {
    "p_inplay_30s": "i_tradeable_30s_{profile}",
    "p_inplay_2m": "i_tradeable_2m_{profile}",
    "p_inplay_5m": "i_tradeable_5m_{profile}",
    "p_toxic": "y_toxic",
}

# Prediction heads in order
PREDICTION_HEADS: tuple[str, ...] = (
    "p_inplay_30s",
    "p_inplay_2m",
    "p_inplay_5m",
    "p_toxic",
)


class FeatureSchemaError(Exception):
    """Raised when feature schema validation fails."""


def compute_feature_hash(features: list[str] | tuple[str, ...] | None = None) -> str:
    """Compute deterministic SHA256 hash of feature schema.

    The hash is used in model version strings to detect feature drift.

    Args:
        features: Feature names in order. Uses FEATURE_ORDER if not provided.

    Returns:
        First 16 chars of SHA256 hex digest.
    """
    if features is None:
        features = FEATURE_ORDER

    # Deterministic: join with newlines, encode as UTF-8
    content = "\n".join(features)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return digest[:16]


def save_features_json(
    path: Path,
    features: list[str] | tuple[str, ...] | None = None,
    schema_version: str | None = None,
) -> str:
    """Write features.json artifact.

    Args:
        path: Output path for features.json.
        features: Feature names in order. Uses FEATURE_ORDER if not provided.
        schema_version: Schema version. Uses FEATURE_SCHEMA_VERSION if not provided.

    Returns:
        Feature hash (for inclusion in version string).
    """
    if features is None:
        features = list(FEATURE_ORDER)
    if schema_version is None:
        schema_version = FEATURE_SCHEMA_VERSION

    feature_hash = compute_feature_hash(features)

    data: dict[str, Any] = {
        "schema_version": schema_version,
        "features": list(features),
        "feature_hash": feature_hash,
    }

    # Write with sorted keys for determinism
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")

    return feature_hash


def load_features_json(path: Path) -> dict[str, Any]:
    """Load features.json artifact.

    Args:
        path: Path to features.json.

    Returns:
        Dict with schema_version, features, feature_hash.

    Raises:
        FileNotFoundError: If file doesn't exist.
        FeatureSchemaError: If file is malformed.
    """
    with path.open() as f:
        data: dict[str, Any] = json.load(f)

    # Validate required fields
    required = {"schema_version", "features", "feature_hash"}
    missing = required - set(data.keys())
    if missing:
        raise FeatureSchemaError(f"features.json missing required fields: {missing}")

    if not isinstance(data["features"], list):
        raise FeatureSchemaError("features must be a list")

    return data


def validate_feature_compatibility(
    features_json_path: Path,
    expected_features: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Verify features.json matches expected feature order.

    This ensures a trained model is compatible with MLRunner.

    Args:
        features_json_path: Path to features.json.
        expected_features: Expected feature names. Uses FEATURE_ORDER if not provided.

    Raises:
        FeatureSchemaError: If features don't match.
    """
    if expected_features is None:
        expected_features = FEATURE_ORDER

    data = load_features_json(features_json_path)
    loaded_features = data["features"]

    # Check count
    if len(loaded_features) != len(expected_features):
        raise FeatureSchemaError(
            f"Feature count mismatch: expected {len(expected_features)}, got {len(loaded_features)}"
        )

    # Check order and names
    for i, (expected, actual) in enumerate(zip(expected_features, loaded_features, strict=True)):
        if expected != actual:
            raise FeatureSchemaError(
                f"Feature mismatch at index {i}: expected '{expected}', got '{actual}'"
            )

    # Verify hash integrity
    expected_hash = compute_feature_hash(expected_features)
    actual_hash = data["feature_hash"]
    if expected_hash != actual_hash:
        raise FeatureSchemaError(
            f"Feature hash mismatch: expected {expected_hash}, got {actual_hash}"
        )


def get_label_column(head_name: str, profile: str = "a") -> str:
    """Get label column name for a prediction head.

    Args:
        head_name: Prediction head name (e.g., "p_inplay_30s").
        profile: Execution profile ("a" or "b").

    Returns:
        Label column name (e.g., "i_tradeable_30s_a").

    Raises:
        ValueError: If head_name is unknown.
    """
    if head_name not in HEAD_TO_LABEL:
        raise ValueError(f"Unknown prediction head: {head_name}")

    template = HEAD_TO_LABEL[head_name]
    return template.format(profile=profile)
