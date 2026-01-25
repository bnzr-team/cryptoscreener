"""ModelPackage loader and validation.

Single entry point for loading model artifacts with fail-fast validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptoscreener.registry.manifest import (
    Manifest,
    load_manifest,
    validate_manifest,
)
from cryptoscreener.registry.version import ModelVersion, parse_model_version


class PackageError(Exception):
    """Base exception for package operations."""


class PackageValidationError(PackageError):
    """Raised when package validation fails."""


@dataclass
class FeatureSpec:
    """Feature specification from features.json.

    Attributes:
        features: Ordered list of feature names.
        version: Feature spec version.
    """

    features: list[str]
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "features": self.features,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSpec:
        """Create from dictionary."""
        return cls(
            features=data["features"],
            version=data.get("version", "1.0.0"),
        )


@dataclass
class SchemaVersion:
    """Schema version from schema_version.json.

    Attributes:
        schema_version: Package schema version.
        model_version: Model version string.
        compatible_schemas: List of compatible schema versions.
    """

    schema_version: str
    model_version: str
    compatible_schemas: list[str] = field(default_factory=list)

    def is_compatible(self, other_version: str) -> bool:
        """Check if this schema is compatible with another version."""
        if other_version == self.schema_version:
            return True
        return other_version in self.compatible_schemas

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "compatible_schemas": self.compatible_schemas,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaVersion:
        """Create from dictionary."""
        return cls(
            schema_version=data["schema_version"],
            model_version=data["model_version"],
            compatible_schemas=data.get("compatible_schemas", []),
        )


@dataclass
class ModelPackage:
    """Loaded model package with validated artifacts.

    Attributes:
        path: Path to package directory.
        manifest: Package manifest.
        schema: Schema version info.
        features: Feature specification.
        version: Parsed model version.
        calibrators: Dict of head name to calibrator path.
    """

    path: Path
    manifest: Manifest
    schema: SchemaVersion
    features: FeatureSpec
    version: ModelVersion
    calibrators: dict[str, Path] = field(default_factory=dict)

    @property
    def model_path(self) -> Path | None:
        """Get path to model.bin if it exists."""
        model_path = self.path / "model.bin"
        return model_path if model_path.exists() else None

    def get_calibrator_path(self, head: str) -> Path | None:
        """Get path to calibrator for a specific head."""
        return self.calibrators.get(head)

    def has_calibrator(self, head: str) -> bool:
        """Check if calibrator exists for a head."""
        return head in self.calibrators


def validate_package(
    package_dir: Path,
    *,
    expected_schema_version: str | None = None,
    expected_features: list[str] | None = None,
) -> list[str]:
    """Validate a model package directory.

    Performs comprehensive validation:
    1. Manifest exists and is valid
    2. All artifact hashes match
    3. schema_version.json is valid
    4. features.json is valid and ordered correctly (if expected_features provided)
    5. Schema version is compatible (if expected_schema_version provided)

    Args:
        package_dir: Path to package directory.
        expected_schema_version: Required schema version (optional).
        expected_features: Expected feature list in exact order (optional).

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []
    package_dir = Path(package_dir)

    if not package_dir.exists():
        errors.append(f"Package directory does not exist: {package_dir}")
        return errors

    if not package_dir.is_dir():
        errors.append(f"Package path is not a directory: {package_dir}")
        return errors

    # Load and validate manifest
    try:
        manifest = load_manifest(package_dir)
    except Exception as e:
        errors.append(f"Failed to load manifest: {e}")
        return errors

    # Validate manifest against files
    manifest_errors = validate_manifest(manifest, package_dir)
    errors.extend(manifest_errors)

    # Validate schema_version.json
    schema_path = package_dir / "schema_version.json"
    if schema_path.exists():
        try:
            with schema_path.open() as f:
                schema_data = json.load(f)
            schema = SchemaVersion.from_dict(schema_data)

            if expected_schema_version and not schema.is_compatible(expected_schema_version):
                errors.append(
                    f"Schema version {schema.schema_version} is not compatible "
                    f"with expected {expected_schema_version}"
                )
        except Exception as e:
            errors.append(f"Invalid schema_version.json: {e}")
    else:
        errors.append("Missing schema_version.json")

    # Validate features.json
    features_path = package_dir / "features.json"
    if features_path.exists():
        try:
            with features_path.open() as f:
                features_data = json.load(f)
            features = FeatureSpec.from_dict(features_data)

            if expected_features and features.features != expected_features:
                errors.append(
                    f"Feature mismatch: expected {expected_features}, got {features.features}"
                )
        except Exception as e:
            errors.append(f"Invalid features.json: {e}")
    else:
        errors.append("Missing features.json")

    return errors


def load_package(
    package_dir: Path,
    *,
    validate: bool = True,
    expected_schema_version: str | None = None,
    expected_features: list[str] | None = None,
) -> ModelPackage:
    """Load a model package with optional validation.

    This is the single entry point for loading model artifacts.

    Args:
        package_dir: Path to package directory.
        validate: Whether to validate the package (default: True).
        expected_schema_version: Required schema version (optional).
        expected_features: Expected feature list in exact order (optional).

    Returns:
        Loaded ModelPackage.

    Raises:
        PackageError: If package cannot be loaded.
        PackageValidationError: If validation fails (when validate=True).
    """
    package_dir = Path(package_dir)

    # Validate if requested
    if validate:
        errors = validate_package(
            package_dir,
            expected_schema_version=expected_schema_version,
            expected_features=expected_features,
        )
        if errors:
            msg = "Package validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise PackageValidationError(msg)

    # Load manifest
    try:
        manifest = load_manifest(package_dir)
    except Exception as e:
        msg = f"Failed to load manifest from {package_dir}: {e}"
        raise PackageError(msg) from e

    # Load schema_version.json
    schema_path = package_dir / "schema_version.json"
    try:
        with schema_path.open() as f:
            schema_data = json.load(f)
        schema = SchemaVersion.from_dict(schema_data)
    except Exception as e:
        msg = f"Failed to load schema_version.json: {e}"
        raise PackageError(msg) from e

    # Load features.json
    features_path = package_dir / "features.json"
    try:
        with features_path.open() as f:
            features_data = json.load(f)
        features = FeatureSpec.from_dict(features_data)
    except Exception as e:
        msg = f"Failed to load features.json: {e}"
        raise PackageError(msg) from e

    # Parse model version
    try:
        version = parse_model_version(schema.model_version)
    except ValueError:
        # Fallback for non-standard version strings
        version = ModelVersion(
            semver="0.0.0",
            git_sha="0000000",
            data_cutoff="00000000",
            train_hash="00000000",
            raw=schema.model_version,
        )

    # Find calibrators
    calibrators: dict[str, Path] = {}
    for artifact in manifest.artifacts:
        if artifact.name.startswith("calibrator_") and artifact.name.endswith(".pkl"):
            # Extract head name: calibrator_{head}.pkl -> head
            head = artifact.name[len("calibrator_") : -len(".pkl")]
            calibrators[head] = package_dir / artifact.name

    return ModelPackage(
        path=package_dir,
        manifest=manifest,
        schema=schema,
        features=features,
        version=version,
        calibrators=calibrators,
    )
