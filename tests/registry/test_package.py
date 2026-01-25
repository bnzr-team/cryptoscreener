"""Tests for ModelPackage loader and validation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cryptoscreener.registry.manifest import (
    ArtifactEntry,
    Manifest,
    compute_file_sha256,
    save_manifest,
)
from cryptoscreener.registry.package import (
    FeatureSpec,
    ModelPackage,
    PackageValidationError,
    SchemaVersion,
    load_package,
    validate_package,
)


def create_valid_package(tmp_path: Path) -> Path:
    """Create a valid package directory with all required files.

    The checksums.txt file presents a chicken-and-egg problem: we need its hash
    in the manifest, but the manifest determines its content. Solution:
    1. Create all files except checksums.txt
    2. Generate checksums.txt with hashes for those files
    3. Calculate checksums.txt's own hash
    4. Create final manifest.json with all hashes including checksums.txt
    """
    # Create schema_version.json
    schema_data = {
        "schema_version": "1.0.0",
        "model_version": "1.0.0+abc1234+20260125+12345678",
        "compatible_schemas": [],
    }
    schema_path = tmp_path / "schema_version.json"
    schema_path.write_text(json.dumps(schema_data, indent=2))

    # Create features.json
    features_data = {
        "features": ["spread_bps", "book_imbalance", "flow_imbalance"],
        "version": "1.0.0",
    }
    features_path = tmp_path / "features.json"
    features_path.write_text(json.dumps(features_data, indent=2))

    # Create model.bin
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"fake model data")

    # Create calibrator
    calibrator_path = tmp_path / "calibrator_p_inplay_2m.pkl"
    calibrator_path.write_bytes(b"fake calibrator data")

    # Compute hashes for non-checksums files
    non_checksums_files = [schema_path, features_path, model_path, calibrator_path]
    artifacts = []
    for file_path in non_checksums_files:
        artifacts.append(
            ArtifactEntry(
                name=file_path.name,
                sha256=compute_file_sha256(file_path),
                size_bytes=file_path.stat().st_size,
            )
        )

    # Create a temporary manifest to generate checksums.txt content
    temp_manifest = Manifest(
        schema_version="1.0.0",
        model_version="1.0.0+abc1234+20260125+12345678",
        created_at="2026-01-25T12:00:00Z",
        artifacts=artifacts,
    )

    # Generate checksums.txt content and write it
    from cryptoscreener.registry.manifest import generate_checksums_txt

    checksums_content = generate_checksums_txt(temp_manifest)
    checksums_path = tmp_path / "checksums.txt"
    checksums_path.write_text(checksums_content)

    # Now add checksums.txt to artifacts with its actual hash
    artifacts.append(
        ArtifactEntry(
            name="checksums.txt",
            sha256=compute_file_sha256(checksums_path),
            size_bytes=checksums_path.stat().st_size,
        )
    )

    # Create final manifest with all artifacts
    final_manifest = Manifest(
        schema_version="1.0.0",
        model_version="1.0.0+abc1234+20260125+12345678",
        created_at="2026-01-25T12:00:00Z",
        artifacts=artifacts,
    )

    # Write manifest.json only (don't regenerate checksums.txt)
    manifest_path = tmp_path / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(final_manifest.to_dict(), f, indent=2)

    return tmp_path


class TestFeatureSpec:
    """Tests for FeatureSpec dataclass."""

    def test_from_dict(self) -> None:
        """Creates FeatureSpec from dictionary."""
        data = {
            "features": ["a", "b", "c"],
            "version": "1.0.0",
        }
        spec = FeatureSpec.from_dict(data)

        assert spec.features == ["a", "b", "c"]
        assert spec.version == "1.0.0"

    def test_from_dict_default_version(self) -> None:
        """Uses default version when not provided."""
        data = {"features": ["x"]}
        spec = FeatureSpec.from_dict(data)

        assert spec.version == "1.0.0"

    def test_to_dict_roundtrip(self) -> None:
        """Roundtrips through to_dict/from_dict."""
        original = FeatureSpec(features=["a", "b"], version="2.0.0")
        data = original.to_dict()
        restored = FeatureSpec.from_dict(data)

        assert restored.features == original.features
        assert restored.version == original.version


class TestSchemaVersion:
    """Tests for SchemaVersion dataclass."""

    def test_from_dict(self) -> None:
        """Creates SchemaVersion from dictionary."""
        data = {
            "schema_version": "1.0.0",
            "model_version": "test-version",
            "compatible_schemas": ["0.9.0"],
        }
        schema = SchemaVersion.from_dict(data)

        assert schema.schema_version == "1.0.0"
        assert schema.model_version == "test-version"
        assert schema.compatible_schemas == ["0.9.0"]

    def test_is_compatible_exact_match(self) -> None:
        """Exact version match is compatible."""
        schema = SchemaVersion(
            schema_version="1.0.0",
            model_version="test",
            compatible_schemas=[],
        )

        assert schema.is_compatible("1.0.0")

    def test_is_compatible_in_list(self) -> None:
        """Version in compatible list is compatible."""
        schema = SchemaVersion(
            schema_version="1.0.0",
            model_version="test",
            compatible_schemas=["0.9.0", "0.8.0"],
        )

        assert schema.is_compatible("0.9.0")
        assert schema.is_compatible("0.8.0")

    def test_is_compatible_not_in_list(self) -> None:
        """Version not in list is incompatible."""
        schema = SchemaVersion(
            schema_version="1.0.0",
            model_version="test",
            compatible_schemas=["0.9.0"],
        )

        assert not schema.is_compatible("0.7.0")

    def test_to_dict_roundtrip(self) -> None:
        """Roundtrips through to_dict/from_dict."""
        original = SchemaVersion(
            schema_version="1.0.0",
            model_version="test",
            compatible_schemas=["0.9.0"],
        )
        data = original.to_dict()
        restored = SchemaVersion.from_dict(data)

        assert restored.schema_version == original.schema_version
        assert restored.model_version == original.model_version
        assert restored.compatible_schemas == original.compatible_schemas


class TestValidatePackage:
    """Tests for validate_package function."""

    def test_valid_package_no_errors(self, tmp_path: Path) -> None:
        """Valid package returns no errors."""
        pkg_path = create_valid_package(tmp_path)
        errors = validate_package(pkg_path)

        assert errors == [], f"Unexpected errors: {errors}"

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Nonexistent directory returns error."""
        errors = validate_package(tmp_path / "nonexistent")

        assert len(errors) == 1
        assert "does not exist" in errors[0]

    def test_not_a_directory(self, tmp_path: Path) -> None:
        """File instead of directory returns error."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("not a directory")

        errors = validate_package(file_path)

        assert len(errors) == 1
        assert "not a directory" in errors[0]

    def test_missing_manifest(self, tmp_path: Path) -> None:
        """Missing manifest files returns error."""
        errors = validate_package(tmp_path)

        assert len(errors) >= 1
        assert any("manifest" in e.lower() for e in errors)

    def test_missing_schema_version(self, tmp_path: Path) -> None:
        """Missing schema_version.json returns error."""
        # Create minimal manifest without schema_version.json
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[],
        )
        save_manifest(manifest, tmp_path)

        errors = validate_package(tmp_path)

        assert any("Missing schema_version.json" in e for e in errors)

    def test_missing_features(self, tmp_path: Path) -> None:
        """Missing features.json returns error."""
        # Create schema_version.json only
        schema_data = {"schema_version": "1.0.0", "model_version": "test"}
        (tmp_path / "schema_version.json").write_text(json.dumps(schema_data))

        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[],
        )
        save_manifest(manifest, tmp_path)

        errors = validate_package(tmp_path)

        assert any("Missing features.json" in e for e in errors)

    def test_schema_version_mismatch(self, tmp_path: Path) -> None:
        """Schema version mismatch returns error."""
        pkg_path = create_valid_package(tmp_path)

        errors = validate_package(
            pkg_path,
            expected_schema_version="2.0.0",  # Package has 1.0.0
        )

        assert any("not compatible" in e for e in errors)

    def test_feature_list_mismatch(self, tmp_path: Path) -> None:
        """Feature list mismatch returns error."""
        pkg_path = create_valid_package(tmp_path)

        errors = validate_package(
            pkg_path,
            expected_features=["wrong", "features"],
        )

        assert any("Feature mismatch" in e for e in errors)


class TestLoadPackage:
    """Tests for load_package function."""

    def test_loads_valid_package(self, tmp_path: Path) -> None:
        """Loads valid package successfully."""
        pkg_path = create_valid_package(tmp_path)
        package = load_package(pkg_path)

        assert isinstance(package, ModelPackage)
        assert package.path == pkg_path
        assert package.schema.schema_version == "1.0.0"
        assert package.features.features == ["spread_bps", "book_imbalance", "flow_imbalance"]

    def test_parses_model_version(self, tmp_path: Path) -> None:
        """Parses model version correctly."""
        pkg_path = create_valid_package(tmp_path)
        package = load_package(pkg_path)

        assert package.version.semver == "1.0.0"
        assert package.version.git_sha == "abc1234"
        assert package.version.data_cutoff == "20260125"
        assert package.version.train_hash == "12345678"

    def test_finds_calibrators(self, tmp_path: Path) -> None:
        """Finds calibrator files."""
        pkg_path = create_valid_package(tmp_path)
        package = load_package(pkg_path)

        assert package.has_calibrator("p_inplay_2m")
        assert package.get_calibrator_path("p_inplay_2m") == pkg_path / "calibrator_p_inplay_2m.pkl"

    def test_model_path_property(self, tmp_path: Path) -> None:
        """model_path property returns correct path."""
        pkg_path = create_valid_package(tmp_path)
        package = load_package(pkg_path)

        assert package.model_path == pkg_path / "model.bin"

    def test_model_path_none_when_missing(self, tmp_path: Path) -> None:
        """model_path returns None when model.bin missing."""
        pkg_path = create_valid_package(tmp_path)
        (pkg_path / "model.bin").unlink()

        # Load without validation to allow missing model.bin
        package = load_package(pkg_path, validate=False)

        assert package.model_path is None

    def test_validation_failure_raises(self, tmp_path: Path) -> None:
        """Validation failure raises PackageValidationError."""
        # Create incomplete package
        schema_data = {"schema_version": "1.0.0", "model_version": "test"}
        (tmp_path / "schema_version.json").write_text(json.dumps(schema_data))

        features_data = {"features": ["a"]}
        (tmp_path / "features.json").write_text(json.dumps(features_data))

        (tmp_path / "checksums.txt").write_text("")

        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[],  # Missing required artifacts in manifest
        )
        save_manifest(manifest, tmp_path)

        with pytest.raises(PackageValidationError):
            load_package(tmp_path)

    def test_skip_validation(self, tmp_path: Path) -> None:
        """Can load package without validation."""
        # Create minimal package
        schema_data = {"schema_version": "1.0.0", "model_version": "test"}
        (tmp_path / "schema_version.json").write_text(json.dumps(schema_data))

        features_data = {"features": ["a"]}
        (tmp_path / "features.json").write_text(json.dumps(features_data))

        (tmp_path / "checksums.txt").write_text("")

        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[],
        )
        save_manifest(manifest, tmp_path)

        # Should not raise with validate=False
        package = load_package(tmp_path, validate=False)

        assert package.schema.model_version == "test"

    def test_missing_schema_file_raises(self, tmp_path: Path) -> None:
        """Missing schema_version.json raises PackageError."""
        # Create package without schema_version.json
        (tmp_path / "features.json").write_text('{"features": []}')
        (tmp_path / "checksums.txt").write_text("")

        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[],
        )
        save_manifest(manifest, tmp_path)

        with pytest.raises(PackageValidationError):
            load_package(tmp_path)

    def test_missing_features_file_raises(self, tmp_path: Path) -> None:
        """Missing features.json raises PackageError."""
        # Create package without features.json
        schema_data = {"schema_version": "1.0.0", "model_version": "test"}
        (tmp_path / "schema_version.json").write_text(json.dumps(schema_data))
        (tmp_path / "checksums.txt").write_text("")

        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[],
        )
        save_manifest(manifest, tmp_path)

        with pytest.raises(PackageValidationError):
            load_package(tmp_path)

    def test_invalid_model_version_fallback(self, tmp_path: Path) -> None:
        """Invalid model version falls back to default."""
        pkg_path = create_valid_package(tmp_path)

        # Update schema with invalid version
        schema_data = {
            "schema_version": "1.0.0",
            "model_version": "invalid-version-string",
        }
        (pkg_path / "schema_version.json").write_text(json.dumps(schema_data))

        # Recalculate hash and update manifest
        manifest_path = pkg_path / "manifest.json"
        with manifest_path.open() as f:
            manifest_data = json.load(f)

        for artifact in manifest_data["artifacts"]:
            if artifact["name"] == "schema_version.json":
                artifact["sha256"] = compute_file_sha256(pkg_path / "schema_version.json")
                artifact["size_bytes"] = (pkg_path / "schema_version.json").stat().st_size

        manifest_data["model_version"] = "invalid-version-string"
        with manifest_path.open("w") as f:
            json.dump(manifest_data, f, indent=2)

        package = load_package(pkg_path, validate=False)

        # Should fallback to default version
        assert package.version.semver == "0.0.0"
        assert package.version.raw == "invalid-version-string"


class TestLoadPackageIntegration:
    """Integration tests for the complete package loading flow."""

    def test_complete_roundtrip(self, tmp_path: Path) -> None:
        """Complete package creation and loading roundtrip."""
        # Create package
        pkg_path = create_valid_package(tmp_path)

        # Load and validate
        package = load_package(pkg_path)

        # Verify all components
        assert package.manifest.model_version == "1.0.0+abc1234+20260125+12345678"
        assert package.schema.schema_version == "1.0.0"
        assert len(package.features.features) == 3
        assert package.version.semver == "1.0.0"
        assert len(package.calibrators) == 1

    def test_validation_checks_hashes(self, tmp_path: Path) -> None:
        """Validation catches hash mismatches."""
        pkg_path = create_valid_package(tmp_path)

        # Corrupt a file after creation
        model_path = pkg_path / "model.bin"
        model_path.write_bytes(b"corrupted data")

        with pytest.raises(PackageValidationError) as exc_info:
            load_package(pkg_path)

        assert "SHA256 mismatch" in str(exc_info.value)

    def test_expected_schema_validation(self, tmp_path: Path) -> None:
        """Expected schema version is validated."""
        pkg_path = create_valid_package(tmp_path)

        # Should pass with correct schema
        package = load_package(pkg_path, expected_schema_version="1.0.0")
        assert package is not None

        # Should fail with wrong schema
        with pytest.raises(PackageValidationError):
            load_package(pkg_path, expected_schema_version="2.0.0")

    def test_expected_features_validation(self, tmp_path: Path) -> None:
        """Expected features list is validated."""
        pkg_path = create_valid_package(tmp_path)

        # Should pass with correct features
        package = load_package(
            pkg_path,
            expected_features=["spread_bps", "book_imbalance", "flow_imbalance"],
        )
        assert package is not None

        # Should fail with wrong features
        with pytest.raises(PackageValidationError):
            load_package(pkg_path, expected_features=["wrong", "features"])
