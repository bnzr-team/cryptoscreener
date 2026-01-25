"""Manifest management for model artifacts.

Implements checksums.txt + manifest.json dual format per MODEL_REGISTRY_VERSIONING.md.

checksums.txt format (SSOT for hashes):
    sha256  filename
    abc123...  model.bin
    def456...  calibrator_p_inplay_2m.pkl

manifest.json format (metadata + checksums reference):
    {
        "schema_version": "1.0.0",
        "model_version": "1.0.0+abc123+20260125+a1b2c3d4",
        "created_at": "2026-01-25T12:00:00Z",
        "artifacts": [
            {"name": "model.bin", "sha256": "abc123...", "size_bytes": 12345},
            ...
        ]
    }
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


class ManifestError(Exception):
    """Base exception for manifest operations."""


class ManifestValidationError(ManifestError):
    """Raised when manifest validation fails."""


MANIFEST_SCHEMA_VERSION = "1.0.0"

# Required artifacts per MODEL_REGISTRY_VERSIONING.md
REQUIRED_ARTIFACTS = frozenset(
    {
        "schema_version.json",
        "features.json",
        "checksums.txt",
    }
)

# Optional but expected artifacts
OPTIONAL_ARTIFACTS = frozenset(
    {
        "model.bin",
        "training_report.md",
    }
)


@dataclass
class ArtifactEntry:
    """Single artifact entry in manifest.

    Attributes:
        name: Artifact filename.
        sha256: SHA256 hash of artifact content.
        size_bytes: File size in bytes.
    """

    name: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactEntry:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            sha256=data["sha256"],
            size_bytes=data["size_bytes"],
        )


@dataclass
class Manifest:
    """Model package manifest.

    Attributes:
        schema_version: Manifest schema version.
        model_version: Model version string.
        created_at: ISO 8601 timestamp.
        artifacts: List of artifact entries.
        metadata: Additional metadata (optional).
    """

    schema_version: str
    model_version: str
    created_at: str
    artifacts: list[ArtifactEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_artifact(self, name: str) -> ArtifactEntry | None:
        """Get artifact by name."""
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        return None

    def get_sha256(self, name: str) -> str | None:
        """Get SHA256 hash for an artifact."""
        artifact = self.get_artifact(name)
        return artifact.sha256 if artifact else None

    @property
    def artifact_names(self) -> set[str]:
        """Get set of all artifact names."""
        return {a.name for a in self.artifacts}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        """Create from dictionary."""
        return cls(
            schema_version=data["schema_version"],
            model_version=data["model_version"],
            created_at=data["created_at"],
            artifacts=[ArtifactEntry.from_dict(a) for a in data.get("artifacts", [])],
            metadata=data.get("metadata", {}),
        )


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        filepath: Path to file.

    Returns:
        Hex-encoded SHA256 hash (64 chars).

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_checksums_txt(content: str) -> dict[str, str]:
    """Parse checksums.txt content into filename -> sha256 mapping.

    Format: sha256  filename (two spaces between hash and name)

    Args:
        content: checksums.txt file content.

    Returns:
        Dict mapping filename to sha256 hash.
    """
    checksums: dict[str, str] = {}
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: sha256  filename (two spaces)
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            sha256, filename = parts
            checksums[filename.strip()] = sha256.strip().lower()
    return checksums


def generate_checksums_txt(manifest: Manifest) -> str:
    """Generate checksums.txt content from manifest.

    Args:
        manifest: Manifest with artifacts.

    Returns:
        checksums.txt formatted string.
    """
    lines = [f"# checksums.txt - SHA256 hashes for {manifest.model_version}"]
    lines.append(f"# Generated: {manifest.created_at}")
    lines.append("")
    for artifact in sorted(manifest.artifacts, key=lambda a: a.name):
        lines.append(f"{artifact.sha256}  {artifact.name}")
    return "\n".join(lines) + "\n"


def load_manifest(package_dir: Path) -> Manifest:
    """Load manifest from package directory.

    Tries manifest.json first, falls back to generating from checksums.txt.

    Args:
        package_dir: Path to package directory.

    Returns:
        Loaded Manifest.

    Raises:
        ManifestError: If neither manifest.json nor checksums.txt exists.
    """
    manifest_path = package_dir / "manifest.json"
    checksums_path = package_dir / "checksums.txt"

    if manifest_path.exists():
        with manifest_path.open() as f:
            data = json.load(f)
        return Manifest.from_dict(data)

    if checksums_path.exists():
        # Generate manifest from checksums.txt
        with checksums_path.open() as f:
            checksums = parse_checksums_txt(f.read())

        # Try to get model version from schema_version.json
        model_version = "unknown"
        schema_path = package_dir / "schema_version.json"
        if schema_path.exists():
            with schema_path.open() as f:
                schema_data = json.load(f)
                model_version = schema_data.get("model_version", "unknown")

        artifacts = []
        for filename, sha256 in checksums.items():
            filepath = package_dir / filename
            size_bytes = filepath.stat().st_size if filepath.exists() else 0
            artifacts.append(
                ArtifactEntry(
                    name=filename,
                    sha256=sha256,
                    size_bytes=size_bytes,
                )
            )

        return Manifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            model_version=model_version,
            created_at=datetime.now(UTC).isoformat(),
            artifacts=artifacts,
        )

    msg = f"No manifest.json or checksums.txt found in {package_dir}"
    raise ManifestError(msg)


def save_manifest(manifest: Manifest, package_dir: Path) -> None:
    """Save manifest to package directory.

    Writes both manifest.json and checksums.txt for compatibility.

    Args:
        manifest: Manifest to save.
        package_dir: Path to package directory.
    """
    package_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest.json
    manifest_path = package_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    # Save checksums.txt
    checksums_path = package_dir / "checksums.txt"
    with checksums_path.open("w") as f:
        f.write(generate_checksums_txt(manifest))


def validate_manifest(manifest: Manifest, package_dir: Path) -> list[str]:
    """Validate manifest against actual files.

    Checks:
    1. Required artifacts exist
    2. Artifact names are safe (no path traversal)
    3. Files are not symlinks (security)
    4. SHA256 hashes match
    5. File sizes match

    Args:
        manifest: Manifest to validate.
        package_dir: Path to package directory.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    # Check required artifacts
    missing = REQUIRED_ARTIFACTS - manifest.artifact_names
    for name in missing:
        errors.append(f"Missing required artifact: {name}")

    # Resolve package_dir for path containment checks
    package_dir_resolved = package_dir.resolve()

    # Validate each artifact
    for artifact in manifest.artifacts:
        # Security: Validate artifact name using PurePath
        from pathlib import PurePath

        pure_name = PurePath(artifact.name)

        # Reject empty names
        if not artifact.name or artifact.name.isspace():
            errors.append(f"Invalid artifact name (empty or whitespace): {artifact.name!r}")
            continue

        # Reject . and .. (check before parts check since PurePath('.').parts == ())
        if artifact.name in (".", ".."):
            errors.append(f"Invalid artifact name (special directory): {artifact.name}")
            continue

        # Reject absolute paths
        if pure_name.is_absolute():
            errors.append(f"Invalid artifact name (absolute path): {artifact.name}")
            continue

        # Reject paths with subdirectories or .. components
        # A valid artifact name should have exactly 1 part (just the filename)
        if len(pure_name.parts) != 1 or ".." in pure_name.parts:
            errors.append(f"Invalid artifact name (contains path separator): {artifact.name}")
            continue

        filepath = package_dir / artifact.name

        # Security: Verify file is inside package directory using relative_to()
        # This is immune to prefix collision attacks (e.g., /pkg/dir vs /pkg/dir_evil)
        try:
            filepath_resolved = filepath.resolve()
            filepath_resolved.relative_to(package_dir_resolved)
        except ValueError:
            errors.append(f"Artifact path escapes package directory: {artifact.name}")
            continue
        except OSError:
            errors.append(f"Cannot resolve artifact path: {artifact.name}")
            continue

        if not filepath.exists():
            errors.append(f"Artifact file missing: {artifact.name}")
            continue

        # Security: Reject symlinks
        if filepath.is_symlink():
            errors.append(f"Artifact is a symlink (not allowed): {artifact.name}")
            continue

        # Check SHA256
        actual_sha256 = compute_file_sha256(filepath)
        if actual_sha256 != artifact.sha256:
            errors.append(
                f"SHA256 mismatch for {artifact.name}: "
                f"expected {artifact.sha256}, got {actual_sha256}"
            )

        # Check size
        actual_size = filepath.stat().st_size
        if actual_size != artifact.size_bytes:
            errors.append(
                f"Size mismatch for {artifact.name}: "
                f"expected {artifact.size_bytes}, got {actual_size}"
            )

    return errors
