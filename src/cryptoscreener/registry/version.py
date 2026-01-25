"""Model version parsing and validation.

Version string format per MODEL_REGISTRY_VERSIONING.md:
    {semver}+{git_sha}+{data_cutoff}+{train_hash}

Example:
    1.0.0+abc123def+20260125+a1b2c3d4
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVersion:
    """Parsed model version components.

    Attributes:
        semver: Semantic version (e.g., "1.0.0").
        git_sha: Git commit SHA (7-12 chars).
        data_cutoff: Data cutoff date (YYYYMMDD format).
        train_hash: Training run hash (8 chars).
        raw: Original version string.
    """

    semver: str
    git_sha: str
    data_cutoff: str
    train_hash: str
    raw: str

    def __str__(self) -> str:
        """Return the raw version string."""
        return self.raw

    @property
    def major(self) -> int:
        """Extract major version number."""
        parts = self.semver.split(".")
        return int(parts[0]) if parts else 0

    @property
    def minor(self) -> int:
        """Extract minor version number."""
        parts = self.semver.split(".")
        return int(parts[1]) if len(parts) > 1 else 0

    @property
    def patch(self) -> int:
        """Extract patch version number."""
        parts = self.semver.split(".")
        return int(parts[2]) if len(parts) > 2 else 0


# Regex for version string parsing
# Format: {semver}+{git_sha}+{data_cutoff}+{train_hash}
VERSION_PATTERN = re.compile(
    r"^"
    r"(?P<semver>\d+\.\d+\.\d+)"  # semver: X.Y.Z
    r"\+"
    r"(?P<git_sha>[a-f0-9]{7,12})"  # git_sha: 7-12 hex chars
    r"\+"
    r"(?P<data_cutoff>\d{8})"  # data_cutoff: YYYYMMDD
    r"\+"
    r"(?P<train_hash>[a-f0-9]{8})"  # train_hash: 8 hex chars
    r"$",
    re.IGNORECASE,
)

# Simpler version patterns for baseline/fallback
BASELINE_PATTERN = re.compile(
    r"^baseline-v(?P<semver>\d+\.\d+\.\d+)\+(?P<git_sha>[a-f0-9]{7,12})$",
    re.IGNORECASE,
)


def parse_model_version(version_str: str) -> ModelVersion:
    """Parse a model version string into components.

    Args:
        version_str: Version string in format {semver}+{git_sha}+{data_cutoff}+{train_hash}
                     or baseline-v{semver}+{git_sha} for baseline models.

    Returns:
        ModelVersion with parsed components.

    Raises:
        ValueError: If version string doesn't match expected format.
    """
    # Try full version format first
    match = VERSION_PATTERN.match(version_str)
    if match:
        return ModelVersion(
            semver=match.group("semver"),
            git_sha=match.group("git_sha"),
            data_cutoff=match.group("data_cutoff"),
            train_hash=match.group("train_hash"),
            raw=version_str,
        )

    # Try baseline format
    baseline_match = BASELINE_PATTERN.match(version_str)
    if baseline_match:
        return ModelVersion(
            semver=baseline_match.group("semver"),
            git_sha=baseline_match.group("git_sha"),
            data_cutoff="00000000",  # Placeholder for baseline
            train_hash="00000000",  # Placeholder for baseline
            raw=version_str,
        )

    msg = (
        f"Invalid version string: {version_str!r}. "
        f"Expected format: {{semver}}+{{git_sha}}+{{data_cutoff}}+{{train_hash}} "
        f"or baseline-v{{semver}}+{{git_sha}}"
    )
    raise ValueError(msg)
