"""Tests for model version parsing."""

from __future__ import annotations

import pytest

from cryptoscreener.registry.version import parse_model_version


class TestParseModelVersion:
    """Tests for parse_model_version function."""

    def test_full_version_string(self) -> None:
        """Parse full version string with all components."""
        version = parse_model_version("1.0.0+abc123def+20260125+a1b2c3d4")

        assert version.semver == "1.0.0"
        assert version.git_sha == "abc123def"
        assert version.data_cutoff == "20260125"
        assert version.train_hash == "a1b2c3d4"
        assert version.raw == "1.0.0+abc123def+20260125+a1b2c3d4"

    def test_version_with_7_char_sha(self) -> None:
        """Parse version with 7-character git SHA."""
        version = parse_model_version("2.1.3+abc1234+20260101+12345678")

        assert version.semver == "2.1.3"
        assert version.git_sha == "abc1234"

    def test_version_with_12_char_sha(self) -> None:
        """Parse version with 12-character git SHA."""
        version = parse_model_version("1.0.0+abcdef123456+20260125+a1b2c3d4")

        assert version.git_sha == "abcdef123456"

    def test_baseline_version(self) -> None:
        """Parse baseline version format."""
        version = parse_model_version("baseline-v1.0.0+abc1234")

        assert version.semver == "1.0.0"
        assert version.git_sha == "abc1234"
        assert version.data_cutoff == "00000000"  # Placeholder
        assert version.train_hash == "00000000"  # Placeholder

    def test_baseline_version_longer_sha(self) -> None:
        """Parse baseline version with longer SHA."""
        version = parse_model_version("baseline-v2.3.4+abcdef123456")

        assert version.semver == "2.3.4"
        assert version.git_sha == "abcdef123456"

    def test_invalid_version_raises(self) -> None:
        """Invalid version string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid version string"):
            parse_model_version("invalid")

    def test_missing_components_raises(self) -> None:
        """Version with missing components raises ValueError."""
        with pytest.raises(ValueError, match="Invalid version string"):
            parse_model_version("1.0.0+abc123")  # Missing data_cutoff and train_hash

    def test_invalid_semver_raises(self) -> None:
        """Invalid semver format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid version string"):
            parse_model_version("1.0+abc123def+20260125+a1b2c3d4")


class TestModelVersion:
    """Tests for ModelVersion dataclass."""

    def test_str_returns_raw(self) -> None:
        """__str__ returns raw version string."""
        version = parse_model_version("1.2.3+abc1234+20260125+a1b2c3d4")
        assert str(version) == "1.2.3+abc1234+20260125+a1b2c3d4"

    def test_major_version(self) -> None:
        """Extract major version number."""
        version = parse_model_version("2.3.4+abc1234+20260125+a1b2c3d4")
        assert version.major == 2

    def test_minor_version(self) -> None:
        """Extract minor version number."""
        version = parse_model_version("2.3.4+abc1234+20260125+a1b2c3d4")
        assert version.minor == 3

    def test_patch_version(self) -> None:
        """Extract patch version number."""
        version = parse_model_version("2.3.4+abc1234+20260125+a1b2c3d4")
        assert version.patch == 4

    def test_frozen_dataclass(self) -> None:
        """ModelVersion is immutable."""
        version = parse_model_version("1.0.0+abc1234+20260125+a1b2c3d4")
        with pytest.raises(AttributeError):
            version.semver = "2.0.0"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Two versions with same components are equal."""
        v1 = parse_model_version("1.0.0+abc1234+20260125+a1b2c3d4")
        v2 = parse_model_version("1.0.0+abc1234+20260125+a1b2c3d4")
        assert v1 == v2

    def test_inequality(self) -> None:
        """Two versions with different components are not equal."""
        v1 = parse_model_version("1.0.0+abc1234+20260125+a1b2c3d4")
        v2 = parse_model_version("1.0.1+abc1234+20260125+a1b2c3d4")
        assert v1 != v2
