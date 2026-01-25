"""Tests for manifest management."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cryptoscreener.registry.manifest import (
    ArtifactEntry,
    Manifest,
    ManifestError,
    compute_file_sha256,
    generate_checksums_txt,
    load_manifest,
    parse_checksums_txt,
    save_manifest,
    validate_manifest,
)


class TestComputeFileSha256:
    """Tests for compute_file_sha256 function."""

    def test_computes_correct_hash(self, tmp_path: Path) -> None:
        """Computes correct SHA256 hash of file content."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        # Known SHA256 of "hello world"
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert compute_file_sha256(test_file) == expected

    def test_binary_file(self, tmp_path: Path) -> None:
        """Works with binary files."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        result = compute_file_sha256(test_file)
        assert len(result) == 64  # SHA256 hex is 64 chars

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            compute_file_sha256(tmp_path / "nonexistent.txt")

    def test_deterministic(self, tmp_path: Path) -> None:
        """Same content produces same hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("deterministic content")

        hash1 = compute_file_sha256(test_file)
        hash2 = compute_file_sha256(test_file)
        assert hash1 == hash2


class TestParseChecksumsTxt:
    """Tests for parse_checksums_txt function."""

    def test_parses_standard_format(self) -> None:
        """Parses standard checksums.txt format."""
        content = """abc123  model.bin
def456  features.json
"""
        result = parse_checksums_txt(content)

        assert result == {
            "model.bin": "abc123",
            "features.json": "def456",
        }

    def test_ignores_comments(self) -> None:
        """Ignores comment lines starting with #."""
        content = """# This is a comment
abc123  model.bin
# Another comment
def456  features.json
"""
        result = parse_checksums_txt(content)

        assert len(result) == 2
        assert "model.bin" in result

    def test_ignores_empty_lines(self) -> None:
        """Ignores empty lines."""
        content = """
abc123  model.bin

def456  features.json

"""
        result = parse_checksums_txt(content)

        assert len(result) == 2

    def test_normalizes_to_lowercase(self) -> None:
        """Normalizes hashes to lowercase."""
        content = "ABC123DEF  model.bin"
        result = parse_checksums_txt(content)

        assert result["model.bin"] == "abc123def"

    def test_handles_filenames_with_spaces(self) -> None:
        """Handles filenames that might have leading/trailing spaces."""
        content = "abc123   model.bin "
        result = parse_checksums_txt(content)

        assert "model.bin" in result


class TestGenerateChecksumsTxt:
    """Tests for generate_checksums_txt function."""

    def test_generates_valid_format(self) -> None:
        """Generates valid checksums.txt format."""
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="1.0.0+abc1234+20260125+12345678",
            created_at="2026-01-25T12:00:00Z",
            artifacts=[
                ArtifactEntry(name="model.bin", sha256="abc123", size_bytes=100),
                ArtifactEntry(name="features.json", sha256="def456", size_bytes=50),
            ],
        )

        result = generate_checksums_txt(manifest)

        assert "abc123  model.bin" in result
        assert "def456  features.json" in result

    def test_sorts_artifacts_by_name(self) -> None:
        """Sorts artifacts alphabetically by name."""
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25T12:00:00Z",
            artifacts=[
                ArtifactEntry(name="z_file.bin", sha256="zzz", size_bytes=1),
                ArtifactEntry(name="a_file.bin", sha256="aaa", size_bytes=1),
            ],
        )

        result = generate_checksums_txt(manifest)
        lines = [line for line in result.split("\n") if line and not line.startswith("#")]

        assert "a_file.bin" in lines[0]
        assert "z_file.bin" in lines[1]

    def test_includes_header_comments(self) -> None:
        """Includes header with version and timestamp."""
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test-version",
            created_at="2026-01-25T12:00:00Z",
            artifacts=[],
        )

        result = generate_checksums_txt(manifest)

        assert "test-version" in result
        assert "2026-01-25T12:00:00Z" in result


class TestManifest:
    """Tests for Manifest dataclass."""

    def test_get_artifact(self) -> None:
        """Gets artifact by name."""
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[
                ArtifactEntry(name="model.bin", sha256="abc", size_bytes=100),
            ],
        )

        artifact = manifest.get_artifact("model.bin")
        assert artifact is not None
        assert artifact.sha256 == "abc"

    def test_get_artifact_missing(self) -> None:
        """Returns None for missing artifact."""
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[],
        )

        assert manifest.get_artifact("nonexistent") is None

    def test_get_sha256(self) -> None:
        """Gets SHA256 for artifact."""
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[
                ArtifactEntry(name="model.bin", sha256="abc123", size_bytes=100),
            ],
        )

        assert manifest.get_sha256("model.bin") == "abc123"
        assert manifest.get_sha256("nonexistent") is None

    def test_artifact_names(self) -> None:
        """Gets set of artifact names."""
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[
                ArtifactEntry(name="model.bin", sha256="abc", size_bytes=100),
                ArtifactEntry(name="features.json", sha256="def", size_bytes=50),
            ],
        )

        assert manifest.artifact_names == {"model.bin", "features.json"}

    def test_to_dict_roundtrip(self) -> None:
        """Roundtrip through to_dict/from_dict."""
        original = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[
                ArtifactEntry(name="model.bin", sha256="abc", size_bytes=100),
            ],
            metadata={"key": "value"},
        )

        data = original.to_dict()
        restored = Manifest.from_dict(data)

        assert restored.schema_version == original.schema_version
        assert restored.model_version == original.model_version
        assert len(restored.artifacts) == len(original.artifacts)
        assert restored.metadata == original.metadata


class TestLoadManifest:
    """Tests for load_manifest function."""

    def test_loads_manifest_json(self, tmp_path: Path) -> None:
        """Loads manifest from manifest.json."""
        manifest_data = {
            "schema_version": "1.0.0",
            "model_version": "test",
            "created_at": "2026-01-25",
            "artifacts": [
                {"name": "model.bin", "sha256": "abc", "size_bytes": 100},
            ],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest_data))

        manifest = load_manifest(tmp_path)

        assert manifest.model_version == "test"
        assert len(manifest.artifacts) == 1

    def test_loads_from_checksums_txt_fallback(self, tmp_path: Path) -> None:
        """Falls back to checksums.txt when manifest.json missing."""
        # Create checksums.txt
        (tmp_path / "checksums.txt").write_text("abc123  model.bin\n")
        # Create model.bin for size calculation
        (tmp_path / "model.bin").write_bytes(b"test")
        # Create schema_version.json for model_version
        schema_data = {"schema_version": "1.0.0", "model_version": "from-schema"}
        (tmp_path / "schema_version.json").write_text(json.dumps(schema_data))

        manifest = load_manifest(tmp_path)

        assert manifest.model_version == "from-schema"
        assert len(manifest.artifacts) == 1

    def test_raises_when_no_manifest(self, tmp_path: Path) -> None:
        """Raises ManifestError when no manifest files exist."""
        with pytest.raises(ManifestError, match=r"No manifest\.json or checksums\.txt"):
            load_manifest(tmp_path)


class TestSaveManifest:
    """Tests for save_manifest function."""

    def test_saves_both_files(self, tmp_path: Path) -> None:
        """Saves both manifest.json and checksums.txt."""
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[
                ArtifactEntry(name="model.bin", sha256="abc", size_bytes=100),
            ],
        )

        save_manifest(manifest, tmp_path)

        assert (tmp_path / "manifest.json").exists()
        assert (tmp_path / "checksums.txt").exists()

    def test_manifest_json_is_valid(self, tmp_path: Path) -> None:
        """Saved manifest.json is valid JSON."""
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[],
        )

        save_manifest(manifest, tmp_path)

        with (tmp_path / "manifest.json").open() as f:
            data = json.load(f)

        assert data["model_version"] == "test"

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Save and load produces equivalent manifest."""
        original = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[
                ArtifactEntry(name="model.bin", sha256="abc", size_bytes=100),
            ],
        )

        save_manifest(original, tmp_path)
        loaded = load_manifest(tmp_path)

        assert loaded.model_version == original.model_version
        assert len(loaded.artifacts) == len(original.artifacts)


class TestValidateManifest:
    """Tests for validate_manifest function."""

    def test_valid_manifest_no_errors(self, tmp_path: Path) -> None:
        """Valid manifest with matching files returns no errors."""
        # Create required files
        (tmp_path / "schema_version.json").write_text('{"schema_version": "1.0.0"}')
        (tmp_path / "features.json").write_text('{"features": []}')
        (tmp_path / "checksums.txt").write_text("")

        # Compute actual hashes
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[
                ArtifactEntry(
                    name="schema_version.json",
                    sha256=compute_file_sha256(tmp_path / "schema_version.json"),
                    size_bytes=(tmp_path / "schema_version.json").stat().st_size,
                ),
                ArtifactEntry(
                    name="features.json",
                    sha256=compute_file_sha256(tmp_path / "features.json"),
                    size_bytes=(tmp_path / "features.json").stat().st_size,
                ),
                ArtifactEntry(
                    name="checksums.txt",
                    sha256=compute_file_sha256(tmp_path / "checksums.txt"),
                    size_bytes=(tmp_path / "checksums.txt").stat().st_size,
                ),
            ],
        )

        errors = validate_manifest(manifest, tmp_path)

        assert errors == []

    def test_missing_required_artifact(self, tmp_path: Path) -> None:
        """Reports missing required artifacts."""
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[],  # Missing required artifacts
        )

        errors = validate_manifest(manifest, tmp_path)

        assert any("Missing required artifact" in e for e in errors)

    def test_sha256_mismatch(self, tmp_path: Path) -> None:
        """Reports SHA256 mismatch."""
        (tmp_path / "test.txt").write_text("content")

        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[
                ArtifactEntry(name="test.txt", sha256="wrong_hash", size_bytes=7),
            ],
        )

        errors = validate_manifest(manifest, tmp_path)

        assert any("SHA256 mismatch" in e for e in errors)

    def test_size_mismatch(self, tmp_path: Path) -> None:
        """Reports size mismatch."""
        (tmp_path / "test.txt").write_text("content")
        actual_hash = compute_file_sha256(tmp_path / "test.txt")

        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[
                ArtifactEntry(name="test.txt", sha256=actual_hash, size_bytes=999),
            ],
        )

        errors = validate_manifest(manifest, tmp_path)

        assert any("Size mismatch" in e for e in errors)

    def test_missing_file(self, tmp_path: Path) -> None:
        """Reports missing artifact file."""
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[
                ArtifactEntry(name="nonexistent.bin", sha256="abc", size_bytes=100),
            ],
        )

        errors = validate_manifest(manifest, tmp_path)

        assert any("Artifact file missing" in e for e in errors)

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        """Rejects artifact names with path traversal."""
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[
                ArtifactEntry(name="../etc/passwd", sha256="abc", size_bytes=100),
            ],
        )

        errors = validate_manifest(manifest, tmp_path)

        assert any("path traversal" in e.lower() for e in errors)

    def test_rejects_path_with_slash(self, tmp_path: Path) -> None:
        """Rejects artifact names with forward slash."""
        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[
                ArtifactEntry(name="subdir/file.bin", sha256="abc", size_bytes=100),
            ],
        )

        errors = validate_manifest(manifest, tmp_path)

        assert any("path traversal" in e.lower() for e in errors)

    def test_rejects_symlinks(self, tmp_path: Path) -> None:
        """Rejects symlinked artifacts."""
        import os

        # Create a real file and a symlink to it
        real_file = tmp_path / "real.txt"
        real_file.write_text("real content")

        symlink = tmp_path / "linked.txt"
        os.symlink(real_file, symlink)

        manifest = Manifest(
            schema_version="1.0.0",
            model_version="test",
            created_at="2026-01-25",
            artifacts=[
                ArtifactEntry(
                    name="linked.txt",
                    sha256=compute_file_sha256(real_file),
                    size_bytes=real_file.stat().st_size,
                ),
            ],
        )

        errors = validate_manifest(manifest, tmp_path)

        assert any("symlink" in e.lower() for e in errors)
