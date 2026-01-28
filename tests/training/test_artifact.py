"""Tests for model artifact packaging (DEC-038)."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np

from cryptoscreener.training.artifact import (
    ArtifactBuildResult,
    build_model_package,
    generate_model_version,
)
from cryptoscreener.training.feature_schema import (
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    compute_feature_hash,
)
from cryptoscreener.training.trainer import Trainer, TrainingConfig


def make_sample_rows(n: int = 100, seed: int = 42) -> list[dict]:
    """Generate sample data rows for testing."""
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(n):
        row = {
            "spread_bps": rng.uniform(1, 20),
            "mid": rng.uniform(40000, 50000),
            "book_imbalance": rng.uniform(-1, 1),
            "flow_imbalance": rng.uniform(-1, 1),
            "natr_14_5m": rng.uniform(0.5, 3.0),
            "impact_bps_q": rng.uniform(0, 15),
            "regime_vol_binary": float(rng.integers(0, 2)),
            "regime_trend_binary": float(rng.integers(0, 2)),
            "i_tradeable_30s_a": int(rng.random() > 0.7),
            "i_tradeable_2m_a": int(rng.random() > 0.6),
            "i_tradeable_5m_a": int(rng.random() > 0.5),
            "y_toxic": int(rng.random() > 0.9),
        }
        rows.append(row)

    return rows


class TestGenerateModelVersion:
    """Tests for generate_model_version()."""

    def test_version_format(self) -> None:
        """Version string matches expected format."""
        version = generate_model_version(major=1, minor=2, patch=3)

        # Format: 1.2.3+gitsha+YYYYMMDD+featurehash
        pattern = r"^\d+\.\d+\.\d+\+\w+\+\d{8}\+\w{8}$"
        assert re.match(pattern, version)

    def test_default_version(self) -> None:
        """Default version starts with 1.0.0."""
        version = generate_model_version()
        assert version.startswith("1.0.0+")

    def test_custom_version_numbers(self) -> None:
        """Custom major.minor.patch are included."""
        version = generate_model_version(major=2, minor=3, patch=4)
        assert version.startswith("2.3.4+")

    def test_custom_git_sha(self) -> None:
        """Custom git_sha is included."""
        version = generate_model_version(git_sha="abc1234")
        parts = version.split("+")
        assert parts[1] == "abc1234"

    def test_custom_feature_hash(self) -> None:
        """Custom feature_hash is truncated to 8 chars."""
        version = generate_model_version(feature_hash="abcdef1234567890")
        parts = version.split("+")
        assert parts[3] == "abcdef12"

    def test_uses_computed_feature_hash(self) -> None:
        """Default uses computed feature hash."""
        version = generate_model_version()
        parts = version.split("+")
        expected_hash = compute_feature_hash(FEATURE_ORDER)[:8]
        assert parts[3] == expected_hash


class TestBuildModelPackage:
    """Tests for build_model_package()."""

    def test_creates_required_files(self) -> None:
        """build_model_package creates all required files."""
        config = TrainingConfig(n_estimators=10)
        trainer = Trainer(config)
        rows = make_sample_rows(n=100)
        X, y = trainer.prepare_data(rows)
        model = trainer.train(X, y)
        metrics = trainer.evaluate(model, X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            build_model_package(
                output_dir=Path(tmpdir),
                model=model,
                config=config,
                metrics=metrics,
                train_samples=80,
                val_samples=20,
            )

            # Check all files exist
            assert (Path(tmpdir) / "model.pkl").exists()
            assert (Path(tmpdir) / "features.json").exists()
            assert (Path(tmpdir) / "schema_version.json").exists()
            assert (Path(tmpdir) / "checksums.txt").exists()
            assert (Path(tmpdir) / "manifest.json").exists()
            assert (Path(tmpdir) / "training_report.md").exists()

    def test_features_json_content(self) -> None:
        """features.json contains correct content."""
        config = TrainingConfig(n_estimators=10)
        trainer = Trainer(config)
        rows = make_sample_rows(n=100)
        X, y = trainer.prepare_data(rows)
        model = trainer.train(X, y)
        metrics = trainer.evaluate(model, X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            build_model_package(
                output_dir=Path(tmpdir),
                model=model,
                config=config,
                metrics=metrics,
            )

            features_path = Path(tmpdir) / "features.json"
            with features_path.open() as f:
                data = json.load(f)

            assert data["schema_version"] == FEATURE_SCHEMA_VERSION
            assert data["features"] == list(FEATURE_ORDER)
            assert len(data["feature_hash"]) == 16

    def test_manifest_json_content(self) -> None:
        """manifest.json contains correct content."""
        config = TrainingConfig(n_estimators=10)
        trainer = Trainer(config)
        rows = make_sample_rows(n=100)
        X, y = trainer.prepare_data(rows)
        model = trainer.train(X, y)
        metrics = trainer.evaluate(model, X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            build_model_package(
                output_dir=Path(tmpdir),
                model=model,
                config=config,
                metrics=metrics,
                train_samples=80,
                val_samples=20,
            )

            manifest_path = Path(tmpdir) / "manifest.json"
            with manifest_path.open() as f:
                data = json.load(f)

            assert "schema_version" in data
            assert "model_version" in data
            assert "created_at" in data
            assert "artifacts" in data
            assert len(data["artifacts"]) >= 5  # At least 5 files

            # Check metadata
            assert data["metadata"]["model_type"] == "random_forest"
            assert data["metadata"]["train_samples"] == 80
            assert data["metadata"]["val_samples"] == 20

    def test_checksums_txt_format(self) -> None:
        """checksums.txt has correct format."""
        config = TrainingConfig(n_estimators=10)
        trainer = Trainer(config)
        rows = make_sample_rows(n=100)
        X, y = trainer.prepare_data(rows)
        model = trainer.train(X, y)
        metrics = trainer.evaluate(model, X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            build_model_package(
                output_dir=Path(tmpdir),
                model=model,
                config=config,
                metrics=metrics,
            )

            checksums_path = Path(tmpdir) / "checksums.txt"
            content = checksums_path.read_text()

            # Should have header comments
            assert content.startswith("#")

            # Should have lines like "sha256  filename"
            lines = [line for line in content.split("\n") if line and not line.startswith("#")]
            assert len(lines) >= 4

            for line in lines:
                parts = line.split()
                assert len(parts) == 2
                sha256, filename = parts
                assert len(sha256) == 64  # SHA256 hex
                assert filename.endswith((".pkl", ".json", ".md", ".txt"))

    def test_training_report_content(self) -> None:
        """training_report.md contains expected sections."""
        config = TrainingConfig(n_estimators=10, model_type="random_forest")
        trainer = Trainer(config)
        rows = make_sample_rows(n=100)
        X, y = trainer.prepare_data(rows)
        model = trainer.train(X, y)
        metrics = trainer.evaluate(model, X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            build_model_package(
                output_dir=Path(tmpdir),
                model=model,
                config=config,
                metrics=metrics,
                train_samples=80,
                val_samples=20,
            )

            report_path = Path(tmpdir) / "training_report.md"
            content = report_path.read_text()

            assert "# Training Report" in content
            assert "## Configuration" in content
            assert "## Dataset" in content
            assert "## Feature Schema" in content
            assert "## Metrics by Head" in content
            assert "random_forest" in content
            assert "80" in content  # train samples
            assert "20" in content  # val samples

    def test_custom_model_version(self) -> None:
        """Custom model_version is used."""
        config = TrainingConfig(n_estimators=10)
        trainer = Trainer(config)
        rows = make_sample_rows(n=100)
        X, y = trainer.prepare_data(rows)
        model = trainer.train(X, y)
        metrics = trainer.evaluate(model, X, y)

        custom_version = "2.0.0+custom+20260101+abcd1234"

        with tempfile.TemporaryDirectory() as tmpdir:
            result = build_model_package(
                output_dir=Path(tmpdir),
                model=model,
                config=config,
                metrics=metrics,
                model_version=custom_version,
            )

            assert result.model_version == custom_version

            manifest_path = Path(tmpdir) / "manifest.json"
            with manifest_path.open() as f:
                data = json.load(f)
            assert data["model_version"] == custom_version

    def test_returns_artifact_build_result(self) -> None:
        """build_model_package returns ArtifactBuildResult."""
        config = TrainingConfig(n_estimators=10)
        trainer = Trainer(config)
        rows = make_sample_rows(n=100)
        X, y = trainer.prepare_data(rows)
        model = trainer.train(X, y)
        metrics = trainer.evaluate(model, X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = build_model_package(
                output_dir=Path(tmpdir),
                model=model,
                config=config,
                metrics=metrics,
            )

            assert isinstance(result, ArtifactBuildResult)
            assert result.output_dir == Path(tmpdir)
            assert result.model_version is not None
            assert result.manifest is not None
            assert "model.pkl" in result.checksums
            assert "features.json" in result.checksums

    def test_creates_output_dir_if_not_exists(self) -> None:
        """build_model_package creates output directory if needed."""
        config = TrainingConfig(n_estimators=10)
        trainer = Trainer(config)
        rows = make_sample_rows(n=100)
        X, y = trainer.prepare_data(rows)
        model = trainer.train(X, y)
        metrics = trainer.evaluate(model, X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "path" / "model"
            assert not nested_dir.exists()

            build_model_package(
                output_dir=nested_dir,
                model=model,
                config=config,
                metrics=metrics,
            )

            assert nested_dir.exists()
            assert (nested_dir / "model.pkl").exists()


class TestArtifactDeterminism:
    """Tests for artifact determinism."""

    def test_same_inputs_produce_same_checksums(self) -> None:
        """Same inputs produce same file checksums."""
        config = TrainingConfig(n_estimators=10, seed=42)

        # Train once
        trainer1 = Trainer(config)
        rows = make_sample_rows(n=100, seed=99)
        X1, y1 = trainer1.prepare_data(rows)
        model1 = trainer1.train(X1, y1)
        metrics1 = trainer1.evaluate(model1, X1, y1)

        # Train again with same seed
        trainer2 = Trainer(config)
        X2, y2 = trainer2.prepare_data(rows)
        model2 = trainer2.train(X2, y2)
        metrics2 = trainer2.evaluate(model2, X2, y2)

        # Build artifacts
        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            result1 = build_model_package(
                output_dir=Path(tmpdir1),
                model=model1,
                config=config,
                metrics=metrics1,
                model_version="1.0.0+test+20260101+12345678",
            )

            result2 = build_model_package(
                output_dir=Path(tmpdir2),
                model=model2,
                config=config,
                metrics=metrics2,
                model_version="1.0.0+test+20260101+12345678",
            )

            # model.pkl should have same hash
            assert result1.checksums["model.pkl"] == result2.checksums["model.pkl"]

            # features.json should have same hash
            assert result1.checksums["features.json"] == result2.checksums["features.json"]
