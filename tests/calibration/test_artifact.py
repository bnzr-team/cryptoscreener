"""Tests for calibration artifact storage and roundtrip."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cryptoscreener.calibration.artifact import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationArtifact,
    CalibrationMetadata,
    create_calibration_metadata,
    load_calibration_artifact,
    save_calibration_artifact,
)
from cryptoscreener.calibration.platt import PlattCalibrator


class TestCalibrationMetadata:
    """Tests for CalibrationMetadata."""

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """Metadata should survive serialization roundtrip."""
        metadata = CalibrationMetadata(
            schema_version="1.0.0",
            git_sha="abc123def456",
            config_hash="config123456789",
            data_hash="data1234567890",
            calibration_timestamp="2026-01-25T12:00:00+00:00",
            method="platt",
            heads=["p_inplay_30s", "p_toxic"],
            n_samples=1000,
            metrics_before={
                "p_inplay_30s": {"brier": 0.25, "ece": 0.15},
                "p_toxic": {"brier": 0.20, "ece": 0.10},
            },
            metrics_after={
                "p_inplay_30s": {"brier": 0.22, "ece": 0.08},
                "p_toxic": {"brier": 0.18, "ece": 0.05},
            },
        )

        data = metadata.to_dict()
        restored = CalibrationMetadata.from_dict(data)

        assert restored.schema_version == metadata.schema_version
        assert restored.git_sha == metadata.git_sha
        assert restored.config_hash == metadata.config_hash
        assert restored.data_hash == metadata.data_hash
        assert restored.method == metadata.method
        assert restored.heads == metadata.heads
        assert restored.n_samples == metadata.n_samples
        assert restored.metrics_before == metadata.metrics_before
        assert restored.metrics_after == metadata.metrics_after

    def test_create_metadata_computes_hashes(self) -> None:
        """create_calibration_metadata should compute valid hashes."""
        config = {"method": "platt", "max_iter": 100}
        val_data = [
            {"ts": 1000, "p_inplay_30s": 0.5, "i_tradeable_30s_a": 1},
            {"ts": 2000, "p_inplay_30s": 0.3, "i_tradeable_30s_a": 0},
        ]

        metadata = create_calibration_metadata(
            method="platt",
            heads=["p_inplay_30s"],
            n_samples=2,
            config=config,
            val_data=val_data,
        )

        assert metadata.schema_version == CALIBRATION_SCHEMA_VERSION
        assert len(metadata.config_hash) == 16
        assert len(metadata.data_hash) == 16
        assert metadata.method == "platt"
        assert metadata.heads == ["p_inplay_30s"]
        assert metadata.n_samples == 2

    def test_different_data_different_hash(self) -> None:
        """Different data should produce different hashes."""
        config = {"method": "platt"}
        data1 = [{"ts": 1000, "value": 1}]
        data2 = [{"ts": 2000, "value": 2}]

        meta1 = create_calibration_metadata("platt", ["test"], 1, config, data1)
        meta2 = create_calibration_metadata("platt", ["test"], 1, config, data2)

        assert meta1.data_hash != meta2.data_hash

    def test_different_config_different_hash(self) -> None:
        """Different config should produce different hashes."""
        data = [{"ts": 1000, "value": 1}]
        config1 = {"method": "platt", "max_iter": 100}
        config2 = {"method": "platt", "max_iter": 200}

        meta1 = create_calibration_metadata("platt", ["test"], 1, config1, data)
        meta2 = create_calibration_metadata("platt", ["test"], 1, config2, data)

        assert meta1.config_hash != meta2.config_hash


class TestCalibrationArtifact:
    """Tests for CalibrationArtifact."""

    def test_transform_applies_correct_calibrator(self) -> None:
        """transform should use the right calibrator for each head."""
        cal1 = PlattCalibrator(a=1.0, b=0.0, head_name="head1")
        cal2 = PlattCalibrator(a=2.0, b=1.0, head_name="head2")

        metadata = CalibrationMetadata(
            schema_version="1.0.0",
            git_sha="test",
            config_hash="test",
            data_hash="test",
            calibration_timestamp="2026-01-25T12:00:00+00:00",
            method="platt",
            heads=["head1", "head2"],
            n_samples=100,
        )

        artifact = CalibrationArtifact(
            calibrators={"head1": cal1, "head2": cal2},
            metadata=metadata,
        )

        # Results should differ because calibrators differ
        p = 0.7
        result1 = artifact.transform("head1", p)
        result2 = artifact.transform("head2", p)

        assert result1 != result2
        assert result1 == cal1.transform(p)
        assert result2 == cal2.transform(p)

    def test_transform_unknown_head_raises(self) -> None:
        """transform with unknown head should raise KeyError."""
        cal = PlattCalibrator(a=1.0, b=0.0, head_name="known")
        metadata = CalibrationMetadata(
            schema_version="1.0.0",
            git_sha="test",
            config_hash="test",
            data_hash="test",
            calibration_timestamp="2026-01-25T12:00:00+00:00",
            method="platt",
            heads=["known"],
            n_samples=100,
        )

        artifact = CalibrationArtifact(
            calibrators={"known": cal},
            metadata=metadata,
        )

        with pytest.raises(KeyError, match="unknown"):
            artifact.transform("unknown", 0.5)

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """Artifact should survive serialization roundtrip."""
        cal1 = PlattCalibrator(a=1.234, b=-0.567, head_name="p_inplay_30s")
        cal2 = PlattCalibrator(a=0.987, b=0.123, head_name="p_toxic")

        metadata = CalibrationMetadata(
            schema_version="1.0.0",
            git_sha="abc123",
            config_hash="config123",
            data_hash="data123",
            calibration_timestamp="2026-01-25T12:00:00+00:00",
            method="platt",
            heads=["p_inplay_30s", "p_toxic"],
            n_samples=500,
            metrics_before={"p_inplay_30s": {"brier": 0.25}},
            metrics_after={"p_inplay_30s": {"brier": 0.20}},
        )

        artifact = CalibrationArtifact(
            calibrators={"p_inplay_30s": cal1, "p_toxic": cal2},
            metadata=metadata,
        )

        data = artifact.to_dict()
        restored = CalibrationArtifact.from_dict(data)

        # Check calibrators preserved
        assert len(restored.calibrators) == 2
        assert restored.calibrators["p_inplay_30s"].a == cal1.a
        assert restored.calibrators["p_inplay_30s"].b == cal1.b
        assert restored.calibrators["p_toxic"].a == cal2.a
        assert restored.calibrators["p_toxic"].b == cal2.b

        # Check metadata preserved
        assert restored.metadata.schema_version == metadata.schema_version
        assert restored.metadata.heads == metadata.heads
        assert restored.metadata.n_samples == metadata.n_samples

    def test_transform_matches_after_roundtrip(self) -> None:
        """Transform results should match after serialization."""
        cal = PlattCalibrator(a=1.5, b=-0.3, head_name="test")
        metadata = CalibrationMetadata(
            schema_version="1.0.0",
            git_sha="test",
            config_hash="test",
            data_hash="test",
            calibration_timestamp="2026-01-25T12:00:00+00:00",
            method="platt",
            heads=["test"],
            n_samples=100,
        )

        artifact = CalibrationArtifact(
            calibrators={"test": cal},
            metadata=metadata,
        )

        # Transform before roundtrip
        probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        results_before = [artifact.transform("test", p) for p in probs]

        # Roundtrip
        data = artifact.to_dict()
        restored = CalibrationArtifact.from_dict(data)

        # Transform after roundtrip
        results_after = [restored.transform("test", p) for p in probs]

        assert results_before == results_after


class TestArtifactIO:
    """Tests for save/load functions."""

    def test_save_load_roundtrip(self) -> None:
        """Artifact should survive file save/load."""
        cal = PlattCalibrator(a=1.5, b=-0.3, head_name="p_inplay_30s")
        metadata = CalibrationMetadata(
            schema_version="1.0.0",
            git_sha="abc123",
            config_hash="config123",
            data_hash="data123",
            calibration_timestamp="2026-01-25T12:00:00+00:00",
            method="platt",
            heads=["p_inplay_30s"],
            n_samples=500,
            metrics_before={"p_inplay_30s": {"brier": 0.25, "ece": 0.10}},
            metrics_after={"p_inplay_30s": {"brier": 0.20, "ece": 0.05}},
        )

        artifact = CalibrationArtifact(
            calibrators={"p_inplay_30s": cal},
            metadata=metadata,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"

            save_calibration_artifact(artifact, path)
            assert path.exists()

            loaded = load_calibration_artifact(path)

            # Verify calibrators
            assert len(loaded.calibrators) == 1
            assert loaded.calibrators["p_inplay_30s"].a == cal.a
            assert loaded.calibrators["p_inplay_30s"].b == cal.b

            # Verify metadata
            assert loaded.metadata.schema_version == metadata.schema_version
            assert loaded.metadata.git_sha == metadata.git_sha
            assert loaded.metadata.n_samples == metadata.n_samples
            assert loaded.metadata.metrics_before == metadata.metrics_before
            assert loaded.metadata.metrics_after == metadata.metrics_after

    def test_load_nonexistent_raises(self) -> None:
        """Loading nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_calibration_artifact(Path("/nonexistent/calibration.json"))

    def test_save_creates_parent_dirs(self) -> None:
        """save should create parent directories if needed."""
        cal = PlattCalibrator(a=1.0, b=0.0, head_name="test")
        metadata = CalibrationMetadata(
            schema_version="1.0.0",
            git_sha="test",
            config_hash="test",
            data_hash="test",
            calibration_timestamp="2026-01-25T12:00:00+00:00",
            method="platt",
            heads=["test"],
            n_samples=10,
        )

        artifact = CalibrationArtifact(
            calibrators={"test": cal},
            metadata=metadata,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "calibration.json"
            save_calibration_artifact(artifact, path)
            assert path.exists()


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_hash(self) -> None:
        """Same input data should produce same hash."""
        config = {"method": "platt", "max_iter": 100}
        val_data = [
            {"ts": 1000, "value": 1},
            {"ts": 2000, "value": 2},
        ]

        meta1 = create_calibration_metadata("platt", ["test"], 2, config, val_data)
        meta2 = create_calibration_metadata("platt", ["test"], 2, config, val_data)

        assert meta1.config_hash == meta2.config_hash
        assert meta1.data_hash == meta2.data_hash

    def test_artifact_roundtrip_deterministic(self) -> None:
        """Multiple roundtrips should produce identical results."""
        cal = PlattCalibrator(a=1.5, b=-0.3, head_name="test")
        metadata = CalibrationMetadata(
            schema_version="1.0.0",
            git_sha="test",
            config_hash="test123",
            data_hash="data456",
            calibration_timestamp="2026-01-25T12:00:00+00:00",
            method="platt",
            heads=["test"],
            n_samples=100,
        )

        artifact = CalibrationArtifact(
            calibrators={"test": cal},
            metadata=metadata,
        )

        # Multiple roundtrips
        data1 = artifact.to_dict()
        restored1 = CalibrationArtifact.from_dict(data1)
        data2 = restored1.to_dict()
        restored2 = CalibrationArtifact.from_dict(data2)
        data3 = restored2.to_dict()

        # All should be equal
        assert data1 == data2 == data3
