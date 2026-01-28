"""Tests for feature schema (DEC-038)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cryptoscreener.training.feature_schema import (
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    HEAD_TO_LABEL,
    PREDICTION_HEADS,
    FeatureSchemaError,
    compute_feature_hash,
    get_label_column,
    load_features_json,
    save_features_json,
    validate_feature_compatibility,
)


class TestFeatureOrder:
    """Tests for canonical feature ordering."""

    def test_feature_order_has_8_features(self) -> None:
        """FEATURE_ORDER should have exactly 8 features."""
        assert len(FEATURE_ORDER) == 8

    def test_feature_order_matches_mlrunner(self) -> None:
        """FEATURE_ORDER must match MLRunner._extract_features() order.

        This is critical for training/inference compatibility.
        """
        expected = (
            "spread_bps",
            "mid",
            "book_imbalance",
            "flow_imbalance",
            "natr_14_5m",
            "impact_bps_q",
            "regime_vol_binary",
            "regime_trend_binary",
        )
        assert expected == FEATURE_ORDER

    def test_feature_order_is_tuple(self) -> None:
        """FEATURE_ORDER should be immutable (tuple)."""
        assert isinstance(FEATURE_ORDER, tuple)

    def test_prediction_heads_has_4_heads(self) -> None:
        """PREDICTION_HEADS should have exactly 4 heads."""
        assert len(PREDICTION_HEADS) == 4
        assert "p_inplay_30s" in PREDICTION_HEADS
        assert "p_inplay_2m" in PREDICTION_HEADS
        assert "p_inplay_5m" in PREDICTION_HEADS
        assert "p_toxic" in PREDICTION_HEADS


class TestComputeFeatureHash:
    """Tests for feature hash computation."""

    def test_hash_is_deterministic(self) -> None:
        """Same features should produce same hash."""
        hash1 = compute_feature_hash(list(FEATURE_ORDER))
        hash2 = compute_feature_hash(list(FEATURE_ORDER))
        assert hash1 == hash2

    def test_hash_uses_default_features(self) -> None:
        """compute_feature_hash() without args uses FEATURE_ORDER."""
        hash_default = compute_feature_hash()
        hash_explicit = compute_feature_hash(FEATURE_ORDER)
        assert hash_default == hash_explicit

    def test_hash_length_is_16(self) -> None:
        """Hash should be 16 characters (truncated SHA256)."""
        feature_hash = compute_feature_hash()
        assert len(feature_hash) == 16

    def test_hash_is_hex(self) -> None:
        """Hash should be hexadecimal."""
        feature_hash = compute_feature_hash()
        int(feature_hash, 16)  # Should not raise

    def test_different_order_produces_different_hash(self) -> None:
        """Feature order matters for hash."""
        hash1 = compute_feature_hash(["a", "b", "c"])
        hash2 = compute_feature_hash(["c", "b", "a"])
        assert hash1 != hash2

    def test_different_features_produce_different_hash(self) -> None:
        """Different features produce different hashes."""
        hash1 = compute_feature_hash(["feature_a", "feature_b"])
        hash2 = compute_feature_hash(["feature_x", "feature_y"])
        assert hash1 != hash2


class TestSaveLoadFeaturesJson:
    """Tests for features.json I/O."""

    def test_save_creates_valid_json(self) -> None:
        """save_features_json creates valid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            save_features_json(path)

            assert path.exists()
            with path.open() as f:
                data = json.load(f)

            assert "schema_version" in data
            assert "features" in data
            assert "feature_hash" in data

    def test_save_uses_defaults(self) -> None:
        """save_features_json uses FEATURE_ORDER and FEATURE_SCHEMA_VERSION by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            save_features_json(path)

            data = load_features_json(path)
            assert data["schema_version"] == FEATURE_SCHEMA_VERSION
            assert data["features"] == list(FEATURE_ORDER)
            assert data["feature_hash"] == compute_feature_hash()

    def test_save_custom_features(self) -> None:
        """save_features_json accepts custom features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            custom_features = ["feat_a", "feat_b"]
            save_features_json(path, features=custom_features)

            data = load_features_json(path)
            assert data["features"] == custom_features
            assert data["feature_hash"] == compute_feature_hash(custom_features)

    def test_load_missing_file_raises(self) -> None:
        """load_features_json raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_features_json(Path("/nonexistent/features.json"))

    def test_load_invalid_json_raises(self) -> None:
        """load_features_json raises on invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            path.write_text("not valid json")

            with pytest.raises(json.JSONDecodeError):
                load_features_json(path)

    def test_load_missing_fields_raises(self) -> None:
        """load_features_json raises FeatureSchemaError for missing fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            path.write_text('{"schema_version": "1.0.0"}')

            with pytest.raises(FeatureSchemaError, match="missing required fields"):
                load_features_json(path)

    def test_roundtrip_preserves_data(self) -> None:
        """Save then load preserves all data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            original_hash = save_features_json(path)

            data = load_features_json(path)

            assert data["schema_version"] == FEATURE_SCHEMA_VERSION
            assert data["features"] == list(FEATURE_ORDER)
            assert data["feature_hash"] == original_hash


class TestValidateFeatureCompatibility:
    """Tests for feature compatibility validation."""

    def test_valid_features_pass(self) -> None:
        """validate_feature_compatibility passes for matching features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            save_features_json(path)

            # Should not raise
            validate_feature_compatibility(path)

    def test_wrong_count_raises(self) -> None:
        """validate_feature_compatibility raises on wrong feature count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            save_features_json(path, features=["a", "b"])  # Wrong count

            with pytest.raises(FeatureSchemaError, match="Feature count mismatch"):
                validate_feature_compatibility(path)

    def test_wrong_order_raises(self) -> None:
        """validate_feature_compatibility raises on wrong feature order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            # Swap first two features
            wrong_order = list(FEATURE_ORDER)
            wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
            save_features_json(path, features=wrong_order)

            with pytest.raises(FeatureSchemaError, match="Feature mismatch"):
                validate_feature_compatibility(path)

    def test_wrong_names_raises(self) -> None:
        """validate_feature_compatibility raises on wrong feature names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            wrong_names = ["wrong_" + f for f in FEATURE_ORDER]
            save_features_json(path, features=wrong_names)

            with pytest.raises(FeatureSchemaError, match="Feature mismatch"):
                validate_feature_compatibility(path)

    def test_custom_expected_features(self) -> None:
        """validate_feature_compatibility accepts custom expected features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            custom_features = ["custom_a", "custom_b"]
            save_features_json(path, features=custom_features)

            # Should not raise when expected matches actual
            validate_feature_compatibility(path, expected_features=custom_features)

    def test_hash_mismatch_raises(self) -> None:
        """validate_feature_compatibility raises on hash mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            # Manually create file with wrong hash
            data = {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "features": list(FEATURE_ORDER),
                "feature_hash": "wrong_hash_value_",
            }
            with path.open("w") as f:
                json.dump(data, f)

            with pytest.raises(FeatureSchemaError, match="Feature hash mismatch"):
                validate_feature_compatibility(path)


class TestGetLabelColumn:
    """Tests for label column name generation."""

    def test_p_inplay_30s_profile_a(self) -> None:
        """get_label_column returns correct column for p_inplay_30s."""
        assert get_label_column("p_inplay_30s", "a") == "i_tradeable_30s_a"

    def test_p_inplay_2m_profile_b(self) -> None:
        """get_label_column returns correct column for p_inplay_2m."""
        assert get_label_column("p_inplay_2m", "b") == "i_tradeable_2m_b"

    def test_p_inplay_5m(self) -> None:
        """get_label_column returns correct column for p_inplay_5m."""
        assert get_label_column("p_inplay_5m", "a") == "i_tradeable_5m_a"

    def test_p_toxic_ignores_profile(self) -> None:
        """get_label_column returns y_toxic regardless of profile."""
        assert get_label_column("p_toxic", "a") == "y_toxic"
        assert get_label_column("p_toxic", "b") == "y_toxic"

    def test_default_profile_is_a(self) -> None:
        """get_label_column uses profile 'a' by default."""
        assert get_label_column("p_inplay_30s") == "i_tradeable_30s_a"

    def test_unknown_head_raises(self) -> None:
        """get_label_column raises ValueError for unknown head."""
        with pytest.raises(ValueError, match="Unknown prediction head"):
            get_label_column("unknown_head")

    def test_all_heads_have_mappings(self) -> None:
        """All PREDICTION_HEADS have label mappings."""
        for head in PREDICTION_HEADS:
            assert head in HEAD_TO_LABEL
            # Should not raise
            get_label_column(head)
