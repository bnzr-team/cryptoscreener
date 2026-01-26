"""Tests for dataset loading and schema validation."""

import tempfile
from pathlib import Path
from typing import Any

import orjson

from cryptoscreener.training.dataset import (
    DEFAULT_SCHEMA,
    get_feature_columns,
    get_label_columns,
    load_labeled_dataset,
    validate_schema,
)


def create_valid_labeled_data(n_rows: int = 10) -> list[dict[str, Any]]:
    """Create valid labeled data matching expected schema."""
    rows = []

    for i in range(n_rows):
        row = {
            "ts": i * 1000,
            "symbol": "BTCUSDT",
            "mid_price": 50000.0 + i,
            "spread_bps": 2.0,
            "feature_vol": 100.0 + i,
            "feature_imb": 0.5,
        }

        # Add all expected label columns
        for h in ["30s", "2m", "5m"]:
            for p in ["a", "b"]:
                row[f"i_tradeable_{h}_{p}"] = 1 if i % 2 == 0 else 0
                row[f"net_edge_bps_{h}_{p}"] = 10.0 + i
                row[f"mfe_bps_{h}_{p}"] = 20.0 + i
                row[f"mae_bps_{h}_{p}"] = 5.0
                row[f"cost_bps_{h}_{p}"] = 8.0

        row["y_toxic"] = 0
        row["severity_toxic_bps"] = 0.0

        rows.append(row)

    return rows


class TestValidateSchema:
    """Tests for schema validation."""

    def test_valid_data_passes(self) -> None:
        """Valid data should pass validation."""
        rows = create_valid_labeled_data()
        result = validate_schema(rows)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_empty_data_fails(self) -> None:
        """Empty data should fail validation."""
        result = validate_schema([])

        assert not result.is_valid
        assert "empty" in result.errors[0].lower()

    def test_missing_required_column(self) -> None:
        """Missing required column should fail."""
        rows = [{"ts": 1000, "symbol": "BTC"}]  # Missing mid_price, spread_bps
        result = validate_schema(rows)

        assert not result.is_valid
        assert any("mid_price" in str(e) or "Missing" in str(e) for e in result.errors)

    def test_missing_ts_fails(self) -> None:
        """Missing timestamp should fail."""
        rows = [{"symbol": "BTC", "mid_price": 100, "spread_bps": 2}]
        result = validate_schema(rows)

        assert not result.is_valid
        assert any("ts" in str(e) for e in result.errors)

    def test_extra_columns_allowed_by_default(self) -> None:
        """Extra columns should be allowed by default."""
        rows = create_valid_labeled_data()
        rows[0]["extra_column"] = "value"

        result = validate_schema(rows, strict=False)

        assert result.is_valid
        assert "extra_column" in result.extra_columns

    def test_extra_columns_fail_strict(self) -> None:
        """Extra columns should fail in strict mode."""
        rows = create_valid_labeled_data()
        rows[0]["extra_column"] = "value"

        result = validate_schema(rows, strict=True)

        assert not result.is_valid
        assert any("Unexpected" in str(e) for e in result.errors)


class TestLoadLabeledDataset:
    """Tests for loading labeled datasets."""

    def test_load_jsonl(self) -> None:
        """Load JSONL file should work."""
        rows = create_valid_labeled_data()

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            for row in rows:
                f.write(orjson.dumps(row))
                f.write(b"\n")
            tmp_path = Path(f.name)

        try:
            loaded, validation = load_labeled_dataset(tmp_path)

            assert len(loaded) == len(rows)
            assert validation is not None
            assert validation.is_valid
        finally:
            tmp_path.unlink()

    def test_load_nonexistent_fails(self) -> None:
        """Loading nonexistent file should raise error."""
        try:
            load_labeled_dataset(Path("/nonexistent/file.jsonl"))
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass

    def test_load_unsupported_format_fails(self) -> None:
        """Unsupported format should raise error."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"a,b,c\n1,2,3\n")
            tmp_path = Path(f.name)

        try:
            load_labeled_dataset(tmp_path)
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "Unsupported" in str(e)
        finally:
            tmp_path.unlink()

    def test_skip_validation(self) -> None:
        """Skip validation should return None for validation result."""
        rows = [{"ts": 1, "symbol": "X"}]  # Invalid data

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            for row in rows:
                f.write(orjson.dumps(row))
                f.write(b"\n")
            tmp_path = Path(f.name)

        try:
            loaded, validation = load_labeled_dataset(tmp_path, validate=False)

            assert len(loaded) == 1
            assert validation is None
        finally:
            tmp_path.unlink()


class TestGetColumns:
    """Tests for column extraction functions."""

    def test_get_feature_columns(self) -> None:
        """Feature columns should exclude labels and metadata."""
        rows = create_valid_labeled_data()
        features = get_feature_columns(rows)

        # Should include custom features
        assert "feature_vol" in features
        assert "feature_imb" in features

        # Should exclude labels
        assert "i_tradeable_30s_a" not in features
        assert "y_toxic" not in features

        # Should exclude metadata
        assert "ts" not in features
        assert "symbol" not in features

    def test_get_label_columns(self) -> None:
        """Label columns should be correctly organized."""
        rows = create_valid_labeled_data()
        labels = get_label_columns(rows)

        assert "tradeable" in labels
        assert "edge" in labels
        assert "toxicity" in labels

        assert len(labels["tradeable"]) == 6  # 3 horizons * 2 profiles
        assert len(labels["edge"]) == 6
        assert len(labels["toxicity"]) == 2  # y_toxic, severity_toxic_bps

    def test_get_label_columns_filtered(self) -> None:
        """Label columns with horizon filter should work."""
        rows = create_valid_labeled_data()
        labels = get_label_columns(rows, horizons=["30s"], profiles=["a"])

        assert len(labels["tradeable"]) == 1
        assert labels["tradeable"][0] == "i_tradeable_30s_a"

    def test_empty_rows(self) -> None:
        """Empty rows should return empty lists."""
        features = get_feature_columns([])
        labels = get_label_columns([])

        assert features == []
        assert labels == {"tradeable": [], "edge": [], "toxicity": []}


class TestDefaultSchema:
    """Tests for default schema definition."""

    def test_schema_version(self) -> None:
        """Schema should have version."""
        assert DEFAULT_SCHEMA.version == "1.0.0"

    def test_required_columns(self) -> None:
        """Required columns should be defined."""
        assert "ts" in DEFAULT_SCHEMA.required_columns
        assert "symbol" in DEFAULT_SCHEMA.required_columns
        assert "mid_price" in DEFAULT_SCHEMA.required_columns

    def test_get_all_columns(self) -> None:
        """get_all_columns should expand placeholders."""
        columns = DEFAULT_SCHEMA.get_all_columns()

        # Should include base columns
        assert "ts" in columns
        assert "symbol" in columns

        # Should include expanded label columns
        assert "i_tradeable_30s_a" in columns
        assert "i_tradeable_5m_b" in columns
        assert "net_edge_bps_2m_a" in columns

        # Should include toxicity
        assert "y_toxic" in columns
