"""Tests for time-based dataset splitting.

Includes strict anti-leakage verification tests.
"""

import tempfile
from pathlib import Path
from typing import Any

import orjson

from cryptoscreener.training.split import (
    SplitConfig,
    load_split_metadata,
    save_split,
    time_based_split,
)


def create_test_data(n_rows: int = 100, n_symbols: int = 2) -> list[dict[str, Any]]:
    """Create test data with sequential timestamps."""
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    rows = []

    for i in range(n_rows):
        ts = i * 1000  # Sequential timestamps
        symbol = symbols[i % n_symbols]

        row = {
            "ts": ts,
            "symbol": symbol,
            "mid_price": 100.0 + i * 0.1,
            "spread_bps": 2.0,
            "feature_a": i * 0.5,
            "feature_b": (i % 10) * 2.0,
        }

        # Add label columns
        for h in ["30s", "2m", "5m"]:
            for p in ["a", "b"]:
                row[f"i_tradeable_{h}_{p}"] = 1 if i % 3 == 0 else 0
                row[f"net_edge_bps_{h}_{p}"] = (i % 5) * 10 - 20

        row["y_toxic"] = 1 if i % 7 == 0 else 0
        row["severity_toxic_bps"] = 15.0 if i % 7 == 0 else 0.0

        rows.append(row)

    return rows


class TestSplitConfig:
    """Tests for SplitConfig validation."""

    def test_default_config(self) -> None:
        """Default config should have valid ratios."""
        config = SplitConfig()
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.15
        assert config.test_ratio == 0.15
        assert abs(config.train_ratio + config.val_ratio + config.test_ratio - 1.0) < 1e-6

    def test_custom_config(self) -> None:
        """Custom config with valid ratios should work."""
        config = SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        assert config.train_ratio == 0.8

    def test_invalid_ratios_sum(self) -> None:
        """Ratios not summing to 1 should raise error."""
        try:
            SplitConfig(train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "sum to 1.0" in str(e)

    def test_negative_ratio(self) -> None:
        """Negative ratios should raise error."""
        try:
            SplitConfig(train_ratio=1.2, val_ratio=-0.1, test_ratio=-0.1)
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "positive" in str(e)


class TestTimeBasedSplit:
    """Tests for time_based_split function."""

    def test_basic_split(self) -> None:
        """Basic split should produce correct row counts."""
        rows = create_test_data(100)
        result = time_based_split(rows)

        assert result.metadata.train_rows == 70
        assert result.metadata.val_rows == 15
        assert result.metadata.test_rows == 15
        assert len(result.train) == 70
        assert len(result.val) == 15
        assert len(result.test) == 15

    def test_no_leakage_basic(self) -> None:
        """Basic split should have no temporal leakage."""
        rows = create_test_data(100)
        result = time_based_split(rows)

        assert result.verify_no_leakage()

        # Explicit check: max(train_ts) < min(val_ts) < min(test_ts)
        train_max_ts = max(r["ts"] for r in result.train)
        val_min_ts = min(r["ts"] for r in result.val)
        val_max_ts = max(r["ts"] for r in result.val)
        test_min_ts = min(r["ts"] for r in result.test)

        assert train_max_ts < val_min_ts, (
            f"Leakage: train_max={train_max_ts} >= val_min={val_min_ts}"
        )
        assert val_max_ts < test_min_ts, f"Leakage: val_max={val_max_ts} >= test_min={test_min_ts}"

    def test_no_leakage_with_purge_gap(self) -> None:
        """Split with purge gap should have larger temporal separation."""
        rows = create_test_data(1000)
        config = SplitConfig(purge_gap_ms=5000)  # 5 second gap
        result = time_based_split(rows, config)

        assert result.verify_no_leakage()

        # Verify by ACTUAL DATA, not just metadata
        actual_train_max = max(r["ts"] for r in result.train)
        actual_val_min = min(r["ts"] for r in result.val)
        actual_val_max = max(r["ts"] for r in result.val)
        actual_test_min = min(r["ts"] for r in result.test)

        # Metadata should match actual data
        assert result.metadata.train_ts_range[1] == actual_train_max
        assert result.metadata.val_ts_range[0] == actual_val_min
        assert result.metadata.val_ts_range[1] == actual_val_max
        assert result.metadata.test_ts_range[0] == actual_test_min

        # With purge gap, there should be at least purge_gap_ms between splits
        assert actual_val_min - actual_train_max >= config.purge_gap_ms
        assert actual_test_min - actual_val_max >= config.purge_gap_ms

    def test_no_leakage_all_symbols(self) -> None:
        """Split should have no leakage for any symbol."""
        rows = create_test_data(200, n_symbols=5)
        result = time_based_split(rows)

        # Group by symbol - only track actual timestamps, no defaults
        train_by_sym: dict[str, list[int]] = {}
        val_by_sym: dict[str, list[int]] = {}
        test_by_sym: dict[str, list[int]] = {}

        for r in result.train:
            train_by_sym.setdefault(r["symbol"], []).append(r["ts"])
        for r in result.val:
            val_by_sym.setdefault(r["symbol"], []).append(r["ts"])
        for r in result.test:
            test_by_sym.setdefault(r["symbol"], []).append(r["ts"])

        # Check each symbol that appears in multiple splits
        all_symbols = set(train_by_sym.keys()) | set(val_by_sym.keys()) | set(test_by_sym.keys())

        for sym in all_symbols:
            # Only check boundaries for symbols present in both splits
            if sym in train_by_sym and sym in val_by_sym:
                train_max = max(train_by_sym[sym])
                val_min = min(val_by_sym[sym])
                assert train_max < val_min, (
                    f"Symbol {sym}: train_max={train_max} >= val_min={val_min}"
                )

            if sym in val_by_sym and sym in test_by_sym:
                val_max = max(val_by_sym[sym])
                test_min = min(test_by_sym[sym])
                assert val_max < test_min, f"Symbol {sym}: val_max={val_max} >= test_min={test_min}"

    def test_empty_dataset_raises(self) -> None:
        """Empty dataset should raise error."""
        try:
            time_based_split([])
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "empty" in str(e).lower()

    def test_too_small_dataset_raises(self) -> None:
        """Dataset too small for split ratios should raise error."""
        # Only 3 rows - not enough for 70/15/15 split with all non-empty
        rows = [{"ts": i * 1000, "value": i} for i in range(3)]
        try:
            time_based_split(rows)
            raise AssertionError("Expected ValueError for small dataset")
        except ValueError as e:
            assert "empty" in str(e).lower()

    def test_large_purge_gap_raises(self) -> None:
        """Purge gap too large for dataset should raise error."""
        # 100 rows but purge gap consumes everything
        rows = [{"ts": i * 100, "value": i} for i in range(100)]  # ts: 0-9900
        config = SplitConfig(purge_gap_ms=50000)  # Gap larger than data range
        try:
            time_based_split(rows, config)
            raise AssertionError("Expected ValueError for large purge gap")
        except ValueError as e:
            assert "empty" in str(e).lower()

    def test_missing_timestamp_raises(self) -> None:
        """Missing timestamp column should raise error."""
        rows = [{"a": 1, "b": 2}]
        try:
            time_based_split(rows)
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "ts" in str(e)

    def test_custom_timestamp_column(self) -> None:
        """Custom timestamp column should work."""
        rows = [{"custom_ts": i * 1000, "value": i} for i in range(100)]
        config = SplitConfig(timestamp_col="custom_ts")
        result = time_based_split(rows, config)

        assert result.verify_no_leakage()
        assert result.metadata.train_rows > 0

    def test_determinism(self) -> None:
        """Same input should produce same output."""
        rows = create_test_data(100)

        result1 = time_based_split(rows)
        result2 = time_based_split(rows)

        # Row counts should match
        assert result1.metadata.train_rows == result2.metadata.train_rows
        assert result1.metadata.val_rows == result2.metadata.val_rows
        assert result1.metadata.test_rows == result2.metadata.test_rows

        # Data hash should match
        assert result1.metadata.data_hash == result2.metadata.data_hash

        # Config hash should match
        assert result1.metadata.config_hash == result2.metadata.config_hash

        # FULL ROW EQUALITY - not just timestamps
        assert result1.train == result2.train, "Train splits differ"
        assert result1.val == result2.val, "Val splits differ"
        assert result1.test == result2.test, "Test splits differ"


class TestSplitMetadata:
    """Tests for split metadata."""

    def test_metadata_fields(self) -> None:
        """Metadata should have all required fields."""
        rows = create_test_data(100)
        result = time_based_split(rows)

        assert result.metadata.schema_version == "1.0.0"
        assert result.metadata.git_sha  # Not empty
        assert result.metadata.config_hash  # Not empty
        assert result.metadata.data_hash  # Not empty
        assert result.metadata.split_timestamp  # Not empty

    def test_metadata_ts_ranges(self) -> None:
        """Timestamp ranges should be accurate."""
        rows = create_test_data(100)
        result = time_based_split(rows)

        # Verify ranges match actual data
        train_min = min(r["ts"] for r in result.train)
        train_max = max(r["ts"] for r in result.train)
        assert result.metadata.train_ts_range == (train_min, train_max)

        val_min = min(r["ts"] for r in result.val)
        val_max = max(r["ts"] for r in result.val)
        assert result.metadata.val_ts_range == (val_min, val_max)

        test_min = min(r["ts"] for r in result.test)
        test_max = max(r["ts"] for r in result.test)
        assert result.metadata.test_ts_range == (test_min, test_max)

    def test_to_dict(self) -> None:
        """to_dict should produce valid dict."""
        rows = create_test_data(100)
        result = time_based_split(rows)

        d = result.metadata.to_dict()

        assert "schema_version" in d
        assert "git_sha" in d
        assert "train_rows" in d
        assert "train_ts_range" in d
        assert isinstance(d["train_ts_range"], list)


class TestSaveAndLoad:
    """Tests for saving and loading splits."""

    def test_save_jsonl(self) -> None:
        """Save as JSONL should create correct files."""
        rows = create_test_data(100)
        result = time_based_split(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_split(result, output_dir, format="jsonl")

            assert (output_dir / "train.jsonl").exists()
            assert (output_dir / "val.jsonl").exists()
            assert (output_dir / "test.jsonl").exists()
            assert (output_dir / "metadata.json").exists()

            # Verify row counts
            with (output_dir / "train.jsonl").open("rb") as f:
                train_rows = [orjson.loads(line) for line in f if line.strip()]
            assert len(train_rows) == result.metadata.train_rows

    def test_load_metadata(self) -> None:
        """Load metadata should restore original values."""
        rows = create_test_data(100)
        result = time_based_split(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_split(result, output_dir)

            loaded = load_split_metadata(output_dir / "metadata.json")

            assert loaded.schema_version == result.metadata.schema_version
            assert loaded.train_rows == result.metadata.train_rows
            assert loaded.val_rows == result.metadata.val_rows
            assert loaded.test_rows == result.metadata.test_rows
            assert loaded.data_hash == result.metadata.data_hash
            assert loaded.config_hash == result.metadata.config_hash


class TestAntiLeakageStrict:
    """Strict anti-leakage tests.

    These tests verify that NO future information leaks into training.
    """

    def test_train_val_boundary_strict(self) -> None:
        """Strictly verify train-val boundary has no overlap."""
        rows = create_test_data(1000)
        result = time_based_split(rows)

        train_timestamps = {r["ts"] for r in result.train}
        val_timestamps = {r["ts"] for r in result.val}

        # No overlap
        overlap = train_timestamps & val_timestamps
        assert len(overlap) == 0, f"Train-val timestamp overlap: {overlap}"

        # Strict ordering
        assert max(train_timestamps) < min(val_timestamps)

    def test_val_test_boundary_strict(self) -> None:
        """Strictly verify val-test boundary has no overlap."""
        rows = create_test_data(1000)
        result = time_based_split(rows)

        val_timestamps = {r["ts"] for r in result.val}
        test_timestamps = {r["ts"] for r in result.test}

        # No overlap
        overlap = val_timestamps & test_timestamps
        assert len(overlap) == 0, f"Val-test timestamp overlap: {overlap}"

        # Strict ordering
        assert max(val_timestamps) < min(test_timestamps)

    def test_no_future_in_train(self) -> None:
        """Verify train set has NO samples from future."""
        rows = create_test_data(1000)
        result = time_based_split(rows)

        train_max_ts = max(r["ts"] for r in result.train)

        # Check no val or test sample has timestamp <= train_max
        for r in result.val:
            assert r["ts"] > train_max_ts, (
                f"Val sample has ts={r['ts']} <= train_max={train_max_ts}"
            )

        for r in result.test:
            assert r["ts"] > train_max_ts, (
                f"Test sample has ts={r['ts']} <= train_max={train_max_ts}"
            )

    def test_shuffled_input_still_correct(self) -> None:
        """Even shuffled input should produce correct temporal split."""
        import random

        rows = create_test_data(500)
        random.shuffle(rows)  # Shuffle input

        result = time_based_split(rows)

        assert result.verify_no_leakage()

        # Verify strict ordering after split
        train_max = max(r["ts"] for r in result.train)
        val_min = min(r["ts"] for r in result.val)

        assert train_max < val_min, "Shuffled input caused leakage"

    def test_duplicate_timestamps_no_leakage(self) -> None:
        """Duplicate timestamps should NOT be split across train/val/test.

        This is critical: if same ts appears in train and val, it's leakage.
        The boundary shift logic should keep all same-ts rows in one split.
        """
        # Create data with many unique timestamps + some duplicates at boundaries
        # Use enough data (200 rows) to ensure all splits are non-empty
        rows = []
        # 70 rows with unique ts for train (0-69)
        for i in range(70):
            rows.append({"ts": i * 100, "value": i, "symbol": "A"})
        # 10 rows at same ts near train/val boundary
        for i in range(10):
            rows.append({"ts": 7000, "value": 70 + i, "symbol": "A"})
        # 30 rows with unique ts for val
        for i in range(30):
            rows.append({"ts": 8000 + i * 100, "value": 80 + i, "symbol": "A"})
        # 10 rows at same ts near val/test boundary
        for i in range(10):
            rows.append({"ts": 11000, "value": 110 + i, "symbol": "A"})
        # 30 rows with unique ts for test
        for i in range(30):
            rows.append({"ts": 12000 + i * 100, "value": 120 + i, "symbol": "A"})

        # Total: 150 rows
        result = time_based_split(rows)

        # Verify no leakage
        assert result.verify_no_leakage(), "Leakage detected with duplicate timestamps"

        # Critical check: same timestamp should NOT appear in multiple splits
        train_ts = {r["ts"] for r in result.train}
        val_ts = {r["ts"] for r in result.val}
        test_ts = {r["ts"] for r in result.test}

        assert len(train_ts & val_ts) == 0, f"Same ts in train and val: {train_ts & val_ts}"
        assert len(val_ts & test_ts) == 0, f"Same ts in val and test: {val_ts & test_ts}"
        assert len(train_ts & test_ts) == 0, f"Same ts in train and test: {train_ts & test_ts}"

        # Verify strict temporal ordering (only if splits are non-empty)
        if train_ts and val_ts:
            assert max(train_ts) < min(val_ts), "train_max >= val_min"
        if val_ts and test_ts:
            assert max(val_ts) < min(test_ts), "val_max >= test_min"

    def test_many_duplicates_at_boundary(self) -> None:
        """Large cluster of duplicates at split boundary should shift correctly."""
        rows = []
        # First 60 rows with unique ts (will be in train)
        for i in range(60):
            rows.append({"ts": i * 100, "value": i})
        # Next 20 rows ALL with same ts (at ~70% boundary)
        for i in range(20):
            rows.append({"ts": 7000, "value": 60 + i})  # All same ts
        # Remaining 20 rows with unique ts
        for i in range(20):
            rows.append({"ts": 8000 + i * 100, "value": 80 + i})

        result = time_based_split(rows)

        # No leakage
        assert result.verify_no_leakage()

        # All ts=7000 rows should be in same split (not divided)
        ts_7000_in_train = sum(1 for r in result.train if r["ts"] == 7000)
        ts_7000_in_val = sum(1 for r in result.val if r["ts"] == 7000)
        ts_7000_in_test = sum(1 for r in result.test if r["ts"] == 7000)

        # Should be in exactly ONE split
        splits_with_7000 = sum(
            [
                ts_7000_in_train > 0,
                ts_7000_in_val > 0,
                ts_7000_in_test > 0,
            ]
        )
        assert splits_with_7000 == 1, (
            f"ts=7000 split across multiple sets: "
            f"train={ts_7000_in_train}, val={ts_7000_in_val}, test={ts_7000_in_test}"
        )
