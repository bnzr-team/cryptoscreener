"""
Time-based dataset splitting for ML training.

Implements strict temporal splits to prevent data leakage:
- Train/val/test split by timestamp
- Optional purge gap between splits
- Per-symbol or global split
- Deterministic and reproducible

Per PRD §11 and DATASET_BUILD_PIPELINE.md:
"Split by time into train/val/test"
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING, Any

import orjson

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for time-based splitting.

    Attributes:
        train_ratio: Fraction of data for training (by time).
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        purge_gap_ms: Gap between splits to prevent leakage (default 0).
        timestamp_col: Column name for timestamps.
        group_col: Optional column for grouped splits (e.g., symbol).
    """

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    purge_gap_ms: int = 0
    timestamp_col: str = "ts"
    group_col: str | None = None

    def __post_init__(self) -> None:
        """Validate ratios sum to 1."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        if self.train_ratio <= 0 or self.val_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("All ratios must be positive")


@dataclass(frozen=True)
class SplitMetadata:
    """Metadata for a dataset split.

    Attributes:
        schema_version: Version of the split schema.
        git_sha: Git commit SHA at split time.
        config_hash: SHA256 of split configuration.
        split_timestamp: When the split was created.
        train_rows: Number of training rows.
        val_rows: Number of validation rows.
        test_rows: Number of test rows.
        train_ts_range: (min, max) timestamp in train set.
        val_ts_range: (min, max) timestamp in val set.
        test_ts_range: (min, max) timestamp in test set.
        data_hash: SHA256 of input data.
        config: Original split configuration.
    """

    schema_version: str
    git_sha: str
    config_hash: str
    split_timestamp: str
    train_rows: int
    val_rows: int
    test_rows: int
    train_ts_range: tuple[int, int]
    val_ts_range: tuple[int, int]
    test_ts_range: tuple[int, int]
    data_hash: str
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "split_timestamp": self.split_timestamp,
            "train_rows": self.train_rows,
            "val_rows": self.val_rows,
            "test_rows": self.test_rows,
            "train_ts_range": list(self.train_ts_range),
            "val_ts_range": list(self.val_ts_range),
            "test_ts_range": list(self.test_ts_range),
            "data_hash": self.data_hash,
            "config": self.config,
        }


@dataclass
class SplitResult:
    """Result of time-based split.

    Attributes:
        train: Training data rows.
        val: Validation data rows.
        test: Test data rows.
        metadata: Split metadata.
    """

    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]
    metadata: SplitMetadata

    def verify_no_leakage(self) -> bool:
        """Verify no temporal leakage between splits.

        Returns:
            True if max(train_ts) < min(val_ts) < min(test_ts).

        Note:
            This method assumes all splits are non-empty. Empty splits
            should be caught earlier by time_based_split() which raises
            ValueError if any split would be empty.
        """
        train_max = self.metadata.train_ts_range[1]
        val_min = self.metadata.val_ts_range[0]
        val_max = self.metadata.val_ts_range[1]
        test_min = self.metadata.test_ts_range[0]

        return train_max < val_min and val_max < test_min


def _compute_data_hash(rows: Sequence[dict[str, Any]]) -> str:
    """Compute SHA256 hash of data rows."""
    hasher = hashlib.sha256()
    for row in rows:
        hasher.update(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
    return hasher.hexdigest()[:16]


def _compute_config_hash(config: SplitConfig) -> str:
    """Compute SHA256 hash of configuration."""
    config_dict = {
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "test_ratio": config.test_ratio,
        "purge_gap_ms": config.purge_gap_ms,
        "timestamp_col": config.timestamp_col,
        "group_col": config.group_col,
    }
    data = orjson.dumps(config_dict, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(data).hexdigest()[:16]


def _get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def _get_ts_range(rows: Sequence[dict[str, Any]], ts_col: str) -> tuple[int, int]:
    """Get (min, max) timestamp from rows."""
    if not rows:
        return (0, 0)
    timestamps = [int(row[ts_col]) for row in rows]
    return (min(timestamps), max(timestamps))


def _find_boundary_shift(
    sorted_rows: Sequence[dict[str, Any]],
    target_idx: int,
    ts_col: str,
) -> int:
    """Find index where timestamp changes to avoid splitting same ts across sets.

    If rows at target_idx and target_idx-1 have same timestamp, shift forward
    to first row with different (larger) timestamp.

    Args:
        sorted_rows: Rows sorted by timestamp.
        target_idx: Initial split boundary index.
        ts_col: Timestamp column name.

    Returns:
        Adjusted index that doesn't split same timestamp across sets.
    """
    if target_idx <= 0 or target_idx >= len(sorted_rows):
        return target_idx

    boundary_ts = int(sorted_rows[target_idx - 1][ts_col])

    # Shift forward while timestamps are the same
    while target_idx < len(sorted_rows) and int(sorted_rows[target_idx][ts_col]) == boundary_ts:
        target_idx += 1

    return target_idx


def time_based_split(
    rows: Sequence[dict[str, Any]],
    config: SplitConfig | None = None,
) -> SplitResult:
    """Split data by timestamp into train/val/test sets.

    Implements strict temporal ordering to prevent data leakage:
    - All training samples have timestamps before all validation samples
    - All validation samples have timestamps before all test samples
    - Optional purge gap removes samples near split boundaries

    Args:
        rows: Input data rows (must have timestamp column).
        config: Split configuration. Uses defaults if not provided.

    Returns:
        SplitResult with train/val/test sets and metadata.

    Raises:
        ValueError: If rows are empty or missing timestamp column.
    """
    if config is None:
        config = SplitConfig()

    if not rows:
        raise ValueError("Cannot split empty dataset")

    ts_col = config.timestamp_col

    # Verify timestamp column exists
    if ts_col not in rows[0]:
        raise ValueError(f"Timestamp column '{ts_col}' not found in data")

    # Sort by timestamp
    sorted_rows = sorted(rows, key=lambda r: int(r[ts_col]))

    n = len(sorted_rows)
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))

    # CRITICAL: Shift boundaries to avoid splitting same timestamp across sets
    # This prevents leakage when multiple rows have identical timestamps
    train_end = _find_boundary_shift(sorted_rows, train_end, ts_col)
    val_end = _find_boundary_shift(sorted_rows, val_end, ts_col)

    # Apply purge gap if configured
    if config.purge_gap_ms > 0:
        # Find split boundaries by timestamp
        train_cutoff_ts = int(sorted_rows[train_end - 1][ts_col])
        val_cutoff_ts = int(sorted_rows[val_end - 1][ts_col])

        # Adjust boundaries to exclude samples within purge gap
        train_rows = [
            r
            for r in sorted_rows[:train_end]
            if int(r[ts_col]) <= train_cutoff_ts - config.purge_gap_ms
        ]

        val_rows = [
            r
            for r in sorted_rows[train_end:val_end]
            if int(r[ts_col]) >= train_cutoff_ts + config.purge_gap_ms
            and int(r[ts_col]) <= val_cutoff_ts - config.purge_gap_ms
        ]

        test_rows = [
            r
            for r in sorted_rows[val_end:]
            if int(r[ts_col]) >= val_cutoff_ts + config.purge_gap_ms
        ]
    else:
        train_rows = list(sorted_rows[:train_end])
        val_rows = list(sorted_rows[train_end:val_end])
        test_rows = list(sorted_rows[val_end:])

    # FAIL-FAST: Reject empty splits
    # Empty splits indicate dataset too small or purge_gap too large
    if not train_rows:
        raise ValueError(
            "Train split is empty. Dataset may be too small or boundary shift "
            "consumed all training data. Try reducing purge_gap_ms or using more data."
        )
    if not val_rows:
        raise ValueError(
            "Validation split is empty. Dataset may be too small, purge_gap_ms too large, "
            "or boundary shift consumed all validation data."
        )
    if not test_rows:
        raise ValueError(
            "Test split is empty. Dataset may be too small, purge_gap_ms too large, "
            "or boundary shift consumed all test data."
        )

    # Compute metadata
    data_hash = _compute_data_hash(sorted_rows)
    config_hash = _compute_config_hash(config)

    metadata = SplitMetadata(
        schema_version="1.0.0",
        git_sha=_get_git_sha(),
        config_hash=config_hash,
        split_timestamp=datetime.now(UTC).isoformat(),
        train_rows=len(train_rows),
        val_rows=len(val_rows),
        test_rows=len(test_rows),
        train_ts_range=_get_ts_range(train_rows, ts_col),
        val_ts_range=_get_ts_range(val_rows, ts_col),
        test_ts_range=_get_ts_range(test_rows, ts_col),
        data_hash=data_hash,
        config={
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "purge_gap_ms": config.purge_gap_ms,
            "timestamp_col": config.timestamp_col,
            "group_col": config.group_col,
        },
    )

    return SplitResult(
        train=train_rows,
        val=val_rows,
        test=test_rows,
        metadata=metadata,
    )


def save_split(
    result: SplitResult,
    output_dir: Path,
    format: str = "jsonl",
) -> None:
    """Save split result to files.

    Args:
        result: Split result to save.
        output_dir: Directory to save files.
        format: Output format ('jsonl' or 'parquet').
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        for name, data in [
            ("train", result.train),
            ("val", result.val),
            ("test", result.test),
        ]:
            path = output_dir / f"{name}.jsonl"
            with path.open("wb") as f:
                for row in data:
                    f.write(orjson.dumps(row))
                    f.write(b"\n")

    elif format == "parquet":
        try:
            import pyarrow as pa  # type: ignore[import-not-found]
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "pyarrow required for parquet. Install with: pip install pyarrow"
            ) from e

        for name, data in [
            ("train", result.train),
            ("val", result.val),
            ("test", result.test),
        ]:
            if data:
                table = pa.Table.from_pylist(data)
                pq.write_table(table, output_dir / f"{name}.parquet")

    else:
        raise ValueError(f"Unsupported format: {format}")

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("wb") as f:
        f.write(orjson.dumps(result.metadata.to_dict(), option=orjson.OPT_INDENT_2))


def load_split_metadata(metadata_path: Path) -> SplitMetadata:
    """Load split metadata from JSON file.

    Args:
        metadata_path: Path to metadata.json file.

    Returns:
        SplitMetadata object.
    """
    with metadata_path.open("rb") as f:
        data = orjson.loads(f.read())

    return SplitMetadata(
        schema_version=data["schema_version"],
        git_sha=data["git_sha"],
        config_hash=data["config_hash"],
        split_timestamp=data["split_timestamp"],
        train_rows=data["train_rows"],
        val_rows=data["val_rows"],
        test_rows=data["test_rows"],
        train_ts_range=tuple(data["train_ts_range"]),
        val_ts_range=tuple(data["val_ts_range"]),
        test_ts_range=tuple(data["test_ts_range"]),
        data_hash=data["data_hash"],
        config=data.get("config", {}),
    )
