"""
Dataset loading and schema validation for ML training.

Loads labeled data from label_builder output and validates schema.
Ensures compatibility between offline training and online inference.

Per DATASET_BUILD_PIPELINE.md:
"Build features using the SAME feature library as online"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING, Any

import orjson

if TYPE_CHECKING:
    from collections.abc import Sequence

# Schema version for dataset format
DATASET_SCHEMA_VERSION = "1.0.0"

# Required columns from label_builder output
REQUIRED_LABEL_COLUMNS = [
    "ts",
    "symbol",
    "mid_price",
    "spread_bps",
]

# Label columns for each horizon/profile
HORIZON_PROFILE_COLUMNS = [
    "i_tradeable_{h}_{p}",
    "net_edge_bps_{h}_{p}",
    "mfe_bps_{h}_{p}",
    "mae_bps_{h}_{p}",
    "cost_bps_{h}_{p}",
]

# Toxicity columns
TOXICITY_COLUMNS = [
    "y_toxic",
    "severity_toxic_bps",
]


@dataclass(frozen=True)
class DatasetSchema:
    """Schema definition for labeled dataset.

    Attributes:
        version: Schema version string.
        required_columns: List of required columns.
        label_columns: List of label columns (with {h}/{p} placeholders).
        horizons: Available horizons.
        profiles: Available profiles.
    """

    version: str
    required_columns: list[str]
    label_columns: list[str]
    horizons: list[str]
    profiles: list[str]

    def get_all_columns(self) -> list[str]:
        """Get all expected columns after expanding placeholders."""
        columns = list(self.required_columns)

        for h in self.horizons:
            for p in self.profiles:
                for col_template in self.label_columns:
                    col = col_template.format(h=h, p=p)
                    columns.append(col)

        columns.extend(TOXICITY_COLUMNS)
        return columns


# Default schema for label_builder output
DEFAULT_SCHEMA = DatasetSchema(
    version=DATASET_SCHEMA_VERSION,
    required_columns=REQUIRED_LABEL_COLUMNS,
    label_columns=HORIZON_PROFILE_COLUMNS,
    horizons=["30s", "2m", "5m"],
    profiles=["a", "b"],
)


@dataclass
class ValidationResult:
    """Result of schema validation.

    Attributes:
        is_valid: Whether validation passed.
        missing_columns: List of missing required columns.
        extra_columns: List of unexpected columns.
        row_count: Number of rows validated.
        errors: List of validation error messages.
    """

    is_valid: bool
    missing_columns: list[str]
    extra_columns: list[str]
    row_count: int
    errors: list[str]


def validate_schema(
    rows: Sequence[dict[str, Any]],
    schema: DatasetSchema | None = None,
    strict: bool = False,
) -> ValidationResult:
    """Validate dataset against schema.

    Args:
        rows: Data rows to validate.
        schema: Schema to validate against. Uses DEFAULT_SCHEMA if not provided.
        strict: If True, fail on extra columns.

    Returns:
        ValidationResult with validation details.
    """
    if schema is None:
        schema = DEFAULT_SCHEMA

    if not rows:
        return ValidationResult(
            is_valid=False,
            missing_columns=[],
            extra_columns=[],
            row_count=0,
            errors=["Dataset is empty"],
        )

    expected_columns = set(schema.get_all_columns())
    actual_columns = set(rows[0].keys())

    missing = list(expected_columns - actual_columns)
    extra = list(actual_columns - expected_columns)

    errors: list[str] = []

    if missing:
        errors.append(f"Missing columns: {missing}")

    if strict and extra:
        errors.append(f"Unexpected columns: {extra}")

    # Check for required base columns (always required)
    for col in schema.required_columns:
        if col not in actual_columns:
            errors.append(f"Missing required column: {col}")

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        missing_columns=missing,
        extra_columns=extra,
        row_count=len(rows),
        errors=errors,
    )


def load_labeled_dataset(
    input_path: Path,
    validate: bool = True,
    schema: DatasetSchema | None = None,
) -> tuple[list[dict[str, Any]], ValidationResult | None]:
    """Load labeled dataset from file.

    Args:
        input_path: Path to labeled data file (.parquet or .jsonl).
        validate: Whether to validate schema.
        schema: Schema to validate against.

    Returns:
        Tuple of (rows, validation_result).
        validation_result is None if validate=False.

    Raises:
        ValueError: If file format is unsupported.
        FileNotFoundError: If file doesn't exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "pyarrow required for parquet. Install with: pip install pyarrow"
            ) from e

        table = pq.read_table(input_path)
        rows: list[dict[str, Any]] = table.to_pylist()

    elif suffix in (".jsonl", ".json"):
        rows = []
        with input_path.open("rb") as f:
            for line in f:
                if line.strip():
                    rows.append(orjson.loads(line))

    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    validation_result = None
    if validate:
        validation_result = validate_schema(rows, schema)

    return rows, validation_result


def get_feature_columns(
    rows: Sequence[dict[str, Any]],
    exclude_prefixes: Sequence[str] | None = None,
) -> list[str]:
    """Get feature column names from dataset.

    Extracts columns that are not labels or metadata.

    Args:
        rows: Data rows.
        exclude_prefixes: Column prefixes to exclude.

    Returns:
        List of feature column names.
    """
    if not rows:
        return []

    if exclude_prefixes is None:
        exclude_prefixes = [
            "i_tradeable_",
            "net_edge_bps_",
            "mfe_bps_",
            "mae_bps_",
            "cost_bps_",
            "y_toxic",
            "severity_toxic_bps",
        ]

    all_columns = list(rows[0].keys())

    feature_columns = []
    for col in all_columns:
        is_excluded = any(col.startswith(prefix) for prefix in exclude_prefixes)
        if not is_excluded and col not in ("ts", "symbol"):
            feature_columns.append(col)

    return sorted(feature_columns)


def get_label_columns(
    rows: Sequence[dict[str, Any]],
    horizons: Sequence[str] | None = None,
    profiles: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Get label column names organized by type.

    Args:
        rows: Data rows.
        horizons: Horizons to include. Defaults to ["30s", "2m", "5m"].
        profiles: Profiles to include. Defaults to ["a", "b"].

    Returns:
        Dictionary with keys 'tradeable', 'edge', 'toxicity' mapping to column lists.
    """
    if not rows:
        return {"tradeable": [], "edge": [], "toxicity": []}

    if horizons is None:
        horizons = ["30s", "2m", "5m"]
    if profiles is None:
        profiles = ["a", "b"]

    all_columns = set(rows[0].keys())

    tradeable_cols = []
    edge_cols = []

    for h in horizons:
        for p in profiles:
            t_col = f"i_tradeable_{h}_{p}"
            e_col = f"net_edge_bps_{h}_{p}"
            if t_col in all_columns:
                tradeable_cols.append(t_col)
            if e_col in all_columns:
                edge_cols.append(e_col)

    toxicity_cols = [c for c in TOXICITY_COLUMNS if c in all_columns]

    return {
        "tradeable": tradeable_cols,
        "edge": edge_cols,
        "toxicity": toxicity_cols,
    }
