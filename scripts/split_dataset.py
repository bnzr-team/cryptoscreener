#!/usr/bin/env python3
"""
CLI script for splitting labeled datasets into train/val/test sets.

Performs time-based splitting with strict temporal ordering to prevent
data leakage. Outputs splits with metadata for reproducibility.

Usage:
    python scripts/split_dataset.py data/labels.parquet --output-dir data/splits/
    python scripts/split_dataset.py data/labels.jsonl --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

Per PRD §11 Milestone 3 and DATASET_BUILD_PIPELINE.md:
"Split by time into train/val/test"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cryptoscreener.training.dataset import load_labeled_dataset
from cryptoscreener.training.split import SplitConfig, save_split, time_based_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Split labeled dataset into train/val/test sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/output
    parser.add_argument(
        "input",
        type=Path,
        help="Path to labeled data file (.parquet or .jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output directory for split files",
    )

    # Split ratios
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of data for training (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of data for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of data for testing (default: 0.15)",
    )

    # Options
    parser.add_argument(
        "--purge-gap-ms",
        type=int,
        default=0,
        help="Gap in milliseconds between splits to prevent leakage (default: 0)",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="ts",
        help="Timestamp column name (default: ts)",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "parquet"],
        default="jsonl",
        help="Output format (default: jsonl)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip schema validation",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        logger.error(f"Ratios must sum to 1.0, got {total_ratio}")
        return 1

    # Load data
    logger.info(f"Loading dataset from {args.input}")
    try:
        rows, validation_result = load_labeled_dataset(
            args.input,
            validate=not args.skip_validation,
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1

    logger.info(f"Loaded {len(rows)} rows")

    # Check validation - FAIL by default on schema errors
    if validation_result and not validation_result.is_valid:
        logger.error(f"Schema validation FAILED: {validation_result.errors}")
        if validation_result.missing_columns:
            logger.error(f"Missing columns: {validation_result.missing_columns}")
        logger.error("Use --skip-validation to bypass schema checks (not recommended)")
        return 1

    if len(rows) == 0:
        logger.error("Dataset is empty")
        return 1

    # Create split config
    config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        purge_gap_ms=args.purge_gap_ms,
        timestamp_col=args.timestamp_col,
    )

    # Perform split
    logger.info("Performing time-based split...")
    try:
        result = time_based_split(rows, config)
    except Exception as e:
        logger.error(f"Split failed: {e}")
        return 1

    # Verify no leakage
    if not result.verify_no_leakage():
        logger.error("CRITICAL: Temporal leakage detected in split!")
        logger.error(f"Train max ts: {result.metadata.train_ts_range[1]}")
        logger.error(f"Val min ts: {result.metadata.val_ts_range[0]}")
        logger.error(f"Test min ts: {result.metadata.test_ts_range[0]}")
        return 1

    # Print summary
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.format}")
    print()
    print(f"Train: {result.metadata.train_rows} rows")
    print(f"  Timestamp range: {result.metadata.train_ts_range}")
    print(f"Val:   {result.metadata.val_rows} rows")
    print(f"  Timestamp range: {result.metadata.val_ts_range}")
    print(f"Test:  {result.metadata.test_rows} rows")
    print(f"  Timestamp range: {result.metadata.test_ts_range}")
    print()
    print(f"Config hash: {result.metadata.config_hash}")
    print(f"Data hash: {result.metadata.data_hash}")
    print(f"Git SHA: {result.metadata.git_sha}")
    print()

    # Verify temporal ordering
    train_max = result.metadata.train_ts_range[1]
    val_min = result.metadata.val_ts_range[0]
    val_max = result.metadata.val_ts_range[1]
    test_min = result.metadata.test_ts_range[0]

    print("LEAKAGE CHECK:")
    print(f"  max(train_ts) < min(val_ts): {train_max} < {val_min} = {train_max < val_min}")
    print(f"  max(val_ts) < min(test_ts): {val_max} < {test_min} = {val_max < test_min}")
    print(f"  Status: {'PASS' if result.verify_no_leakage() else 'FAIL'}")
    print("=" * 60)

    # Save splits
    logger.info(f"Saving splits to {args.output_dir}")
    try:
        save_split(result, args.output_dir, format=args.format)
    except Exception as e:
        logger.error(f"Failed to save splits: {e}")
        return 1

    logger.info("Split complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
