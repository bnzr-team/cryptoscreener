#!/usr/bin/env python3
"""
CLI script for fitting probability calibrators on validation data.

Fits Platt scaling calibrators for specified prediction heads
and outputs calibration artifacts with metrics before/after.

Usage:
    python scripts/fit_calibration.py data/splits/val.jsonl --output calibration.json
    python scripts/fit_calibration.py data/splits/val.jsonl --heads p_inplay_30s p_toxic

Per PRD §11 Milestone 3: "Training pipeline skeleton"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import orjson

from cryptoscreener.backtest.metrics import compute_brier_score, compute_ece
from cryptoscreener.calibration.artifact import (
    CalibrationArtifact,
    create_calibration_metadata,
    save_calibration_artifact,
)
from cryptoscreener.calibration.platt import fit_platt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default heads to calibrate (from LABELS_SPEC.md)
DEFAULT_HEADS = [
    "p_inplay_30s",
    "p_inplay_2m",
    "p_inplay_5m",
    "p_toxic",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file."""
    rows = []
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                rows.append(orjson.loads(line))
    return rows


def extract_head_data(
    rows: list[dict[str, Any]],
    prob_col: str,
    label_col: str,
) -> tuple[list[int], list[float]]:
    """Extract labels and probabilities for a head.

    Args:
        rows: Data rows.
        prob_col: Column name for probabilities.
        label_col: Column name for labels.

    Returns:
        Tuple of (y_true, p_raw).

    Raises:
        ValueError: If columns are missing or data is invalid.
    """
    y_true = []
    p_raw = []

    for i, row in enumerate(rows):
        if prob_col not in row:
            raise ValueError(f"Row {i}: missing probability column '{prob_col}'")
        if label_col not in row:
            raise ValueError(f"Row {i}: missing label column '{label_col}'")

        p = float(row[prob_col])
        y = int(row[label_col])

        if not 0 <= p <= 1:
            raise ValueError(f"Row {i}: probability {p} not in [0, 1]")
        if y not in (0, 1):
            raise ValueError(f"Row {i}: label {y} not in {{0, 1}}")

        p_raw.append(p)
        y_true.append(y)

    return y_true, p_raw


def compute_metrics(y_true: list[int], probs: list[float]) -> dict[str, float]:
    """Compute Brier score and ECE."""
    brier = compute_brier_score(y_true, probs)
    calibration = compute_ece(y_true, probs)

    return {"brier": brier, "ece": calibration.ece}


def get_label_column(head_name: str) -> str:
    """Map probability column to label column.

    Mapping:
        p_inplay_{horizon} -> i_tradeable_{horizon}_a (using profile 'a')
        p_toxic -> y_toxic
    """
    if head_name.startswith("p_inplay_"):
        horizon = head_name.replace("p_inplay_", "")
        return f"i_tradeable_{horizon}_a"
    elif head_name == "p_toxic":
        return "y_toxic"
    else:
        raise ValueError(f"Unknown head: {head_name}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fit probability calibrators on validation data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to validation data file (.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("calibration.json"),
        help="Output path for calibration artifact (default: calibration.json)",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        default=DEFAULT_HEADS,
        help=f"Prediction heads to calibrate (default: {DEFAULT_HEADS})",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum iterations for Platt fitting (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for Platt fitting (default: 0.1)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Load data
    logger.info(f"Loading validation data from {args.input}")
    try:
        rows = load_jsonl(args.input)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    if len(rows) == 0:
        logger.error("Validation data is empty")
        return 1

    logger.info(f"Loaded {len(rows)} rows")

    # Check which heads are available
    available_heads = []
    for head in args.heads:
        prob_col = head
        try:
            label_col = get_label_column(head)
        except ValueError as e:
            logger.warning(f"Skipping {head}: {e}")
            continue

        if prob_col not in rows[0]:
            logger.warning(f"Skipping {head}: probability column '{prob_col}' not in data")
            continue
        if label_col not in rows[0]:
            logger.warning(f"Skipping {head}: label column '{label_col}' not in data")
            continue

        available_heads.append(head)

    if not available_heads:
        logger.error("No valid heads found in data")
        return 1

    logger.info(f"Calibrating heads: {available_heads}")

    # Fit calibrators
    from cryptoscreener.calibration.platt import PlattCalibrator

    calibrators: dict[str, PlattCalibrator] = {}
    metrics_before: dict[str, dict[str, float]] = {}
    metrics_after: dict[str, dict[str, float]] = {}

    for head in available_heads:
        prob_col = head
        label_col = get_label_column(head)

        logger.info(f"Fitting calibrator for {head}...")

        try:
            y_true, p_raw = extract_head_data(rows, prob_col, label_col)
        except ValueError as e:
            logger.error(f"Failed to extract data for {head}: {e}")
            return 1

        # Compute metrics before calibration
        before = compute_metrics(y_true, p_raw)
        metrics_before[head] = before
        logger.info(f"  Before: Brier={before['brier']:.4f}, ECE={before['ece']:.4f}")

        # Fit calibrator
        try:
            calibrator = fit_platt(
                y_true,
                p_raw,
                head,
                max_iter=args.max_iter,
                lr=args.lr,
            )
        except ValueError as e:
            logger.error(f"Failed to fit calibrator for {head}: {e}")
            return 1

        calibrators[head] = calibrator
        logger.info(f"  Fitted: a={calibrator.a:.4f}, b={calibrator.b:.4f}")

        # Compute metrics after calibration
        p_cal = calibrator.transform_batch(p_raw)
        after = compute_metrics(y_true, p_cal)
        metrics_after[head] = after
        logger.info(f"  After:  Brier={after['brier']:.4f}, ECE={after['ece']:.4f}")

        # Report improvement
        brier_delta = after["brier"] - before["brier"]
        ece_delta = after["ece"] - before["ece"]
        logger.info(
            f"  Delta:  Brier={brier_delta:+.4f}, ECE={ece_delta:+.4f}"
        )

    # Create metadata
    config = {
        "method": "platt",
        "max_iter": args.max_iter,
        "lr": args.lr,
        "heads": available_heads,
    }

    metadata = create_calibration_metadata(
        method="platt",
        heads=available_heads,
        n_samples=len(rows),
        config=config,
        val_data=rows,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
    )

    # Create artifact
    artifact = CalibrationArtifact(
        calibrators=calibrators,
        metadata=metadata,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("Method: platt")
    print(f"Samples: {len(rows)}")
    print()
    print("HEADS:")
    for head in available_heads:
        cal = calibrators[head]
        before = metrics_before[head]
        after = metrics_after[head]
        print(f"  {head}:")
        print(f"    Parameters: a={cal.a:.4f}, b={cal.b:.4f}")
        print(f"    Brier: {before['brier']:.4f} -> {after['brier']:.4f}")
        print(f"    ECE:   {before['ece']:.4f} -> {after['ece']:.4f}")
    print()
    print(f"Config hash: {metadata.config_hash}")
    print(f"Data hash: {metadata.data_hash}")
    print(f"Git SHA: {metadata.git_sha}")
    print("=" * 60)

    # Save artifact
    logger.info(f"Saving calibration artifact to {args.output}")
    try:
        save_calibration_artifact(artifact, args.output)
    except Exception as e:
        logger.error(f"Failed to save artifact: {e}")
        return 1

    logger.info("Calibration complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
