#!/usr/bin/env python3
"""
CLI script for training ML models (DEC-038).

Trains sklearn models with calibration and produces versioned artifact packages.

Usage:
    python scripts/train_model.py --data data/labels.parquet --output models/v1.0.0/
    python scripts/train_model.py --data data/labels.parquet --output models/v1.0.0/ --seed 42 --model-type logistic

Per PRD §11 Milestone 3: "Training pipeline skeleton"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from cryptoscreener.calibration.artifact import (
    CalibrationArtifact,
    create_calibration_metadata,
)
from cryptoscreener.calibration.platt import NegativeSlopeError, PlattCalibrator, fit_platt
from cryptoscreener.training.artifact import build_model_package, generate_model_version
from cryptoscreener.training.dataset import load_labeled_dataset
from cryptoscreener.training.feature_schema import (
    PREDICTION_HEADS,
)
from cryptoscreener.training.trainer import HeadMetrics, Trainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def time_split_data(
    rows: list[dict[str, Any]],
    val_ratio: float,
    timestamp_col: str = "ts",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split data by time into train and validation sets.

    Args:
        rows: Data rows sorted by timestamp.
        val_ratio: Fraction of data for validation.
        timestamp_col: Column name for timestamps.

    Returns:
        Tuple of (train_rows, val_rows).
    """
    # Sort by timestamp
    sorted_rows = sorted(rows, key=lambda r: r[timestamp_col])

    n = len(sorted_rows)
    split_idx = int(n * (1 - val_ratio))

    train_rows = sorted_rows[:split_idx]
    val_rows = sorted_rows[split_idx:]

    logger.info(
        f"Time-based split: {len(train_rows)} train, {len(val_rows)} val (val_ratio={val_ratio})"
    )

    if train_rows and val_rows:
        train_max_ts = train_rows[-1][timestamp_col]
        val_min_ts = val_rows[0][timestamp_col]
        logger.info(f"Split boundary: train_max_ts={train_max_ts}, val_min_ts={val_min_ts}")

    return train_rows, val_rows


def fit_calibration(
    trainer: Trainer,
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_rows: list[dict[str, Any]],
    config: TrainingConfig,
) -> tuple[CalibrationArtifact, dict[str, HeadMetrics]]:
    """Fit Platt calibrators on validation set.

    Args:
        trainer: Trainer instance (for predict_proba).
        model: Trained model.
        X_val: Validation features.
        y_val: Validation labels.
        val_rows: Validation data rows for provenance.
        config: Training configuration for metadata.

    Returns:
        Tuple of (CalibrationArtifact, calibrated_metrics).
    """
    n_samples = len(val_rows)
    proba_dict = trainer.predict_proba(model, X_val)

    calibrators: dict[str, PlattCalibrator] = {}
    metrics_before: dict[str, dict[str, float]] = {}
    metrics_after: dict[str, dict[str, float]] = {}
    calibrated_metrics: dict[str, HeadMetrics] = {}

    for i, head in enumerate(PREDICTION_HEADS):
        y_true = y_val[:, i].tolist()
        p_raw = proba_dict[head].tolist()

        # Compute metrics before calibration
        from cryptoscreener.backtest.metrics import compute_brier_score, compute_ece

        brier_before = compute_brier_score(y_true, p_raw)
        ece_before = compute_ece(y_true, p_raw).ece

        metrics_before[head] = {"brier": brier_before, "ece": ece_before}

        # Fit Platt calibrator
        try:
            calibrator = fit_platt(y_true, p_raw, head)
            logger.info(f"{head}: Platt fit a={calibrator.a:.4f}, b={calibrator.b:.4f}")
        except NegativeSlopeError:
            logger.warning(f"{head}: Platt fitting failed (negative slope), using identity")
            calibrator = PlattCalibrator(head_name=head, a=1.0, b=0.0)

        calibrators[head] = calibrator

        # Compute metrics after calibration
        p_cal = [calibrator.transform(p) for p in p_raw]
        brier_after = compute_brier_score(y_true, p_cal)
        ece_after = compute_ece(y_true, p_cal).ece

        metrics_after[head] = {"brier": brier_after, "ece": ece_after}

        logger.info(
            f"{head}: Brier {brier_before:.4f} -> {brier_after:.4f}, "
            f"ECE {ece_before:.4f} -> {ece_after:.4f}"
        )

        # Store calibrated metrics
        from cryptoscreener.backtest.metrics import compute_auc, compute_pr_auc

        calibrated_metrics[head] = HeadMetrics(
            head_name=head,
            auc=compute_auc(y_true, p_cal),
            pr_auc=compute_pr_auc(y_true, p_cal),
            brier=brier_after,
            ece=ece_after,
            n_samples=len(y_true),
            n_positives=sum(y_true),
        )

    # Create calibration metadata
    config_dict = {
        "model_type": config.model_type,
        "seed": config.seed,
        "val_ratio": config.val_ratio,
        "profile": config.profile,
    }
    metadata = create_calibration_metadata(
        method="platt",
        heads=list(PREDICTION_HEADS),
        n_samples=n_samples,
        config=config_dict,
        val_data=val_rows,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
    )

    artifact = CalibrationArtifact(calibrators=calibrators, metadata=metadata)

    return artifact, calibrated_metrics


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train ML model and produce artifact package.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/output
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to labeled data file (.parquet or .jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for model artifacts",
    )

    # Training options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)",
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "logistic"],
        default="random_forest",
        help="Model type (default: random_forest)",
    )
    parser.add_argument(
        "--profile",
        choices=["a", "b"],
        default="a",
        help="Label profile to use (default: a)",
    )

    # RandomForest options
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees for RandomForest (default: 100)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Max tree depth for RandomForest (default: 10)",
    )

    # Logistic options
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization strength for Logistic (default: 1.0)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Max iterations for Logistic (default: 1000)",
    )

    # Version
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Model version string (auto-generated if not provided)",
    )

    # Flags
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip calibration fitting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load data
    logger.info(f"Loading data from {args.data}")
    rows, validation = load_labeled_dataset(args.data, validate=True)

    if validation and not validation.is_valid:
        logger.error(f"Dataset validation failed: {validation.errors}")
        return 1

    logger.info(f"Loaded {len(rows)} rows")

    # Time-based split
    train_rows, val_rows = time_split_data(rows, args.val_ratio)

    # Create training config
    config = TrainingConfig(
        seed=args.seed,
        val_ratio=args.val_ratio,
        model_type=args.model_type,
        profile=args.profile,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        C=args.C,
        max_iter=args.max_iter,
    )

    # Train model
    logger.info("Training model...")
    trainer = Trainer(config)

    X_train, y_train = trainer.prepare_data(train_rows)
    X_val, y_val = trainer.prepare_data(val_rows)

    result = trainer.train_and_evaluate(X_train, y_train, X_val, y_val)

    logger.info("Raw model metrics:")
    for head, m in result.metrics.items():
        logger.info(f"  {head}: AUC={m.auc:.4f}, PR-AUC={m.pr_auc:.4f}")

    # Fit calibration
    calibration: CalibrationArtifact | None = None
    calibrated_metrics: dict[str, HeadMetrics] | None = None

    if not args.no_calibration:
        logger.info("Fitting calibration...")
        calibration, calibrated_metrics = fit_calibration(
            trainer=trainer,
            model=result.model,
            X_val=X_val,
            y_val=y_val,
            val_rows=val_rows,
            config=config,
        )

    # Generate version
    model_version = args.version or generate_model_version()
    logger.info(f"Model version: {model_version}")

    # Build artifact package
    logger.info(f"Building artifact package in {args.output}")
    artifact_result = build_model_package(
        output_dir=args.output,
        model=result.model,
        config=config,
        metrics=result.metrics,
        calibration=calibration,
        metrics_calibrated=calibrated_metrics,
        train_samples=len(train_rows),
        val_samples=len(val_rows),
        model_version=model_version,
    )

    logger.info("Artifact package contents:")
    for name, sha256 in sorted(artifact_result.checksums.items()):
        logger.info(f"  {name}: {sha256[:16]}...")

    logger.info(f"Training complete. Artifacts saved to {args.output}")
    logger.info(f"  Version: {artifact_result.model_version}")
    logger.info(f"  Model: {args.output / 'model.pkl'}")
    if calibration:
        logger.info(f"  Calibration: {args.output / 'calibration.json'}")
    logger.info(f"  Report: {args.output / 'training_report.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
