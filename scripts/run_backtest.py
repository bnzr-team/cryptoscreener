#!/usr/bin/env python3
"""
CLI script for running offline backtests.

Evaluates model predictions (or label quality) against ground truth.

Usage:
    # Evaluate label quality (no model):
    python scripts/run_backtest.py data/labels.parquet --output results/backtest.json

    # With model predictions (future):
    python scripts/run_backtest.py data/labels.parquet --predictions preds.jsonl --output results.json

Per PRD §11 Milestone 2: "Label builder + offline backtest harness"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cryptoscreener.backtest.harness import (
    BacktestConfig,
    BacktestHarness,
    load_labeled_data,
    print_backtest_summary,
    save_backtest_result,
)
from cryptoscreener.cost_model.calculator import Profile
from cryptoscreener.label_builder import Horizon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run offline backtest evaluation on labeled data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/output
    parser.add_argument(
        "input",
        type=Path,
        help="Path to labeled data file (.parquet or .jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for results JSON (default: stdout only)",
    )

    # Config options
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="K value for top-K metrics (default: 20)",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="30s,2m,5m",
        help="Comma-separated horizons to evaluate (default: 30s,2m,5m)",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="A,B",
        help="Comma-separated profiles to evaluate (default: A,B)",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of bins for ECE calculation (default: 10)",
    )

    # Mode
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress summary output",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Parse horizons
    horizon_map = {
        "30s": Horizon.H_30S,
        "2m": Horizon.H_2M,
        "5m": Horizon.H_5M,
    }
    try:
        horizons = tuple(horizon_map[h.strip()] for h in args.horizons.split(",") if h.strip())
    except KeyError as e:
        logger.error(f"Invalid horizon: {e}. Valid: 30s, 2m, 5m")
        return 1

    # Parse profiles
    profile_map = {"A": Profile.A, "B": Profile.B}
    try:
        profiles = tuple(
            profile_map[p.strip().upper()] for p in args.profiles.split(",") if p.strip()
        )
    except KeyError as e:
        logger.error(f"Invalid profile: {e}. Valid: A, B")
        return 1

    # Create config
    config = BacktestConfig(
        horizons=horizons,
        profiles=profiles,
        top_k=args.top_k,
        calibration_bins=args.calibration_bins,
    )

    logger.info(f"Loading labeled data from {args.input}")
    try:
        rows = load_labeled_data(args.input)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    logger.info(f"Loaded {len(rows)} rows")

    if len(rows) == 0:
        logger.error("No data to evaluate")
        return 1

    # Run evaluation
    logger.info("Running backtest evaluation...")
    harness = BacktestHarness(config)
    result = harness.evaluate_labels_only(rows)

    # Print summary
    if not args.quiet:
        print_backtest_summary(result)

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_backtest_result(result, args.output)
        logger.info(f"Results saved to {args.output}")

    # Return exit code based on acceptance criteria
    # ECE should be < 5% for all horizons
    all_pass = all(r.metrics.calibration.ece < 0.05 for r in result.results)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
