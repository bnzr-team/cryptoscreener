"""
Backtest harness for evaluating model predictions against labels.

This module provides the infrastructure to:
1. Load labeled datasets (from label_builder output)
2. Run model predictions on historical data
3. Evaluate predictions using backtest metrics
4. Generate evaluation reports

Per PRD §11 Milestone 2: "Label builder + offline backtest harness"
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING, Any, Protocol

import orjson

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from cryptoscreener.backtest.metrics import (
    BacktestMetrics,
    compute_all_metrics,
)
from cryptoscreener.cost_model.calculator import Profile
from cryptoscreener.label_builder import Horizon


class Predictor(Protocol):
    """Protocol for model predictors.

    Models must implement this interface to be evaluated by the harness.
    """

    def predict(
        self,
        features: dict[str, Any],
    ) -> dict[str, float]:
        """Generate predictions for a single sample.

        Args:
            features: Feature dictionary for the sample.

        Returns:
            Dictionary with prediction keys like:
            - p_inplay_30s, p_inplay_2m, p_inplay_5m
            - p_toxic
        """
        ...


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for backtest evaluation.

    Attributes:
        horizons: Which horizons to evaluate.
        profiles: Which profiles to evaluate.
        top_k: Value of K for top-K metrics.
        calibration_bins: Number of bins for ECE calculation.
    """

    horizons: tuple[Horizon, ...] = (Horizon.H_30S, Horizon.H_2M, Horizon.H_5M)
    profiles: tuple[Profile, ...] = (Profile.A, Profile.B)
    top_k: int = 20
    calibration_bins: int = 10


@dataclass
class HorizonProfileResult:
    """Evaluation results for a specific horizon/profile combination.

    Attributes:
        horizon: Prediction horizon.
        profile: Execution profile.
        metrics: Computed backtest metrics.
    """

    horizon: Horizon
    profile: Profile
    metrics: BacktestMetrics


@dataclass
class BacktestResult:
    """Complete backtest evaluation result.

    Attributes:
        results: Per-horizon/profile results.
        toxicity_metrics: Metrics for toxicity prediction.
        config: Configuration used.
        metadata: Run metadata (git sha, timestamp, etc.).
    """

    results: list[HorizonProfileResult]
    toxicity_metrics: BacktestMetrics | None
    config: BacktestConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_result(
        self,
        horizon: Horizon,
        profile: Profile,
    ) -> HorizonProfileResult | None:
        """Get result for specific horizon/profile."""
        for r in self.results:
            if r.horizon == horizon and r.profile == profile:
                return r
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "config": {
                "horizons": [h.value for h in self.config.horizons],
                "profiles": [p.value for p in self.config.profiles],
                "top_k": self.config.top_k,
                "calibration_bins": self.config.calibration_bins,
            },
            "metadata": self.metadata,
            "results": {},
        }

        for r in self.results:
            key = f"{r.horizon.value}_{r.profile.value}"
            result["results"][key] = {
                "auc": r.metrics.auc,
                "pr_auc": r.metrics.pr_auc,
                "brier_score": r.metrics.calibration.brier_score,
                "ece": r.metrics.calibration.ece,
                "mce": r.metrics.calibration.mce,
                "topk_capture": r.metrics.topk.capture_rate,
                "topk_mean_edge_bps": r.metrics.topk.mean_edge_bps,
                "topk_precision": r.metrics.topk.precision_at_k,
                "n_samples": r.metrics.n_samples,
                "n_positives": r.metrics.n_positives,
            }

            if r.metrics.churn:
                result["results"][key]["churn"] = {
                    "state_changes_per_step": r.metrics.churn.state_changes_per_step,
                    "jaccard_similarity": r.metrics.churn.jaccard_similarity,
                }

        if self.toxicity_metrics:
            result["toxicity"] = {
                "auc": self.toxicity_metrics.auc,
                "pr_auc": self.toxicity_metrics.pr_auc,
                "brier_score": self.toxicity_metrics.calibration.brier_score,
                "ece": self.toxicity_metrics.calibration.ece,
                "n_samples": self.toxicity_metrics.n_samples,
                "n_positives": self.toxicity_metrics.n_positives,
            }

        return result


def load_labeled_data(
    input_path: Path,
) -> list[dict[str, Any]]:
    """Load labeled dataset from parquet or JSONL.

    Args:
        input_path: Path to labeled data file.

    Returns:
        List of label rows as dictionaries.
    """
    suffix = input_path.suffix.lower()

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "pyarrow required for parquet. Install with: pip install pyarrow"
            ) from e

        table = pq.read_table(input_path)
        result: list[dict[str, Any]] = table.to_pylist()
        return result

    elif suffix in (".jsonl", ".json"):
        rows: list[dict[str, Any]] = []
        with input_path.open("rb") as f:
            for line in f:
                if line.strip():
                    rows.append(orjson.loads(line))
        return rows

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


class BacktestHarness:
    """Harness for running offline backtests.

    Evaluates model predictions against labeled ground truth data.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        """Initialize harness.

        Args:
            config: Backtest configuration. Uses defaults if not provided.
        """
        self.config = config or BacktestConfig()

    def evaluate_labels_only(
        self,
        rows: Sequence[dict[str, Any]],
    ) -> BacktestResult:
        """Evaluate using labels only (no model predictions).

        This is useful for baseline analysis: evaluating label quality
        and understanding the distribution of tradeable events.

        Uses net_edge_bps as a "perfect score" predictor.

        Args:
            rows: Labeled data rows from label_builder.

        Returns:
            BacktestResult with metrics for each horizon/profile.
        """
        results: list[HorizonProfileResult] = []

        for horizon in self.config.horizons:
            for profile in self.config.profiles:
                h_key = horizon.value.replace("m", "m").replace("s", "s")
                p_key = profile.value.lower()

                # Extract labels and "scores" (using net_edge as perfect predictor)
                i_key = f"i_tradeable_{h_key}_{p_key}"
                edge_key = f"net_edge_bps_{h_key}_{p_key}"

                y_true: list[int] = []
                y_scores: list[float] = []
                net_edges: list[float] = []
                timestamps: list[int] = []
                symbols: list[str] = []

                for row in rows:
                    if i_key in row and edge_key in row:
                        y_true.append(int(row[i_key]))
                        # Use net_edge as score (higher edge = better)
                        edge = float(row[edge_key])
                        net_edges.append(edge)
                        # Normalize to [0, 1] for "probability-like" score
                        # Assuming edge typically in [-100, 100] bps
                        y_scores.append(max(0.0, min(1.0, (edge + 50) / 100)))
                        timestamps.append(int(row.get("ts", 0)))
                        symbols.append(str(row.get("symbol", "")))

                if not y_true:
                    continue

                metrics = compute_all_metrics(
                    y_true=y_true,
                    y_probs=y_scores,
                    net_edge_bps=net_edges,
                    k=self.config.top_k,
                    timestamps=timestamps if any(timestamps) else None,
                    symbols=symbols if any(symbols) else None,
                )

                results.append(
                    HorizonProfileResult(
                        horizon=horizon,
                        profile=profile,
                        metrics=metrics,
                    )
                )

        # Toxicity metrics
        toxicity_metrics = None
        y_toxic: list[int] = []
        toxic_scores: list[float] = []
        toxic_edges: list[float] = []

        for row in rows:
            if "y_toxic" in row:
                y_toxic.append(int(row["y_toxic"]))
                # Use severity as score
                severity = float(row.get("severity_toxic_bps", 0))
                toxic_scores.append(max(0.0, min(1.0, severity / 50)))
                toxic_edges.append(0.0)  # No edge concept for toxicity

        if y_toxic:
            toxicity_metrics = compute_all_metrics(
                y_true=y_toxic,
                y_probs=toxic_scores,
                net_edge_bps=toxic_edges,
                k=self.config.top_k,
            )

        # Build metadata
        metadata = self._build_metadata(len(rows))

        return BacktestResult(
            results=results,
            toxicity_metrics=toxicity_metrics,
            config=self.config,
            metadata=metadata,
        )

    def evaluate_with_predictor(
        self,
        rows: Sequence[dict[str, Any]],
        predictor: Predictor,
        feature_extractor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> BacktestResult:
        """Evaluate model predictions against labels.

        Args:
            rows: Labeled data rows from label_builder.
            predictor: Model predictor implementing Predictor protocol.
            feature_extractor: Optional function to extract features from rows.
                If not provided, uses row directly as features.

        Returns:
            BacktestResult with metrics for each horizon/profile.
        """
        # Generate predictions for all rows
        predictions: list[dict[str, float]] = []

        for row in rows:
            features = feature_extractor(row) if feature_extractor else row
            pred = predictor.predict(features)
            predictions.append(pred)

        # Evaluate each horizon/profile
        results: list[HorizonProfileResult] = []

        for horizon in self.config.horizons:
            for profile in self.config.profiles:
                h_key = horizon.value
                p_key = profile.value.lower()

                # Build column keys
                i_key = f"i_tradeable_{h_key}_{p_key}"
                edge_key = f"net_edge_bps_{h_key}_{p_key}"
                pred_key = f"p_inplay_{h_key}"

                y_true: list[int] = []
                y_scores: list[float] = []
                net_edges: list[float] = []
                timestamps: list[int] = []
                symbols: list[str] = []

                for row, pred in zip(rows, predictions, strict=True):
                    if i_key in row and pred_key in pred:
                        y_true.append(int(row[i_key]))
                        y_scores.append(float(pred[pred_key]))
                        net_edges.append(float(row.get(edge_key, 0)))
                        timestamps.append(int(row.get("ts", 0)))
                        symbols.append(str(row.get("symbol", "")))

                if not y_true:
                    continue

                metrics = compute_all_metrics(
                    y_true=y_true,
                    y_probs=y_scores,
                    net_edge_bps=net_edges,
                    k=self.config.top_k,
                    timestamps=timestamps if any(timestamps) else None,
                    symbols=symbols if any(symbols) else None,
                )

                results.append(
                    HorizonProfileResult(
                        horizon=horizon,
                        profile=profile,
                        metrics=metrics,
                    )
                )

        # Toxicity evaluation
        toxicity_metrics = None
        y_toxic: list[int] = []
        toxic_scores: list[float] = []
        toxic_edges: list[float] = []

        for row, pred in zip(rows, predictions, strict=True):
            if "y_toxic" in row and "p_toxic" in pred:
                y_toxic.append(int(row["y_toxic"]))
                toxic_scores.append(float(pred["p_toxic"]))
                toxic_edges.append(0.0)

        if y_toxic:
            toxicity_metrics = compute_all_metrics(
                y_true=y_toxic,
                y_probs=toxic_scores,
                net_edge_bps=toxic_edges,
                k=self.config.top_k,
            )

        metadata = self._build_metadata(len(rows))

        return BacktestResult(
            results=results,
            toxicity_metrics=toxicity_metrics,
            config=self.config,
            metadata=metadata,
        )

    def _build_metadata(self, n_rows: int) -> dict[str, Any]:
        """Build run metadata."""
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_sha": self._get_git_sha(),
            "n_rows": n_rows,
        }

    def _get_git_sha(self) -> str:
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


def save_backtest_result(
    result: BacktestResult,
    output_path: Path,
) -> None:
    """Save backtest results to JSON file.

    Args:
        result: Backtest result to save.
        output_path: Path to output JSON file.
    """
    result_dict = result.to_dict()
    with output_path.open("wb") as f:
        f.write(orjson.dumps(result_dict, option=orjson.OPT_INDENT_2))


def print_backtest_summary(result: BacktestResult) -> None:
    """Print human-readable backtest summary to stdout."""
    print("\n" + "=" * 70)
    print("BACKTEST EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Timestamp: {result.metadata.get('timestamp', 'N/A')}")
    print(f"Git SHA: {result.metadata.get('git_sha', 'N/A')}")
    print(f"Rows evaluated: {result.metadata.get('n_rows', 0)}")
    print(f"Top-K: {result.config.top_k}")
    print()

    print("-" * 70)
    print(
        f"{'Horizon/Profile':<15} {'AUC':>8} {'PR-AUC':>8} {'Brier':>8} {'ECE':>8} {'TopK%':>8} {'Edge':>8}"
    )
    print("-" * 70)

    for r in result.results:
        key = f"{r.horizon.value}/{r.profile.value}"
        m = r.metrics
        print(
            f"{key:<15} "
            f"{m.auc:>8.4f} "
            f"{m.pr_auc:>8.4f} "
            f"{m.calibration.brier_score:>8.4f} "
            f"{m.calibration.ece:>8.4f} "
            f"{m.topk.capture_rate * 100:>7.1f}% "
            f"{m.topk.mean_edge_bps:>7.1f}"
        )

    if result.toxicity_metrics:
        print("-" * 70)
        t = result.toxicity_metrics
        print(
            f"{'Toxicity':<15} "
            f"{t.auc:>8.4f} "
            f"{t.pr_auc:>8.4f} "
            f"{t.calibration.brier_score:>8.4f} "
            f"{t.calibration.ece:>8.4f} "
            f"{'N/A':>8} "
            f"{'N/A':>8}"
        )

    print("=" * 70)

    # Acceptance criteria check (per PRD §10.3)
    print("\nACCEPTANCE CRITERIA:")
    for r in result.results:
        # ECE < 5% target
        ece_ok = r.metrics.calibration.ece < 0.05
        status = "✓" if ece_ok else "✗"
        print(
            f"  {status} {r.horizon.value}/{r.profile.value} ECE={r.metrics.calibration.ece:.3f} (target < 0.05)"
        )
