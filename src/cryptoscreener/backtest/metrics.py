"""
Evaluation metrics for offline backtest.

Implements metrics per PRD §10 and EVALUATION_METRICS.md:

Offline metrics:
- AUC, PR-AUC per horizon (classification quality)
- Brier score (probabilistic accuracy)
- ECE (Expected Calibration Error)
- Top-K capture: fraction of tradeable events in top-K predictions
- Mean net_edge_bps in top-K
- Churn: rank/state change frequency

These metrics evaluate model quality before deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class CalibrationMetrics:
    """Calibration quality metrics.

    Attributes:
        brier_score: Mean squared error between predicted prob and outcome.
        ece: Expected Calibration Error (weighted avg bin error).
        mce: Maximum Calibration Error (max bin error).
        bin_accuracies: Per-bin actual positive rate.
        bin_confidences: Per-bin mean predicted probability.
        bin_counts: Per-bin sample count.
    """

    brier_score: float
    ece: float
    mce: float
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]


@dataclass(frozen=True)
class TopKMetrics:
    """Top-K selection metrics.

    Attributes:
        capture_rate: Fraction of all positives captured in top-K.
        mean_edge_bps: Average net_edge_bps in top-K.
        precision_at_k: Fraction of top-K that are positive.
        k: Value of K used.
    """

    capture_rate: float
    mean_edge_bps: float
    precision_at_k: float
    k: int


@dataclass(frozen=True)
class ChurnMetrics:
    """Rank stability metrics.

    Attributes:
        rank_changes_per_step: Average rank changes between consecutive steps.
        state_changes_per_step: Average state changes (in/out of top-K).
        jaccard_similarity: Average Jaccard similarity of top-K sets.
    """

    rank_changes_per_step: float
    state_changes_per_step: float
    jaccard_similarity: float


@dataclass(frozen=True)
class BacktestMetrics:
    """Complete backtest evaluation results.

    Attributes:
        auc: Area Under ROC Curve.
        pr_auc: Area Under Precision-Recall Curve.
        calibration: Calibration metrics (Brier, ECE, MCE).
        topk: Top-K capture metrics.
        churn: Rank stability metrics.
        n_samples: Total samples evaluated.
        n_positives: Total positive labels.
    """

    auc: float
    pr_auc: float
    calibration: CalibrationMetrics
    topk: TopKMetrics
    churn: ChurnMetrics | None
    n_samples: int
    n_positives: int


def compute_auc(
    y_true: Sequence[int],
    y_scores: Sequence[float],
) -> float:
    """Compute Area Under ROC Curve.

    Uses trapezoidal approximation with sorted thresholds.

    Args:
        y_true: Binary labels (0 or 1).
        y_scores: Predicted probabilities or scores.

    Returns:
        AUC value in [0, 1]. Returns 0.5 if degenerate.
    """
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have same length")

    if len(y_true) == 0:
        return 0.5

    # Count positives and negatives
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Degenerate case

    # Sort by score descending
    paired = sorted(zip(y_scores, y_true, strict=True), reverse=True)

    # Compute AUC via Wilcoxon-Mann-Whitney statistic
    # Count pairs where positive has higher score than negative
    tp = 0
    fp = 0
    auc_sum = 0.0
    prev_score = float("inf")

    for score, label in paired:
        if score != prev_score:
            prev_score = score
        if label == 1:
            tp += 1
        else:
            fp += 1
            # This negative is ranked below all previous positives
            auc_sum += tp

    return auc_sum / (n_pos * n_neg)


def compute_pr_auc(
    y_true: Sequence[int],
    y_scores: Sequence[float],
) -> float:
    """Compute Area Under Precision-Recall Curve.

    Uses trapezoidal approximation.

    Args:
        y_true: Binary labels (0 or 1).
        y_scores: Predicted probabilities or scores.

    Returns:
        PR-AUC value in [0, 1]. Returns 0 if no positives.
    """
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have same length")

    if len(y_true) == 0:
        return 0.0

    n_pos = sum(y_true)
    if n_pos == 0:
        return 0.0

    # Sort by score descending
    paired = sorted(zip(y_scores, y_true, strict=True), reverse=True)

    # Compute precision-recall points
    tp = 0
    fp = 0
    precisions = []
    recalls = []

    for _score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / n_pos

        precisions.append(precision)
        recalls.append(recall)

    # Compute AUC using trapezoidal rule
    # We integrate precision as a function of recall
    auc = 0.0
    prev_recall = 0.0

    for _i, (prec, rec) in enumerate(zip(precisions, recalls, strict=True)):
        if rec > prev_recall:
            auc += prec * (rec - prev_recall)
            prev_recall = rec

    return auc


def compute_brier_score(
    y_true: Sequence[int],
    y_probs: Sequence[float],
) -> float:
    """Compute Brier score (mean squared error of probabilities).

    Brier score = (1/N) * sum((p_i - y_i)^2)

    Lower is better. Perfect calibration with perfect predictions = 0.

    Args:
        y_true: Binary labels (0 or 1).
        y_probs: Predicted probabilities in [0, 1].

    Returns:
        Brier score in [0, 1].
    """
    if len(y_true) != len(y_probs):
        raise ValueError("y_true and y_probs must have same length")

    if len(y_true) == 0:
        return 0.0

    total = sum((p - y) ** 2 for p, y in zip(y_probs, y_true, strict=True))
    return total / len(y_true)


def compute_ece(
    y_true: Sequence[int],
    y_probs: Sequence[float],
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute Expected Calibration Error and related metrics.

    ECE = sum(|bin_count / N| * |accuracy - confidence|)

    Args:
        y_true: Binary labels (0 or 1).
        y_probs: Predicted probabilities in [0, 1].
        n_bins: Number of bins for calibration (default 10).

    Returns:
        CalibrationMetrics with ECE, MCE, and per-bin stats.
    """
    if len(y_true) != len(y_probs):
        raise ValueError("y_true and y_probs must have same length")

    n = len(y_true)
    if n == 0:
        return CalibrationMetrics(
            brier_score=0.0,
            ece=0.0,
            mce=0.0,
            bin_accuracies=[],
            bin_confidences=[],
            bin_counts=[],
        )

    # Initialize bins
    bin_sums: list[float] = [0.0] * n_bins
    bin_correct: list[int] = [0] * n_bins
    bin_counts: list[int] = [0] * n_bins

    # Assign samples to bins
    for prob, label in zip(y_probs, y_true, strict=True):
        # Bin index: [0, 0.1) -> 0, [0.1, 0.2) -> 1, ..., [0.9, 1.0] -> 9
        bin_idx = min(int(prob * n_bins), n_bins - 1)
        bin_sums[bin_idx] += prob
        bin_correct[bin_idx] += label
        bin_counts[bin_idx] += 1

    # Compute per-bin accuracy and confidence
    bin_accuracies: list[float] = []
    bin_confidences: list[float] = []

    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_accuracies.append(bin_correct[i] / bin_counts[i])
            bin_confidences.append(bin_sums[i] / bin_counts[i])
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((i + 0.5) / n_bins)  # Bin midpoint

    # Compute ECE and MCE
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        if bin_counts[i] > 0:
            gap = abs(bin_accuracies[i] - bin_confidences[i])
            weight = bin_counts[i] / n
            ece += weight * gap
            mce = max(mce, gap)

    # Also compute Brier score
    brier = compute_brier_score(y_true, y_probs)

    return CalibrationMetrics(
        brier_score=brier,
        ece=ece,
        mce=mce,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )


def compute_topk_capture(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    k: int,
) -> float:
    """Compute top-K capture rate.

    Capture rate = (# positives in top-K) / (# total positives)

    Args:
        y_true: Binary labels (0 or 1).
        y_scores: Predicted probabilities or scores.
        k: Number of top predictions to consider.

    Returns:
        Capture rate in [0, 1].
    """
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have same length")

    n_pos = sum(y_true)
    if n_pos == 0:
        return 0.0

    # Sort by score descending and take top-K
    paired = sorted(zip(y_scores, y_true, strict=True), reverse=True)
    top_k = paired[:k]

    # Count positives in top-K
    captured = sum(label for _, label in top_k)

    return captured / n_pos


def compute_topk_mean_edge(
    y_scores: Sequence[float],
    net_edge_bps: Sequence[float],
    k: int,
) -> float:
    """Compute mean net_edge_bps in top-K predictions.

    Args:
        y_scores: Predicted probabilities or scores.
        net_edge_bps: Net edge in basis points for each sample.
        k: Number of top predictions to consider.

    Returns:
        Mean net_edge_bps in top-K.
    """
    if len(y_scores) != len(net_edge_bps):
        raise ValueError("y_scores and net_edge_bps must have same length")

    if len(y_scores) == 0 or k == 0:
        return 0.0

    # Sort by score descending and take top-K
    paired = sorted(zip(y_scores, net_edge_bps, strict=True), reverse=True)
    top_k = paired[: min(k, len(paired))]

    # Compute mean edge
    total_edge = sum(edge for _, edge in top_k)
    return total_edge / len(top_k)


def compute_topk_metrics(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    net_edge_bps: Sequence[float],
    k: int,
) -> TopKMetrics:
    """Compute all top-K metrics.

    Args:
        y_true: Binary labels (0 or 1).
        y_scores: Predicted probabilities or scores.
        net_edge_bps: Net edge in basis points for each sample.
        k: Number of top predictions to consider.

    Returns:
        TopKMetrics with capture rate, mean edge, and precision.
    """
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have same length")
    if len(y_scores) != len(net_edge_bps):
        raise ValueError("y_scores and net_edge_bps must have same length")

    capture_rate = compute_topk_capture(y_true, y_scores, k)
    mean_edge = compute_topk_mean_edge(y_scores, net_edge_bps, k)

    # Precision at K
    if len(y_true) == 0 or k == 0:
        precision = 0.0
    else:
        paired = sorted(zip(y_scores, y_true, strict=True), reverse=True)
        top_k = paired[: min(k, len(paired))]
        n_pos_in_k = sum(label for _, label in top_k)
        precision = n_pos_in_k / len(top_k)

    return TopKMetrics(
        capture_rate=capture_rate,
        mean_edge_bps=mean_edge,
        precision_at_k=precision,
        k=k,
    )


def compute_churn_metrics(
    timestamps: Sequence[int],
    symbols: Sequence[str],
    scores: Sequence[float],
    k: int,
) -> ChurnMetrics:
    """Compute churn/stability metrics for ranked predictions.

    Measures how stable the top-K set is over time.

    Args:
        timestamps: Timestamp for each prediction.
        symbols: Symbol for each prediction.
        scores: Predicted score for each prediction.
        k: Size of top-K set.

    Returns:
        ChurnMetrics with rank changes, state changes, Jaccard similarity.
    """
    if len(timestamps) != len(symbols) or len(symbols) != len(scores):
        raise ValueError("All sequences must have same length")

    if len(timestamps) == 0:
        return ChurnMetrics(
            rank_changes_per_step=0.0,
            state_changes_per_step=0.0,
            jaccard_similarity=1.0,
        )

    # Group by timestamp
    from collections import defaultdict

    by_ts: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for ts, sym, score in zip(timestamps, symbols, scores, strict=True):
        by_ts[ts].append((sym, score))

    # Sort timestamps
    sorted_ts = sorted(by_ts.keys())

    if len(sorted_ts) < 2:
        return ChurnMetrics(
            rank_changes_per_step=0.0,
            state_changes_per_step=0.0,
            jaccard_similarity=1.0,
        )

    # Compute top-K set at each timestamp
    topk_sets: list[set[str]] = []
    for ts in sorted_ts:
        items = by_ts[ts]
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        topk = {sym for sym, _ in sorted_items[:k]}
        topk_sets.append(topk)

    # Compute metrics between consecutive steps
    total_state_changes = 0
    total_jaccard = 0.0
    n_transitions = len(sorted_ts) - 1

    for i in range(n_transitions):
        prev_set = topk_sets[i]
        curr_set = topk_sets[i + 1]

        # State changes: symbols that entered or left top-K
        entered = curr_set - prev_set
        exited = prev_set - curr_set
        total_state_changes += len(entered) + len(exited)

        # Jaccard similarity
        intersection = len(prev_set & curr_set)
        union = len(prev_set | curr_set)
        jaccard = intersection / union if union > 0 else 1.0
        total_jaccard += jaccard

    avg_state_changes = total_state_changes / n_transitions
    avg_jaccard = total_jaccard / n_transitions

    # Rank changes: count position changes within top-K
    # For simplicity, we use state changes as proxy for rank instability
    # A more sophisticated metric would track actual position changes

    return ChurnMetrics(
        rank_changes_per_step=avg_state_changes / 2,  # Approximation
        state_changes_per_step=avg_state_changes,
        jaccard_similarity=avg_jaccard,
    )


def compute_all_metrics(
    y_true: Sequence[int],
    y_probs: Sequence[float],
    net_edge_bps: Sequence[float],
    k: int,
    timestamps: Sequence[int] | None = None,
    symbols: Sequence[str] | None = None,
) -> BacktestMetrics:
    """Compute all backtest evaluation metrics.

    Args:
        y_true: Binary labels (0 or 1).
        y_probs: Predicted probabilities in [0, 1].
        net_edge_bps: Net edge in basis points for each sample.
        k: Number of top predictions for top-K metrics.
        timestamps: Optional timestamps for churn calculation.
        symbols: Optional symbols for churn calculation.

    Returns:
        BacktestMetrics with all evaluation results.
    """
    # Basic metrics
    auc = compute_auc(y_true, y_probs)
    pr_auc = compute_pr_auc(y_true, y_probs)
    calibration = compute_ece(y_true, y_probs)
    topk = compute_topk_metrics(y_true, y_probs, net_edge_bps, k)

    # Churn metrics (optional)
    churn = None
    if timestamps is not None and symbols is not None:
        churn = compute_churn_metrics(timestamps, symbols, y_probs, k)

    return BacktestMetrics(
        auc=auc,
        pr_auc=pr_auc,
        calibration=calibration,
        topk=topk,
        churn=churn,
        n_samples=len(y_true),
        n_positives=sum(y_true),
    )
