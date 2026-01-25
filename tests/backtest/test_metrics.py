"""Tests for backtest evaluation metrics."""

from cryptoscreener.backtest.metrics import (
    compute_auc,
    compute_brier_score,
    compute_churn_metrics,
    compute_ece,
    compute_pr_auc,
    compute_topk_capture,
    compute_topk_mean_edge,
    compute_topk_metrics,
)


class TestAUC:
    """Tests for AUC computation."""

    def test_perfect_auc(self) -> None:
        """Perfect classifier should have AUC = 1."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        auc = compute_auc(y_true, y_scores)
        assert auc == 1.0

    def test_random_auc(self) -> None:
        """Random-ish classifier should have AUC near 0.5."""
        # Balanced mix: some positives above negatives, some below
        # P at 0.9 > N at 0.1, 0.3 (2 correct pairs)
        # P at 0.2 < N at 0.8, 0.6 (2 wrong pairs)
        # P at 0.5 mixed with N at 0.4, 0.7 (1 correct, 1 wrong)
        y_true = [1, 0, 1, 0, 1, 0]
        y_scores = [0.9, 0.8, 0.2, 0.6, 0.5, 0.4]
        auc = compute_auc(y_true, y_scores)
        # AUC should reflect mixed ranking
        assert 0.3 <= auc <= 0.7

    def test_worst_auc(self) -> None:
        """Inverted classifier should have AUC = 0."""
        y_true = [1, 1, 1, 0, 0, 0]
        y_scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        auc = compute_auc(y_true, y_scores)
        assert auc == 0.0

    def test_degenerate_all_positive(self) -> None:
        """All positives should return 0.5."""
        y_true = [1, 1, 1]
        y_scores = [0.5, 0.7, 0.9]
        auc = compute_auc(y_true, y_scores)
        assert auc == 0.5

    def test_degenerate_all_negative(self) -> None:
        """All negatives should return 0.5."""
        y_true = [0, 0, 0]
        y_scores = [0.5, 0.7, 0.9]
        auc = compute_auc(y_true, y_scores)
        assert auc == 0.5

    def test_empty(self) -> None:
        """Empty input should return 0.5."""
        auc = compute_auc([], [])
        assert auc == 0.5

    def test_realistic_auc(self) -> None:
        """Realistic classifier should have 0.5 < AUC < 1."""
        # Mix of correct and incorrect rankings
        y_true = [0, 0, 1, 0, 1, 1, 0, 1]
        y_scores = [0.2, 0.6, 0.5, 0.4, 0.8, 0.3, 0.7, 0.9]
        auc = compute_auc(y_true, y_scores)
        # Not perfect (some negatives ranked higher), not random
        assert 0.5 < auc < 1.0


class TestPRAUC:
    """Tests for PR-AUC computation."""

    def test_perfect_pr_auc(self) -> None:
        """Perfect classifier should have PR-AUC = 1."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        pr_auc = compute_pr_auc(y_true, y_scores)
        assert pr_auc == 1.0

    def test_no_positives(self) -> None:
        """No positives should return 0."""
        y_true = [0, 0, 0]
        y_scores = [0.5, 0.6, 0.7]
        pr_auc = compute_pr_auc(y_true, y_scores)
        assert pr_auc == 0.0

    def test_empty(self) -> None:
        """Empty input should return 0."""
        pr_auc = compute_pr_auc([], [])
        assert pr_auc == 0.0

    def test_realistic_pr_auc(self) -> None:
        """Realistic classifier should have 0 < PR-AUC < 1."""
        # Mix of correct and incorrect rankings
        y_true = [0, 0, 1, 0, 1, 1, 0, 1]
        y_scores = [0.2, 0.6, 0.5, 0.4, 0.8, 0.3, 0.7, 0.9]
        pr_auc = compute_pr_auc(y_true, y_scores)
        # Should be imperfect (some negatives ranked high)
        assert 0.0 < pr_auc < 1.0


class TestBrierScore:
    """Tests for Brier score computation."""

    def test_perfect_brier(self) -> None:
        """Perfect predictions should have Brier = 0."""
        y_true = [0, 0, 1, 1]
        y_probs = [0.0, 0.0, 1.0, 1.0]
        brier = compute_brier_score(y_true, y_probs)
        assert brier == 0.0

    def test_worst_brier(self) -> None:
        """Worst predictions should have Brier = 1."""
        y_true = [0, 0, 1, 1]
        y_probs = [1.0, 1.0, 0.0, 0.0]
        brier = compute_brier_score(y_true, y_probs)
        assert brier == 1.0

    def test_uniform_brier(self) -> None:
        """Uniform 0.5 predictions should have Brier = 0.25."""
        y_true = [0, 1]
        y_probs = [0.5, 0.5]
        brier = compute_brier_score(y_true, y_probs)
        assert brier == 0.25

    def test_empty(self) -> None:
        """Empty input should return 0."""
        brier = compute_brier_score([], [])
        assert brier == 0.0


class TestECE:
    """Tests for Expected Calibration Error."""

    def test_perfectly_calibrated(self) -> None:
        """Perfectly calibrated predictions should have ECE = 0."""
        # 4 samples in bin [0.2, 0.3): 2 positive = 50% accuracy
        # Predict 0.25 (midpoint) for all = 25% confidence
        # This isn't perfectly calibrated
        # Let's make truly calibrated: predict 0.5, half are positive
        y_true = [0, 1, 0, 1]
        y_probs = [0.5, 0.5, 0.5, 0.5]
        result = compute_ece(y_true, y_probs, n_bins=10)
        # All in bin 5 ([0.5, 0.6)): accuracy = 50%, confidence = 50%
        assert result.ece == 0.0

    def test_miscalibrated_overconfident(self) -> None:
        """Overconfident predictions should have ECE > 0."""
        # All predict 0.9 but only 50% are positive
        y_true = [0, 1, 0, 1]
        y_probs = [0.9, 0.9, 0.9, 0.9]
        result = compute_ece(y_true, y_probs, n_bins=10)
        # Bin 9: accuracy = 50%, confidence = 90%, gap = 40%
        assert result.ece == 0.4
        assert result.mce == 0.4

    def test_miscalibrated_underconfident(self) -> None:
        """Underconfident predictions should have ECE > 0."""
        # All predict 0.1 but all are positive
        y_true = [1, 1, 1, 1]
        y_probs = [0.1, 0.1, 0.1, 0.1]
        result = compute_ece(y_true, y_probs, n_bins=10)
        # Bin 1: accuracy = 100%, confidence = 10%, gap = 90%
        assert result.ece == 0.9
        assert result.mce == 0.9

    def test_empty(self) -> None:
        """Empty input should return zeros."""
        result = compute_ece([], [], n_bins=10)
        assert result.ece == 0.0
        assert result.mce == 0.0
        assert result.brier_score == 0.0

    def test_includes_brier(self) -> None:
        """ECE result should include Brier score."""
        y_true = [0, 1, 0, 1]
        y_probs = [0.2, 0.8, 0.3, 0.7]
        result = compute_ece(y_true, y_probs)
        # Brier = mean((0.2-0)^2, (0.8-1)^2, (0.3-0)^2, (0.7-1)^2)
        # = mean(0.04, 0.04, 0.09, 0.09) = 0.065
        assert 0.06 < result.brier_score < 0.07


class TestTopKCapture:
    """Tests for top-K capture rate."""

    def test_perfect_capture(self) -> None:
        """All positives in top-K should give capture = 1."""
        y_true = [1, 1, 0, 0, 0]
        y_scores = [0.9, 0.8, 0.3, 0.2, 0.1]
        capture = compute_topk_capture(y_true, y_scores, k=2)
        assert capture == 1.0

    def test_partial_capture(self) -> None:
        """Some positives in top-K should give partial capture."""
        y_true = [1, 1, 1, 0, 0]
        y_scores = [0.9, 0.5, 0.8, 0.4, 0.1]
        # Top 2: indices 0 (score 0.9, label 1) and 2 (score 0.8, label 1)
        # Captured: 2 of 3 positives
        capture = compute_topk_capture(y_true, y_scores, k=2)
        assert abs(capture - 2 / 3) < 0.01

    def test_no_capture(self) -> None:
        """No positives in top-K should give capture = 0."""
        y_true = [0, 0, 1, 1]
        y_scores = [0.9, 0.8, 0.2, 0.1]
        capture = compute_topk_capture(y_true, y_scores, k=2)
        assert capture == 0.0

    def test_no_positives(self) -> None:
        """No positives total should give capture = 0."""
        y_true = [0, 0, 0]
        y_scores = [0.9, 0.8, 0.7]
        capture = compute_topk_capture(y_true, y_scores, k=2)
        assert capture == 0.0


class TestTopKMeanEdge:
    """Tests for top-K mean edge calculation."""

    def test_mean_edge(self) -> None:
        """Mean edge should be average of top-K edges."""
        y_scores = [0.9, 0.8, 0.7, 0.1]
        net_edge_bps = [20.0, 15.0, 10.0, 5.0]
        # Top 2: scores 0.9 (edge 20) and 0.8 (edge 15)
        mean_edge = compute_topk_mean_edge(y_scores, net_edge_bps, k=2)
        assert mean_edge == 17.5

    def test_empty(self) -> None:
        """Empty input should return 0."""
        mean_edge = compute_topk_mean_edge([], [], k=2)
        assert mean_edge == 0.0


class TestTopKMetrics:
    """Tests for combined top-K metrics."""

    def test_combined_metrics(self) -> None:
        """Combined metrics should include all top-K measures."""
        y_true = [1, 1, 0, 0]
        y_scores = [0.9, 0.8, 0.3, 0.1]
        net_edge_bps = [20.0, 15.0, 5.0, 2.0]

        result = compute_topk_metrics(y_true, y_scores, net_edge_bps, k=2)

        assert result.k == 2
        assert result.capture_rate == 1.0  # Both positives captured
        assert result.mean_edge_bps == 17.5  # (20 + 15) / 2
        assert result.precision_at_k == 1.0  # Both top-2 are positive


class TestChurnMetrics:
    """Tests for churn/stability metrics."""

    def test_no_churn(self) -> None:
        """Identical rankings should have no churn."""
        # Two timestamps, same rankings
        timestamps = [1000, 1000, 1000, 2000, 2000, 2000]
        symbols = ["A", "B", "C", "A", "B", "C"]
        scores = [0.9, 0.8, 0.7, 0.9, 0.8, 0.7]

        result = compute_churn_metrics(timestamps, symbols, scores, k=2)

        assert result.state_changes_per_step == 0.0
        assert result.jaccard_similarity == 1.0

    def test_complete_churn(self) -> None:
        """Completely different rankings should have max churn."""
        # Two timestamps, completely different top-K
        timestamps = [1000, 1000, 1000, 2000, 2000, 2000]
        symbols = ["A", "B", "C", "D", "E", "F"]
        scores = [0.9, 0.8, 0.7, 0.9, 0.8, 0.7]

        result = compute_churn_metrics(timestamps, symbols, scores, k=2)

        # Top-2 at t1: {A, B}, Top-2 at t2: {D, E}
        # State changes: 2 exited + 2 entered = 4
        assert result.state_changes_per_step == 4.0
        assert result.jaccard_similarity == 0.0

    def test_partial_churn(self) -> None:
        """Partial overlap should have partial churn."""
        timestamps = [1000, 1000, 1000, 2000, 2000, 2000]
        symbols = ["A", "B", "C", "A", "D", "C"]
        scores = [0.9, 0.8, 0.7, 0.9, 0.8, 0.7]

        result = compute_churn_metrics(timestamps, symbols, scores, k=2)

        # Top-2 at t1: {A, B}, Top-2 at t2: {A, D}
        # State changes: 1 exited (B) + 1 entered (D) = 2
        assert result.state_changes_per_step == 2.0
        # Jaccard: |{A}| / |{A, B, D}| = 1/3
        assert abs(result.jaccard_similarity - 1 / 3) < 0.01

    def test_single_timestamp(self) -> None:
        """Single timestamp should return defaults."""
        timestamps = [1000, 1000, 1000]
        symbols = ["A", "B", "C"]
        scores = [0.9, 0.8, 0.7]

        result = compute_churn_metrics(timestamps, symbols, scores, k=2)

        assert result.state_changes_per_step == 0.0
        assert result.jaccard_similarity == 1.0

    def test_empty(self) -> None:
        """Empty input should return defaults."""
        result = compute_churn_metrics([], [], [], k=2)
        assert result.state_changes_per_step == 0.0
        assert result.jaccard_similarity == 1.0
