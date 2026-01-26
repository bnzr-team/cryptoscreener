"""Tests for Platt scaling calibrator."""

from __future__ import annotations

import math

import pytest

from cryptoscreener.calibration.platt import NegativeSlopeError, PlattCalibrator, fit_platt


class TestPlattCalibrator:
    """Tests for PlattCalibrator class."""

    def test_identity_transform(self) -> None:
        """Calibrator with a=1, b=0 should approximate identity."""
        cal = PlattCalibrator(a=1.0, b=0.0, head_name="test")

        # Identity transform: sigmoid(logit(p)) ≈ p
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = cal.transform(p)
            assert abs(result - p) < 0.01, f"p={p}, result={result}"

    def test_transform_stays_in_bounds(self) -> None:
        """Transformed probabilities must be in [0, 1]."""
        # Various parameter combinations
        test_cases = [
            (1.0, 0.0),
            (2.0, 1.0),
            (0.5, -1.0),
            (3.0, -2.0),
            (-1.0, 2.0),  # Adversarial: inverted slope
        ]

        for a, b in test_cases:
            cal = PlattCalibrator(a=a, b=b, head_name="test")

            # Test full range including extreme values
            for p in [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]:
                result = cal.transform(p)
                assert 0.0 <= result <= 1.0, f"a={a}, b={b}, p={p}: result={result} out of bounds"

    def test_transform_batch(self) -> None:
        """Batch transform should match individual transforms."""
        cal = PlattCalibrator(a=1.5, b=0.5, head_name="test")
        probs = [0.1, 0.3, 0.5, 0.7, 0.9]

        batch_result = cal.transform_batch(probs)
        individual_results = [cal.transform(p) for p in probs]

        assert batch_result == individual_results

    def test_serialization_roundtrip(self) -> None:
        """to_dict and from_dict should preserve all fields."""
        cal = PlattCalibrator(a=1.234, b=-0.567, head_name="p_inplay_30s")

        data = cal.to_dict()
        restored = PlattCalibrator.from_dict(data)

        assert restored.a == cal.a
        assert restored.b == cal.b
        assert restored.head_name == cal.head_name

    def test_transform_monotonic_for_positive_slope(self) -> None:
        """With positive slope, transform should be monotonic."""
        cal = PlattCalibrator(a=2.0, b=0.5, head_name="test")

        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = [cal.transform(p) for p in probs]

        # Check monotonically increasing
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], (
                f"Not monotonic at i={i}: {results[i]} >= {results[i + 1]}"
            )


class TestFitPlatt:
    """Tests for fit_platt function."""

    def test_fit_on_calibrated_data(self) -> None:
        """Fitting on already-calibrated data should learn near-identity."""
        # Generate calibrated data: p ≈ observed frequency
        n = 1000
        probs = [i / (n + 1) for i in range(1, n + 1)]
        labels = [1 if (i % 10) < int(p * 10) else 0 for i, p in enumerate(probs)]

        cal = fit_platt(labels, probs, "test")

        # Parameters should be close to identity (a≈1, b≈0)
        # Allow some tolerance since it's an optimization
        assert 0.5 < cal.a < 2.0, f"a={cal.a} far from 1.0"
        assert -1.0 < cal.b < 1.0, f"b={cal.b} far from 0.0"

    def test_fit_on_overconfident_data(self) -> None:
        """Fitting on overconfident predictions should learn to temper."""
        # Overconfident model: predicts 0.9 but only 50% are positive
        n = 200
        probs = [0.9] * n
        labels = [1 if i % 2 == 0 else 0 for i in range(n)]

        cal = fit_platt(labels, probs, "test")

        # Calibrated output should be closer to 0.5
        calibrated = cal.transform(0.9)
        assert 0.3 < calibrated < 0.7, f"Expected ~0.5, got {calibrated}"

    def test_fit_on_underconfident_data(self) -> None:
        """Fitting on underconfident predictions should boost confidence."""
        # Underconfident model: predicts 0.5 but 90% are positive
        n = 200
        probs = [0.5] * n
        labels = [1 if i < 180 else 0 for i in range(n)]  # 90% positive

        cal = fit_platt(labels, probs, "test")

        # Calibrated output should be higher
        calibrated = cal.transform(0.5)
        assert calibrated > 0.6, f"Expected >0.6, got {calibrated}"

    def test_fit_requires_minimum_samples(self) -> None:
        """Fitting with < 2 samples should raise error."""
        with pytest.raises(ValueError, match="at least 2"):
            fit_platt([1], [0.5], "test")

    def test_fit_requires_both_classes(self) -> None:
        """Fitting with only one class should raise error."""
        with pytest.raises(ValueError, match="positive and negative"):
            fit_platt([1, 1, 1], [0.5, 0.6, 0.7], "test")

        with pytest.raises(ValueError, match="positive and negative"):
            fit_platt([0, 0, 0], [0.5, 0.6, 0.7], "test")

    def test_fit_length_mismatch(self) -> None:
        """Mismatched lengths should raise error."""
        with pytest.raises(ValueError, match="Length mismatch"):
            fit_platt([1, 0], [0.5, 0.6, 0.7], "test")

    def test_fit_improves_brier_on_miscalibrated(self) -> None:
        """Calibration should improve Brier score on miscalibrated data."""
        # Create miscalibrated predictions
        n = 500
        probs_raw = []
        labels = []

        for i in range(n):
            # True probability varies by position
            true_p = i / n
            # Model is systematically overconfident
            model_p = min(0.99, true_p + 0.2)
            probs_raw.append(model_p)
            labels.append(1 if (i % 100) < int(true_p * 100) else 0)

        # Compute Brier before
        brier_before = sum((p - y) ** 2 for p, y in zip(probs_raw, labels, strict=True)) / n

        # Fit and calibrate
        cal = fit_platt(labels, probs_raw, "test")
        probs_cal = cal.transform_batch(probs_raw)

        # Compute Brier after
        brier_after = sum((p - y) ** 2 for p, y in zip(probs_cal, labels, strict=True)) / n

        assert brier_after < brier_before, (
            f"Calibration should improve Brier: {brier_before:.4f} -> {brier_after:.4f}"
        )

    def test_fit_deterministic(self) -> None:
        """Same input should produce same output."""
        labels = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
        probs = [0.8, 0.2, 0.7, 0.3, 0.9, 0.6, 0.4, 0.1, 0.75, 0.25]

        cal1 = fit_platt(labels, probs, "test")
        cal2 = fit_platt(labels, probs, "test")

        assert cal1.a == cal2.a
        assert cal1.b == cal2.b

    def test_calibrated_probs_in_bounds(self) -> None:
        """All calibrated probabilities must be in [0, 1]."""
        # Well-correlated data (higher prob → higher label rate)
        n = 100
        labels = []
        probs = []
        for i in range(n):
            p = (i + 1) / (n + 1)
            probs.append(p)
            # Label rate increases with probability (positive correlation)
            labels.append(1 if (i % 10) < int(p * 10) else 0)

        cal = fit_platt(labels, probs, "test")

        # Test on edge cases
        for p in [0.0, 0.001, 0.5, 0.999, 1.0]:
            result = cal.transform(p)
            assert 0.0 <= result <= 1.0, f"p={p}: result={result} out of bounds"


class TestRankingPreservation:
    """Tests for ranking/ordering preservation.

    CRITICAL: Calibration must not invert rankings. If higher raw probability
    means higher true probability, calibrated outputs must preserve this order.
    """

    def test_fit_preserves_ranking_on_well_ordered_data(self) -> None:
        """Fitted calibrator should preserve ranking on positively-correlated data.

        When raw probabilities are positively correlated with outcomes
        (higher p_raw → higher P(y=1)), the fitted calibrator must have a >= 0
        to preserve ranking order.
        """
        # Create well-ordered data: higher probability → higher label rate
        n = 200
        labels = []
        probs = []

        for i in range(n):
            # Probability increases with i
            p = (i + 1) / (n + 1)
            probs.append(p)
            # Label rate also increases with i (positive correlation)
            labels.append(1 if (i % 10) < int(p * 10) else 0)

        cal = fit_platt(labels, probs, "test")

        # a must be positive for ranking preservation
        assert cal.a > 0, (
            f"Fitted a={cal.a} is non-positive on well-ordered data. This would invert rankings!"
        )

        # Verify ranking is actually preserved
        test_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        calibrated = cal.transform_batch(test_probs)

        for i in range(len(calibrated) - 1):
            assert calibrated[i] < calibrated[i + 1], (
                f"Ranking inverted: cal({test_probs[i]})={calibrated[i]} >= "
                f"cal({test_probs[i + 1]})={calibrated[i + 1]}"
            )

    def test_ranking_preserved_after_calibration(self) -> None:
        """Rankings between samples should be preserved post-calibration.

        This is the key invariant: if p1_raw > p2_raw, then p1_cal >= p2_cal
        (weak inequality allows for ties at extremes).
        """
        labels = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0] * 20
        probs = [0.9, 0.1, 0.8, 0.2, 0.95, 0.7, 0.3, 0.15, 0.85, 0.25] * 20

        cal = fit_platt(labels, probs, "test")

        # For any pair where p1 > p2, calibrated should satisfy cal(p1) >= cal(p2)
        test_pairs = [(0.2, 0.8), (0.3, 0.7), (0.1, 0.9), (0.4, 0.6)]

        for p_low, p_high in test_pairs:
            cal_low = cal.transform(p_low)
            cal_high = cal.transform(p_high)
            assert cal_low <= cal_high, (
                f"Ranking inverted: cal({p_low})={cal_low} > cal({p_high})={cal_high}"
            )

    def test_negative_slope_inverts_ranking(self) -> None:
        """Document behavior: negative slope inverts rankings.

        This test documents that a negative slope WILL invert rankings.
        In production, a < 0 indicates the model is anti-correlated with outcomes
        and should be rejected or investigated.
        """
        cal = PlattCalibrator(a=-1.0, b=0.0, head_name="test")

        # With negative slope, higher input → lower output (inversion)
        result_low = cal.transform(0.2)
        result_high = cal.transform(0.8)

        # Ranking IS inverted with negative slope
        assert result_low > result_high, "Expected ranking inversion with negative slope"


class TestNegativeSlopeRejection:
    """Tests for negative slope rejection (anti-ranking-inversion safety)."""

    def test_anti_correlated_data_raises_negative_slope_error(self) -> None:
        """Anti-correlated data should raise NegativeSlopeError.

        When higher raw probability correlates with LOWER true probability,
        the optimizer will find a < 0. This is rejected by default.
        """
        # Create anti-correlated data: higher probability → lower label rate
        n = 200
        labels = []
        probs = []

        for i in range(n):
            # Probability increases with i
            p = (i + 1) / (n + 1)
            probs.append(p)
            # BUT label rate DECREASES with i (anti-correlation)
            labels.append(1 if (i % 10) >= int(p * 10) else 0)

        with pytest.raises(NegativeSlopeError) as exc_info:
            fit_platt(labels, probs, "anti_correlated")

        # Check exception contains useful info
        assert "anti_correlated" in str(exc_info.value)
        assert exc_info.value.head_name == "anti_correlated"
        assert exc_info.value.a <= 0

    def test_negative_slope_allowed_when_disabled(self) -> None:
        """Negative slope should be allowed when reject_negative_slope=False.

        This is for diagnostic/debugging purposes only.
        """
        # Create anti-correlated data
        n = 200
        labels = []
        probs = []

        for i in range(n):
            p = (i + 1) / (n + 1)
            probs.append(p)
            labels.append(1 if (i % 10) >= int(p * 10) else 0)

        # Should NOT raise when rejection is disabled
        cal = fit_platt(labels, probs, "test", reject_negative_slope=False)

        # Slope should be negative
        assert cal.a < 0, f"Expected negative slope, got a={cal.a}"

    def test_error_message_is_actionable(self) -> None:
        """NegativeSlopeError message should be actionable."""
        error = NegativeSlopeError("p_inplay_30s", a=-0.5, b=0.1)

        msg = str(error)
        assert "p_inplay_30s" in msg
        assert "-0.5" in msg or "negative" in msg.lower()
        assert "invert" in msg.lower() or "ranking" in msg.lower()


class TestNumericalStability:
    """Tests for numerical edge cases."""

    def test_extreme_probabilities(self) -> None:
        """Calibrator should handle probabilities near 0 and 1."""
        cal = PlattCalibrator(a=1.0, b=0.0, head_name="test")

        # Near zero
        result = cal.transform(1e-10)
        assert 0.0 < result < 0.1
        assert math.isfinite(result)

        # Near one
        result = cal.transform(1.0 - 1e-10)
        assert 0.9 < result < 1.0
        assert math.isfinite(result)

    def test_extreme_parameters(self) -> None:
        """Large parameters should not cause overflow."""
        cal = PlattCalibrator(a=100.0, b=50.0, head_name="test")

        for p in [0.1, 0.5, 0.9]:
            result = cal.transform(p)
            assert math.isfinite(result)
            assert 0.0 <= result <= 1.0

    def test_negative_slope(self) -> None:
        """Negative slope should still produce valid probabilities."""
        cal = PlattCalibrator(a=-1.0, b=0.0, head_name="test")

        for p in [0.1, 0.5, 0.9]:
            result = cal.transform(p)
            assert math.isfinite(result)
            assert 0.0 <= result <= 1.0
