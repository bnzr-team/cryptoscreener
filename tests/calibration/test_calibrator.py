"""Tests for high-level calibrator interface."""

from __future__ import annotations

from cryptoscreener.calibration.calibrator import (
    CalibratorMethod,
    fit_calibrator,
)


class TestFitCalibrator:
    """Tests for fit_calibrator function."""

    def test_fit_platt_method(self) -> None:
        """fit_calibrator with PLATT should return PlattCalibrator."""
        labels = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
        probs = [0.8, 0.2, 0.7, 0.3, 0.9, 0.6, 0.4, 0.1, 0.75, 0.25]

        cal = fit_calibrator(
            labels,
            probs,
            "test_head",
            method=CalibratorMethod.PLATT,
        )

        assert cal.head_name == "test_head"
        assert hasattr(cal, "a")
        assert hasattr(cal, "b")

    def test_fit_default_method_is_platt(self) -> None:
        """Default method should be PLATT."""
        labels = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
        probs = [0.8, 0.2, 0.7, 0.3, 0.9, 0.6, 0.4, 0.1, 0.75, 0.25]

        cal = fit_calibrator(labels, probs, "test_head")

        # Should be a PlattCalibrator
        from cryptoscreener.calibration.platt import PlattCalibrator

        assert isinstance(cal, PlattCalibrator)

    def test_fit_passes_kwargs(self) -> None:
        """kwargs should be passed to underlying fit function."""
        labels = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
        probs = [0.8, 0.2, 0.7, 0.3, 0.9, 0.6, 0.4, 0.1, 0.75, 0.25]

        # Different max_iter should produce different results
        # (in practice, may converge to same, but at least it shouldn't error)
        cal1 = fit_calibrator(labels, probs, "test", max_iter=10)
        cal2 = fit_calibrator(labels, probs, "test", max_iter=100)

        # Both should be valid calibrators
        assert 0.0 <= cal1.transform(0.5) <= 1.0
        assert 0.0 <= cal2.transform(0.5) <= 1.0


class TestCalibratorProtocol:
    """Tests for Calibrator protocol conformance."""

    def test_platt_satisfies_protocol(self) -> None:
        """PlattCalibrator should satisfy Calibrator protocol."""
        from cryptoscreener.calibration.platt import PlattCalibrator

        cal = PlattCalibrator(a=1.0, b=0.0, head_name="test")

        # Check protocol methods exist and work
        assert hasattr(cal, "head_name")
        assert cal.head_name == "test"

        result = cal.transform(0.5)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

        batch_result = cal.transform_batch([0.3, 0.5, 0.7])
        assert isinstance(batch_result, list)
        assert len(batch_result) == 3

        dict_result = cal.to_dict()
        assert isinstance(dict_result, dict)
        assert "a" in dict_result
        assert "b" in dict_result
        assert "head_name" in dict_result


class TestCalibratorMethod:
    """Tests for CalibratorMethod enum."""

    def test_platt_value(self) -> None:
        """PLATT should have correct string value."""
        assert CalibratorMethod.PLATT.value == "platt"

    def test_enum_is_string(self) -> None:
        """CalibratorMethod value should be usable as string."""
        method = CalibratorMethod.PLATT
        assert f"Method: {method.value}" == "Method: platt"
