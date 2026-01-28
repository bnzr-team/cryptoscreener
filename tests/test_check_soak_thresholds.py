"""
DEC-032: Tests for scripts/check_soak_thresholds.py threshold checking logic.
"""
from __future__ import annotations

from typing import Any

from scripts.check_soak_thresholds import check_baseline, check_overload


def _baseline_summary(**overrides: Any) -> dict[str, Any]:
    """Return a passing baseline summary with optional overrides."""
    base: dict[str, Any] = {
        "events_dropped": 0,
        "snapshots_dropped": 0,
        "max_event_queue_depth": 100,
        "max_snapshot_queue_depth": 10,
        "max_rss_mb": 80.0,
        "max_tick_drift_ms": 50,
        "max_reconnect_rate_per_min": 0.0,
    }
    base.update(overrides)
    return base


def _overload_summary(**overrides: Any) -> dict[str, Any]:
    """Return a passing overload summary with optional overrides."""
    base: dict[str, Any] = {
        "events_dropped": 500,
        "snapshots_dropped": 10,
        "max_event_queue_depth": 9000,
        "max_snapshot_queue_depth": 800,
        "max_rss_mb": 150.0,
        "max_tick_drift_ms": 500,
        "max_reconnect_rate_per_min": 2.0,
    }
    base.update(overrides)
    return base


DEFAULT_BASELINE_THRESHOLDS: dict[str, Any] = {
    "events_dropped_max": 0,
    "snapshots_dropped_max": 0,
    "max_event_queue_depth": 2000,
    "max_snapshot_queue_depth": 500,
    "max_rss_mb": 200,
    "max_tick_drift_ms": 200,
    "max_reconnect_rate_per_min": 6.0,
}

DEFAULT_OVERLOAD_THRESHOLDS: dict[str, Any] = {
    "events_dropped_min": 1,
    "max_event_queue_depth": 10000,
    "max_snapshot_queue_depth": 1000,
    "max_rss_mb": 300,
    "max_tick_drift_ms": 1000,
    "max_reconnect_rate_per_min": 6.0,
}


class TestBaselineThresholds:
    """DEC-032: Baseline soak threshold checks."""

    def test_all_pass(self) -> None:
        failures = check_baseline(_baseline_summary(), DEFAULT_BASELINE_THRESHOLDS)
        assert failures == []

    def test_events_dropped_fails(self) -> None:
        failures = check_baseline(
            _baseline_summary(events_dropped=5), DEFAULT_BASELINE_THRESHOLDS
        )
        assert len(failures) == 1
        assert "events_dropped" in failures[0]

    def test_queue_depth_fails(self) -> None:
        failures = check_baseline(
            _baseline_summary(max_event_queue_depth=3000), DEFAULT_BASELINE_THRESHOLDS
        )
        assert len(failures) == 1
        assert "max_event_queue_depth" in failures[0]

    def test_rss_fails(self) -> None:
        failures = check_baseline(
            _baseline_summary(max_rss_mb=250.0), DEFAULT_BASELINE_THRESHOLDS
        )
        assert len(failures) == 1
        assert "max_rss_mb" in failures[0]

    def test_tick_drift_fails(self) -> None:
        failures = check_baseline(
            _baseline_summary(max_tick_drift_ms=300), DEFAULT_BASELINE_THRESHOLDS
        )
        assert len(failures) == 1
        assert "max_tick_drift_ms" in failures[0]

    def test_reconnect_rate_fails(self) -> None:
        failures = check_baseline(
            _baseline_summary(max_reconnect_rate_per_min=10.0), DEFAULT_BASELINE_THRESHOLDS
        )
        assert len(failures) == 1
        assert "max_reconnect_rate_per_min" in failures[0]

    def test_multiple_failures(self) -> None:
        failures = check_baseline(
            _baseline_summary(events_dropped=5, max_rss_mb=250.0), DEFAULT_BASELINE_THRESHOLDS
        )
        assert len(failures) == 2

    def test_missing_field_reported(self) -> None:
        summary: dict[str, Any] = {}
        failures = check_baseline(summary, DEFAULT_BASELINE_THRESHOLDS)
        assert any("MISSING" in f for f in failures)


class TestOverloadThresholds:
    """DEC-032: Overload soak threshold checks."""

    def test_all_pass(self) -> None:
        failures = check_overload(_overload_summary(), DEFAULT_OVERLOAD_THRESHOLDS)
        assert failures == []

    def test_no_drops_fails(self) -> None:
        """Under overload, events_dropped must be > 0 (backpressure engaged)."""
        failures = check_overload(
            _overload_summary(events_dropped=0), DEFAULT_OVERLOAD_THRESHOLDS
        )
        assert len(failures) == 1
        assert "events_dropped" in failures[0]

    def test_queue_depth_at_cap_passes(self) -> None:
        failures = check_overload(
            _overload_summary(max_event_queue_depth=10000), DEFAULT_OVERLOAD_THRESHOLDS
        )
        assert failures == []

    def test_queue_depth_exceeds_cap_fails(self) -> None:
        failures = check_overload(
            _overload_summary(max_event_queue_depth=10001), DEFAULT_OVERLOAD_THRESHOLDS
        )
        assert len(failures) == 1

    def test_rss_fails(self) -> None:
        failures = check_overload(
            _overload_summary(max_rss_mb=350.0), DEFAULT_OVERLOAD_THRESHOLDS
        )
        assert len(failures) == 1
        assert "max_rss_mb" in failures[0]

    def test_tick_drift_at_limit_passes(self) -> None:
        failures = check_overload(
            _overload_summary(max_tick_drift_ms=1000), DEFAULT_OVERLOAD_THRESHOLDS
        )
        assert failures == []

    def test_missing_field_reported(self) -> None:
        summary: dict[str, Any] = {}
        failures = check_overload(summary, DEFAULT_OVERLOAD_THRESHOLDS)
        assert any("MISSING" in f for f in failures)
