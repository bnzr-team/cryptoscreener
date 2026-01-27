"""
Tests for Prometheus metrics exporter.

DEC-025: Validates exporter correctness:
- No forbidden high-cardinality labels
- Metric names match alert_rules.yml exactly
- Metrics output snapshot
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from prometheus_client import generate_latest
from prometheus_client.registry import CollectorRegistry

from cryptoscreener.connectors.backoff import CircuitBreaker, RestGovernor, RestGovernorConfig
from cryptoscreener.connectors.binance.types import ConnectorMetrics
from cryptoscreener.connectors.exporter import (
    FORBIDDEN_LABELS,
    REQUIRED_METRIC_NAMES,
    MetricsExporter,
)


class TestNoForbiddenLabels:
    """DEC-025: Verify no high-cardinality labels are used."""

    def test_exporter_has_no_forbidden_labels(self):
        """Exporter metrics must not contain forbidden labels."""
        registry = CollectorRegistry()
        exporter = MetricsExporter(registry=registry)

        # Create minimal components to populate metrics
        governor = RestGovernor(config=RestGovernorConfig())
        circuit_breaker = CircuitBreaker()
        connector_metrics = ConnectorMetrics()

        # Update metrics
        exporter.update(
            governor=governor,
            circuit_breaker=circuit_breaker,
            connector_metrics=connector_metrics,
        )

        # Generate metrics output
        output = generate_latest(registry).decode("utf-8")

        # Parse all label names from output
        # Format: metric_name{label1="value1",label2="value2"} value
        label_pattern = re.compile(r"\{([^}]+)\}")
        found_labels: set[str] = set()

        for match in label_pattern.finditer(output):
            label_str = match.group(1)
            # Parse label names (before =)
            for pair in label_str.split(","):
                if "=" in pair:
                    label_name = pair.split("=")[0].strip()
                    found_labels.add(label_name)

        # Check no forbidden labels present
        forbidden_found = found_labels & FORBIDDEN_LABELS
        assert not forbidden_found, (
            f"Forbidden labels found in metrics output: {forbidden_found}\n"
            f"All found labels: {found_labels}\n"
            f"Metrics output:\n{output}"
        )

    def test_forbidden_labels_list_is_comprehensive(self):
        """FORBIDDEN_LABELS contains expected high-cardinality labels."""
        expected = {
            "symbol",
            "endpoint",
            "path",
            "query",
            "ip",
            "request_id",
            "user_id",
            "token",
            "stream",
            "shard_id",
        }
        assert expected == FORBIDDEN_LABELS


class TestMetricNamesMatchRules:
    """DEC-025: Verify metric names match alert_rules.yml."""

    def test_required_metric_names_list(self):
        """REQUIRED_METRIC_NAMES matches DEC-025 specification with _total suffix for counters."""
        expected = {
            # Gauges (no _total suffix)
            "cryptoscreener_gov_current_queue_depth",
            "cryptoscreener_gov_max_queue_depth",
            "cryptoscreener_gov_current_concurrent",
            "cryptoscreener_gov_max_concurrent_requests",
            "cryptoscreener_cb_last_open_duration_ms",
            # Counters (prometheus_client adds _total suffix)
            "cryptoscreener_gov_requests_allowed_total",
            "cryptoscreener_gov_requests_dropped_total",
            "cryptoscreener_cb_transitions_closed_to_open_total",
            "cryptoscreener_ws_total_disconnects_total",
            "cryptoscreener_ws_total_reconnect_attempts_total",
            "cryptoscreener_ws_total_ping_timeouts_total",
            "cryptoscreener_ws_total_subscribe_delayed_total",
        }
        assert expected == REQUIRED_METRIC_NAMES

    def test_exporter_exports_all_required_metrics(self):
        """Exporter must export all metrics needed by alert_rules.yml."""
        registry = CollectorRegistry()
        exporter = MetricsExporter(registry=registry)

        # Create minimal components
        governor = RestGovernor(config=RestGovernorConfig())
        circuit_breaker = CircuitBreaker()
        connector_metrics = ConnectorMetrics()

        # Update to populate metrics
        exporter.update(
            governor=governor,
            circuit_breaker=circuit_breaker,
            connector_metrics=connector_metrics,
        )

        # Generate output
        output = generate_latest(registry).decode("utf-8")

        # Check each required metric is present
        missing_metrics = []
        for metric_name in REQUIRED_METRIC_NAMES:
            # Metric should appear as either:
            # - cryptoscreener_gov_xxx (for gauge)
            # - cryptoscreener_gov_xxx_total (for counter, prometheus convention)
            # Or in HELP/TYPE lines
            if metric_name not in output and f"{metric_name}_total" not in output:
                missing_metrics.append(metric_name)

        assert not missing_metrics, (
            f"Missing required metrics: {missing_metrics}\nMetrics output:\n{output}"
        )

    def test_metric_names_match_alert_rules_yml(self):
        """Cross-check metric names against actual alert_rules.yml file."""
        # Find alert_rules.yml
        project_root = Path(__file__).parent.parent.parent
        alert_rules_path = project_root / "monitoring" / "alert_rules.yml"

        if not alert_rules_path.exists():
            pytest.skip(f"alert_rules.yml not found at {alert_rules_path}")

        content = alert_rules_path.read_text()

        # Extract metric names from expr: blocks only (actual PromQL expressions)
        # Pattern: matches cryptoscreener_xxx_yyy (full metric names, not group names)
        # Must have at least 3 underscore-separated parts and end with alphanumeric
        metric_pattern = re.compile(r"cryptoscreener_(?:gov|cb|ws)_[a-z][a-z0-9_]*[a-z0-9]")
        referenced_metrics = set(metric_pattern.findall(content))

        # Filter out partial matches (e.g., cryptoscreener_gov_ without suffix)
        referenced_metrics = {m for m in referenced_metrics if not m.endswith("_")}

        # Verify all referenced metrics are in REQUIRED_METRIC_NAMES
        missing_from_exporter = referenced_metrics - REQUIRED_METRIC_NAMES
        assert not missing_from_exporter, (
            f"alert_rules.yml references metrics not in REQUIRED_METRIC_NAMES: "
            f"{missing_from_exporter}"
        )


class TestMetricsSnapshot:
    """DEC-025: Verify metrics output format and values."""

    def test_metrics_snapshot_format(self):
        """Metrics output has correct Prometheus format."""
        registry = CollectorRegistry()
        exporter = MetricsExporter(registry=registry)

        # Setup components with known values
        governor = RestGovernor(
            config=RestGovernorConfig(
                max_queue_depth=50,
                max_concurrent_requests=10,
            )
        )
        governor.metrics.current_queue_depth = 5
        governor.metrics.current_concurrent = 3
        governor.metrics.requests_allowed = 100
        governor.metrics.requests_dropped = 2

        circuit_breaker = CircuitBreaker()
        circuit_breaker.metrics.transitions_closed_to_open = 1
        circuit_breaker.metrics.last_open_duration_ms = 5000

        connector_metrics = ConnectorMetrics(
            total_disconnects=10,
            total_reconnect_attempts=15,
            total_ping_timeouts=2,
            total_subscribe_delayed=8,
        )

        # Update exporter
        exporter.update(
            governor=governor,
            circuit_breaker=circuit_breaker,
            connector_metrics=connector_metrics,
        )

        # Generate output
        output = generate_latest(registry).decode("utf-8")

        # Verify format: lines should be # HELP, # TYPE, or metric{labels} value
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            assert (
                line.startswith("# HELP")
                or line.startswith("# TYPE")
                or re.match(r"^cryptoscreener_\w+", line)
            ), f"Invalid line format: {line}"

    def test_gauge_values_are_set_correctly(self):
        """Gauge metrics reflect current component state."""
        registry = CollectorRegistry()
        exporter = MetricsExporter(registry=registry)

        governor = RestGovernor(
            config=RestGovernorConfig(
                max_queue_depth=100,
                max_concurrent_requests=20,
            )
        )
        governor.metrics.current_queue_depth = 42
        governor.metrics.current_concurrent = 7

        circuit_breaker = CircuitBreaker()
        circuit_breaker.metrics.last_open_duration_ms = 12345

        exporter.update(governor=governor, circuit_breaker=circuit_breaker)

        output = generate_latest(registry).decode("utf-8")

        # Check gauge values
        assert "cryptoscreener_gov_current_queue_depth 42.0" in output
        assert "cryptoscreener_gov_max_queue_depth 100.0" in output
        assert "cryptoscreener_gov_current_concurrent 7.0" in output
        assert "cryptoscreener_gov_max_concurrent_requests 20.0" in output
        assert "cryptoscreener_cb_last_open_duration_ms 12345.0" in output

    def test_counter_values_increment_correctly(self):
        """Counter metrics increment based on delta from source."""
        registry = CollectorRegistry()
        exporter = MetricsExporter(registry=registry)

        governor = RestGovernor(config=RestGovernorConfig())
        governor.metrics.requests_allowed = 50
        governor.metrics.requests_dropped = 5

        connector_metrics = ConnectorMetrics(
            total_disconnects=10,
            total_reconnect_attempts=20,
        )

        # First update
        exporter.update(governor=governor, connector_metrics=connector_metrics)
        output1 = generate_latest(registry).decode("utf-8")

        # Counters should show initial values (or _total suffix)
        assert "cryptoscreener_gov_requests_allowed_total 50.0" in output1
        assert "cryptoscreener_gov_requests_dropped_total 5.0" in output1
        assert "cryptoscreener_ws_total_disconnects_total 10.0" in output1

        # Increment source values
        governor.metrics.requests_allowed = 75
        governor.metrics.requests_dropped = 8
        connector_metrics.total_disconnects = 15
        connector_metrics.total_reconnect_attempts = 25

        # Second update
        exporter.update(governor=governor, connector_metrics=connector_metrics)
        output2 = generate_latest(registry).decode("utf-8")

        # Counters should have incremented by delta
        assert "cryptoscreener_gov_requests_allowed_total 75.0" in output2
        assert "cryptoscreener_gov_requests_dropped_total 8.0" in output2
        assert "cryptoscreener_ws_total_disconnects_total 15.0" in output2

    def test_empty_components_export_zeros(self):
        """Fresh components with no activity export zero values."""
        registry = CollectorRegistry()
        exporter = MetricsExporter(registry=registry)

        governor = RestGovernor(config=RestGovernorConfig())
        circuit_breaker = CircuitBreaker()
        connector_metrics = ConnectorMetrics()

        exporter.update(
            governor=governor,
            circuit_breaker=circuit_breaker,
            connector_metrics=connector_metrics,
        )

        output = generate_latest(registry).decode("utf-8")

        # All gauges/counters should be 0
        assert "cryptoscreener_gov_current_queue_depth 0.0" in output
        assert "cryptoscreener_gov_current_concurrent 0.0" in output
        assert "cryptoscreener_cb_last_open_duration_ms 0.0" in output
        assert "cryptoscreener_ws_total_disconnects_total 0.0" in output

    def test_reset_counter_tracking(self):
        """reset_counter_tracking allows re-sync after component reset."""
        registry = CollectorRegistry()
        exporter = MetricsExporter(registry=registry)

        governor = RestGovernor(config=RestGovernorConfig())
        governor.metrics.requests_allowed = 100

        exporter.update(governor=governor)

        # Simulate component reset (value goes back to 0)
        governor.metrics.requests_allowed = 0

        # Without reset_counter_tracking, delta would be negative (ignored)
        exporter.update(governor=governor)
        output1 = generate_latest(registry).decode("utf-8")
        assert "cryptoscreener_gov_requests_allowed_total 100.0" in output1

        # Reset tracking
        exporter.reset_counter_tracking()

        # Now set to new value
        governor.metrics.requests_allowed = 50
        exporter.update(governor=governor)
        output2 = generate_latest(registry).decode("utf-8")
        # Counter should now show cumulative (100 + 50 = 150)
        assert "cryptoscreener_gov_requests_allowed_total 150.0" in output2


class TestMetricsSnapshotOutput:
    """DEC-025: Full metrics snapshot for documentation."""

    def test_full_metrics_snapshot(self):
        """Generate and display full metrics output for verification."""
        registry = CollectorRegistry()
        exporter = MetricsExporter(registry=registry)

        # Setup with representative values
        governor = RestGovernor(
            config=RestGovernorConfig(
                max_queue_depth=50,
                max_concurrent_requests=10,
            )
        )
        governor.metrics.current_queue_depth = 3
        governor.metrics.current_concurrent = 2
        governor.metrics.requests_allowed = 1000
        governor.metrics.requests_dropped = 5

        circuit_breaker = CircuitBreaker()
        circuit_breaker.metrics.transitions_closed_to_open = 2
        circuit_breaker.metrics.last_open_duration_ms = 15000

        connector_metrics = ConnectorMetrics(
            total_disconnects=8,
            total_reconnect_attempts=12,
            total_ping_timeouts=1,
            total_subscribe_delayed=3,
        )

        exporter.update(
            governor=governor,
            circuit_breaker=circuit_breaker,
            connector_metrics=connector_metrics,
        )

        output = generate_latest(registry).decode("utf-8")

        # Verify all required metrics are present
        for metric_name in REQUIRED_METRIC_NAMES:
            assert metric_name in output or f"{metric_name}_total" in output, (
                f"Missing metric: {metric_name}"
            )

        # Print snapshot for manual verification (visible in pytest -v output)
        print("\n=== METRICS SNAPSHOT ===")
        print(output)
        print("=== END SNAPSHOT ===\n")

        # Basic sanity checks on output
        assert "# HELP" in output
        assert "# TYPE" in output
        assert "cryptoscreener_gov_" in output
        assert "cryptoscreener_cb_" in output
        assert "cryptoscreener_ws_" in output
