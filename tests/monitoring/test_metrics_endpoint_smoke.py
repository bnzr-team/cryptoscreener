"""
End-to-end smoke test for DEC-025 metrics exporter + HTTP endpoint.

Verifies:
- All 12 REQUIRED_METRIC_NAMES are present in /metrics output
- Metric TYPE annotations are correct (counter vs gauge)
- Counter monotonicity after sequential updates
- Gauge values reflect latest state
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
from prometheus_client.registry import CollectorRegistry

from cryptoscreener.connectors.exporter import REQUIRED_METRIC_NAMES, MetricsExporter
from cryptoscreener.connectors.metrics_server import create_metrics_app

# --- Stub types matching real interfaces used by MetricsExporter.update() ---


@dataclass
class _StubGovernorMetrics:
    current_queue_depth: int = 0
    current_concurrent: int = 0
    requests_allowed: int = 0
    requests_dropped: int = 0


@dataclass
class _StubGovernorConfig:
    max_queue_depth: int = 100
    max_concurrent_requests: int = 10


@dataclass
class _StubGovernor:
    metrics: _StubGovernorMetrics = field(default_factory=_StubGovernorMetrics)
    config: _StubGovernorConfig = field(default_factory=_StubGovernorConfig)


@dataclass
class _StubCBMetrics:
    transitions_closed_to_open: int = 0
    last_open_duration_ms: int = 0


@dataclass
class _StubCircuitBreaker:
    metrics: _StubCBMetrics = field(default_factory=_StubCBMetrics)


@dataclass
class _StubConnectorMetrics:
    total_disconnects: int = 0
    total_reconnect_attempts: int = 0
    total_ping_timeouts: int = 0
    total_subscribe_delayed: int = 0


# --- Expected TYPE annotations ---

EXPECTED_GAUGES: frozenset[str] = frozenset(
    {
        "cryptoscreener_gov_current_queue_depth",
        "cryptoscreener_gov_max_queue_depth",
        "cryptoscreener_gov_current_concurrent",
        "cryptoscreener_gov_max_concurrent_requests",
        "cryptoscreener_cb_last_open_duration_ms",
    }
)

EXPECTED_COUNTERS: frozenset[str] = frozenset(
    {
        "cryptoscreener_gov_requests_allowed_total",
        "cryptoscreener_gov_requests_dropped_total",
        "cryptoscreener_cb_transitions_closed_to_open_total",
        "cryptoscreener_ws_total_disconnects_total",
        "cryptoscreener_ws_total_reconnect_attempts_total",
        "cryptoscreener_ws_total_ping_timeouts_total",
        "cryptoscreener_ws_total_subscribe_delayed_total",
    }
)

# Counter TYPE line names.
# prometheus_client emits: # TYPE <name>_total counter
COUNTER_TYPE_NAMES: frozenset[str] = frozenset(
    {
        "cryptoscreener_gov_requests_allowed_total",
        "cryptoscreener_gov_requests_dropped_total",
        "cryptoscreener_cb_transitions_closed_to_open_total",
        "cryptoscreener_ws_total_disconnects_total",
        "cryptoscreener_ws_total_reconnect_attempts_total",
        "cryptoscreener_ws_total_ping_timeouts_total",
        "cryptoscreener_ws_total_subscribe_delayed_total",
    }
)


def _parse_type_lines(body: str) -> dict[str, str]:
    """Extract {metric_name: type} from # TYPE lines."""
    result: dict[str, str] = {}
    for match in re.finditer(r"^# TYPE (\S+) (\S+)$", body, re.MULTILINE):
        result[match.group(1)] = match.group(2)
    return result


def _parse_metric_values(body: str, name: str) -> float:
    """Extract the numeric value for a metric sample line."""
    pattern = rf"^{re.escape(name)}\s+([\d.eE+\-]+)$"
    match = re.search(pattern, body, re.MULTILINE)
    assert match is not None, f"Metric sample '{name}' not found in /metrics output"
    return float(match.group(1))


class TestMetricsEndpointSmoke(AioHTTPTestCase):
    """E2E smoke: exporter + /metrics endpoint runtime correctness."""

    async def get_application(self) -> None:  # type: ignore[override]
        self.registry = CollectorRegistry()
        self.exporter = MetricsExporter(registry=self.registry)
        return create_metrics_app(self.registry)

    # -- helpers --

    async def _fetch_metrics(self) -> str:
        resp = await self.client.get("/metrics")
        assert resp.status == 200
        ct = resp.headers.get("Content-Type", "")
        assert ct.startswith("text/plain"), f"Unexpected Content-Type: {ct}"
        body = await resp.text()
        return body

    # -- tests --

    @unittest_run_loop
    async def test_all_required_metric_names_present(self) -> None:
        """All 12 REQUIRED_METRIC_NAMES appear in /metrics after update."""
        gov = _StubGovernor()
        gov.metrics.requests_allowed = 5
        gov.metrics.requests_dropped = 1
        cb = _StubCircuitBreaker()
        cb.metrics.transitions_closed_to_open = 2
        cb.metrics.last_open_duration_ms = 500
        cm = _StubConnectorMetrics(
            total_disconnects=3,
            total_reconnect_attempts=4,
            total_ping_timeouts=1,
            total_subscribe_delayed=2,
        )
        self.exporter.update(governor=gov, circuit_breaker=cb, connector_metrics=cm)

        body = await self._fetch_metrics()
        missing = []
        for name in sorted(REQUIRED_METRIC_NAMES):
            if name not in body:
                missing.append(name)
        assert not missing, f"Missing metrics in /metrics output: {missing}"

    @unittest_run_loop
    async def test_type_annotations_correct(self) -> None:
        """# TYPE lines have correct counter/gauge annotations."""
        gov = _StubGovernor()
        cb = _StubCircuitBreaker()
        cm = _StubConnectorMetrics()
        self.exporter.update(governor=gov, circuit_breaker=cb, connector_metrics=cm)

        body = await self._fetch_metrics()
        types = _parse_type_lines(body)

        for gauge_name in sorted(EXPECTED_GAUGES):
            assert types.get(gauge_name) == "gauge", (
                f"{gauge_name}: expected TYPE gauge, got {types.get(gauge_name)!r}"
            )

        for cname in sorted(COUNTER_TYPE_NAMES):
            assert types.get(cname) == "counter", (
                f"{cname}: expected TYPE counter, got {types.get(cname)!r}"
            )

    @unittest_run_loop
    async def test_counters_monotonic_after_two_updates(self) -> None:
        """Counters never decrease between sequential updates."""
        gov = _StubGovernor()
        cb = _StubCircuitBreaker()
        cm = _StubConnectorMetrics()

        # Update 1: initial values
        gov.metrics.requests_allowed = 10
        gov.metrics.requests_dropped = 2
        cb.metrics.transitions_closed_to_open = 1
        cm.total_disconnects = 5
        cm.total_reconnect_attempts = 3
        cm.total_ping_timeouts = 1
        cm.total_subscribe_delayed = 2
        self.exporter.update(governor=gov, circuit_breaker=cb, connector_metrics=cm)

        body1 = await self._fetch_metrics()

        # Update 2: increase all counters
        gov.metrics.requests_allowed = 20
        gov.metrics.requests_dropped = 5
        cb.metrics.transitions_closed_to_open = 3
        cm.total_disconnects = 8
        cm.total_reconnect_attempts = 6
        cm.total_ping_timeouts = 4
        cm.total_subscribe_delayed = 7
        self.exporter.update(governor=gov, circuit_breaker=cb, connector_metrics=cm)

        body2 = await self._fetch_metrics()

        for counter_name in sorted(EXPECTED_COUNTERS):
            v1 = _parse_metric_values(body1, counter_name)
            v2 = _parse_metric_values(body2, counter_name)
            assert v2 >= v1, (
                f"{counter_name}: decreased from {v1} to {v2} (not monotonic)"
            )
            # Also verify counters actually increased
            assert v2 > v1, (
                f"{counter_name}: did not increase ({v1} -> {v2})"
            )

    @unittest_run_loop
    async def test_gauges_reflect_latest_values(self) -> None:
        """Gauges reflect the most recent update values."""
        gov = _StubGovernor()
        cb = _StubCircuitBreaker()
        cm = _StubConnectorMetrics()

        # Update 1
        gov.metrics.current_queue_depth = 5
        gov.config.max_queue_depth = 50
        gov.metrics.current_concurrent = 3
        gov.config.max_concurrent_requests = 10
        cb.metrics.last_open_duration_ms = 1000
        self.exporter.update(governor=gov, circuit_breaker=cb, connector_metrics=cm)

        body1 = await self._fetch_metrics()
        assert _parse_metric_values(body1, "cryptoscreener_gov_current_queue_depth") == 5.0
        assert _parse_metric_values(body1, "cryptoscreener_gov_max_queue_depth") == 50.0
        assert _parse_metric_values(body1, "cryptoscreener_cb_last_open_duration_ms") == 1000.0

        # Update 2: change gauge values
        gov.metrics.current_queue_depth = 12
        gov.config.max_queue_depth = 50
        gov.metrics.current_concurrent = 8
        gov.config.max_concurrent_requests = 10
        cb.metrics.last_open_duration_ms = 2500
        self.exporter.update(governor=gov, circuit_breaker=cb, connector_metrics=cm)

        body2 = await self._fetch_metrics()
        assert _parse_metric_values(body2, "cryptoscreener_gov_current_queue_depth") == 12.0
        assert _parse_metric_values(body2, "cryptoscreener_gov_current_concurrent") == 8.0
        assert _parse_metric_values(body2, "cryptoscreener_cb_last_open_duration_ms") == 2500.0
