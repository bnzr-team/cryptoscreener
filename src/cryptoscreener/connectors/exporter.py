"""
Prometheus metrics exporter for CryptoScreener infrastructure.

DEC-025: Exports low-cardinality metrics for Governor, CircuitBreaker, and WebSocket
components. No high-cardinality labels (symbol, endpoint, path, etc.).

Metric names are aligned 1:1 with monitoring/alert_rules.yml.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge
from prometheus_client.registry import CollectorRegistry

if TYPE_CHECKING:
    from cryptoscreener.connectors.backoff import CircuitBreaker, RestGovernor
    from cryptoscreener.connectors.binance.types import ConnectorMetrics


# Forbidden labels that would cause cardinality explosion
FORBIDDEN_LABELS = frozenset(
    {
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
)


class MetricsExporter:
    """
    Prometheus metrics exporter for CryptoScreener connector infrastructure.

    DEC-025: Exports only low-cardinality metrics. No symbol, endpoint, or other
    dynamic labels that could cause cardinality explosion.

    Metric names match monitoring/alert_rules.yml exactly:
    - cryptoscreener_gov_* : RestGovernor metrics
    - cryptoscreener_cb_*  : CircuitBreaker metrics
    - cryptoscreener_ws_*  : WebSocket/connector metrics

    Usage:
        registry = CollectorRegistry()
        exporter = MetricsExporter(registry=registry)
        exporter.update(governor=gov, circuit_breaker=cb, connector_metrics=cm)
        # generate_latest(registry) -> bytes for /metrics endpoint
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """
        Initialize metrics exporter.

        Args:
            registry: Prometheus CollectorRegistry. If None, uses default registry.
        """
        self._registry = registry or CollectorRegistry()

        # === Governor metrics (cryptoscreener_gov_*) ===
        self._gov_current_queue_depth = Gauge(
            "cryptoscreener_gov_current_queue_depth",
            "Current number of requests waiting in governor queue",
            registry=self._registry,
        )
        self._gov_max_queue_depth = Gauge(
            "cryptoscreener_gov_max_queue_depth",
            "Maximum queue depth configured for governor",
            registry=self._registry,
        )
        self._gov_requests_allowed = Counter(
            "cryptoscreener_gov_requests_allowed",
            "Total requests allowed through governor",
            registry=self._registry,
        )
        self._gov_requests_dropped = Counter(
            "cryptoscreener_gov_requests_dropped",
            "Total requests dropped by governor (queue full, timeout, breaker)",
            registry=self._registry,
        )
        self._gov_current_concurrent = Gauge(
            "cryptoscreener_gov_current_concurrent",
            "Current number of concurrent requests in flight",
            registry=self._registry,
        )
        self._gov_max_concurrent_requests = Gauge(
            "cryptoscreener_gov_max_concurrent_requests",
            "Maximum concurrent requests configured for governor",
            registry=self._registry,
        )

        # === Circuit Breaker metrics (cryptoscreener_cb_*) ===
        self._cb_transitions_closed_to_open = Counter(
            "cryptoscreener_cb_transitions_closed_to_open",
            "Number of times circuit breaker transitioned from CLOSED to OPEN",
            registry=self._registry,
        )
        self._cb_last_open_duration_ms = Gauge(
            "cryptoscreener_cb_last_open_duration_ms",
            "Duration of the most recent OPEN period in milliseconds",
            registry=self._registry,
        )

        # === WebSocket metrics (cryptoscreener_ws_*) ===
        self._ws_total_disconnects = Counter(
            "cryptoscreener_ws_total_disconnects",
            "Total unintentional WebSocket disconnection events",
            registry=self._registry,
        )
        self._ws_total_reconnect_attempts = Counter(
            "cryptoscreener_ws_total_reconnect_attempts",
            "Total WebSocket reconnect attempts (allowed + denied)",
            registry=self._registry,
        )
        self._ws_total_ping_timeouts = Counter(
            "cryptoscreener_ws_total_ping_timeouts",
            "Total WebSocket ping/pong timeout events",
            registry=self._registry,
        )
        self._ws_total_subscribe_delayed = Counter(
            "cryptoscreener_ws_total_subscribe_delayed",
            "Total outbound subscribe requests delayed by throttler",
            registry=self._registry,
        )

        # === DEC-028: Pipeline backpressure metrics ===
        self._pipeline_event_queue_depth = Gauge(
            "cryptoscreener_pipeline_event_queue_depth",
            "Current event queue depth (WS → pipeline)",
            registry=self._registry,
        )
        self._pipeline_snapshot_queue_depth = Gauge(
            "cryptoscreener_pipeline_snapshot_queue_depth",
            "Current snapshot queue depth (feature engine → consumer)",
            registry=self._registry,
        )
        self._pipeline_tick_drift_ms = Gauge(
            "cryptoscreener_pipeline_tick_drift_ms",
            "Last observed main-loop tick drift in milliseconds",
            registry=self._registry,
        )
        self._pipeline_rss_mb = Gauge(
            "cryptoscreener_pipeline_rss_mb",
            "Current process RSS in megabytes",
            registry=self._registry,
        )
        self._pipeline_events_dropped = Counter(
            "cryptoscreener_pipeline_events_dropped",
            "Total events dropped due to backpressure (queue full)",
            registry=self._registry,
        )
        self._pipeline_snapshots_dropped = Counter(
            "cryptoscreener_pipeline_snapshots_dropped",
            "Total snapshots dropped due to backpressure (queue full)",
            registry=self._registry,
        )

        # Track last seen values for counter increments (counters are monotonic)
        self._last_gov_requests_allowed = 0
        self._last_gov_requests_dropped = 0
        self._last_cb_transitions_closed_to_open = 0
        self._last_ws_total_disconnects = 0
        self._last_ws_total_reconnect_attempts = 0
        self._last_ws_total_ping_timeouts = 0
        self._last_ws_total_subscribe_delayed = 0
        self._last_pipeline_events_dropped = 0
        self._last_pipeline_snapshots_dropped = 0

    @property
    def registry(self) -> CollectorRegistry:
        """Get the Prometheus CollectorRegistry."""
        return self._registry

    def update(
        self,
        governor: RestGovernor | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        connector_metrics: ConnectorMetrics | None = None,
        *,
        pipeline_metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Update all metrics from component states.

        Call this periodically (e.g., every scrape or on a timer) to sync
        internal component metrics to Prometheus.

        Args:
            governor: RestGovernor instance for queue/concurrency metrics.
            circuit_breaker: CircuitBreaker instance for breaker metrics.
            connector_metrics: ConnectorMetrics for aggregated WebSocket metrics.
            pipeline_metrics: DEC-028 pipeline health gauges (dict with keys:
                event_queue_depth, snapshot_queue_depth, tick_drift_ms, rss_mb,
                events_dropped, snapshots_dropped).
        """
        if governor is not None:
            self._update_governor_metrics(governor)

        if circuit_breaker is not None:
            self._update_circuit_breaker_metrics(circuit_breaker)

        if connector_metrics is not None:
            self._update_websocket_metrics(connector_metrics)

        if pipeline_metrics is not None:
            self._update_pipeline_metrics(pipeline_metrics)

    def _update_governor_metrics(self, governor: RestGovernor) -> None:
        """Update governor metrics (gauges and counters)."""
        # Gauges: set directly
        self._gov_current_queue_depth.set(governor.metrics.current_queue_depth)
        self._gov_max_queue_depth.set(governor.config.max_queue_depth)
        self._gov_current_concurrent.set(governor.metrics.current_concurrent)
        self._gov_max_concurrent_requests.set(governor.config.max_concurrent_requests)

        # Counters: increment by delta since last update
        current_allowed = governor.metrics.requests_allowed
        delta_allowed = current_allowed - self._last_gov_requests_allowed
        if delta_allowed > 0:
            self._gov_requests_allowed.inc(delta_allowed)
        self._last_gov_requests_allowed = current_allowed

        current_dropped = governor.metrics.requests_dropped
        delta_dropped = current_dropped - self._last_gov_requests_dropped
        if delta_dropped > 0:
            self._gov_requests_dropped.inc(delta_dropped)
        self._last_gov_requests_dropped = current_dropped

    def _update_circuit_breaker_metrics(self, circuit_breaker: CircuitBreaker) -> None:
        """Update circuit breaker metrics."""
        # Gauge: last open duration
        self._cb_last_open_duration_ms.set(circuit_breaker.metrics.last_open_duration_ms)

        # Counter: transitions to OPEN
        current_transitions = circuit_breaker.metrics.transitions_closed_to_open
        delta = current_transitions - self._last_cb_transitions_closed_to_open
        if delta > 0:
            self._cb_transitions_closed_to_open.inc(delta)
        self._last_cb_transitions_closed_to_open = current_transitions

    def _update_websocket_metrics(self, connector_metrics: ConnectorMetrics) -> None:
        """Update WebSocket metrics from aggregated connector metrics."""
        # Counter: total disconnects
        current_disconnects = connector_metrics.total_disconnects
        delta_disconnects = current_disconnects - self._last_ws_total_disconnects
        if delta_disconnects > 0:
            self._ws_total_disconnects.inc(delta_disconnects)
        self._last_ws_total_disconnects = current_disconnects

        # Counter: reconnect attempts
        current_attempts = connector_metrics.total_reconnect_attempts
        delta_attempts = current_attempts - self._last_ws_total_reconnect_attempts
        if delta_attempts > 0:
            self._ws_total_reconnect_attempts.inc(delta_attempts)
        self._last_ws_total_reconnect_attempts = current_attempts

        # Counter: ping timeouts
        current_timeouts = connector_metrics.total_ping_timeouts
        delta_timeouts = current_timeouts - self._last_ws_total_ping_timeouts
        if delta_timeouts > 0:
            self._ws_total_ping_timeouts.inc(delta_timeouts)
        self._last_ws_total_ping_timeouts = current_timeouts

        # Counter: subscribe delayed
        current_delayed = connector_metrics.total_subscribe_delayed
        delta_delayed = current_delayed - self._last_ws_total_subscribe_delayed
        if delta_delayed > 0:
            self._ws_total_subscribe_delayed.inc(delta_delayed)
        self._last_ws_total_subscribe_delayed = current_delayed

    def _update_pipeline_metrics(self, pm: dict[str, float]) -> None:
        """DEC-028: Update pipeline backpressure gauges and counters."""
        # Gauges: set directly
        self._pipeline_event_queue_depth.set(pm.get("event_queue_depth", 0))
        self._pipeline_snapshot_queue_depth.set(pm.get("snapshot_queue_depth", 0))
        self._pipeline_tick_drift_ms.set(pm.get("tick_drift_ms", 0))
        self._pipeline_rss_mb.set(pm.get("rss_mb", 0))

        # Counters: increment by delta
        current_events_dropped = int(pm.get("events_dropped", 0))
        delta = current_events_dropped - self._last_pipeline_events_dropped
        if delta > 0:
            self._pipeline_events_dropped.inc(delta)
        self._last_pipeline_events_dropped = current_events_dropped

        current_snapshots_dropped = int(pm.get("snapshots_dropped", 0))
        delta = current_snapshots_dropped - self._last_pipeline_snapshots_dropped
        if delta > 0:
            self._pipeline_snapshots_dropped.inc(delta)
        self._last_pipeline_snapshots_dropped = current_snapshots_dropped

    def reset_counter_tracking(self) -> None:
        """
        Reset internal counter tracking.

        Use when components are reset or for testing.
        Does NOT reset the Prometheus counters themselves.
        """
        self._last_gov_requests_allowed = 0
        self._last_gov_requests_dropped = 0
        self._last_cb_transitions_closed_to_open = 0
        self._last_ws_total_disconnects = 0
        self._last_ws_total_reconnect_attempts = 0
        self._last_ws_total_ping_timeouts = 0
        self._last_ws_total_subscribe_delayed = 0
        self._last_pipeline_events_dropped = 0
        self._last_pipeline_snapshots_dropped = 0


# Required metric names that must exist for alert_rules.yml to work
# Note: Counters are exported with _total suffix by prometheus_client
REQUIRED_METRIC_NAMES: frozenset[str] = frozenset(
    {
        # Governor (Gauges)
        "cryptoscreener_gov_current_queue_depth",
        "cryptoscreener_gov_max_queue_depth",
        "cryptoscreener_gov_current_concurrent",
        "cryptoscreener_gov_max_concurrent_requests",
        # Governor (Counters - exported with _total suffix)
        "cryptoscreener_gov_requests_allowed_total",
        "cryptoscreener_gov_requests_dropped_total",
        # CircuitBreaker (Counter - exported with _total suffix)
        "cryptoscreener_cb_transitions_closed_to_open_total",
        # CircuitBreaker (Gauge)
        "cryptoscreener_cb_last_open_duration_ms",
        # WebSocket (Counters - exported with _total suffix)
        "cryptoscreener_ws_total_disconnects_total",
        "cryptoscreener_ws_total_reconnect_attempts_total",
        "cryptoscreener_ws_total_ping_timeouts_total",
        "cryptoscreener_ws_total_subscribe_delayed_total",
        # Pipeline backpressure (Gauges) — DEC-028
        "cryptoscreener_pipeline_event_queue_depth",
        "cryptoscreener_pipeline_snapshot_queue_depth",
        "cryptoscreener_pipeline_tick_drift_ms",
        "cryptoscreener_pipeline_rss_mb",
        # Pipeline backpressure (Counters) — DEC-028
        "cryptoscreener_pipeline_events_dropped_total",
        "cryptoscreener_pipeline_snapshots_dropped_total",
    }
)
