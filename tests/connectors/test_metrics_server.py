"""
Smoke tests for Prometheus /metrics HTTP endpoint.

DEC-025: Validates GET /metrics returns correct format and content.
"""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer
from prometheus_client import Counter, Gauge
from prometheus_client.registry import CollectorRegistry

from cryptoscreener.connectors.metrics_server import create_metrics_app


@pytest.fixture()
def registry() -> CollectorRegistry:
    """Fresh Prometheus registry with sample metrics."""
    reg = CollectorRegistry()
    # Register a gauge and counter to verify they appear in output
    g = Gauge(
        "cryptoscreener_gov_current_queue_depth",
        "test gauge",
        registry=reg,
    )
    g.set(42.0)
    c = Counter(
        "cryptoscreener_ws_total_disconnects",
        "test counter",
        registry=reg,
    )
    c.inc(7)
    return reg


@pytest.fixture()
def empty_registry() -> CollectorRegistry:
    """Empty Prometheus registry."""
    return CollectorRegistry()


class TestMetricsEndpointSmoke:
    """DEC-025: GET /metrics returns 200 with correct content."""

    @pytest.mark.asyncio
    async def test_metrics_returns_200(self, registry: CollectorRegistry) -> None:
        """GET /metrics returns HTTP 200."""
        app = create_metrics_app(registry)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/metrics")
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_metrics_content_type(self, registry: CollectorRegistry) -> None:
        """GET /metrics returns Prometheus exposition content type."""
        app = create_metrics_app(registry)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/metrics")
            assert resp.content_type == "text/plain"
            # Full header includes version
            ct = resp.headers.get("Content-Type", "")
            assert ct.startswith("text/plain")
            assert "version=0.0.4" in ct

    @pytest.mark.asyncio
    async def test_metrics_contains_cryptoscreener_lines(self, registry: CollectorRegistry) -> None:
        """GET /metrics output contains registered cryptoscreener_* metrics."""
        app = create_metrics_app(registry)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/metrics")
            body = await resp.text()

            # Gauge should appear as-is
            assert "cryptoscreener_gov_current_queue_depth 42.0" in body

            # Counter should appear with _total suffix
            assert "cryptoscreener_ws_total_disconnects_total 7.0" in body

    @pytest.mark.asyncio
    async def test_metrics_has_help_and_type(self, registry: CollectorRegistry) -> None:
        """GET /metrics includes HELP and TYPE metadata lines."""
        app = create_metrics_app(registry)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/metrics")
            body = await resp.text()

            assert "# HELP cryptoscreener_gov_current_queue_depth" in body
            assert "# TYPE cryptoscreener_gov_current_queue_depth gauge" in body
            assert "# TYPE cryptoscreener_ws_total_disconnects_total counter" in body

    @pytest.mark.asyncio
    async def test_metrics_empty_registry_returns_200(
        self, empty_registry: CollectorRegistry
    ) -> None:
        """GET /metrics with empty registry returns 200 with empty body."""
        app = create_metrics_app(empty_registry)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/metrics")
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self, registry: CollectorRegistry) -> None:
        """Non-/metrics paths return 404."""
        app = create_metrics_app(registry)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/unknown")
            assert resp.status == 404
