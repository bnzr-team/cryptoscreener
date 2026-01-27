"""
Minimal HTTP server for Prometheus /metrics endpoint.

DEC-025: Serves generate_latest(registry) on GET /metrics.
No new metrics, labels, or business logic. Just wiring.

Uses aiohttp.web (already a project dependency for WS/REST client).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from aiohttp import web
from prometheus_client import generate_latest

if TYPE_CHECKING:
    from prometheus_client.registry import CollectorRegistry

logger = logging.getLogger(__name__)

# Prometheus exposition format content type
METRICS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Type alias for aiohttp handler
_Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


def _make_metrics_handler(
    registry: CollectorRegistry,
) -> _Handler:
    """Create GET /metrics handler bound to a registry."""

    async def handler(request: web.Request) -> web.Response:
        body = generate_latest(registry)
        return web.Response(
            body=body,
            content_type="text/plain; version=0.0.4",
            charset="utf-8",
        )

    return handler


def create_metrics_app(registry: CollectorRegistry) -> web.Application:
    """
    Create aiohttp Application with a single GET /metrics route.

    Args:
        registry: Prometheus CollectorRegistry to serve.

    Returns:
        aiohttp.web.Application ready to be started.
    """
    app = web.Application()
    app.router.add_get("/metrics", _make_metrics_handler(registry))
    return app


async def start_metrics_server(
    registry: CollectorRegistry,
    host: str = "0.0.0.0",
    port: int = 9090,
) -> web.AppRunner:
    """
    Start the metrics HTTP server.

    Args:
        registry: Prometheus CollectorRegistry to serve.
        host: Bind address (default: 0.0.0.0).
        port: Bind port (default: 9090).

    Returns:
        AppRunner (call runner.cleanup() on shutdown).
    """
    app = create_metrics_app(registry)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("Metrics server started on http://%s:%d/metrics", host, port)
    return runner


async def stop_metrics_server(runner: web.AppRunner) -> None:
    """
    Stop the metrics HTTP server.

    Args:
        runner: AppRunner returned by start_metrics_server.
    """
    await runner.cleanup()
    logger.info("Metrics server stopped")
