"""
Minimal HTTP server for Prometheus /metrics and /healthz endpoints.

DEC-025: Serves generate_latest(registry) on GET /metrics.
DEC-029: Serves pipeline health JSON on GET /healthz.

Uses aiohttp.web (already a project dependency for WS/REST client).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from aiohttp import web
from prometheus_client import generate_latest

if TYPE_CHECKING:
    from prometheus_client.registry import CollectorRegistry

logger = logging.getLogger(__name__)

# Prometheus exposition format content type
METRICS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Type alias for aiohttp handler
_Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]

# Type alias for health info callback
HealthFn = Callable[[], dict[str, Any]]


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


def _make_healthz_handler(
    health_fn: HealthFn | None = None,
) -> _Handler:
    """Create GET /healthz handler.

    Args:
        health_fn: Optional callback returning pipeline health dict.
            If None, returns a minimal {"status": "ok"} response.
    """

    async def handler(request: web.Request) -> web.Response:
        info = health_fn() if health_fn is not None else {"status": "ok"}
        return web.Response(
            body=json.dumps(info),
            content_type="application/json",
        )

    return handler


def create_metrics_app(
    registry: CollectorRegistry,
    *,
    health_fn: HealthFn | None = None,
) -> web.Application:
    """
    Create aiohttp Application with /metrics and /healthz routes.

    Args:
        registry: Prometheus CollectorRegistry to serve.
        health_fn: Optional callback for /healthz pipeline health info.

    Returns:
        aiohttp.web.Application ready to be started.
    """
    app = web.Application()
    app.router.add_get("/metrics", _make_metrics_handler(registry))
    app.router.add_get("/healthz", _make_healthz_handler(health_fn))
    return app


async def start_metrics_server(
    registry: CollectorRegistry,
    host: str = "0.0.0.0",
    port: int = 9090,
    *,
    health_fn: HealthFn | None = None,
) -> web.AppRunner:
    """
    Start the metrics HTTP server.

    Args:
        registry: Prometheus CollectorRegistry to serve.
        host: Bind address (default: 0.0.0.0).
        port: Bind port (default: 9090).
        health_fn: Optional callback for /healthz pipeline health info.

    Returns:
        AppRunner (call runner.cleanup() on shutdown).
    """
    app = create_metrics_app(registry, health_fn=health_fn)
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
