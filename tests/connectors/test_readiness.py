"""
DEC-030: Readiness endpoint transition tests.

Tests that /readyz returns 503 when pipeline is not ready and 200 when ready,
including transitions based on WS connectivity and event staleness.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from prometheus_client.registry import CollectorRegistry

from cryptoscreener.connectors.metrics_server import create_metrics_app


class TestReadinessTransitions:
    """DEC-030: /readyz transitions between 200 and 503."""

    @pytest.fixture()
    def registry(self) -> CollectorRegistry:
        return CollectorRegistry()

    @pytest.mark.asyncio
    async def test_not_ready_then_ready(self, registry: CollectorRegistry) -> None:
        """Readyz transitions from 503 to 200 when ready_fn changes."""
        state: dict[str, Any] = {"ready": False, "reason": "warming up"}

        def ready_fn() -> tuple[bool, dict[str, Any]]:
            return bool(state.get("ready", False)), state

        app = create_metrics_app(registry, ready_fn=ready_fn)
        async with TestClient(TestServer(app)) as client:
            # Initially not ready
            resp = await client.get("/readyz")
            assert resp.status == 503

            # Transition to ready
            state["ready"] = True
            state.pop("reason", None)

            resp = await client.get("/readyz")
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_ready_then_stale(self, registry: CollectorRegistry) -> None:
        """Readyz transitions from 200 to 503 when feed goes stale."""
        state: dict[str, Any] = {"ready": True, "ws_connected": True}

        def ready_fn() -> tuple[bool, dict[str, Any]]:
            return bool(state.get("ready", False)), state

        app = create_metrics_app(registry, ready_fn=ready_fn)
        async with TestClient(TestServer(app)) as client:
            # Initially ready
            resp = await client.get("/readyz")
            assert resp.status == 200

            # Transition to stale
            state["ready"] = False
            state["reason"] = "stale: last event 45.0s ago"

            resp = await client.get("/readyz")
            assert resp.status == 503
            data = await resp.json()
            assert "stale" in data.get("reason", "")

    @pytest.mark.asyncio
    async def test_readyz_body_always_json(self, registry: CollectorRegistry) -> None:
        """Both 200 and 503 responses are valid JSON."""
        is_ready = False

        def ready_fn() -> tuple[bool, dict[str, Any]]:
            return is_ready, {"ready": is_ready}

        app = create_metrics_app(registry, ready_fn=ready_fn)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/readyz")
            assert resp.status == 503
            data = await resp.json()
            assert isinstance(data, dict)


class TestGetReadyInfoUnit:
    """DEC-030: Unit tests for LivePipeline.get_ready_info()."""

    def _make_pipeline(
        self,
        *,
        running: bool = True,
        active_shards: int = 1,
        last_snapshot_ts: int = 0,
        staleness_s: int = 30,
    ) -> Any:
        """Create a mock pipeline with controllable state."""
        from scripts.run_live import LivePipelineConfig

        pipeline = MagicMock()
        pipeline._running = running
        pipeline._config = LivePipelineConfig(readiness_staleness_s=staleness_s)
        pipeline._start_monotonic = 100.0

        # Mock metrics
        pipeline._metrics = MagicMock()
        pipeline._metrics.last_snapshot_ts = last_snapshot_ts

        # Mock stream manager
        cm = MagicMock()
        cm.active_shards = active_shards
        pipeline._stream_manager = MagicMock()
        pipeline._stream_manager.get_metrics.return_value = cm

        return pipeline

    def test_not_running_returns_not_ready(self) -> None:
        from scripts.run_live import LivePipeline

        pipeline = self._make_pipeline(running=False)
        is_ready, info = LivePipeline.get_ready_info(pipeline)
        assert is_ready is False
        assert "not running" in info["reason"]

    def test_no_shards_returns_not_ready(self) -> None:
        from scripts.run_live import LivePipeline

        pipeline = self._make_pipeline(active_shards=0)
        is_ready, info = LivePipeline.get_ready_info(pipeline)
        assert is_ready is False
        assert info["ws_connected"] is False

    def test_no_events_returns_not_ready(self) -> None:
        from scripts.run_live import LivePipeline

        pipeline = self._make_pipeline(last_snapshot_ts=0)
        is_ready, info = LivePipeline.get_ready_info(pipeline)
        assert is_ready is False
        assert "no events" in info["reason"]

    def test_fresh_events_returns_ready(self) -> None:
        import time

        from scripts.run_live import LivePipeline

        now_ms = int(time.time() * 1000)
        pipeline = self._make_pipeline(last_snapshot_ts=now_ms - 5000)  # 5s ago
        is_ready, info = LivePipeline.get_ready_info(pipeline)
        assert is_ready is True
        assert info["ready"] is True

    def test_stale_events_returns_not_ready(self) -> None:
        import time

        from scripts.run_live import LivePipeline

        now_ms = int(time.time() * 1000)
        pipeline = self._make_pipeline(
            last_snapshot_ts=now_ms - 60000,  # 60s ago
            staleness_s=30,
        )
        is_ready, info = LivePipeline.get_ready_info(pipeline)
        assert is_ready is False
        assert "stale" in info["reason"]
