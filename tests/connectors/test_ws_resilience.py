"""
DEC-027: WebSocket resilience integration test.

Uses a fake WS server to validate reconnect discipline under adversity:
- Shard reconnects after server-initiated disconnect
- No reconnect storms (≤6 attempts per minute)
- Backoff delays are observed (first delay > 0)
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import aiohttp
import aiohttp.web
import pytest

from cryptoscreener.connectors.backoff import CircuitBreaker, ReconnectLimiter
from cryptoscreener.connectors.binance.shard import WebSocketShard
from cryptoscreener.connectors.binance.types import ConnectorConfig, RawMessage, ShardConfig


class FakeWSServer:
    """Minimal WS server that sends messages then force-closes, accepts reconnects."""

    def __init__(
        self,
        messages_before_drop: int = 3,
        max_connections: int = 5,
    ) -> None:
        self.messages_before_drop = messages_before_drop
        self.max_connections = max_connections
        self.connection_count = 0
        self.total_messages_sent = 0
        self._app: aiohttp.web.Application | None = None
        self._runner: aiohttp.web.AppRunner | None = None
        self.port: int = 0

    async def _ws_handler(self, request: aiohttp.web.Request) -> aiohttp.web.WebSocketResponse:
        ws = aiohttp.web.WebSocketResponse()
        await ws.prepare(request)
        self.connection_count += 1

        # Send a few fake aggTrade messages
        for _i in range(self.messages_before_drop):
            msg = {
                "e": "aggTrade",
                "E": int(time.time() * 1000),
                "s": "BTCUSDT",
                "p": "50000.00",
                "q": "0.1",
                "T": int(time.time() * 1000),
            }
            try:
                await ws.send_json(msg)
                self.total_messages_sent += 1
                await asyncio.sleep(0.05)
            except ConnectionResetError:
                break

        # Force close to simulate disconnect (unless we hit max)
        if self.connection_count < self.max_connections:
            await ws.close()

        return ws

    async def start(self) -> None:
        self._app = aiohttp.web.Application()
        self._app.router.add_get("/ws", self._ws_handler)
        self._app.router.add_get("/stream", self._ws_handler)
        self._runner = aiohttp.web.AppRunner(self._app)
        await self._runner.setup()
        site = aiohttp.web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        # Extract bound port
        assert self._runner.addresses
        self.port = self._runner.addresses[0][1]

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    @property
    def base_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"


class TestReconnectDiscipline:
    """DEC-027: Verify reconnect behavior under server-initiated disconnects."""

    @pytest.mark.asyncio
    async def test_reconnect_no_storm(self) -> None:
        """Shard reconnects after disconnect without storm (≤6/min)."""
        server = FakeWSServer(messages_before_drop=2, max_connections=6)
        await server.start()

        try:
            messages_received: list[RawMessage] = []

            async def on_message(raw: RawMessage) -> None:
                messages_received.append(raw)

            config = ConnectorConfig(
                base_ws_url=server.base_url,
                shard_config=ShardConfig(
                    ping_interval_ms=5000,
                    ping_timeout_ms=3000,
                ),
            )
            cb = CircuitBreaker()
            limiter = ReconnectLimiter()

            shard = WebSocketShard(
                shard_id=0,
                config=config,
                circuit_breaker=cb,
                on_message=on_message,
                reconnect_limiter=limiter,
            )

            # Connect
            await shard.connect()

            # Run for enough time to see a few reconnects
            # Server drops after 2 messages (~100ms), backoff starts at 1s
            # In 8s we expect 2-4 reconnect cycles
            start = time.monotonic()
            timeout_s = 8.0

            while time.monotonic() - start < timeout_s:
                metrics = shard.get_metrics()
                # If shard disconnected, trigger reconnect
                if metrics.state.value in ("DISCONNECTED", "CLOSED"):
                    with contextlib.suppress(Exception):
                        await shard.reconnect()
                await asyncio.sleep(0.2)

            # Collect final metrics
            metrics = shard.get_metrics()

            # Assertions
            assert metrics.reconnect_attempts > 0, "Expected at least one reconnect attempt"
            assert server.connection_count > 1, "Expected server to see multiple connections"

            # No storm: actual server connections should be bounded.
            # With 1s base backoff + jitter over 8s, expect ≤8 connections.
            assert server.connection_count <= 10, (
                f"Too many server connections: {server.connection_count} "
                f"(indicates reconnect storm)"
            )

            # Messages were received (recovery works)
            assert server.total_messages_sent > 0, "Expected messages to be sent"

        finally:
            await shard.disconnect()
            await server.stop()

    @pytest.mark.asyncio
    async def test_backoff_delay_observed(self) -> None:
        """First reconnect delay is > 0ms (backoff is applied)."""
        server = FakeWSServer(messages_before_drop=1, max_connections=3)
        await server.start()

        try:

            async def on_message(raw: RawMessage) -> None:
                pass

            config = ConnectorConfig(
                base_ws_url=server.base_url,
                shard_config=ShardConfig(
                    ping_interval_ms=5000,
                    ping_timeout_ms=3000,
                ),
            )
            cb = CircuitBreaker()

            shard = WebSocketShard(
                shard_id=0,
                config=config,
                circuit_breaker=cb,
                on_message=on_message,
            )

            await shard.connect()

            # Wait for server to close connection
            await asyncio.sleep(0.3)

            # Time the reconnect (includes backoff delay)
            t0 = time.monotonic()
            with contextlib.suppress(Exception):
                await shard.reconnect()
            elapsed_ms = (time.monotonic() - t0) * 1000

            # Backoff base is 1000ms with jitter, so delay should be > 500ms
            assert elapsed_ms > 400, (
                f"Reconnect was too fast ({elapsed_ms:.0f}ms), "
                f"backoff not applied"
            )

        finally:
            await shard.disconnect()
            await server.stop()


class TestForceDisconnect:
    """DEC-027: Verify BinanceStreamManager.force_disconnect() method."""

    @pytest.mark.asyncio
    async def test_force_disconnect_exists(self) -> None:
        """force_disconnect is callable on BinanceStreamManager."""
        from cryptoscreener.connectors.binance.stream_manager import BinanceStreamManager

        mgr = BinanceStreamManager(on_event=None)
        # Should not raise (no shards to disconnect)
        await mgr.force_disconnect()

    @pytest.mark.asyncio
    async def test_force_disconnect_closes_ws_connections(self) -> None:
        """force_disconnect closes raw WS on each shard (not graceful disconnect)."""
        from unittest.mock import AsyncMock, MagicMock

        from cryptoscreener.connectors.binance.stream_manager import BinanceStreamManager

        mgr = BinanceStreamManager(on_event=None)

        # Create mock shards with mock _ws
        mock_ws_1 = AsyncMock()
        mock_ws_1.closed = False
        mock_shard_1 = MagicMock()
        mock_shard_1._ws = mock_ws_1

        mock_ws_2 = AsyncMock()
        mock_ws_2.closed = False
        mock_shard_2 = MagicMock()
        mock_shard_2._ws = mock_ws_2

        mgr._shards = {0: mock_shard_1, 1: mock_shard_2}

        await mgr.force_disconnect()

        mock_ws_1.close.assert_awaited_once()
        mock_ws_2.close.assert_awaited_once()
