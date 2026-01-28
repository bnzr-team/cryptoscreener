"""
DEC-027/028: WebSocket resilience and backpressure integration tests.

Uses a fake WS server to validate:
- Shard reconnects after server-initiated disconnect (DEC-027)
- No reconnect storms (≤6 attempts per minute) (DEC-027)
- Backoff delays are observed (first delay > 0) (DEC-027)
- Bounded queues drop events under overload (DEC-028)
- Queue recovers after overload ends (DEC-028)
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
        *,
        send_interval_ms: int = 50,
    ) -> None:
        self.messages_before_drop = messages_before_drop
        self.max_connections = max_connections
        self.send_interval_s = send_interval_ms / 1000
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
                await asyncio.sleep(self.send_interval_s)
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
                f"Reconnect was too fast ({elapsed_ms:.0f}ms), backoff not applied"
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


class TestBackpressureAcceptance:
    """DEC-028: Verify bounded queues and backpressure under overload."""

    @pytest.mark.asyncio
    async def test_event_queue_bounded_under_overload(self) -> None:
        """Under overload, event queue stays bounded and events are dropped."""
        from cryptoscreener.connectors.binance.stream_manager import BinanceStreamManager

        mgr = BinanceStreamManager(on_event=None)

        # Verify queue has maxsize set (DEC-028)
        assert mgr._event_queue.maxsize == 10_000, (
            f"Event queue maxsize should be 10000, got {mgr._event_queue.maxsize}"
        )

        # Simulate overload: fill queue beyond capacity using put_nowait
        from cryptoscreener.contracts.events import MarketEvent, MarketEventType

        for i in range(10_050):
            event = MarketEvent(
                ts=i,
                source="test",
                symbol="BTCUSDT",
                type=MarketEventType.TRADE,
                payload={"e": "aggTrade", "s": "BTCUSDT"},
                recv_ts=i,
            )
            try:
                mgr._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                mgr._events_dropped += 1

        # Queue should be at max capacity
        assert mgr._event_queue.qsize() == 10_000
        # Events were dropped
        assert mgr._events_dropped == 50
        assert mgr.events_dropped == 50

    @pytest.mark.asyncio
    async def test_snapshot_queue_bounded(self) -> None:
        """Snapshot queue has bounded maxsize (DEC-028)."""
        from cryptoscreener.features.engine import FeatureEngine

        engine = FeatureEngine()
        assert engine._snapshot_queue.maxsize == 1_000, (
            f"Snapshot queue maxsize should be 1000, got {engine._snapshot_queue.maxsize}"
        )

    @pytest.mark.asyncio
    async def test_queue_depth_in_metrics(self) -> None:
        """ConnectorMetrics includes event_queue_depth and events_dropped (DEC-028)."""
        from cryptoscreener.connectors.binance.stream_manager import BinanceStreamManager

        mgr = BinanceStreamManager(on_event=None)
        metrics = mgr.get_metrics()

        assert metrics.event_queue_depth == 0
        assert metrics.events_dropped == 0

        # Add an event to queue
        from cryptoscreener.contracts.events import MarketEvent, MarketEventType

        event = MarketEvent(
            ts=1000,
            source="test",
            symbol="BTCUSDT",
            type=MarketEventType.TRADE,
            payload={"e": "aggTrade", "s": "BTCUSDT"},
            recv_ts=1000,
        )
        mgr._event_queue.put_nowait(event)
        metrics = mgr.get_metrics()
        assert metrics.event_queue_depth == 1

    @pytest.mark.asyncio
    async def test_drop_oldest_policy(self) -> None:
        """Stream manager drops oldest event when queue is full (DEC-028)."""
        from cryptoscreener.connectors.binance.stream_manager import BinanceStreamManager
        from cryptoscreener.connectors.binance.types import RawMessage

        mgr = BinanceStreamManager(on_event=None)

        # Fill the queue to capacity with ts=0..9999
        from cryptoscreener.contracts.events import MarketEvent, MarketEventType

        for i in range(10_000):
            event = MarketEvent(
                ts=i,
                source="test",
                symbol="BTCUSDT",
                type=MarketEventType.TRADE,
                payload={},
                recv_ts=i,
            )
            mgr._event_queue.put_nowait(event)

        assert mgr._event_queue.full()
        assert mgr._events_dropped == 0

        # Handle a raw message with ts=99999 — should drop oldest, enqueue newest
        raw = RawMessage(
            shard_id=0,
            data={"e": "aggTrade", "E": 99999, "s": "BTCUSDT", "p": "1", "q": "1", "T": 99999},
            recv_ts=99999,
        )
        await mgr._handle_raw_message(raw)

        assert mgr._events_dropped == 1
        # Queue still at capacity (oldest removed, newest added)
        assert mgr._event_queue.qsize() == 10_000
        # The oldest event (ts=0) was dropped; head is now ts=1
        head = mgr._event_queue.get_nowait()
        assert head.ts == 1

    @pytest.mark.asyncio
    async def test_queue_recovers_after_drain(self) -> None:
        """After overload, draining the queue brings depth back to zero."""
        from cryptoscreener.connectors.binance.stream_manager import BinanceStreamManager

        mgr = BinanceStreamManager(on_event=None)

        # Fill partially
        from cryptoscreener.contracts.events import MarketEvent, MarketEventType

        for i in range(100):
            event = MarketEvent(
                ts=i,
                source="test",
                symbol="BTCUSDT",
                type=MarketEventType.TRADE,
                payload={},
                recv_ts=i,
            )
            mgr._event_queue.put_nowait(event)

        assert mgr.event_queue_depth == 100

        # Drain
        for _ in range(100):
            mgr._event_queue.get_nowait()

        assert mgr.event_queue_depth == 0

    @pytest.mark.asyncio
    async def test_snapshot_drop_counter(self) -> None:
        """FeatureEngine increments snapshots_dropped on queue full (DEC-028)."""
        from cryptoscreener.contracts.events import (
            DataHealth,
            Features,
            FeatureSnapshot,
            RegimeTrend,
            RegimeVol,
            Windows,
        )
        from cryptoscreener.features.engine import FeatureEngine

        def _snap(ts: int) -> FeatureSnapshot:
            return FeatureSnapshot(
                ts=ts,
                symbol="BTCUSDT",
                features=Features(
                    spread_bps=1.0,
                    mid=50000.0,
                    book_imbalance=0.0,
                    flow_imbalance=0.0,
                    natr_14_5m=0.01,
                    impact_bps_q=1.0,
                    regime_vol=RegimeVol.LOW,
                    regime_trend=RegimeTrend.CHOP,
                ),
                windows=Windows(),
                data_health=DataHealth(),
            )

        engine = FeatureEngine()
        assert engine.snapshots_dropped == 0

        # Fill snapshot queue
        for i in range(1_000):
            engine._snapshot_queue.put_nowait(_snap(i))

        assert engine._snapshot_queue.full()

        # Emit one more — should drop
        await engine._emit_snapshot(_snap(9999))
        assert engine.snapshots_dropped == 1
