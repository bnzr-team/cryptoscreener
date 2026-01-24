"""Tests for WebSocket shard."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from cryptoscreener.connectors.backoff import CircuitBreaker
from cryptoscreener.connectors.binance.shard import WebSocketShard
from cryptoscreener.connectors.binance.types import (
    ConnectionState,
    ConnectorConfig,
    RawMessage,
    StreamSubscription,
    StreamType,
)


class TestWebSocketShard:
    """Tests for WebSocketShard."""

    @pytest.fixture
    def config(self) -> ConnectorConfig:
        """Create connector config."""
        return ConnectorConfig()

    @pytest.fixture
    def circuit_breaker(self) -> CircuitBreaker:
        """Create circuit breaker."""
        return CircuitBreaker()

    @pytest.fixture
    def message_callback(self) -> AsyncMock:
        """Create async message callback."""
        return AsyncMock()

    @pytest.fixture
    def shard(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> WebSocketShard:
        """Create WebSocket shard instance."""
        return WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=message_callback,
        )

    def test_initial_state(self, shard: WebSocketShard) -> None:
        """Initial shard state."""
        assert shard.shard_id == 0
        assert shard.state == ConnectionState.DISCONNECTED
        assert shard.stream_count == 0
        assert shard.can_add_streams is True
        assert shard.available_capacity == 800

    def test_stream_subscription_management(self, shard: WebSocketShard) -> None:
        """Subscriptions are tracked correctly."""
        sub1 = StreamSubscription(symbol="BTCUSDT", stream_type=StreamType.TRADE)
        sub2 = StreamSubscription(symbol="ETHUSDT", stream_type=StreamType.TRADE)

        # Add subscriptions (not connected, so won't send WS command)
        shard._subscriptions.add(sub1)
        shard._subscriptions.add(sub2)
        shard._active_streams.add(sub1.to_stream_name())
        shard._active_streams.add(sub2.to_stream_name())

        assert shard.stream_count == 2
        assert shard.available_capacity == 798

    def test_max_streams_capacity(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> None:
        """Cannot add streams beyond max capacity."""
        config.shard_config.max_streams = 2

        shard = WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=message_callback,
        )

        # Add max streams
        shard._active_streams.add("btcusdt@aggTrade")
        shard._active_streams.add("ethusdt@aggTrade")

        assert shard.stream_count == 2
        assert shard.can_add_streams is False
        assert shard.available_capacity == 0

    def test_build_ws_url_no_streams(self, shard: WebSocketShard) -> None:
        """WS URL without streams."""
        url = shard._build_ws_url()
        assert url == "wss://fstream.binance.com/ws"

    def test_build_ws_url_with_streams(self, shard: WebSocketShard) -> None:
        """WS URL with combined streams."""
        shard._active_streams.add("btcusdt@aggTrade")
        shard._active_streams.add("ethusdt@aggTrade")

        url = shard._build_ws_url()

        assert "wss://fstream.binance.com/stream?streams=" in url
        assert "btcusdt@aggTrade" in url
        assert "ethusdt@aggTrade" in url

    @pytest.mark.asyncio
    async def test_connect_circuit_breaker_open(
        self,
        shard: WebSocketShard,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        """Connect fails when circuit breaker is open."""
        circuit_breaker.force_open(60000, "test")

        with pytest.raises(ConnectionError) as exc_info:
            await shard.connect()

        assert "Circuit breaker open" in str(exc_info.value)
        assert shard.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_idempotent(self, shard: WebSocketShard) -> None:
        """Disconnect can be called multiple times."""
        await shard.disconnect()
        await shard.disconnect()

        assert shard.state == ConnectionState.CLOSED

    def test_get_metrics(self, shard: WebSocketShard) -> None:
        """Get shard metrics."""
        shard._active_streams.add("btcusdt@aggTrade")
        shard._metrics.messages_received = 100
        shard._metrics.reconnect_count = 2

        metrics = shard.get_metrics()

        assert metrics.shard_id == 0
        assert metrics.stream_count == 1
        assert metrics.messages_received == 100
        assert metrics.reconnect_count == 2
        assert metrics.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_add_subscriptions_not_connected(
        self,
        shard: WebSocketShard,
    ) -> None:
        """Add subscriptions when not connected."""
        subs = [
            StreamSubscription(symbol="BTCUSDT", stream_type=StreamType.TRADE),
            StreamSubscription(symbol="ETHUSDT", stream_type=StreamType.TRADE),
        ]

        added = await shard.add_subscriptions(subs)

        assert len(added) == 2
        assert shard.stream_count == 2

    @pytest.mark.asyncio
    async def test_add_subscriptions_respects_capacity(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> None:
        """Add subscriptions respects max capacity."""
        config.shard_config.max_streams = 2

        shard = WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=message_callback,
        )

        subs = [
            StreamSubscription(symbol="BTCUSDT", stream_type=StreamType.TRADE),
            StreamSubscription(symbol="ETHUSDT", stream_type=StreamType.TRADE),
            StreamSubscription(symbol="SOLUSDT", stream_type=StreamType.TRADE),
        ]

        added = await shard.add_subscriptions(subs)

        # Only 2 added (max capacity)
        assert len(added) == 2
        assert shard.stream_count == 2

    @pytest.mark.asyncio
    async def test_remove_subscriptions(self, shard: WebSocketShard) -> None:
        """Remove subscriptions."""
        sub = StreamSubscription(symbol="BTCUSDT", stream_type=StreamType.TRADE)

        # Add then remove
        await shard.add_subscriptions([sub])
        assert shard.stream_count == 1

        removed = await shard.remove_subscriptions([sub])
        assert len(removed) == 1
        assert shard.stream_count == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_subscription(
        self,
        shard: WebSocketShard,
    ) -> None:
        """Remove non-existent subscription returns empty."""
        sub = StreamSubscription(symbol="BTCUSDT", stream_type=StreamType.TRADE)

        removed = await shard.remove_subscriptions([sub])
        assert len(removed) == 0

    def test_next_request_id_increments(self, shard: WebSocketShard) -> None:
        """Request ID increments on each call."""
        id1 = shard._next_request_id()
        id2 = shard._next_request_id()
        id3 = shard._next_request_id()

        assert id1 == 1
        assert id2 == 2
        assert id3 == 3

    def test_state_change_callback(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> None:
        """State change callback is called."""
        state_changes: list[tuple[int, ConnectionState]] = []

        def on_state_change(shard_id: int, state: ConnectionState) -> None:
            state_changes.append((shard_id, state))

        shard = WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=message_callback,
            on_state_change=on_state_change,
        )

        shard._set_state(ConnectionState.CONNECTING)
        shard._set_state(ConnectionState.CONNECTED)

        assert len(state_changes) == 2
        assert state_changes[0] == (0, ConnectionState.CONNECTING)
        assert state_changes[1] == (0, ConnectionState.CONNECTED)


class TestWebSocketShardMessageParsing:
    """Tests for message parsing in shard."""

    @pytest.fixture
    def config(self) -> ConnectorConfig:
        """Create connector config."""
        return ConnectorConfig()

    @pytest.fixture
    def circuit_breaker(self) -> CircuitBreaker:
        """Create circuit breaker."""
        return CircuitBreaker()

    @pytest.fixture
    def received_messages(self) -> list[RawMessage]:
        """List to collect received messages."""
        return []

    @pytest.fixture
    def shard(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        received_messages: list[RawMessage],
    ) -> WebSocketShard:
        """Create WebSocket shard with message collector."""

        async def on_message(msg: RawMessage) -> None:
            received_messages.append(msg)

        return WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=on_message,
        )

    @pytest.mark.asyncio
    async def test_combined_stream_format(
        self,
        shard: WebSocketShard,
        received_messages: list[RawMessage],
    ) -> None:
        """Parse combined stream message format."""
        # Simulate receive loop behavior
        msg_data: dict[str, Any] = {
            "stream": "btcusdt@aggTrade",
            "data": {"e": "aggTrade", "s": "BTCUSDT", "p": "50000"},
        }

        recv_ts = 1234567890

        # Extract data from combined format
        inner_data: dict[str, Any] = msg_data.get("data", msg_data)
        raw_msg = RawMessage(
            data=inner_data,
            recv_ts=recv_ts,
            shard_id=0,
        )

        await shard._on_message(raw_msg)

        assert len(received_messages) == 1
        assert received_messages[0].data["e"] == "aggTrade"
        assert received_messages[0].data["s"] == "BTCUSDT"
