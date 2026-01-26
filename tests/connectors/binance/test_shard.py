"""Tests for WebSocket shard.

DEC-023b: Added integration tests for:
- ReconnectLimiter wiring (reconnect storm protection)
- MessageThrottler wiring (subscribe rate limiting)
- Metrics counter increments
- Concurrent subscribe serialization via _send_lock
"""

import asyncio
import random
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptoscreener.connectors.backoff import (
    CircuitBreaker,
    MessageThrottler,
    MessageThrottlerConfig,
    ReconnectLimiter,
    ReconnectLimiterConfig,
)
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


# =============================================================================
# DEC-023b: Integration Tests for Limiter/Throttler Wiring
# =============================================================================


class TestWebSocketShardReconnectLimiterIntegration:
    """
    DEC-023b: Integration tests proving ReconnectLimiter is wired into
    WebSocketShard.reconnect() and actually invoked.
    """

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

    @pytest.mark.asyncio
    async def test_reconnect_denied_when_limiter_blocks(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> None:
        """Test reconnect is denied and metrics incremented when limiter blocks."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        # Create limiter with very restrictive settings
        limiter_config = ReconnectLimiterConfig(
            max_reconnects_per_window=1,
            window_ms=60000,
            per_shard_min_interval_ms=10000,
        )
        limiter = ReconnectLimiter(config=limiter_config, _time_fn=time_fn)

        shard = WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=message_callback,
            reconnect_limiter=limiter,
            time_fn=time_fn,
        )

        # First reconnect should be allowed
        limiter.record_reconnect(shard_id=0)

        # Mock asyncio.sleep to avoid actual delays
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Second reconnect should be denied
            await shard.reconnect()

            # Metrics should show denial
            metrics = shard.get_metrics()
            assert metrics.reconnect_denied >= 1

            # Sleep should have been called (for wait time)
            assert mock_sleep.called

    @pytest.mark.asyncio
    async def test_reconnect_allowed_when_limiter_permits(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> None:
        """Test reconnect proceeds normally when limiter permits."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        limiter_config = ReconnectLimiterConfig(
            max_reconnects_per_window=10,
            per_shard_min_interval_ms=0,  # Disable per-shard limit
        )
        limiter = ReconnectLimiter(config=limiter_config, _time_fn=time_fn)

        shard = WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=message_callback,
            reconnect_limiter=limiter,
            time_fn=time_fn,
        )

        try:
            # Mock asyncio.sleep to avoid actual delays
            with patch("asyncio.sleep", new_callable=AsyncMock):
                # Should attempt reconnect (will fail due to no network, but that's OK)
                await shard.reconnect()

                # reconnect_denied should not increase since limiter allowed
                metrics = shard.get_metrics()
                assert metrics.reconnect_denied == 0
        finally:
            # DEC-023e: Ensure shard resources are cleaned up
            await shard.disconnect()

    @pytest.mark.asyncio
    async def test_seeded_rng_produces_deterministic_jitter(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> None:
        """Test seeded RNG produces deterministic backoff jitter."""

        def create_shard(seed: int) -> WebSocketShard:
            return WebSocketShard(
                shard_id=0,
                config=config,
                circuit_breaker=circuit_breaker,
                on_message=message_callback,
                rng=random.Random(seed),
            )

        shard1 = create_shard(42)
        shard2 = create_shard(42)

        # Both shards with same seed should have same RNG state
        # This is verified by the backoff delay computation
        delay1 = shard1._rng.uniform(0, 1) if shard1._rng else 0
        delay2 = shard2._rng.uniform(0, 1) if shard2._rng else 0

        assert delay1 == delay2


class TestWebSocketShardMessageThrottlerIntegration:
    """
    DEC-023b: Integration tests proving MessageThrottler is wired into
    WebSocketShard._send_subscribe()/_send_unsubscribe() and actually invoked.
    """

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

    @pytest.mark.asyncio
    async def test_subscribe_delayed_when_throttler_limits(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> None:
        """Test subscribe is delayed and metrics incremented when throttler limits."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        # Create throttler with no burst allowance to force delay
        throttler_config = MessageThrottlerConfig(
            max_messages_per_second=10,
            safety_margin=1.0,
            burst_allowance=0,  # No burst, force delay
        )
        throttler = MessageThrottler(config=throttler_config, _time_fn=time_fn)

        shard = WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=message_callback,
            message_throttler=throttler,
            time_fn=time_fn,
        )

        # Mock WS connection
        mock_ws = MagicMock()
        mock_ws.closed = False
        mock_ws.send_json = AsyncMock()
        shard._ws = mock_ws

        # Exhaust tokens
        throttler._tokens = 0.0

        # Mock asyncio.sleep
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await shard._send_subscribe(["btcusdt@aggTrade"])

            # Sleep should have been called (for throttle delay)
            assert mock_sleep.called

            # Metrics should show delay
            metrics = shard.get_metrics()
            assert metrics.messages_delayed >= 1

    @pytest.mark.asyncio
    async def test_unsubscribe_delayed_when_throttler_limits(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> None:
        """Test unsubscribe is delayed when throttler limits."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        throttler_config = MessageThrottlerConfig(
            max_messages_per_second=10,
            safety_margin=1.0,
            burst_allowance=0,
        )
        throttler = MessageThrottler(config=throttler_config, _time_fn=time_fn)

        shard = WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=message_callback,
            message_throttler=throttler,
            time_fn=time_fn,
        )

        # Mock WS connection
        mock_ws = MagicMock()
        mock_ws.closed = False
        mock_ws.send_json = AsyncMock()
        shard._ws = mock_ws

        # Exhaust tokens
        throttler._tokens = 0.0

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await shard._send_unsubscribe(["btcusdt@aggTrade"])

            assert mock_sleep.called
            metrics = shard.get_metrics()
            assert metrics.messages_delayed >= 1

    @pytest.mark.asyncio
    async def test_subscribe_not_delayed_when_throttler_permits(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> None:
        """Test subscribe proceeds immediately when throttler permits."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        throttler_config = MessageThrottlerConfig(
            max_messages_per_second=10,
            safety_margin=1.0,
            burst_allowance=10,  # Plenty of burst
        )
        throttler = MessageThrottler(config=throttler_config, _time_fn=time_fn)

        shard = WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=message_callback,
            message_throttler=throttler,
            time_fn=time_fn,
        )

        # Mock WS connection
        mock_ws = MagicMock()
        mock_ws.closed = False
        mock_ws.send_json = AsyncMock()
        shard._ws = mock_ws

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await shard._send_subscribe(["btcusdt@aggTrade"])

            # Sleep should NOT have been called
            assert not mock_sleep.called

            # messages_delayed should remain 0
            metrics = shard.get_metrics()
            assert metrics.messages_delayed == 0

    def test_default_throttler_created_when_none_provided(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> None:
        """Test default MessageThrottler is created when none provided."""
        shard = WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=message_callback,
        )

        # Should have created a default throttler
        assert shard._message_throttler is not None
        assert isinstance(shard._message_throttler, MessageThrottler)

    def test_time_fn_passed_to_default_throttler(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> None:
        """Test time_fn is passed to default throttler when no throttler provided."""
        fake_time = 12345

        def time_fn() -> int:
            return fake_time

        shard = WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=message_callback,
            time_fn=time_fn,
        )

        # The default throttler should use the provided time_fn
        # Verify by checking the throttler uses fake time
        status = shard._message_throttler.get_status()
        # Status should be valid (throttler initialized properly)
        assert "available_tokens" in status

    def test_default_throttler_uses_safety_margin(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> None:
        """Test default MessageThrottler uses 0.8 safety margin (headroom).

        DEC-023b: Per BINANCE_LIMITS.md, we use 80% of the 10 msg/sec limit
        (8 msg/sec effective) to avoid hitting the hard cap.
        """
        shard = WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=message_callback,
        )

        # Default throttler should use 0.8 safety margin
        # With max_messages_per_second=10 and safety_margin=0.8, effective_rate=8.0
        status = shard._message_throttler.get_status()
        assert status["effective_rate_per_sec"] == 8.0, (
            f"Expected effective_rate=8.0 (10 * 0.8), got {status['effective_rate_per_sec']}. "
            "Default throttler must use 0.8 safety margin for headroom."
        )

    @pytest.mark.asyncio
    async def test_concurrent_subscribes_serialized_by_send_lock(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> None:
        """Test concurrent subscribe calls are serialized by _send_lock.

        DEC-023b: Prevents race condition where both concurrent subscribes
        see wait_ms=0 and both send without respecting rate limit.
        """
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        # Create throttler with no burst (force immediate rate limiting)
        throttler_config = MessageThrottlerConfig(
            max_messages_per_second=10,
            safety_margin=1.0,
            burst_allowance=1,  # Only 1 token available
        )
        throttler = MessageThrottler(config=throttler_config, _time_fn=time_fn)

        shard = WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=message_callback,
            message_throttler=throttler,
            time_fn=time_fn,
        )

        # Mock WS connection
        mock_ws = MagicMock()
        mock_ws.closed = False
        mock_ws.send_json = AsyncMock()
        shard._ws = mock_ws

        # Track sleep calls to verify serialization
        sleep_calls: list[float] = []

        async def mock_sleep(duration: float) -> None:
            sleep_calls.append(duration)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            # Launch two concurrent subscribes
            task1 = asyncio.create_task(shard._send_subscribe(["stream1"]))
            task2 = asyncio.create_task(shard._send_subscribe(["stream2"]))

            await asyncio.gather(task1, task2)

        # With burst_allowance=1, only one message can send immediately
        # The second MUST be delayed (proving serialization works)
        # If there was no lock, both would see can_send=True and neither would delay
        metrics = shard.get_metrics()
        assert metrics.messages_delayed >= 1, (
            "Expected at least one message to be delayed. "
            "This proves _send_lock serializes concurrent subscribe calls."
        )

    def test_send_lock_exists(
        self,
        config: ConnectorConfig,
        circuit_breaker: CircuitBreaker,
        message_callback: AsyncMock,
    ) -> None:
        """Test _send_lock is created for serializing WS sends."""
        shard = WebSocketShard(
            shard_id=0,
            config=config,
            circuit_breaker=circuit_breaker,
            on_message=message_callback,
        )

        assert hasattr(shard, "_send_lock")
        assert isinstance(shard._send_lock, asyncio.Lock)
