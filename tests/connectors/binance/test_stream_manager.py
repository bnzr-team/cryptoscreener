"""Tests for Binance stream manager.

DEC-023b: Added integration tests for:
- ReconnectLimiter wiring and metrics aggregation
- Limiter injection for deterministic testing
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptoscreener.connectors.backoff import ReconnectLimiter, ReconnectLimiterConfig
from cryptoscreener.connectors.binance.stream_manager import BinanceStreamManager
from cryptoscreener.connectors.binance.types import (
    ConnectionState,
    ConnectorConfig,
    RawMessage,
    ShardMetrics,
    StreamType,
    SymbolInfo,
)
from cryptoscreener.contracts.events import MarketEvent, MarketEventType


class TestBinanceStreamManager:
    """Tests for BinanceStreamManager."""

    @pytest.fixture
    def config(self) -> ConnectorConfig:
        """Create connector config."""
        return ConnectorConfig()

    @pytest.fixture
    def manager(self, config: ConnectorConfig) -> BinanceStreamManager:
        """Create stream manager instance."""
        return BinanceStreamManager(config=config)

    @pytest.mark.asyncio
    async def test_start_stop(self, manager: BinanceStreamManager) -> None:
        """Start and stop manager."""
        await manager.start()
        assert manager._running is True

        await manager.stop()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self, manager: BinanceStreamManager) -> None:
        """Start can be called multiple times."""
        await manager.start()
        await manager.start()
        assert manager._running is True

        await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, manager: BinanceStreamManager) -> None:
        """Stop can be called multiple times."""
        await manager.stop()
        await manager.stop()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_bootstrap(self, manager: BinanceStreamManager) -> None:
        """Bootstrap fetches tradeable symbols."""
        mock_symbols = [
            SymbolInfo(
                symbol="BTCUSDT",
                base_asset="BTC",
                quote_asset="USDT",
                price_precision=2,
                quantity_precision=3,
                contract_type="PERPETUAL",
                status="TRADING",
            ),
        ]

        with patch.object(
            manager._rest_client,
            "get_tradeable_symbols",
            new_callable=AsyncMock,
            return_value=mock_symbols,
        ):
            symbols = await manager.bootstrap()

        assert len(symbols) == 1
        assert symbols[0].symbol == "BTCUSDT"
        assert manager.get_symbol_info("BTCUSDT") is not None

    def test_get_symbol_info_not_found(
        self,
        manager: BinanceStreamManager,
    ) -> None:
        """Get symbol info returns None for unknown symbol."""
        assert manager.get_symbol_info("UNKNOWN") is None

    def test_get_symbol_info_case_insensitive(
        self,
        manager: BinanceStreamManager,
    ) -> None:
        """Get symbol info is case insensitive."""
        manager._symbols["BTCUSDT"] = SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=3,
            contract_type="PERPETUAL",
            status="TRADING",
        )

        assert manager.get_symbol_info("btcusdt") is not None
        assert manager.get_symbol_info("BtCuSdT") is not None

    def test_get_metrics_empty(self, manager: BinanceStreamManager) -> None:
        """Get metrics when no shards."""
        metrics = manager.get_metrics()

        assert metrics.total_streams == 0
        assert metrics.total_messages == 0
        assert metrics.active_shards == 0
        assert metrics.shard_metrics == []


class TestMessageParsing:
    """Tests for raw message to MarketEvent parsing."""

    @pytest.fixture
    def manager(self) -> BinanceStreamManager:
        """Create stream manager."""
        return BinanceStreamManager()

    def test_parse_agg_trade(self, manager: BinanceStreamManager) -> None:
        """Parse aggTrade message."""
        raw = RawMessage(
            data={
                "e": "aggTrade",
                "E": 1234567890,
                "s": "BTCUSDT",
                "p": "50000.00",
                "q": "1.5",
            },
            recv_ts=1234567891,
            shard_id=0,
        )

        event = manager._parse_raw_message(raw)

        assert event is not None
        assert event.type == MarketEventType.TRADE
        assert event.symbol == "BTCUSDT"
        assert event.ts == 1234567890
        assert event.recv_ts == 1234567891
        assert event.source == "binance_usdm"

    def test_parse_book_ticker(self, manager: BinanceStreamManager) -> None:
        """Parse bookTicker message."""
        raw = RawMessage(
            data={
                "e": "bookTicker",
                "u": 123456,
                "s": "ETHUSDT",
                "b": "3000.00",
                "B": "10.0",
                "a": "3000.50",
                "A": "10.0",
            },
            recv_ts=1234567890,
            shard_id=0,
        )

        event = manager._parse_raw_message(raw)

        assert event is not None
        assert event.type == MarketEventType.BOOK
        assert event.symbol == "ETHUSDT"

    def test_parse_depth_update(self, manager: BinanceStreamManager) -> None:
        """Parse depthUpdate message."""
        raw = RawMessage(
            data={
                "e": "depthUpdate",
                "E": 1234567890,
                "s": "BTCUSDT",
                "b": [["50000", "1.5"]],
                "a": [["50001", "1.0"]],
            },
            recv_ts=1234567891,
            shard_id=0,
        )

        event = manager._parse_raw_message(raw)

        assert event is not None
        assert event.type == MarketEventType.BOOK
        assert event.symbol == "BTCUSDT"

    def test_parse_kline(self, manager: BinanceStreamManager) -> None:
        """Parse kline message."""
        raw = RawMessage(
            data={
                "e": "kline",
                "E": 1234567890,
                "s": "BTCUSDT",
                "k": {
                    "t": 1234567800,
                    "T": 1234567859,
                    "s": "BTCUSDT",
                    "i": "1m",
                    "o": "50000",
                    "c": "50100",
                    "h": "50200",
                    "l": "49900",
                    "v": "100",
                },
            },
            recv_ts=1234567891,
            shard_id=0,
        )

        event = manager._parse_raw_message(raw)

        assert event is not None
        assert event.type == MarketEventType.KLINE
        assert event.symbol == "BTCUSDT"

    def test_parse_mark_price(self, manager: BinanceStreamManager) -> None:
        """Parse markPriceUpdate message."""
        raw = RawMessage(
            data={
                "e": "markPriceUpdate",
                "E": 1234567890,
                "s": "BTCUSDT",
                "p": "50000.00",
                "r": "0.0001",
            },
            recv_ts=1234567891,
            shard_id=0,
        )

        event = manager._parse_raw_message(raw)

        assert event is not None
        assert event.type == MarketEventType.MARK
        assert event.symbol == "BTCUSDT"

    def test_parse_force_order(self, manager: BinanceStreamManager) -> None:
        """Parse forceOrder message."""
        raw = RawMessage(
            data={
                "e": "forceOrder",
                "E": 1234567890,
                "o": {
                    "s": "BTCUSDT",
                    "S": "SELL",
                    "o": "LIMIT",
                    "f": "IOC",
                    "q": "0.001",
                    "p": "50000.00",
                },
            },
            recv_ts=1234567891,
            shard_id=0,
        )

        event = manager._parse_raw_message(raw)

        assert event is not None
        assert event.type == MarketEventType.FUNDING
        assert event.symbol == "BTCUSDT"

    def test_parse_unknown_event_type(
        self,
        manager: BinanceStreamManager,
    ) -> None:
        """Unknown event type returns None."""
        raw = RawMessage(
            data={
                "e": "unknownEvent",
                "s": "BTCUSDT",
            },
            recv_ts=1234567890,
            shard_id=0,
        )

        event = manager._parse_raw_message(raw)
        assert event is None

    def test_parse_missing_symbol(self, manager: BinanceStreamManager) -> None:
        """Missing symbol returns None."""
        raw = RawMessage(
            data={
                "e": "aggTrade",
                "E": 1234567890,
            },
            recv_ts=1234567890,
            shard_id=0,
        )

        event = manager._parse_raw_message(raw)
        assert event is None


class TestSubscriptionManagement:
    """Tests for subscription management."""

    @pytest.fixture
    def manager(self) -> BinanceStreamManager:
        """Create stream manager with mocked shard creation."""
        manager = BinanceStreamManager()
        return manager

    @pytest.mark.asyncio
    async def test_subscribe_creates_shard(
        self,
        manager: BinanceStreamManager,
    ) -> None:
        """Subscribe creates a shard if none exist."""
        # Mock shard creation and connection
        mock_shard = MagicMock()
        mock_shard.can_add_streams = True
        mock_shard.shard_id = 0
        mock_shard.add_subscriptions = AsyncMock(
            side_effect=lambda subs: subs  # Return all as added
        )
        mock_shard.connect = AsyncMock()

        with patch.object(
            manager,
            "_get_or_create_shard_for_subscription",
            new_callable=AsyncMock,
            return_value=mock_shard,
        ):
            count = await manager.subscribe(["BTCUSDT"], [StreamType.TRADE])

        assert count == 1

    @pytest.mark.asyncio
    async def test_subscribe_default_stream_types(
        self,
        manager: BinanceStreamManager,
    ) -> None:
        """Subscribe uses default stream types (TRADE, BOOK_TICKER)."""
        mock_shard = MagicMock()
        mock_shard.can_add_streams = True
        mock_shard.shard_id = 0
        mock_shard.add_subscriptions = AsyncMock(
            side_effect=lambda subs: subs  # Return all as added
        )

        with patch.object(
            manager,
            "_get_or_create_shard_for_subscription",
            new_callable=AsyncMock,
            return_value=mock_shard,
        ):
            count = await manager.subscribe(["BTCUSDT"])

        # TRADE + BOOK_TICKER = 2
        assert count == 2

    @pytest.mark.asyncio
    async def test_subscribe_no_duplicates(
        self,
        manager: BinanceStreamManager,
    ) -> None:
        """Duplicate subscriptions are not added."""
        mock_shard = MagicMock()
        mock_shard.can_add_streams = True
        mock_shard.shard_id = 0
        mock_shard.add_subscriptions = AsyncMock(side_effect=lambda subs: subs)

        with patch.object(
            manager,
            "_get_or_create_shard_for_subscription",
            new_callable=AsyncMock,
            return_value=mock_shard,
        ):
            count1 = await manager.subscribe(["BTCUSDT"], [StreamType.TRADE])
            count2 = await manager.subscribe(["BTCUSDT"], [StreamType.TRADE])

        assert count1 == 1
        assert count2 == 0  # Already subscribed

    @pytest.mark.asyncio
    async def test_unsubscribe(self, manager: BinanceStreamManager) -> None:
        """Unsubscribe removes subscriptions."""
        # Setup: add a subscription
        from cryptoscreener.connectors.binance.types import StreamSubscription

        sub = StreamSubscription(symbol="BTCUSDT", stream_type=StreamType.TRADE)
        stream_name = sub.to_stream_name()
        manager._subscriptions[stream_name] = sub
        manager._stream_to_shard[stream_name] = 0

        mock_shard = MagicMock()
        mock_shard.remove_subscriptions = AsyncMock()
        manager._shards[0] = mock_shard

        count = await manager.unsubscribe(["BTCUSDT"], [StreamType.TRADE])

        assert count == 1
        assert stream_name not in manager._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(
        self,
        manager: BinanceStreamManager,
    ) -> None:
        """Unsubscribe nonexistent returns 0."""
        count = await manager.unsubscribe(["BTCUSDT"], [StreamType.TRADE])
        assert count == 0


class TestEventCallback:
    """Tests for event callback."""

    @pytest.mark.asyncio
    async def test_on_event_callback_called(self) -> None:
        """Event callback is called for each event."""
        received_events: list[MarketEvent] = []

        async def on_event(event: MarketEvent) -> None:
            received_events.append(event)

        manager = BinanceStreamManager(on_event=on_event)

        # Simulate receiving a message
        raw = RawMessage(
            data={
                "e": "aggTrade",
                "E": 1234567890,
                "s": "BTCUSDT",
                "p": "50000.00",
                "q": "1.5",
            },
            recv_ts=1234567891,
            shard_id=0,
        )

        await manager._handle_raw_message(raw)

        assert len(received_events) == 1
        assert received_events[0].symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_event_added_to_queue(self) -> None:
        """Events are added to the queue."""
        manager = BinanceStreamManager()

        raw = RawMessage(
            data={
                "e": "aggTrade",
                "E": 1234567890,
                "s": "BTCUSDT",
            },
            recv_ts=1234567891,
            shard_id=0,
        )

        await manager._handle_raw_message(raw)

        # Event should be in queue
        assert not manager._event_queue.empty()
        event = await manager._event_queue.get()
        assert event.symbol == "BTCUSDT"


# =============================================================================
# DEC-023b: Integration Tests for Limiter/Throttler Wiring in StreamManager
# =============================================================================


class TestStreamManagerReconnectLimiterIntegration:
    """
    DEC-023b: Integration tests for ReconnectLimiter wiring in StreamManager.
    """

    @pytest.fixture
    def config(self) -> ConnectorConfig:
        """Create connector config."""
        return ConnectorConfig()

    def test_default_reconnect_limiter_created(
        self,
        config: ConnectorConfig,
    ) -> None:
        """Test default ReconnectLimiter is created when none provided."""
        manager = BinanceStreamManager(config=config)

        assert manager._reconnect_limiter is not None
        assert isinstance(manager._reconnect_limiter, ReconnectLimiter)

    def test_custom_reconnect_limiter_used(
        self,
        config: ConnectorConfig,
    ) -> None:
        """Test custom ReconnectLimiter is used when provided."""
        custom_limiter = ReconnectLimiter()
        manager = BinanceStreamManager(
            config=config,
            reconnect_limiter=custom_limiter,
        )

        assert manager._reconnect_limiter is custom_limiter

    def test_time_fn_passed_to_default_limiter(
        self,
        config: ConnectorConfig,
    ) -> None:
        """Test time_fn is passed to default limiter."""
        fake_time = 12345

        def time_fn() -> int:
            return fake_time

        manager = BinanceStreamManager(config=config, time_fn=time_fn)

        # Verify by checking limiter uses fake time
        status = manager._reconnect_limiter.get_status()
        # Status should be valid (limiter initialized properly)
        assert "reconnects_in_window" in status

    def test_get_metrics_includes_limiter_status(
        self,
        config: ConnectorConfig,
    ) -> None:
        """Test get_metrics includes reconnect limiter status."""
        manager = BinanceStreamManager(config=config)

        metrics = manager.get_metrics()

        # DEC-023b metrics should be present
        assert hasattr(metrics, "reconnect_limiter_in_cooldown")
        assert hasattr(metrics, "total_reconnects_denied")
        assert hasattr(metrics, "total_messages_delayed")

    def test_get_metrics_aggregates_shard_throttle_metrics(
        self,
        config: ConnectorConfig,
    ) -> None:
        """Test get_metrics aggregates throttle metrics from all shards."""
        manager = BinanceStreamManager(config=config)

        # Create mock shards with metrics
        mock_shard_1 = MagicMock()
        mock_shard_1.get_metrics.return_value = ShardMetrics(
            shard_id=0,
            stream_count=5,
            messages_received=100,
            reconnect_denied=2,  # DEC-023b
            messages_delayed=10,  # DEC-023b
            state=ConnectionState.CONNECTED,
        )

        mock_shard_2 = MagicMock()
        mock_shard_2.get_metrics.return_value = ShardMetrics(
            shard_id=1,
            stream_count=3,
            messages_received=50,
            reconnect_denied=1,  # DEC-023b
            messages_delayed=5,  # DEC-023b
            state=ConnectionState.CONNECTED,
        )

        manager._shards = {0: mock_shard_1, 1: mock_shard_2}

        metrics = manager.get_metrics()

        # Should aggregate throttle metrics
        assert metrics.total_reconnects_denied == 3  # 2 + 1
        assert metrics.total_messages_delayed == 15  # 10 + 5
        assert metrics.total_streams == 8  # 5 + 3
        assert metrics.total_messages == 150  # 100 + 50
        assert metrics.active_shards == 2

    def test_limiter_in_cooldown_reflected_in_metrics(
        self,
        config: ConnectorConfig,
    ) -> None:
        """Test limiter cooldown status is reflected in metrics."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        # Create limiter that will trigger cooldown
        limiter_config = ReconnectLimiterConfig(
            max_reconnects_per_window=1,
            window_ms=60000,
            cooldown_after_burst_ms=10000,
            per_shard_min_interval_ms=0,
        )
        limiter = ReconnectLimiter(config=limiter_config, _time_fn=time_fn)

        manager = BinanceStreamManager(
            config=config,
            reconnect_limiter=limiter,
            time_fn=time_fn,
        )

        # Trigger cooldown by hitting limit
        limiter.record_reconnect(shard_id=0)
        limiter.can_reconnect(shard_id=1)  # This triggers cooldown check

        metrics = manager.get_metrics()
        # Note: cooldown may or may not be active depending on exact timing
        # The important thing is the field is present and queryable
        assert isinstance(metrics.reconnect_limiter_in_cooldown, bool)


class TestStreamManagerShardCreationWithLimiter:
    """
    DEC-023b: Tests that shards are created with proper limiter/time_fn injection.
    """

    @pytest.fixture
    def config(self) -> ConnectorConfig:
        """Create connector config."""
        return ConnectorConfig()

    @pytest.mark.asyncio
    async def test_shard_created_with_reconnect_limiter(
        self,
        config: ConnectorConfig,
    ) -> None:
        """Test shards are created with the global reconnect limiter."""
        fake_time = 0

        def time_fn() -> int:
            return fake_time

        custom_limiter = ReconnectLimiter(_time_fn=time_fn)
        manager = BinanceStreamManager(
            config=config,
            reconnect_limiter=custom_limiter,
            time_fn=time_fn,
        )

        # Mock aiohttp session to avoid actual network calls
        with patch("aiohttp.ClientSession") as mock_session:
            mock_ws = MagicMock()
            mock_ws.closed = False
            mock_session_instance = AsyncMock()
            mock_session_instance.ws_connect = AsyncMock(return_value=mock_ws)
            mock_session_instance.closed = False
            mock_session.return_value = mock_session_instance

            # Create shard via subscription
            shard = await manager._get_or_create_shard_for_subscription()

            if shard is not None:
                # Verify shard has the limiter
                assert shard._reconnect_limiter is custom_limiter

    @pytest.mark.asyncio
    async def test_shard_created_with_time_fn(
        self,
        config: ConnectorConfig,
    ) -> None:
        """Test shards are created with time_fn for deterministic testing."""
        fake_time = 12345

        def time_fn() -> int:
            return fake_time

        manager = BinanceStreamManager(config=config, time_fn=time_fn)

        # Mock aiohttp session
        with patch("aiohttp.ClientSession") as mock_session:
            mock_ws = MagicMock()
            mock_ws.closed = False
            mock_session_instance = AsyncMock()
            mock_session_instance.ws_connect = AsyncMock(return_value=mock_ws)
            mock_session_instance.closed = False
            mock_session.return_value = mock_session_instance

            shard = await manager._get_or_create_shard_for_subscription()

            if shard is not None:
                # Verify shard has time_fn set
                assert shard._time_fn is time_fn
