"""Tests for Binance stream manager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptoscreener.connectors.binance.stream_manager import BinanceStreamManager
from cryptoscreener.connectors.binance.types import (
    ConnectorConfig,
    RawMessage,
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
        mock_shard.add_subscriptions = AsyncMock(
            side_effect=lambda subs: subs
        )

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
