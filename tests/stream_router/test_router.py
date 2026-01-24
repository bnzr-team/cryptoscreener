"""Tests for StreamRouter."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from cryptoscreener.contracts.events import MarketEvent, MarketEventType
from cryptoscreener.features.engine import FeatureEngine
from cryptoscreener.stream_router.router import StreamRouter, StreamRouterConfig


class TestStreamRouter:
    """Tests for StreamRouter."""

    @pytest.fixture
    def engine(self) -> FeatureEngine:
        """Create a feature engine."""
        return FeatureEngine()

    @pytest.fixture
    def router(self, engine: FeatureEngine) -> StreamRouter:
        """Create a stream router with default config."""
        return StreamRouter(engine)

    @pytest.fixture
    def make_event(self) -> Callable[..., MarketEvent]:
        """Factory for creating MarketEvents."""

        def _make(
            symbol: str = "BTCUSDT",
            ts: int = 1000,
            recv_ts: int = 1050,
            event_type: MarketEventType = MarketEventType.TRADE,
        ) -> MarketEvent:
            return MarketEvent(
                ts=ts,
                source="binance_usdm",
                symbol=symbol,
                type=event_type,
                payload={"p": "50000.0", "q": "1.0", "m": False},
                recv_ts=recv_ts,
            )

        return _make

    def test_default_config(self, router: StreamRouter) -> None:
        """Default configuration values."""
        assert router.config.stale_threshold_ms == 5000
        assert router.config.late_threshold_ms == 1000
        assert router.config.drop_stale is False

    def test_custom_config(self, engine: FeatureEngine) -> None:
        """Custom configuration is applied."""
        config = StreamRouterConfig(
            stale_threshold_ms=1000,
            drop_stale=True,
        )
        router = StreamRouter(engine, config=config)

        assert router.config.stale_threshold_ms == 1000
        assert router.config.drop_stale is True

    @pytest.mark.asyncio
    async def test_route_basic(
        self,
        router: StreamRouter,
        make_event: MagicMock,
    ) -> None:
        """Basic event routing."""
        event = make_event()

        with patch.object(router, "_get_current_ts", return_value=1100):
            result = await router.route(event)

        assert result is True
        assert router.metrics.events_received == 1
        assert router.metrics.events_routed == 1

    @pytest.mark.asyncio
    async def test_route_updates_engine(
        self,
        router: StreamRouter,
        make_event: MagicMock,
    ) -> None:
        """Routing updates feature engine state."""
        event = make_event()

        with patch.object(router, "_get_current_ts", return_value=1100):
            await router.route(event)

        assert "BTCUSDT" in router.feature_engine.symbols

    @pytest.mark.asyncio
    async def test_route_symbol_filter(
        self,
        engine: FeatureEngine,
        make_event: MagicMock,
    ) -> None:
        """Symbol filter blocks untracked symbols."""
        config = StreamRouterConfig(symbol_filter={"BTCUSDT"})
        router = StreamRouter(engine, config=config)

        btc_event = make_event(symbol="BTCUSDT")
        eth_event = make_event(symbol="ETHUSDT")

        with patch.object(router, "_get_current_ts", return_value=1100):
            result_btc = await router.route(btc_event)
            result_eth = await router.route(eth_event)

        assert result_btc is True
        assert result_eth is False
        assert router.metrics.unknown_symbols == 1

    @pytest.mark.asyncio
    async def test_route_stale_event(
        self,
        router: StreamRouter,
        make_event: MagicMock,
    ) -> None:
        """Stale events are detected but not dropped by default."""
        # Event from 10 seconds ago (stale threshold is 5s)
        event = make_event(ts=1000, recv_ts=1050)

        with patch.object(router, "_get_current_ts", return_value=11000):
            result = await router.route(event)

        assert result is True  # Not dropped
        assert router.metrics.stale_events == 1

    @pytest.mark.asyncio
    async def test_route_stale_event_dropped(
        self,
        engine: FeatureEngine,
        make_event: MagicMock,
    ) -> None:
        """Stale events are dropped when configured."""
        config = StreamRouterConfig(drop_stale=True, stale_threshold_ms=5000)
        router = StreamRouter(engine, config=config)

        event = make_event(ts=1000, recv_ts=1050)

        with patch.object(router, "_get_current_ts", return_value=11000):
            result = await router.route(event)

        assert result is False  # Dropped
        assert router.metrics.events_dropped == 1
        assert router.metrics.stale_events == 1

    @pytest.mark.asyncio
    async def test_route_late_event(
        self,
        router: StreamRouter,
        make_event: MagicMock,
    ) -> None:
        """Late (out of order) events are detected."""
        event1 = make_event(ts=2000)
        event2 = make_event(ts=1000)  # Older than event1

        with patch.object(router, "_get_current_ts", return_value=3000):
            await router.route(event1)
            await router.route(event2)

        assert router.metrics.late_events == 1

    @pytest.mark.asyncio
    async def test_route_latency_tracking(
        self,
        router: StreamRouter,
        make_event: MagicMock,
    ) -> None:
        """Latency is tracked correctly."""
        event = make_event(ts=1000, recv_ts=1150)  # 150ms latency

        with patch.object(router, "_get_current_ts", return_value=1200):
            await router.route(event)

        assert router.metrics.latency_samples == 1
        assert router.metrics.avg_latency_ms == 150.0

    @pytest.mark.asyncio
    async def test_route_batch(
        self,
        router: StreamRouter,
        make_event: MagicMock,
    ) -> None:
        """Batch routing."""
        events = [
            make_event(symbol="BTCUSDT", ts=1000),
            make_event(symbol="ETHUSDT", ts=1010),
            make_event(symbol="BNBUSDT", ts=1020),
        ]

        with patch.object(router, "_get_current_ts", return_value=1100):
            routed = await router.route_batch(events)

        assert routed == 3
        assert router.metrics.events_received == 3

    @pytest.mark.asyncio
    async def test_callback_invoked(
        self,
        router: StreamRouter,
        make_event: MagicMock,
    ) -> None:
        """Callbacks are invoked after routing."""
        callback = MagicMock()
        router.on_event(callback)

        event = make_event()

        with patch.object(router, "_get_current_ts", return_value=1100):
            await router.route(event)

        callback.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_callback_error_handled(
        self,
        router: StreamRouter,
        make_event: MagicMock,
    ) -> None:
        """Callback errors don't break routing."""

        def bad_callback(e: MarketEvent) -> None:
            raise ValueError("Callback error")

        router.on_event(bad_callback)
        event = make_event()

        with patch.object(router, "_get_current_ts", return_value=1100):
            result = await router.route(event)

        # Should still succeed
        assert result is True
        assert router.metrics.events_routed == 1

    def test_set_symbol_filter(self, router: StreamRouter) -> None:
        """Set symbol filter."""
        router.set_symbol_filter({"BTCUSDT", "ETHUSDT"})

        assert router.config.symbol_filter == {"BTCUSDT", "ETHUSDT"}

    def test_add_symbols(self, router: StreamRouter) -> None:
        """Add symbols to filter."""
        router.set_symbol_filter({"BTCUSDT"})
        router.add_symbols({"ETHUSDT", "BNBUSDT"})

        assert router.config.symbol_filter == {"BTCUSDT", "ETHUSDT", "BNBUSDT"}

    def test_remove_symbols(self, router: StreamRouter) -> None:
        """Remove symbols from filter."""
        router.set_symbol_filter({"BTCUSDT", "ETHUSDT", "BNBUSDT"})
        router.remove_symbols({"ETHUSDT"})

        assert router.config.symbol_filter == {"BTCUSDT", "BNBUSDT"}

    def test_reset_metrics(self, router: StreamRouter) -> None:
        """Reset metrics."""
        router.metrics.record_event("BTCUSDT", latency_ms=100)
        router.reset_metrics()

        assert router.metrics.events_received == 0

    def test_reset_timestamps(self, router: StreamRouter) -> None:
        """Reset timestamp tracking."""
        router._last_ts["BTCUSDT"] = 1000
        router.reset_timestamps()

        assert len(router._last_ts) == 0

    @pytest.mark.asyncio
    async def test_get_symbol_stats(
        self,
        router: StreamRouter,
        make_event: MagicMock,
    ) -> None:
        """Get per-symbol statistics."""
        event = make_event(symbol="BTCUSDT", ts=1000)

        with patch.object(router, "_get_current_ts", return_value=1100):
            await router.route(event)

        stats = router.get_symbol_stats("BTCUSDT")
        assert stats["event_count"] == 1
        assert stats["last_ts"] == 1000


class TestStreamRouterIntegration:
    """Integration tests for StreamRouter."""

    @pytest.mark.asyncio
    async def test_full_flow(self) -> None:
        """Full flow: events through router to engine."""
        engine = FeatureEngine()
        router = StreamRouter(engine)

        received_events: list[MarketEvent] = []
        router.on_event(lambda e: received_events.append(e))

        # Create events
        events = []
        for i in range(10):
            events.append(
                MarketEvent(
                    ts=1000 + i * 10,
                    source="binance_usdm",
                    symbol="BTCUSDT",
                    type=MarketEventType.BOOK,
                    payload={"b": "50000.0", "B": "1.0", "a": "50001.0", "A": "1.0"},
                    recv_ts=1050 + i * 10,
                )
            )

        with patch.object(router, "_get_current_ts", return_value=1200):
            routed = await router.route_batch(events)

        assert routed == 10
        assert len(received_events) == 10
        assert "BTCUSDT" in engine.symbols
        assert router.metrics.events_routed == 10

    @pytest.mark.asyncio
    async def test_multi_symbol_routing(self) -> None:
        """Route events for multiple symbols."""
        engine = FeatureEngine()
        router = StreamRouter(engine)

        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        with patch.object(router, "_get_current_ts", return_value=1200):
            for i, symbol in enumerate(symbols):
                event = MarketEvent(
                    ts=1000 + i * 10,
                    source="binance_usdm",
                    symbol=symbol,
                    type=MarketEventType.TRADE,
                    payload={"p": "100.0", "q": "1.0", "m": False},
                    recv_ts=1050 + i * 10,
                )
                await router.route(event)

        assert len(engine.symbols) == 3
        for symbol in symbols:
            assert symbol in engine.symbols
