"""Tests for FeatureEngine."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptoscreener.contracts.events import (
    FeatureSnapshot,
    MarketEvent,
    MarketEventType,
)
from cryptoscreener.features.engine import FeatureEngine, FeatureEngineConfig


class TestFeatureEngine:
    """Tests for FeatureEngine."""

    @pytest.fixture
    def engine(self) -> FeatureEngine:
        """Create a fresh FeatureEngine."""
        return FeatureEngine()

    @pytest.fixture
    def config(self) -> FeatureEngineConfig:
        """Create test config."""
        return FeatureEngineConfig(
            snapshot_cadence_ms=100,
            max_symbols=10,
        )

    def test_default_config(self, engine: FeatureEngine) -> None:
        """Default config values."""
        assert engine.config.snapshot_cadence_ms == 1000
        assert engine.config.max_symbols == 500

    def test_custom_config(self, config: FeatureEngineConfig) -> None:
        """Custom config is applied."""
        engine = FeatureEngine(config=config)
        assert engine.config.snapshot_cadence_ms == 100
        assert engine.config.max_symbols == 10

    def test_symbols_initially_empty(self, engine: FeatureEngine) -> None:
        """No symbols tracked initially."""
        assert engine.symbols == []

    @pytest.mark.asyncio
    async def test_process_event_creates_state(self, engine: FeatureEngine) -> None:
        """Processing event creates state for symbol."""
        event = MarketEvent(
            ts=1000,
            source="binance_usdm",
            symbol="BTCUSDT",
            type=MarketEventType.TRADE,
            payload={"p": "50000.0", "q": "1.0", "m": False},
            recv_ts=1001,
        )

        await engine.process_event(event)

        assert "BTCUSDT" in engine.symbols
        assert engine.get_state("BTCUSDT") is not None

    @pytest.mark.asyncio
    async def test_process_event_updates_state(self, engine: FeatureEngine) -> None:
        """Processing event updates existing state."""
        event1 = MarketEvent(
            ts=1000,
            source="binance_usdm",
            symbol="BTCUSDT",
            type=MarketEventType.BOOK,
            payload={"b": "50000.0", "B": "1.0", "a": "50001.0", "A": "1.0"},
            recv_ts=1001,
        )

        event2 = MarketEvent(
            ts=2000,
            source="binance_usdm",
            symbol="BTCUSDT",
            type=MarketEventType.TRADE,
            payload={"p": "50000.5", "q": "0.5", "m": True},
            recv_ts=2001,
        )

        await engine.process_event(event1)
        await engine.process_event(event2)

        state = engine.get_state("BTCUSDT")
        assert state is not None
        assert state.last_book is not None
        assert state.last_trade_ts == 2000

    @pytest.mark.asyncio
    async def test_max_symbols_limit(self, config: FeatureEngineConfig) -> None:
        """Max symbols limit is enforced."""
        engine = FeatureEngine(config=config)

        # Add max_symbols
        for i in range(config.max_symbols):
            event = MarketEvent(
                ts=1000,
                source="binance_usdm",
                symbol=f"SYM{i}USDT",
                type=MarketEventType.TRADE,
                payload={"p": "100.0", "q": "1.0", "m": False},
                recv_ts=1001,
            )
            await engine.process_event(event)

        assert len(engine.symbols) == config.max_symbols

        # Try to add one more
        event = MarketEvent(
            ts=1000,
            source="binance_usdm",
            symbol="EXTRAUSDT",
            type=MarketEventType.TRADE,
            payload={"p": "100.0", "q": "1.0", "m": False},
            recv_ts=1001,
        )
        await engine.process_event(event)

        # Still at max
        assert len(engine.symbols) == config.max_symbols
        assert "EXTRAUSDT" not in engine.symbols

    def test_compute_snapshot(self, engine: FeatureEngine) -> None:
        """Compute snapshot for symbol."""
        # Manually create state
        state = engine._get_or_create_state("BTCUSDT")
        state.last_book = MagicMock()
        state.last_book.bid_price = 50000.0
        state.last_book.bid_qty = 10.0
        state.last_book.ask_price = 50010.0
        state.last_book.ask_qty = 10.0
        state.last_book_ts = 1000

        snapshot = engine.compute_snapshot("BTCUSDT", 1000)

        assert snapshot is not None
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.ts == 1000
        assert snapshot.features.mid == 50005.0

    def test_compute_snapshot_unknown_symbol(self, engine: FeatureEngine) -> None:
        """Compute snapshot returns None for unknown symbol."""
        assert engine.compute_snapshot("UNKNOWN", 1000) is None

    @pytest.mark.asyncio
    async def test_emit_snapshots(self, engine: FeatureEngine) -> None:
        """Emit snapshots for all tracked symbols."""
        # Add events for two symbols
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            event = MarketEvent(
                ts=1000,
                source="binance_usdm",
                symbol=symbol,
                type=MarketEventType.BOOK,
                payload={"b": "1000.0", "B": "1.0", "a": "1001.0", "A": "1.0"},
                recv_ts=1001,
            )
            await engine.process_event(event)

        # First emission
        snapshots = await engine.emit_snapshots(2000)
        assert len(snapshots) == 2

        # Second emission within cadence - no new snapshots
        snapshots = await engine.emit_snapshots(2500)
        assert len(snapshots) == 0

        # After cadence - new snapshots
        snapshots = await engine.emit_snapshots(3001)
        assert len(snapshots) == 2

    @pytest.mark.asyncio
    async def test_callback_invoked(self, engine: FeatureEngine) -> None:
        """Callback is invoked on snapshot emission."""
        callback = MagicMock()
        engine.on_snapshot(callback)

        event = MarketEvent(
            ts=1000,
            source="binance_usdm",
            symbol="BTCUSDT",
            type=MarketEventType.BOOK,
            payload={"b": "50000.0", "B": "1.0", "a": "50001.0", "A": "1.0"},
            recv_ts=1001,
        )
        await engine.process_event(event)

        await engine.emit_snapshots(2000)

        callback.assert_called_once()
        snapshot = callback.call_args[0][0]
        assert isinstance(snapshot, FeatureSnapshot)
        assert snapshot.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_async_callback(self, engine: FeatureEngine) -> None:
        """Async callback is awaited."""
        callback = AsyncMock()
        engine.on_snapshot(callback)

        event = MarketEvent(
            ts=1000,
            source="binance_usdm",
            symbol="BTCUSDT",
            type=MarketEventType.BOOK,
            payload={"b": "50000.0", "B": "1.0", "a": "50001.0", "A": "1.0"},
            recv_ts=1001,
        )
        await engine.process_event(event)

        await engine.emit_snapshots(2000)

        callback.assert_awaited_once()

    def test_clear(self, engine: FeatureEngine) -> None:
        """Clear removes all state."""
        engine._get_or_create_state("BTCUSDT")
        engine._get_or_create_state("ETHUSDT")

        assert len(engine.symbols) == 2

        engine.clear()

        assert len(engine.symbols) == 0

    def test_remove_symbol(self, engine: FeatureEngine) -> None:
        """Remove specific symbol."""
        engine._get_or_create_state("BTCUSDT")
        engine._get_or_create_state("ETHUSDT")

        assert engine.remove_symbol("BTCUSDT") is True
        assert "BTCUSDT" not in engine.symbols
        assert "ETHUSDT" in engine.symbols

        # Remove non-existent
        assert engine.remove_symbol("UNKNOWN") is False

    @pytest.mark.asyncio
    async def test_start_stop(self, config: FeatureEngineConfig) -> None:
        """Start and stop emission loop."""
        engine = FeatureEngine(config=config)

        assert not engine.running

        await engine.start(lambda: 1000)
        assert engine.running

        await asyncio.sleep(0.05)

        await engine.stop()
        assert not engine.running

    @pytest.mark.asyncio
    async def test_double_start(self, config: FeatureEngineConfig) -> None:
        """Starting twice is idempotent."""
        engine = FeatureEngine(config=config)

        await engine.start(lambda: 1000)
        await engine.start(lambda: 1000)  # Should not error

        assert engine.running

        await engine.stop()


class TestFeatureEngineIntegration:
    """Integration tests for FeatureEngine."""

    @pytest.mark.asyncio
    async def test_full_flow(self) -> None:
        """Full flow: events -> snapshots."""
        config = FeatureEngineConfig(snapshot_cadence_ms=50)
        engine = FeatureEngine(config=config)

        received_snapshots: list[FeatureSnapshot] = []
        engine.on_snapshot(lambda s: received_snapshots.append(s))

        # Process several events
        ts = 1000
        for i in range(10):
            # Book event
            book_event = MarketEvent(
                ts=ts + i * 10,
                source="binance_usdm",
                symbol="BTCUSDT",
                type=MarketEventType.BOOK,
                payload={"b": "50000.0", "B": "10.0", "a": "50010.0", "A": "10.0"},
                recv_ts=ts + i * 10,
            )
            await engine.process_event(book_event)

            # Trade event
            trade_event = MarketEvent(
                ts=ts + i * 10 + 5,
                source="binance_usdm",
                symbol="BTCUSDT",
                type=MarketEventType.TRADE,
                payload={"p": "50005.0", "q": "0.1", "m": i % 2 == 0},
                recv_ts=ts + i * 10 + 5,
            )
            await engine.process_event(trade_event)

        # Emit snapshots
        await engine.emit_snapshots(ts + 100)

        assert len(received_snapshots) == 1
        snapshot = received_snapshots[0]
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.features.mid == 50005.0
        assert snapshot.features.spread_bps > 0

    @pytest.mark.asyncio
    async def test_multiple_symbols(self) -> None:
        """Process events for multiple symbols."""
        engine = FeatureEngine()

        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        for symbol in symbols:
            event = MarketEvent(
                ts=1000,
                source="binance_usdm",
                symbol=symbol,
                type=MarketEventType.BOOK,
                payload={"b": "1000.0", "B": "1.0", "a": "1001.0", "A": "1.0"},
                recv_ts=1001,
            )
            await engine.process_event(event)

        assert len(engine.symbols) == 3
        for symbol in symbols:
            assert symbol in engine.symbols

        snapshots = await engine.emit_snapshots(2000)
        assert len(snapshots) == 3
